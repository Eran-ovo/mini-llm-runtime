"""基于自有 ModelRunner 和 LayerKVCache 的最小生成控制循环。"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import torch

from .kv_cache import ContiguousKVCache, LayerKVCache
from .qwen_model_runner import QwenPrefillRunner


@dataclass(frozen=True)
class GreedyGenerationOutput:
    """单请求 greedy generation 的结果与执行计数。"""

    generated_token_ids: torch.Tensor
    stopped_by_eos: bool
    prefill_tokens: int
    decode_steps: int
    cache_length: int
    # 生成开始前，该请求在当时 Cache/pool 状态下最多可达到的 token 数。
    cache_capacity: int
    # 仅供 correctness/debug 使用；默认不保存，避免 logits 长期占用显存。
    step_logits: tuple[torch.Tensor, ...] | None = None


@torch.inference_mode()
def greedy_generate(
    runner: QwenPrefillRunner,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: int | Collection[int] | None = None,
    cache: LayerKVCache | None = None,
    return_step_logits: bool = False,
) -> GreedyGenerationOutput:
    """执行一次 Prefill 和若干次单 token Decode。

    当前阶段故意限制为 batch=1、无 padding。最后一个生成 token 不会再执行
    Decode，因此正常达到 max_new_tokens 时，Cache 不包含最后一个输出 token。
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError(
            "当前 greedy_generate 只支持 batch=1 的非空 [1, prompt_length]"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens 必须 > 0")

    prompt_length = input_ids.shape[1]
    # 第一个 token 来自 Prefill；生成 N 个 token 最多只需 N-1 次 Decode。
    required_capacity = prompt_length + max_new_tokens - 1
    if required_capacity > runner.config.max_position_embeddings:
        raise ValueError(
            "prompt_length + max_new_tokens - 1 超过 max_position_embeddings"
        )

    eos_ids = _normalize_eos_token_ids(eos_token_ids, runner.config.vocab_size)
    if cache is None:
        cache = ContiguousKVCache(
            num_layers=runner.config.num_hidden_layers,
            batch_size=1,
            num_kv_heads=runner.config.num_key_value_heads,
            capacity=required_capacity,
            head_dim=runner.config.head_dim,
            dtype=runner.weights.embedding.dtype,
            device=runner.weights.embedding.device,
        )
    if cache.length != 0 or cache.pending is not None:
        raise ValueError("greedy_generate 只接受空闲且 length=0 的 KV Cache")
    # 对连续 Cache，这是固定 capacity；对 Paged Cache，这是当前请求使用
    # 最后一块余量和 pool 全部空闲块时可达到的容量快照。
    cache_capacity = cache.length + cache.available_token_capacity
    if cache_capacity < required_capacity:
        # 必须在 Prefill 修改 Cache 之前失败，避免生成到中途才发现 block 不足。
        raise RuntimeError(
            f"KV Cache capacity={cache_capacity} 小于生成所需的 "
            f"required_capacity={required_capacity}"
        )

    step = runner.prefill(input_ids, cache=cache)
    generated: list[torch.Tensor] = []
    captured_logits: list[torch.Tensor] | None = (
        [] if return_step_logits else None
    )
    stopped_by_eos = False
    decode_steps = 0

    for index in range(max_new_tokens):
        next_logits = step.logits[:, -1, :]
        if captured_logits is not None:
            # clone 避免 Prefill 最后一行的 view 持有完整 [prompt,vocab] storage。
            captured_logits.append(next_logits.detach().clone())
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        generated.append(next_token.detach())

        if eos_ids:
            # Python 控制流需要知道是否 EOS；这会产生一次 GPU→CPU 同步。
            stopped_by_eos = int(next_token.item()) in eos_ids
        if stopped_by_eos or index == max_new_tokens - 1:
            break

        step = runner.decode_one(next_token, cache=cache)
        decode_steps += 1

    return GreedyGenerationOutput(
        generated_token_ids=torch.cat(generated, dim=1),
        stopped_by_eos=stopped_by_eos,
        prefill_tokens=prompt_length,
        decode_steps=decode_steps,
        cache_length=cache.length,
        cache_capacity=cache_capacity,
        step_logits=(
            tuple(captured_logits) if captured_logits is not None else None
        ),
    )


def _normalize_eos_token_ids(
    eos_token_ids: int | Collection[int] | None, vocab_size: int
) -> frozenset[int]:
    if eos_token_ids is None:
        return frozenset()
    if isinstance(eos_token_ids, int):
        values = {eos_token_ids}
    else:
        values = {int(token_id) for token_id in eos_token_ids}
    invalid = sorted(token_id for token_id in values if not 0 <= token_id < vocab_size)
    if invalid:
        raise ValueError(f"EOS token ID 超出词表范围：{invalid}")
    return frozenset(values)
