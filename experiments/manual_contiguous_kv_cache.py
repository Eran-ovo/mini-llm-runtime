#!/usr/bin/env python3
"""为真实 Qwen 第 0 层手写连续 KV Cache，并对拍 Prefill 与一次 Decode。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from experiments.manual_qwen_attention import (
    apply_rope,
    build_default_rope,
    build_prefill_causal_mask,
    compare,
    repeat_kv,
    to_head_layout,
)


class ContiguousKVCache:
    """单层、固定容量的连续 KV Cache 教学实现。"""

    def __init__(
        self,
        *,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if min(batch_size, num_kv_heads, capacity, head_dim) <= 0:
            raise ValueError("Cache 各维度必须 > 0")
        shape = (batch_size, num_kv_heads, capacity, head_dim)
        self.key = torch.empty(shape, dtype=dtype, device=device)
        self.value = torch.empty(shape, dtype=dtype, device=device)
        self.length = 0

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """把 [B,Hkv,Snew,D] 原地追加到有效前缀；失败时不改变 Cache。"""
        if key.shape != value.shape:
            raise ValueError("新 K/V shape 必须相同")
        if key.ndim != 4:
            raise ValueError("新 K/V 必须是 [B,Hkv,Snew,D]")
        expected = (self.key.shape[0], self.key.shape[1], key.shape[2], self.key.shape[3])
        if tuple(key.shape) != expected:
            raise ValueError(f"新 K/V shape {tuple(key.shape)}，预期 {expected}")
        if key.dtype != self.key.dtype or value.dtype != self.value.dtype:
            raise ValueError("新 K/V dtype 必须与 Cache 相同")
        if key.device != self.key.device or value.device != self.value.device:
            raise ValueError("新 K/V device 必须与 Cache 相同")

        new_length = key.shape[2]
        end = self.length + new_length
        if end > self.capacity:
            raise RuntimeError(
                f"KV Cache 容量不足：length={self.length}, append={new_length}, "
                f"capacity={self.capacity}"
            )

        # 所有验证都完成后才写入，避免失败时只更新 K 或产生半写入状态。
        self.key[:, :, self.length : end, :].copy_(key)
        self.value[:, :, self.length : end, :].copy_(value)
        self.length = end

    def view(self) -> tuple[torch.Tensor, torch.Tensor]:
        """只暴露有效前缀，未初始化的 capacity 尾部绝不能参与 Attention。"""
        return self.key[:, :, : self.length, :], self.value[:, :, : self.length, :]


@dataclass(frozen=True)
class AttentionStep:
    output: torch.Tensor
    probabilities: torch.Tensor
    new_key: torch.Tensor
    new_value: torch.Tensor


def project_and_rotate(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_module: Any,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只计算本次输入 token 的 Q/K/V；历史 token 不经过此函数。"""
    query = to_head_layout(
        F.linear(hidden_states, attention_module.q_proj.weight, attention_module.q_proj.bias),
        num_query_heads,
        head_dim,
    )
    key = to_head_layout(
        F.linear(hidden_states, attention_module.k_proj.weight, attention_module.k_proj.bias),
        num_kv_heads,
        head_dim,
    )
    value = to_head_layout(
        F.linear(hidden_states, attention_module.v_proj.weight, attention_module.v_proj.bias),
        num_kv_heads,
        head_dim,
    )
    cos, sin = build_default_rope(
        position_ids, head_dim=head_dim, theta=rope_theta, dtype=hidden_states.dtype
    )
    query, key = apply_rope(query, key, cos, sin)
    return query, key, value


def attention_with_cache(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_module: Any,
    cache: ContiguousKVCache,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
    is_prefill: bool,
) -> AttentionStep:
    """执行一次 Prefill 或 Decode，并把本次新 K/V 追加到连续 Cache。"""
    batch_size, query_length, hidden_size = hidden_states.shape
    old_length = cache.length
    query, new_key, new_value = project_and_rotate(
        hidden_states,
        position_ids,
        attention_module,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    cache.append(new_key, new_value)
    full_key, full_value = cache.view()

    group_size = num_query_heads // num_kv_heads
    full_key = repeat_kv(full_key, group_size)
    full_value = repeat_kv(full_value, group_size)
    scores = torch.matmul(query, full_key.transpose(-2, -1)) / math.sqrt(head_dim)

    if is_prefill:
        if old_length != 0 or query_length != cache.length:
            raise ValueError("教学版 Prefill 只支持从空 Cache 一次写入完整 prompt")
        scores = scores + build_prefill_causal_mask(
            batch_size,
            query_length,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    elif query_length != 1:
        raise ValueError("教学版 Decode 只支持 query_length=1")
    # 单请求无 padding Decode：唯一 Query 位于序列末尾，可以读取全部有效 Cache。

    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    per_head_output = torch.matmul(probabilities, full_value)
    merged = (
        per_head_output.transpose(1, 2)
        .contiguous()
        .reshape(batch_size, query_length, hidden_size)
    )
    output = F.linear(merged, attention_module.o_proj.weight, attention_module.o_proj.bias)
    return AttentionStep(output, probabilities, new_key, new_value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).cuda().eval()
    attention_module = model.model.layers[0].self_attn

    # 每次 Attention 调用记录一项：第 0 项是 Prefill，第 1 项是第一次 Decode。
    reference_calls: list[dict[str, torch.Tensor]] = []
    pending: dict[str, torch.Tensor] = {}

    def capture_input(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        pending.clear()
        pending["hidden_states"] = kwargs["hidden_states"].detach().clone()
        pending["position_ids"] = kwargs["position_ids"].detach().clone()

    def capture_output(
        _module: Any, _args: Any, _kwargs: Any, output: tuple[torch.Tensor, Any]
    ) -> None:
        record = dict(pending)
        record["output"] = output[0].detach().clone()
        record["probabilities"] = output[1].detach().clone()
        reference_calls.append(record)

    pre_handle = attention_module.register_forward_pre_hook(
        capture_input, with_kwargs=True
    )
    post_handle = attention_module.register_forward_hook(
        capture_output, with_kwargs=True
    )
    try:
        encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            prefill = model(
                **encoded,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        reference_cache = prefill.past_key_values
        reference_prefill_key = reference_cache.layers[0].keys.detach().clone()
        reference_prefill_value = reference_cache.layers[0].values.detach().clone()

        next_token = torch.argmax(prefill.logits[:, -1, :], dim=-1, keepdim=True)
        decode_mask = torch.cat(
            [encoded["attention_mask"], torch.ones_like(next_token)], dim=1
        )
        with torch.inference_mode():
            model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=reference_cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        reference_decode_key = reference_cache.layers[0].keys.detach().clone()
        reference_decode_value = reference_cache.layers[0].values.detach().clone()
    finally:
        pre_handle.remove()
        post_handle.remove()

    if len(reference_calls) != 2:
        raise RuntimeError(f"预期捕获 2 次第 0 层 Attention，实际 {len(reference_calls)} 次")

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    manual_cache = ContiguousKVCache(
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        capacity=encoded["input_ids"].shape[1] + 1,
        head_dim=head_dim,
        dtype=torch.float16,
        device="cuda",
    )

    with torch.inference_mode():
        manual_prefill = attention_with_cache(
            reference_calls[0]["hidden_states"],
            reference_calls[0]["position_ids"],
            attention_module,
            manual_cache,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            is_prefill=True,
        )
        manual_prefill_key = manual_cache.key[:, :, : manual_cache.length, :].clone()
        manual_prefill_value = manual_cache.value[:, :, : manual_cache.length, :].clone()
        old_key_prefix = manual_prefill_key.clone()
        prefill_length = manual_cache.length

        manual_decode = attention_with_cache(
            reference_calls[1]["hidden_states"],
            reference_calls[1]["position_ids"],
            attention_module,
            manual_cache,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
            is_prefill=False,
        )

    full_manual_key, full_manual_value = manual_cache.view()
    history_unchanged = torch.equal(
        old_key_prefix, full_manual_key[:, :, :prefill_length, :]
    )
    reports = {
        "Prefill K Cache": compare(manual_prefill_key, reference_prefill_key),
        "Prefill V Cache": compare(manual_prefill_value, reference_prefill_value),
        "Prefill probability": compare(
            manual_prefill.probabilities, reference_calls[0]["probabilities"]
        ),
        "Prefill output": compare(manual_prefill.output, reference_calls[0]["output"]),
        "Decode K Cache": compare(full_manual_key, reference_decode_key),
        "Decode V Cache": compare(full_manual_value, reference_decode_value),
        "Decode probability": compare(
            manual_decode.probabilities, reference_calls[1]["probabilities"]
        ),
        "Decode output": compare(manual_decode.output, reference_calls[1]["output"]),
    }

    print("===== 手写连续 KV Cache：第 0 层 =====")
    print(f"prompt token ids       = {encoded['input_ids'][0].tolist()}")
    print(f"first generated token  = {next_token.item()} {tokenizer.decode(next_token[0])!r}")
    print(f"allocated cache shape  = {list(manual_cache.key.shape)}")
    print(f"Prefill 后 length      = {prefill_length}")
    print(f"Decode 后 length       = {manual_cache.length}")
    print(f"历史 K 前缀保持不变    = {history_unchanged}")
    print(f"Prefill probability    = {list(manual_prefill.probabilities.shape)}")
    print(f"Decode probability     = {list(manual_decode.probabilities.shape)}")
    print("\n===== 与 Hugging Face DynamicCache 对拍 =====")
    for name, report in reports.items():
        print(
            f"{name:22s} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )

    print("\nDecode Head-0 对全部历史和当前 token 的概率：")
    row = manual_decode.probabilities[0, 0, 0].float().cpu().tolist()
    print("  " + " ".join(f"{value:8.5f}" for value in row))

    if not history_unchanged or not all(report.allclose for report in reports.values()):
        raise SystemExit("对拍失败：从第一个失败的 Cache/Attention 检查点定位")
    print("\n全部检查点通过：Prefill 批量写入，Decode 仅追加一个 K/V。")


if __name__ == "__main__":
    main()

