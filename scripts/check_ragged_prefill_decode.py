#!/usr/bin/env python3
"""用 HF 显式 Prefill/Decode 验证变长 packed Prefill 后的 Paged Decode 续写。"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter, PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


PROMPTS = (
    "你好，GPU",
    "请简要解释 KV Cache。",
    "CUDA 是什么？",
    "Paged Attention 如何管理显存？",
)


def expand_prompts(
    tokenized: tuple[tuple[int, ...], ...], lengths: tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    """确定性循环 token 模板；不插入 padding，保证短请求真的只有一个 token。"""
    if not lengths or len(tokenized) != len(lengths):
        raise ValueError("prompt 模板与长度必须非空且逐请求对应")
    if any(not prompt for prompt in tokenized) or any(length <= 0 for length in lengths):
        raise ValueError("prompt 模板和目标长度必须为正")
    return tuple(
        (prompt * math.ceil(length / len(prompt)))[:length]
        for prompt, length in zip(tokenized, lengths, strict=True)
    )


def evaluate_continuation_gate(
    *,
    prompt_lengths: tuple[int, ...],
    expected_tokens: dict[str, tuple[int, ...]],
    actual_tokens: dict[str, tuple[int, ...]],
    snapshots: list[dict],
    request_ids: tuple[str, ...],
    block_size: int,
    new_tokens: int,
    all_resources_released: bool,
) -> dict:
    """逐 step 检查 token、RoPE/Cache position、block 增长及资源回收。"""
    checks = {
        "tokens_match_hf": expected_tokens == actual_tokens,
        "token_counts": all(
            len(expected_tokens.get(rid, ())) == new_tokens
            and len(actual_tokens.get(rid, ())) == new_tokens
            for rid in request_ids
        ),
        "ragged_prompts": len(set(prompt_lengths)) > 1,
        "prefill_then_decode": (
            len(snapshots) == new_tokens
            and bool(snapshots)
            and snapshots[0]["phase"] == "prefill"
            and all(item["phase"] == "decode" for item in snapshots[1:])
        ),
        "cache_lengths_and_positions": True,
        "distinct_physical_blocks": True,
        "crosses_new_block_on_decode": False,
        "all_resources_released": all_resources_released,
    }
    if set(expected_tokens) != set(request_ids) or set(actual_tokens) != set(request_ids):
        checks["tokens_match_hf"] = False
    for step, snapshot in enumerate(snapshots):
        lengths = snapshot["cache_lengths"]
        positions = snapshot["input_positions"]
        blocks = snapshot["block_ids"]
        for request_id, prompt_length in zip(request_ids, prompt_lengths, strict=True):
            expected_length = prompt_length + step
            if lengths[request_id] != expected_length:
                checks["cache_lengths_and_positions"] = False
            if step > 0 and positions[request_id] != expected_length - 1:
                checks["cache_lengths_and_positions"] = False
            if len(blocks[request_id]) != math.ceil(expected_length / block_size):
                checks["cache_lengths_and_positions"] = False
        all_blocks = [block for request_id in request_ids for block in blocks[request_id]]
        if len(all_blocks) != len(set(all_blocks)):
            checks["distinct_physical_blocks"] = False
        if step > 0 and any(
            len(blocks[request_id]) > len(snapshots[step - 1]["block_ids"][request_id])
            for request_id in request_ids
        ):
            checks["crosses_new_block_on_decode"] = True
    return {"passed": all(checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-lengths", type=int, nargs=4, default=(128, 128, 128, 1))
    parser.add_argument("--new-tokens", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--prefill-backend", choices=("masked", "segmented_sdpa"),
        default="masked",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = tuple(args.prompt_lengths)
    if not torch.cuda.is_available():
        raise SystemExit("此 correctness gate 需要 CUDA")
    if args.block_size <= 0 or args.new_tokens < 2 or any(length <= 0 for length in lengths):
        raise SystemExit("block-size > 0、new-tokens >= 2，且 prompt 长度必须为正")
    if len(set(lengths)) == 1 or not any(length % args.block_size == 0 for length in lengths):
        raise SystemExit("当前 gate 需要变长请求，且至少一条 prompt 在 block 边界结束")

    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    templates = tuple(
        tuple(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in PROMPTS
    )
    prompts = expand_prompts(templates, lengths)
    request_ids = tuple(f"request-{index}" for index in range(len(prompts)))

    # HF 与 Runtime 分时加载：6 GB GPU 不需要同时持有两份模型权重。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).cuda().eval()
    expected_tokens: dict[str, tuple[int, ...]] = {}
    with torch.inference_mode():
        for request_id, prompt in zip(request_ids, prompts, strict=True):
            input_ids = torch.tensor((prompt,), dtype=torch.long, device="cuda")
            output = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            generated = [int(token.item())]
            for _ in range(args.new_tokens - 1):
                output = hf_model(
                    input_ids=token, past_key_values=cache,
                    use_cache=True, return_dict=True,
                )
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                generated.append(int(token.item()))
            expected_tokens[request_id] = tuple(generated)
    del hf_model, output, cache, token, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(model_dir, device="cuda", dtype=torch.float16)
    if max(lengths) + args.new_tokens - 1 > config.max_position_embeddings:
        raise SystemExit("Prefill + Decode 将超过模型最大位置")
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    total_blocks = sum(
        math.ceil((length + args.new_tokens - 1) / args.block_size)
        for length in lengths
    )
    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=args.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    for request_id in request_ids:
        manager.create_request(request_id)

    def snapshot(phase: str, positions: dict[str, int]) -> dict:
        return {
            "phase": phase,
            "input_positions": positions,
            "cache_lengths": {
                rid: manager.get_request(rid).token_count for rid in request_ids
            },
            "block_ids": {
                rid: manager.get_request(rid).block_ids for rid in request_ids
            },
        }

    packed_ids = torch.tensor(
        (tuple(token for prompt in prompts for token in prompt),),
        dtype=torch.long, device="cuda",
    )
    with torch.inference_mode():
        prefill = runner.prefill_batch(
            packed_ids,
            cache=PagedBatchPrefillAdapter(
                manager, request_ids, lengths, write_backend="vectorized"
            ),
            attention_backend=args.prefill_backend,
        )
        next_tokens = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
        actual_lists = {
            rid: [int(next_tokens[index].item())]
            for index, rid in enumerate(request_ids)
        }
        snapshots = [snapshot("prefill", {})]
        for _ in range(args.new_tokens - 1):
            positions = {
                rid: manager.get_request(rid).token_count for rid in request_ids
            }
            output = runner.decode_batch(
                next_tokens, cache=PagedBatchDecodeAdapter(manager, request_ids)
            )
            next_tokens = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            for index, rid in enumerate(request_ids):
                actual_lists[rid].append(int(next_tokens[index].item()))
            snapshots.append(snapshot("decode", positions))

    actual_tokens = {rid: tuple(tokens) for rid, tokens in actual_lists.items()}
    released = {rid: manager.release_request(rid) for rid in request_ids}
    all_released = (
        manager.request_ids == ()
        and manager.allocator.free_count == manager.allocator.total_blocks
    )
    gate = evaluate_continuation_gate(
        prompt_lengths=lengths,
        expected_tokens=expected_tokens,
        actual_tokens=actual_tokens,
        snapshots=snapshots,
        request_ids=request_ids,
        block_size=args.block_size,
        new_tokens=args.new_tokens,
        all_resources_released=all_released,
    )
    result = {
        "schema_version": 1,
        "artifact": "ragged_packed_prefill_to_paged_decode_correctness",
        "reference": "HF eager explicit Prefill/Decode; no generate()",
        "model": args.model,
        "dtype": str(weights.embedding.dtype),
        "prompt_lengths": lengths,
        "prompt_token_ids": {rid: prompt for rid, prompt in zip(request_ids, prompts, strict=True)},
        "new_tokens": args.new_tokens,
        "block_size": args.block_size,
        "prefill_backend": args.prefill_backend,
        "prefill_kv_write_backend": "vectorized",
        "decode_backend": "paged_cuda",
        "expected_tokens": expected_tokens,
        "actual_tokens": actual_tokens,
        "snapshots": snapshots,
        "released_blocks": released,
        "gate": gate,
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "result": str(target)}, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("变长 Prefill→Decode 与 HF 对拍失败；详见 result.json")


if __name__ == "__main__":
    main()
