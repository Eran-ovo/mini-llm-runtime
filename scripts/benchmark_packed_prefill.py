#!/usr/bin/env python3
"""同一模型和请求集合下，交错比较逐请求与 packed Prefill 的 GPU 工作。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.paged_batch import PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.timing import CudaBenchmarkCase, measure_cuda_interleaved


PROMPTS = (
    "你好，GPU",
    "请简要解释 KV Cache。",
    "CUDA 是什么？",
    "Paged Attention 如何管理显存？",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("此 benchmark 需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)

    from transformers import AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    prompts = tuple(
        tuple(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in PROMPTS
    )
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    request_ids = tuple(f"request-{i}" for i in range(len(prompts)))
    lengths = tuple(len(prompt) for prompt in prompts)
    if max(lengths) > config.max_position_embeddings:
        raise SystemExit("prompt 超过模型最大位置")
    single_inputs = tuple(
        torch.tensor((prompt,), dtype=torch.long, device="cuda") for prompt in prompts
    )
    packed_input = torch.tensor(
        (tuple(token for prompt in prompts for token in prompt),),
        dtype=torch.long,
        device="cuda",
    )

    def prepare():
        # 两种路径使用完全相同的物理池容量与请求顺序；Cache 在计时区间外重建。
        manager = PagedKVCacheManager(
            total_blocks=sum(math.ceil(length / args.block_size) for length in lengths),
            block_size=args.block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=weights.embedding.dtype,
            device=weights.embedding.device,
        )
        for request_id, length in zip(request_ids, lengths, strict=True):
            manager.create_request(request_id)
            manager.reserve_request_capacity(request_id, length)
        return manager

    def serial(manager):
        outputs = []
        for request_id, input_ids in zip(request_ids, single_inputs, strict=True):
            result = runner.prefill(
                input_ids, cache=PagedRequestKVCache(manager, request_id)
            )
            outputs.append(result.logits[:, -1])
        return torch.cat(outputs, dim=0)

    def packed(manager):
        adapter = PagedBatchPrefillAdapter(manager, request_ids, lengths)
        return runner.prefill_batch(packed_input, cache=adapter).logits[:, -1]

    # 先检验同一组实际输入的 logits 和所有请求的物理 KV 内容。
    serial_manager = prepare()
    packed_manager = prepare()
    serial_logits = serial(serial_manager)
    packed_logits = packed(packed_manager)
    logits_match = bool(torch.allclose(serial_logits, packed_logits, atol=5e-2, rtol=5e-3))
    tokens_match = bool(torch.equal(serial_logits.argmax(-1), packed_logits.argmax(-1)))
    kv_max_abs_diff = 0.0
    kv_max_relative_l2 = 0.0
    kv_worst_values = None
    for request_id in request_ids:
        serial_kv = serial_manager.gather(request_id)
        packed_kv = packed_manager.gather(request_id)
        for left, right in zip(serial_kv, packed_kv, strict=True):
            difference = (left.float() - right.float()).abs()
            maximum, flat_index = difference.flatten().max(dim=0)
            if float(maximum) > kv_max_abs_diff:
                kv_max_abs_diff = float(maximum)
                kv_worst_values = (
                    float(left.flatten()[flat_index]),
                    float(right.flatten()[flat_index]),
                )
            relative_l2 = float(
                torch.linalg.vector_norm(left.float() - right.float())
                / torch.linalg.vector_norm(left.float()).clamp_min(1e-12)
            )
            kv_max_relative_l2 = max(kv_max_relative_l2, relative_l2)
    # FP16 GEMM 的 M 维变化可能改变舍入；同时限制绝对误差与整体相对误差。
    kv_match = kv_max_abs_diff <= 0.15 and kv_max_relative_l2 <= 0.01
    if not (logits_match and tokens_match and kv_match):
        raise RuntimeError(
            f"正确性失败：logits={logits_match}, tokens={tokens_match}, "
            f"KV={kv_match}, KV max_abs_diff={kv_max_abs_diff}, "
            f"KV max_relative_l2={kv_max_relative_l2}, "
            f"worst_values={kv_worst_values}"
        )
    del serial_manager, packed_manager

    results = measure_cuda_interleaved(
        {
            "serial": CudaBenchmarkCase(operation=serial, prepare=prepare),
            "packed": CudaBenchmarkCase(operation=packed, prepare=prepare),
        },
        warmup=args.warmup,
        repeats=args.repeats,
    )
    environment_after = collect_environment(repo_root)
    clean_same_commit = (
        not environment_before["git_dirty"]
        and not environment_after["git_dirty"]
        and environment_before["git_commit"] == environment_after["git_commit"]
    )
    result = {
        "classification": "formal_clean_tree" if clean_same_commit else "exploratory_dirty_tree",
        "scope": "Prefill ModelRunner + Paged KV write; excludes input construction and scheduling",
        "model": args.model,
        "prompts": list(PROMPTS),
        "prompt_token_ids": [list(prompt) for prompt in prompts],
        "request_ids": list(request_ids),
        "block_size": args.block_size,
        "correctness": {
            "logits_match": logits_match,
            "tokens_match": tokens_match,
            "kv_match": kv_match,
            "kv_max_abs_diff": kv_max_abs_diff,
            "kv_max_relative_l2": kv_max_relative_l2,
            "kv_worst_values": kv_worst_values,
        },
        "cases": {name: timing.to_dict() for name, timing in results.items()},
        "environment_before": environment_before,
        "environment_after": environment_after,
    }
    result["packed_vs_serial_median_percent"] = (
        (results["packed"].median_ms / results["serial"].median_ms - 1) * 100
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "correctness": result["correctness"],
        "serial_median_ms": results["serial"].median_ms,
        "packed_median_ms": results["packed"].median_ms,
        "packed_vs_serial_median_percent": result["packed_vs_serial_median_percent"],
        "result": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
