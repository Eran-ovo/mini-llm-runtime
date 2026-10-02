#!/usr/bin/env python3
"""同一模型和请求集合下，交错比较逐请求与 packed Prefill 的 GPU 工作。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.engine import select_prefill_attention_backend
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


def construct_prompts(
    base_prompts: tuple[tuple[int, ...], ...],
    *,
    fixed_prompt_length: int | None = None,
    prompt_lengths: tuple[int, ...] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """用合法 token ID 构造等长或变长请求；不改变每条请求的 RoPE 起点。"""
    if not base_prompts or any(not prompt for prompt in base_prompts):
        raise ValueError("base_prompts 必须非空，且每条 prompt 至少有一个 token")
    if fixed_prompt_length is not None and prompt_lengths is not None:
        raise ValueError("fixed_prompt_length 与 prompt_lengths 不能同时指定")
    if fixed_prompt_length is not None:
        prompt_lengths = (fixed_prompt_length,) * len(base_prompts)
    if prompt_lengths is None:
        return base_prompts
    if not prompt_lengths or any(length <= 0 for length in prompt_lengths):
        raise ValueError("prompt_lengths 必须是非空的正整数序列")
    expanded = []
    for index, length in enumerate(prompt_lengths):
        template = base_prompts[index % len(base_prompts)]
        expanded.append((template * math.ceil(length / len(template)))[:length])
    return tuple(expanded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--compare-segmented-sdpa",
        action="store_true",
        help="额外与分段 SDPA Prefill 交错比较，只改变 Attention backend",
    )
    parser.add_argument(
        "--compare-vectorized-kv-write",
        action="store_true",
        help="额外与批量物理 slot 写入交错比较，只改变 KV 写入 backend",
    )
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument(
        "--fixed-prompt-length",
        type=int,
        help="把四条 token 序列循环扩展到相同长度；用于扫描 Attention 形状",
    )
    shape.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        help="逐请求指定 token 数；模板 prompt 按四条短句循环复用",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("此 benchmark 需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")
    if args.fixed_prompt_length is not None and args.fixed_prompt_length <= 0:
        raise SystemExit("--fixed-prompt-length 必须 > 0")
    if args.prompt_lengths is not None and any(length <= 0 for length in args.prompt_lengths):
        raise SystemExit("--prompt-lengths 必须全部 > 0")
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)

    from transformers import AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    base_prompts = tuple(
        tuple(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in PROMPTS
    )
    prompts = construct_prompts(
        base_prompts,
        fixed_prompt_length=args.fixed_prompt_length,
        prompt_lengths=(
            tuple(args.prompt_lengths) if args.prompt_lengths is not None else None
        ),
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
        adapter = PagedBatchPrefillAdapter(
            manager, request_ids, lengths, write_backend="scalar"
        )
        return runner.prefill_batch(
            packed_input, cache=adapter, attention_backend="masked"
        ).logits[:, -1]

    def packed_vectorized(manager):
        adapter = PagedBatchPrefillAdapter(
            manager, request_ids, lengths, write_backend="vectorized"
        )
        return runner.prefill_batch(
            packed_input, cache=adapter, attention_backend="masked"
        ).logits[:, -1]

    def segmented(manager):
        adapter = PagedBatchPrefillAdapter(manager, request_ids, lengths)
        return runner.prefill_batch(
            packed_input, cache=adapter, attention_backend="segmented_sdpa"
        ).logits[:, -1]

    # 先检验同一组实际输入的 logits 和所有请求的物理 KV 内容。
    serial_manager = prepare()
    serial_logits = serial(serial_manager)

    def compare_to_serial(operation):
        candidate_manager = prepare()
        candidate_logits = operation(candidate_manager)
        logits_elementwise_close = bool(
            torch.allclose(serial_logits, candidate_logits, atol=5e-2, rtol=5e-3)
        )
        logits_max_abs_diff = float(
            (serial_logits.float() - candidate_logits.float()).abs().max()
        )
        logits_relative_l2 = float(
            torch.linalg.vector_norm(serial_logits.float() - candidate_logits.float())
            / torch.linalg.vector_norm(serial_logits.float()).clamp_min(1e-12)
        )
        logits_match = logits_relative_l2 <= 0.01 and bool(
            torch.isfinite(candidate_logits).all()
        )
        tokens_match = bool(
            torch.equal(serial_logits.argmax(-1), candidate_logits.argmax(-1))
        )
        kv_max_abs_diff = 0.0
        kv_max_relative_l2 = 0.0
        kv_worst_values = None
        for request_id in request_ids:
            serial_kv = serial_manager.gather(request_id)
            candidate_kv = candidate_manager.gather(request_id)
            for left, right in zip(serial_kv, candidate_kv, strict=True):
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
        correctness = {
            "logits_match": logits_match,
            "logits_elementwise_close": logits_elementwise_close,
            "logits_max_abs_diff": logits_max_abs_diff,
            "logits_relative_l2": logits_relative_l2,
            "tokens_match": tokens_match,
            "kv_match": kv_max_relative_l2 <= 0.01,
            "kv_max_abs_diff": kv_max_abs_diff,
            "kv_max_relative_l2": kv_max_relative_l2,
            "kv_worst_values": kv_worst_values,
        }
        if not all(correctness[name] for name in ("logits_match", "tokens_match", "kv_match")):
            raise RuntimeError(f"正确性失败：{correctness}")
        return correctness

    correctness = {"packed": compare_to_serial(packed)}
    if args.compare_segmented_sdpa:
        correctness["segmented"] = compare_to_serial(segmented)
    if args.compare_vectorized_kv_write:
        correctness["packed_vectorized"] = compare_to_serial(packed_vectorized)
    del serial_manager

    cases = {
        "serial": CudaBenchmarkCase(operation=serial, prepare=prepare),
        "packed": CudaBenchmarkCase(operation=packed, prepare=prepare),
    }
    if args.compare_segmented_sdpa:
        cases["segmented"] = CudaBenchmarkCase(operation=segmented, prepare=prepare)
    if args.compare_vectorized_kv_write:
        cases["packed_vectorized"] = CudaBenchmarkCase(
            operation=packed_vectorized, prepare=prepare
        )
    results = measure_cuda_interleaved(
        cases,
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
        "prompts": [PROMPTS[index % len(PROMPTS)] for index in range(len(prompts))],
        "prompt_construction": (
            "original_tokenized"
            if args.fixed_prompt_length is None and args.prompt_lengths is None
            else (
                "cyclic_repetition_to_fixed_length"
                if args.fixed_prompt_length is not None
                else "cyclic_repetition_to_variable_lengths"
            )
        ),
        "fixed_prompt_length": args.fixed_prompt_length,
        "prompt_lengths": list(lengths),
        "auto_attention_backend": select_prefill_attention_backend("auto", lengths),
        "prompt_token_ids": [list(prompt) for prompt in prompts],
        "request_ids": list(request_ids),
        "block_size": args.block_size,
        "correctness": correctness,
        "cases": {name: timing.to_dict() for name, timing in results.items()},
        "environment_before": environment_before,
        "environment_after": environment_after,
    }
    result["packed_vs_serial_median_percent"] = (
        (results["packed"].median_ms / results["serial"].median_ms - 1) * 100
    )
    if args.compare_segmented_sdpa:
        result["segmented_vs_packed_median_percent"] = (
            (results["segmented"].median_ms / results["packed"].median_ms - 1)
            * 100
        )
    if args.compare_vectorized_kv_write:
        result["vectorized_vs_packed_median_percent"] = (
            (results["packed_vectorized"].median_ms / results["packed"].median_ms - 1)
            * 100
        )
    if args.compare_segmented_sdpa and args.compare_vectorized_kv_write:
        # Attention 单变量比较：两边都使用 vectorized KV write。
        result["segmented_vs_vectorized_median_percent"] = (
            (results["segmented"].median_ms / results["packed_vectorized"].median_ms - 1)
            * 100
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "correctness": result["correctness"],
        "serial_median_ms": results["serial"].median_ms,
        "packed_median_ms": results["packed"].median_ms,
        "segmented_median_ms": (
            results["segmented"].median_ms if args.compare_segmented_sdpa else None
        ),
        "packed_vectorized_median_ms": (
            results["packed_vectorized"].median_ms
            if args.compare_vectorized_kv_write else None
        ),
        "packed_vs_serial_median_percent": result["packed_vs_serial_median_percent"],
        "segmented_vs_packed_median_percent": result.get("segmented_vs_packed_median_percent"),
        "vectorized_vs_packed_median_percent": result.get("vectorized_vs_packed_median_percent"),
        "segmented_vs_vectorized_median_percent": result.get("segmented_vs_vectorized_median_percent"),
        "auto_attention_backend": result["auto_attention_backend"],
        "result": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
