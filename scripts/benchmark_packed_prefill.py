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
LOGITS_RELATIVE_L2_LIMIT = 0.01
LEGACY_KV_RELATIVE_L2_LIMIT = 0.01


def compare_model_logits(
    reference: torch.Tensor, actual: torch.Tensor
) -> dict[str, bool | float | None]:
    """跨执行形状只用最终可观察 logits/token 做数值门禁。"""
    if reference.shape != actual.shape:
        raise ValueError("reference 与 candidate logits 形状不同")
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(actual).all())
    tokens_match = bool(torch.equal(reference.argmax(-1), actual.argmax(-1)))
    relative_l2 = None
    max_abs = None
    if finite:
        difference = reference.float() - actual.float()
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
        )
        max_abs = float(difference.abs().max())
    return {
        "passed": finite and tokens_match and relative_l2 is not None
        and relative_l2 <= LOGITS_RELATIVE_L2_LIMIT,
        "finite": finite,
        "tokens_match": tokens_match,
        "logits_relative_l2": relative_l2,
        "logits_max_abs_diff": max_abs,
        "logits_elementwise_close_diagnostic": bool(
            torch.allclose(reference, actual, atol=5e-2, rtol=5e-3)
        ),
    }


def compare_packed_storage_exact(
    reference_manager: PagedKVCacheManager,
    actual_manager: PagedKVCacheManager,
    request_ids: tuple[str, ...],
    reference_logits: torch.Tensor,
    actual_logits: torch.Tensor,
) -> dict[str, bool | str | None]:
    """同一 packed 计算只换写入方式时，逐元素验证逻辑 KV 与 logits。"""
    logits_exact = bool(torch.equal(reference_logits, actual_logits))
    first_mismatch = None
    for request_id in request_ids:
        reference_kv = reference_manager.gather(request_id)
        actual_kv = actual_manager.gather(request_id)
        for kind, left, right in zip(
            ("key", "value"), reference_kv, actual_kv, strict=True
        ):
            if not torch.equal(left, right) and first_mismatch is None:
                first_mismatch = f"{request_id}:{kind}"
    return {
        "passed": logits_exact and first_mismatch is None,
        "logits_exact": logits_exact,
        "kv_exact": first_mismatch is None,
        "first_kv_mismatch": first_mismatch,
    }


def compare_kv_diagnostic(
    reference_manager: PagedKVCacheManager,
    actual_manager: PagedKVCacheManager,
    request_ids: tuple[str, ...],
) -> dict[str, bool | float | str | tuple[float, float] | None]:
    """跨形状 KV 误差只保留为诊断；非有限值仍属于硬错误。"""
    max_abs = 0.0
    max_relative_l2 = 0.0
    worst_values = None
    worst_request = None
    worst_kind = None
    finite = True
    for request_id in request_ids:
        reference_kv = reference_manager.gather(request_id)
        actual_kv = actual_manager.gather(request_id)
        for kind, left, right in zip(
            ("key", "value"), reference_kv, actual_kv, strict=True
        ):
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                finite = False
                continue
            difference = (left.float() - right.float()).abs()
            maximum, flat_index = difference.flatten().max(dim=0)
            if float(maximum) > max_abs:
                max_abs = float(maximum)
                worst_values = (
                    float(left.flatten()[flat_index]),
                    float(right.flatten()[flat_index]),
                )
            relative_l2 = float(
                torch.linalg.vector_norm(left.float() - right.float())
                / torch.linalg.vector_norm(left.float()).clamp_min(1e-12)
            )
            if relative_l2 > max_relative_l2:
                max_relative_l2 = relative_l2
                worst_request = request_id
                worst_kind = kind
    return {
        "finite": finite,
        "kv_max_abs_diff": max_abs if finite else None,
        "kv_max_relative_l2": max_relative_l2 if finite else None,
        "kv_within_legacy_1pct_diagnostic": (
            max_relative_l2 <= LEGACY_KV_RELATIVE_L2_LIMIT if finite else False
        ),
        "kv_worst_values": worst_values,
        "relative_l2_worst_request": worst_request,
        "relative_l2_worst_kind": worst_kind,
    }


def build_layered_correctness(
    model_numerics: dict,
    storage_equivalence: dict,
    kv_diagnostics: dict,
) -> dict:
    """旧 KV 1% 状态只作诊断；严格写入等价与模型结果才是硬门禁。"""
    return {
        "schema_version": 2,
        "model_numerics": model_numerics,
        "storage_equivalence": storage_equivalence,
        "kv_diagnostics": kv_diagnostics,
        "passed": (
            bool(model_numerics)
            and all(item["passed"] for item in model_numerics.values())
            and all(item["passed"] for item in storage_equivalence.values())
            and all(item["finite"] for item in kv_diagnostics.values())
        ),
    }


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
        adapter = PagedBatchPrefillAdapter(
            manager, request_ids, lengths, write_backend="vectorized"
        )
        return runner.prefill_batch(
            packed_input, cache=adapter, attention_backend="segmented_sdpa"
        ).logits[:, -1]

    def segmented_scalar(manager):
        # 只作正确性 oracle，不进入计时；与 segmented 只差 KV 写入方式。
        adapter = PagedBatchPrefillAdapter(
            manager, request_ids, lengths, write_backend="scalar"
        )
        return runner.prefill_batch(
            packed_input, cache=adapter, attention_backend="segmented_sdpa"
        ).logits[:, -1]

    # 数值 oracle 和写入 oracle 分开：前者跨执行形状，后者只变 KV 写入。
    serial_manager = prepare()
    serial_logits = serial(serial_manager)
    candidates = {"packed": packed}
    if args.compare_segmented_sdpa:
        candidates["segmented"] = segmented
    if args.compare_vectorized_kv_write:
        candidates["packed_vectorized"] = packed_vectorized
    candidate_runs = {}
    for name, operation in candidates.items():
        manager = prepare()
        candidate_runs[name] = (manager, operation(manager))

    model_numerics = {
        name: compare_model_logits(serial_logits, logits)
        for name, (_, logits) in candidate_runs.items()
    }
    kv_diagnostics = {
        name: compare_kv_diagnostic(serial_manager, manager, request_ids)
        for name, (manager, _) in candidate_runs.items()
    }
    storage_equivalence = {}
    if args.compare_vectorized_kv_write:
        scalar_manager, scalar_logits = candidate_runs["packed"]
        vector_manager, vector_logits = candidate_runs["packed_vectorized"]
        storage_equivalence["masked_scalar_vs_vectorized"] = compare_packed_storage_exact(
            scalar_manager, vector_manager, request_ids, scalar_logits, vector_logits
        )
    if args.compare_segmented_sdpa:
        segmented_scalar_manager = prepare()
        segmented_scalar_logits = segmented_scalar(segmented_scalar_manager)
        segmented_manager, segmented_logits = candidate_runs["segmented"]
        storage_equivalence["segmented_scalar_vs_vectorized"] = (
            compare_packed_storage_exact(
                segmented_scalar_manager,
                segmented_manager,
                request_ids,
                segmented_scalar_logits,
                segmented_logits,
            )
        )
    correctness = build_layered_correctness(
        model_numerics, storage_equivalence, kv_diagnostics
    )
    if not correctness["passed"]:
        raise RuntimeError(f"正确性失败：{correctness}")
    del serial_manager, candidate_runs

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
