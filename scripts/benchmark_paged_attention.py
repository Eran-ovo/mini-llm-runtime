#!/usr/bin/env python3
"""正式测量 Paged Attention CUDA v1，并与连续 K/V 的 PyTorch SDPA 对照。"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.paged_attention_cuda import (
    _paged_decode_attention_cuda_unchecked,
    paged_decode_attention_cuda,
)
from mini_llm_runtime.timing import CudaBenchmarkCase, measure_cuda_interleaved


@dataclass(frozen=True)
class BenchmarkInputs:
    query: torch.Tensor
    contiguous_key: torch.Tensor
    contiguous_value: torch.Tensor
    paged_key: torch.Tensor
    paged_value: torch.Tensor
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor

    @property
    def input_nbytes(self) -> int:
        tensors = (
            self.query,
            self.contiguous_key,
            self.contiguous_value,
            self.paged_key,
            self.paged_value,
            self.block_table,
            self.sequence_lengths,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def parse_positive_int_list(raw: str, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} 必须是逗号分隔整数") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} 中每个值必须 > 0")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{name} 不能包含重复值")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--sequence-lengths", default="16,128,512,2048")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--iterations-per-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def make_inputs(
    *, batch_size: int, sequence_length: int, block_size: int, seed: int
) -> BenchmarkInputs:
    """生成同值的连续/分页 K/V；物理 block 使用固定 seed 随机打散。"""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    num_query_heads = 14
    num_kv_heads = 2
    head_dim = 64
    query = torch.randn(
        batch_size,
        num_query_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    contiguous_key = torch.randn(
        batch_size,
        num_kv_heads,
        sequence_length,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    contiguous_value = torch.randn(
        contiguous_key.shape,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    blocks_per_request = (sequence_length + block_size - 1) // block_size
    padded_length = blocks_per_request * block_size
    padded_shape = (batch_size, num_kv_heads, padded_length, head_dim)
    padded_key = torch.full(
        padded_shape, torch.nan, device="cuda", dtype=torch.float16
    )
    padded_value = torch.full_like(padded_key, torch.nan)
    padded_key[:, :, :sequence_length].copy_(contiguous_key)
    padded_value[:, :, :sequence_length].copy_(contiguous_value)

    # [B,Hkv,N,D] -> [B,logical_block,Hkv,block_offset,D]
    logical_key_blocks = (
        padded_key.view(
            batch_size, num_kv_heads, blocks_per_request, block_size, head_dim
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(-1, num_kv_heads, block_size, head_dim)
    )
    logical_value_blocks = (
        padded_value.view(
            batch_size, num_kv_heads, blocks_per_request, block_size, head_dim
        )
        .permute(0, 2, 1, 3, 4)
        .contiguous()
        .view(-1, num_kv_heads, block_size, head_dim)
    )
    total_blocks = logical_key_blocks.shape[0]
    physical_ids = torch.randperm(total_blocks, device="cuda", generator=generator)
    paged_key = torch.empty_like(logical_key_blocks)
    paged_value = torch.empty_like(logical_value_blocks)
    paged_key.index_copy_(0, physical_ids, logical_key_blocks)
    paged_value.index_copy_(0, physical_ids, logical_value_blocks)
    block_table = physical_ids.view(batch_size, blocks_per_request).to(torch.int32)
    sequence_lengths = torch.full(
        (batch_size,),
        sequence_length,
        device="cuda",
        dtype=torch.int32,
    )
    return BenchmarkInputs(
        query=query,
        contiguous_key=contiguous_key,
        contiguous_value=contiguous_value,
        paged_key=paged_key,
        paged_value=paged_value,
        block_table=block_table,
        sequence_lengths=sequence_lengths,
    )


def run_sdpa(inputs: BenchmarkInputs) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        inputs.query[:, :, None, :],
        inputs.contiguous_key,
        inputs.contiguous_value,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True,
    )[:, :, 0, :]


def run_paged_checked(inputs: BenchmarkInputs) -> torch.Tensor:
    return paged_decode_attention_cuda(
        inputs.query,
        inputs.paged_key,
        inputs.paged_value,
        inputs.block_table,
        inputs.sequence_lengths,
    )


def run_paged_hot_path(inputs: BenchmarkInputs) -> torch.Tensor:
    return _paged_decode_attention_cuda_unchecked(
        inputs.query,
        inputs.paged_key,
        inputs.paged_value,
        inputs.block_table,
        inputs.sequence_lengths,
    )


def repeat_operation(
    operation: Callable[[BenchmarkInputs], torch.Tensor], iterations: int
) -> Callable[[BenchmarkInputs], None]:
    def repeated(inputs: BenchmarkInputs) -> None:
        output = None
        for _ in range(iterations):
            output = operation(inputs)
        # 保持最后一个输出至少活到所有 launch 已入队，避免 Python 过早回收。
        if output is None:
            raise RuntimeError("iterations 必须 > 0")

    return repeated


def normalized_timing(timing: object, iterations: int, base_memory: int) -> dict:
    total_samples = list(timing.samples_ms)
    per_call_samples = [sample / iterations for sample in total_samples]
    return {
        "warmup_samples": timing.warmup,
        "repeats": timing.repeats,
        "iterations_per_sample": iterations,
        "event_samples_ms": total_samples,
        "per_call_samples_ms": per_call_samples,
        "median_ms": float(statistics.median(per_call_samples)),
        "median_us": float(statistics.median(per_call_samples) * 1000.0),
        "peak_memory_samples_bytes": timing.peak_memory_samples_bytes,
        "peak_memory_bytes": timing.peak_memory_bytes,
        "peak_memory_delta_bytes": max(0, timing.peak_memory_bytes - base_memory),
    }


def render_markdown(result: dict) -> str:
    parameters = result["parameters"]
    lines = [
        "# Paged Attention CUDA v1 Benchmark",
        "",
        f"- warmup / repeats: `{parameters['warmup']} / {parameters['repeats']}`",
        f"- iterations per sample: `{parameters['iterations_per_sample']}`",
        f"- block size: `{parameters['block_size']}`",
        "- dtype / heads / head dim: `FP16 / Hq=14,Hkv=2 / D=64`",
        f"- git commit: `{result['environment_after'].get('git_commit')}` "
        f"(dirty={result['environment_after'].get('git_dirty')})",
        "",
        "| Batch | KV length | Paged v1 (us) | SDPA contiguous (us) | SDPA/Paged | max abs error |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        paged_us = case["timings"]["paged_v1"]["median_us"]
        sdpa_us = case["timings"]["sdpa_contiguous"]["median_us"]
        lines.append(
            f"| {case['batch_size']} | {case['sequence_length']} | "
            f"{paged_us:.3f} | {sdpa_us:.3f} | {sdpa_us / paged_us:.3f}x | "
            f"{case['correctness']['max_abs_error']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Paged v1 直接读取随机打散的物理 block；SDPA 使用预先准备好的连续 K/V，",
            "不计 gather。表中仅为当前 kernel baseline，不能代表优化后的最终性能。",
            "全部 CUDA Event 与显存原始样本见 `result.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 benchmark 要求 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")
    if args.warmup < 0 or args.repeats <= 0 or args.iterations_per_sample <= 0:
        raise SystemExit("warmup 必须 >= 0，repeats 和 iterations 必须 > 0")
    batch_sizes = parse_positive_int_list(args.batch_sizes, "batch-sizes")
    sequence_lengths = parse_positive_int_list(
        args.sequence_lengths, "sequence-lengths"
    )
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    cases = []

    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            inputs = make_inputs(
                batch_size=batch_size,
                sequence_length=sequence_length,
                block_size=args.block_size,
                seed=args.seed + batch_size * 100_000 + sequence_length,
            )

            # correctness gate 同时通过安全入口完成一次 metadata 验证。
            paged_output = run_paged_checked(inputs)
            sdpa_output = run_sdpa(inputs)
            torch.cuda.synchronize()
            error = (paged_output.float() - sdpa_output.float()).abs()
            max_abs_error = float(error.max().item())
            mean_abs_error = float(error.mean().item())
            torch.testing.assert_close(
                paged_output, sdpa_output, rtol=2e-2, atol=2e-2
            )
            del paged_output, sdpa_output, error

            torch.cuda.synchronize()
            base_memory = int(torch.cuda.memory_allocated())
            timings = measure_cuda_interleaved(
                {
                    "paged_v1": CudaBenchmarkCase(
                        repeat_operation(
                            run_paged_hot_path, args.iterations_per_sample
                        ),
                        prepare=lambda current=inputs: current,
                    ),
                    "sdpa_contiguous": CudaBenchmarkCase(
                        repeat_operation(run_sdpa, args.iterations_per_sample),
                        prepare=lambda current=inputs: current,
                    ),
                },
                warmup=args.warmup,
                repeats=args.repeats,
                device="cuda",
            )
            cases.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "blocks_per_request": (
                        sequence_length + args.block_size - 1
                    )
                    // args.block_size,
                    "total_physical_blocks": inputs.paged_key.shape[0],
                    "input_nbytes": inputs.input_nbytes,
                    "base_memory_bytes": base_memory,
                    "correctness": {
                        "passed": True,
                        "reference": "torch.nn.functional.scaled_dot_product_attention",
                        "rtol": 2e-2,
                        "atol": 2e-2,
                        "max_abs_error": max_abs_error,
                        "mean_abs_error": mean_abs_error,
                    },
                    "timings": {
                        name: normalized_timing(
                            timing, args.iterations_per_sample, base_memory
                        )
                        for name, timing in timings.items()
                    },
                }
            )
            del inputs, timings
            gc.collect()
            torch.cuda.empty_cache()

    result = {
        "schema_version": 1,
        "benchmark": "paged_attention_cuda_v1_vs_contiguous_sdpa",
        "parameters": {
            "batch_sizes": list(batch_sizes),
            "sequence_lengths": list(sequence_lengths),
            "block_size": args.block_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "iterations_per_sample": args.iterations_per_sample,
            "seed": args.seed,
            "dtype": "torch.float16",
            "num_query_heads": 14,
            "num_kv_heads": 2,
            "head_dim": 64,
            "case_order": "interleaved; reversed on odd rounds",
        },
        "measurement_definition": {
            "timer": "CUDA Event on current PyTorch stream",
            "sample": "event time around N launches divided by N",
            "paged_v1": (
                "direct physical block-pool read; metadata validated once before timing"
            ),
            "sdpa_contiguous": (
                "prebuilt contiguous K/V; enable_gqa=True; gather excluded"
            ),
            "peak_memory": (
                "torch.cuda.max_memory_allocated; delta subtracts live shared inputs"
            ),
            "jit_compile": "completed before warmup and excluded from all samples",
        },
        "cases": cases,
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    report_path = args.output_dir / "report.md"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = render_markdown(result)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"结果已写入：{result_path}")
    print(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
