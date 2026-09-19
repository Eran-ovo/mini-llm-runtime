"""单变量 split-KV 实验：v1 与 partial+merge 两条完整路径交错测量。"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiments.split_kv_attention import split_kv_attention
from scripts.benchmark_paged_attention import (
    make_inputs, run_paged_checked, run_paged_hot_path, run_sdpa,
    repeat_operation, normalized_timing,
)
from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.timing import CudaBenchmarkCase, measure_cuda_interleaved


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-splits", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_splits <= 64 or min(args.warmup, args.repeats, args.iterations) <= 0:
        parser.error("split 数须在 [1,64]；warmup/repeats/iterations 须为正数")
    root = Path(__file__).resolve().parents[1]
    before = collect_environment(root)

    def split(inputs, checked=False):
        return split_kv_attention(
            inputs.query, inputs.paged_key, inputs.paged_value,
            inputs.block_table, inputs.sequence_lengths,
            num_splits=args.num_splits, validate_metadata=checked,
        )

    records = []
    for n in (16, 2048):
        inputs = make_inputs(batch_size=1, sequence_length=n, block_size=16,
                             seed=2027 + 100_000 + n)
        ref = run_sdpa(inputs)
        v1 = run_paged_checked(inputs)
        candidate = split(inputs, checked=True)
        torch.testing.assert_close(candidate, ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(candidate, v1, atol=2e-3, rtol=2e-3)
        error = (candidate.float() - ref.float()).abs().max().item()
        del ref, v1, candidate
        torch.cuda.synchronize()
        baseline_memory = torch.cuda.memory_allocated()
        cases = {
            name: CudaBenchmarkCase(repeat_operation(operation, args.iterations),
                                    prepare=lambda: inputs)
            for name, operation in (("v1", run_paged_hot_path), ("split_kv", split))
        }
        timings = measure_cuda_interleaved(cases, warmup=args.warmup, repeats=args.repeats)
        record = {
            "batch": 1, "sequence_length": n, "seed": 2027 + 100_000 + n,
            "max_abs_vs_sdpa": error,
            "scratch_bytes": 1 * 14 * args.num_splits * (64 + 2) * 4,
            "environment_after_case": collect_environment(root),
            "timings": {name: normalized_timing(t, args.iterations, baseline_memory)
                        for name, t in timings.items()},
        }
        records.append(record)
        print(f"N={n}: v1={record['timings']['v1']['median_us']:.3f} us; "
              f"split={record['timings']['split_kv']['median_us']:.3f} us", flush=True)
        del inputs, cases

    # dirty 工作区时以源码 hash 补充 commit，明确实验所用实际代码。
    paths = [*sorted((root / "csrc").glob("*")), Path(__file__),
             root / "experiments/split_kv_attention.py",
             root / "scripts/benchmark_paged_attention.py",
             root / "src/mini_llm_runtime/timing.py",
             root / "src/mini_llm_runtime/cuda_extension.py",
             root / "src/mini_llm_runtime/paged_attention_cuda.py"]
    result = {
        "parameters": {**vars(args), "output_dir": str(args.output_dir),
                       "Hq": 14, "Hkv": 2, "D": 64, "dtype": "FP16", "block_size": 16},
        "measurement": "CUDA Event around Python launch loop / iterations; includes host supply gaps; not pure kernel duration",
        "scope": "split path includes scratch allocation, partial kernel, merge kernel; metadata validation excluded for both",
        "memory_scope": "peak allocated delta includes transient live outputs; explicit scratch_bytes reported separately",
        "order": "interleaved; reversed on odd rounds; fixed repeated inputs, warm cache",
        "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths},
        "environment_before": before,
        "environment_after": collect_environment(root),
        "cases": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
