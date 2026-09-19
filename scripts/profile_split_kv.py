"""用 Nsight Compute 分别观察 split-KV 的 partial 与 merge kernel。"""

import argparse
import json
from pathlib import Path

import torch

from experiments.split_kv_attention import split_kv_attention
from mini_llm_runtime.environment import collect_environment
from scripts.benchmark_paged_attention import make_inputs, run_paged_checked, run_sdpa


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-splits", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_splits <= 64:
        parser.error("num-splits 必须在 [1,64] 范围内")
    if min(args.batch_size, args.sequence_length, args.block_size) <= 0:
        parser.error("shape 必须 > 0")
    if args.warmup < 0:
        parser.error("warmup 必须 >= 0")
    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    before = collect_environment(root)
    derived_seed = args.seed + args.batch_size * 100_000 + args.sequence_length
    inputs = make_inputs(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        block_size=args.block_size,
        seed=derived_seed,
    )

    # profiling 前先过两道 correctness gate：既对 SDPA，也对稳定的 v1 kernel。
    expected = run_sdpa(inputs)
    v1 = run_paged_checked(inputs)
    checked = split_kv_attention(
        inputs.query,
        inputs.paged_key,
        inputs.paged_value,
        inputs.block_table,
        inputs.sequence_lengths,
        num_splits=args.num_splits,
        validate_metadata=True,
    )
    torch.testing.assert_close(checked, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(checked, v1, rtol=2e-3, atol=2e-3)
    max_abs = (checked.float() - expected.float()).abs().max().item()

    def run_hot_path() -> torch.Tensor:
        # metadata 在上面的 checked 调用中已经验证；profiling 区间只保留两个 CUDA kernel。
        return split_kv_attention(
            inputs.query,
            inputs.paged_key,
            inputs.paged_value,
            inputs.block_table,
            inputs.sequence_lengths,
            num_splits=args.num_splits,
            validate_metadata=False,
        )

    for _ in range(args.warmup):
        run_hot_path()
    torch.cuda.synchronize()

    # ncu --profile-from-start off 只采集这里的一次完整 split-KV 调用：
    # 第一个 kernel 生成 partial state，第二个 kernel 完成 merge。
    torch.cuda.profiler.start()
    try:
        actual = run_hot_path()
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    torch.testing.assert_close(actual, checked, rtol=0, atol=0)

    record = {
        "parameters": {**vars(args), "output_dir": str(args.output_dir)},
        "derived_seed": derived_seed,
        "correctness": {
            "max_abs_vs_sdpa": max_abs,
            "profiled_matches_checked": True,
        },
        "environment_before": before,
        "environment_after": collect_environment(root),
        "note": (
            "单次 profiling；partial/merge 的硬件计数器可能经过 replay，"
            "不能替代 CUDA Event benchmark"
        ),
    }
    (args.output_dir / "profile_context.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"split={args.num_splits} profile correctness passed; "
        f"max_abs_vs_sdpa={max_abs}"
    )


if __name__ == "__main__":
    main()
