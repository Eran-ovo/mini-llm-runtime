"""为 Nsight Compute 提供单次、固定输入的 Paged Attention 采集区间。"""

import argparse
import json
from pathlib import Path

import torch

from scripts.benchmark_paged_attention import (
    make_inputs, run_paged_checked, run_paged_hot_path, run_sdpa,
)
from mini_llm_runtime.environment import collect_environment


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.sequence_length, args.block_size, args.warmup) <= 0:
        parser.error("shape 和 warmup 必须 > 0")
    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    before = collect_environment(root)
    # 与 benchmark 使用相同的派生 seed，保证同配置数据和物理排列一致。
    derived_seed = args.seed + args.batch_size * 100_000 + args.sequence_length
    inputs = make_inputs(batch_size=args.batch_size,
                         sequence_length=args.sequence_length,
                         block_size=args.block_size, seed=derived_seed)
    expected = run_sdpa(inputs)
    checked = run_paged_checked(inputs)
    torch.testing.assert_close(checked, expected, rtol=2e-3, atol=2e-3)
    max_abs = (checked.float() - expected.float()).abs().max().item()
    for _ in range(args.warmup):
        run_paged_hot_path(inputs)
    torch.cuda.synchronize()

    # ncu --profile-from-start off：只在这个区间采集。同步确保 kernel 已结束
    # 才关闭 profiler；此处耗时属于 profiling，不能替代正式 benchmark。
    torch.cuda.profiler.start()
    try:
        actual = run_paged_hot_path(inputs)
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()
    torch.testing.assert_close(actual, checked, rtol=0, atol=0)
    record = {
        "parameters": {**vars(args), "output_dir": str(args.output_dir)},
        "derived_seed": derived_seed,
        "correctness": {"max_abs_vs_sdpa": max_abs, "profiled_matches_checked": True},
        "environment_before": before,
        "environment_after": collect_environment(root),
        "note": "单次 profiling；硬件计数器可能经过 replay，不作为 benchmark 样本",
    }
    (args.output_dir / "profile_context.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Profile correctness passed; max_abs_vs_sdpa={max_abs}")


if __name__ == "__main__":
    main()
