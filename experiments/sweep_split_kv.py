"""固定 B=1/N=2048，扫描 split 数；所有候选共享输入并轮换测量顺序。"""
import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import torch

from experiments.split_kv_attention import split_kv_attention
from scripts.benchmark_paged_attention import make_inputs, run_paged_checked, run_paged_hot_path, run_sdpa
from mini_llm_runtime.environment import collect_environment


def round_order(names, index):
    """每 K 轮让每个 case 遍历全部 K 个位置，下一组 K 轮反转方向。"""
    shift = index % len(names)
    order = list(names[shift:]) + list(names[:shift])
    return order if (index // len(names)) % 2 == 0 else list(reversed(order))


def parse_splits(value):
    """候选必须唯一且处于 kernel 支持范围，防止重复项覆盖 operation。"""
    try:
        values = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("splits 必须是逗号分隔的整数") from error
    if not values or any(s < 1 or s > 64 for s in values):
        raise argparse.ArgumentTypeError("splits 必须在 [1,64] 范围内")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("splits 不能重复")
    return values


def measure(operation, iterations):
    # 同步和创建 Event 在计时外；Event 区间可能包含 host 提交间隙。
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = None
    for _ in range(iterations):
        output = operation()
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    peak = torch.cuda.max_memory_allocated()
    del output
    return elapsed, peak, peak - baseline


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", type=parse_splits, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--exclude-v1", action="store_true",
                        help="只比较 split 候选；v1 仍参与计时前的正确性验证")
    parser.add_argument("--warmup", type=int, default=14)
    parser.add_argument("--repeats", type=int, default=28)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repeats, args.iterations) <= 0:
        parser.error("warmup/repeats/iterations 必须为正数")
    case_count = len(args.splits) + int(not args.exclude_v1)
    if case_count < 2:
        parser.error("至少需要两个候选")
    if args.warmup % case_count or args.repeats % case_count:
        parser.error(f"warmup/repeats 必须为 {case_count} 的倍数，确保测量位置均衡")
    if not torch.cuda.is_available():
        parser.error("需要 CUDA GPU")
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "result.json").exists():
        parser.error("输出目录已有 result.json，请使用新目录以保留原始样本")
    before = collect_environment(root)
    inputs = make_inputs(batch_size=1, sequence_length=2048, block_size=16, seed=104075)
    reference = run_sdpa(inputs)
    v1 = run_paged_checked(inputs)
    torch.testing.assert_close(v1, reference, atol=2e-3, rtol=2e-3)
    operations = {} if args.exclude_v1 else {"v1": lambda: run_paged_hot_path(inputs)}
    errors = {"v1": (v1.float() - reference.float()).abs().max().item()}
    splits = args.splits
    for count in splits:
        def operation(count=count, checked=False):
            return split_kv_attention(inputs.query, inputs.paged_key, inputs.paged_value,
                                      inputs.block_table, inputs.sequence_lengths,
                                      num_splits=count, validate_metadata=checked)
        candidate = operation(checked=True)
        torch.testing.assert_close(candidate, reference, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(candidate, v1, atol=2e-3, rtol=2e-3)
        name = f"split_{count}"
        errors[name] = (candidate.float() - reference.float()).abs().max().item()
        operations[name] = operation
    del candidate, reference, v1
    names = list(operations)
    for index in range(args.warmup):
        for name in round_order(names, index):
            measure(operations[name], args.iterations)
    after_warmup = collect_environment(root)
    raw = []
    for index in range(args.repeats):
        for position, name in enumerate(round_order(names, index)):
            elapsed, peak, delta = measure(operations[name], args.iterations)
            raw.append({"round": index, "position": position, "case": name,
                        "event_ms": elapsed, "per_call_us": elapsed * 1000 / args.iterations,
                        "peak_allocated_bytes": peak, "peak_delta_bytes": delta})
    summary = []
    for name in names:
        samples = [row["per_call_us"] for row in raw if row["case"] == name]
        count = 0 if name == "v1" else int(name.split("_")[1])
        summary.append({"case": name, "num_splits": count,
                        "median_us": statistics.median(samples),
                        "min_us": min(samples), "max_us": max(samples),
                        "cv_pct": statistics.stdev(samples) / statistics.mean(samples) * 100,
                        "scratch_bytes": 14 * count * 66 * 4,
                        "max_abs_vs_sdpa": errors[name]})
    # 保存 dirty 工作区实际使用的源码摘要，不能仅凭 HEAD 复现实验。
    paths = [*sorted((root / "csrc").glob("*")), Path(__file__),
             root / "experiments/split_kv_attention.py",
             root / "scripts/benchmark_paged_attention.py",
             root / "src/mini_llm_runtime/paged_attention_cuda.py",
             root / "src/mini_llm_runtime/cuda_extension.py"]
    result = {"parameters": {"B": 1, "N": 2048, "Hq": 14, "Hkv": 2, "D": 64,
                             "block_size": 16, "dtype": "FP16", "seed": 104075,
                             "splits": splits, "include_v1": not args.exclude_v1,
                             "warmup": args.warmup,
                             "repeats": args.repeats, "iterations": args.iterations},
              "measurement": "CUDA Event around full Python calls including partial+merge; may include host supply gaps; not pure kernel duration",
              "order": f"cyclic rotation; reverse direction every {case_count} rounds; shared immutable inputs",
              "memory": "scratch allocation included in calls; peak delta also includes live outputs and allocator rounding",
              "correctness": {"rtol": 2e-3, "atol": 2e-3},
              "environment_before": before, "environment_after_warmup": after_warmup,
              "environment_after": collect_environment(root),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in paths},
              "summary": summary, "raw_samples": raw}
    (args.output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (args.output_dir / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    lines = ["# Split-KV 分区数扫描", "", "固定 B=1/N=2048；完整路径 Event 均摊时间。", "",
             "| Case | Median us | CV % | Scratch bytes | max_abs vs SDPA |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['case']} | {row['median_us']:.3f} | {row['cv_pct']:.2f} | "
                     f"{row['scratch_bytes']} | {row['max_abs_vs_sdpa']:.8f} |")
    report = "\n".join(lines) + "\n"
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
