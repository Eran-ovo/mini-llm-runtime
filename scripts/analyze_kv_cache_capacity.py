#!/usr/bin/env python3
"""比较连续预留与 Paged KV Cache 的容量和内部碎片。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from mini_llm_runtime.cache_capacity import (
    CapacitySimulationResult,
    KVCacheGeometry,
    generate_log_uniform_lengths,
    simulate_contiguous_reservation,
    simulate_paged_allocation,
)
from mini_llm_runtime.environment import collect_environment


def parse_positive_int_list(raw: str, *, option_name: str) -> tuple[int, ...]:
    """解析逗号分隔正整数，并尽早报告容易忽略的空项。"""
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{option_name} 必须是逗号分隔整数") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{option_name} 中每个值都必须 > 0")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-budget-mib", type=float, default=64.0)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--block-sizes", default="1,4,8,16,32,64")
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument(
        "--request-lengths",
        help="可选：显式给出逗号分隔长度；设置后忽略 num-requests 和 seed",
    )
    parser.add_argument("--num-layers", type=int, default=24)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--bytes-per-element", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_report(payload: dict[str, object], results: list[CapacitySimulationResult]) -> str:
    config = payload["config"]
    assert isinstance(config, dict)
    workload = payload["workload"]
    assert isinstance(workload, dict)
    lines = [
        "# KV Cache 容量与内部碎片报告",
        "",
        "> 这是确定性容量模拟，不执行 GPU kernel，也不包含 latency 或吞吐量结论。",
        "",
        f"- Cache budget: `{config['cache_budget_bytes']}` bytes",
        "- Budget scope: `K/V tensor storage only`",
        f"- KV bytes/token: `{payload['geometry']['bytes_per_token']}`",
        f"- 请求数: `{workload['request_count']}`",
        f"- 请求来源: `{workload['source']}`",
        f"- 最大序列长度: `{config['max_sequence_length']}`",
        "",
        "| Policy | Unit tokens | Admitted | Allocated units | Unit util. | Slot util. | Fragment tokens | Effective budget util. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| "
            f"{result.policy} | {result.allocation_unit_tokens} | "
            f"{result.admitted_request_count} | {result.allocated_units} | "
            f"{percent(result.allocation_unit_utilization)} | "
            f"{percent(result.slot_utilization)} | "
            f"{result.internal_fragmentation_tokens} | "
            f"{percent(result.effective_budget_utilization)} |"
        )
    lines.extend(
        [
            "",
            "说明：Paged 行中的 allocation unit 是 block；continuous 行中则是一个请求的",
            "完整预留区。Slot utilization 只衡量已分配空间中的有效 token，不代表 kernel",
            "效率。本实验未计入 block table 和 Python allocator metadata。完整原始请求",
            "长度、首个拒绝请求和环境信息见 `result.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.cache_budget_mib <= 0:
        raise ValueError("cache-budget-mib 必须 > 0")

    geometry = KVCacheGeometry(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        bytes_per_element=args.bytes_per_element,
    )
    block_sizes = parse_positive_int_list(args.block_sizes, option_name="block-sizes")
    if len(set(block_sizes)) != len(block_sizes):
        raise ValueError("block-sizes 不能包含重复值")

    if args.request_lengths is None:
        lengths = generate_log_uniform_lengths(
            num_requests=args.num_requests,
            max_sequence_length=args.max_sequence_length,
            seed=args.seed,
        )
        workload_source = "seeded_log_uniform"
    else:
        lengths = parse_positive_int_list(
            args.request_lengths, option_name="request-lengths"
        )
        workload_source = "explicit"
    if any(length > args.max_sequence_length for length in lengths):
        raise ValueError("request length 不能超过 max-sequence-length")

    budget_bytes = int(args.cache_budget_mib * 1024 * 1024)
    results = [
        simulate_contiguous_reservation(
            lengths,
            budget_bytes=budget_bytes,
            reservation_tokens=args.max_sequence_length,
            geometry=geometry,
        )
    ]
    results.extend(
        simulate_paged_allocation(
            lengths,
            budget_bytes=budget_bytes,
            block_size=block_size,
            geometry=geometry,
        )
        for block_size in block_sizes
    )

    repo_root = Path(__file__).resolve().parents[1]
    payload: dict[str, object] = {
        "schema_version": 1,
        "measurement_type": "deterministic_capacity_simulation",
        "budget_scope": "kv_tensor_storage_only",
        "created_at": datetime.now().astimezone().isoformat(),
        "config": {
            "cache_budget_mib": args.cache_budget_mib,
            "cache_budget_bytes": budget_bytes,
            "max_sequence_length": args.max_sequence_length,
            "block_sizes": list(block_sizes),
        },
        "geometry": {
            "num_layers": geometry.num_layers,
            "num_kv_heads": geometry.num_kv_heads,
            "head_dim": geometry.head_dim,
            "bytes_per_element": geometry.bytes_per_element,
            "bytes_per_token": geometry.bytes_per_token,
        },
        "workload": {
            "source": workload_source,
            "distribution": (
                "round(exp(uniform(log(1), log(max_sequence_length))))"
                if workload_source == "seeded_log_uniform"
                else None
            ),
            "seed": args.seed if workload_source == "seeded_log_uniform" else None,
            "request_count": len(lengths),
            "request_lengths": list(lengths),
        },
        "policy": {
            "admission_order": "FIFO",
            "oom_behavior": "stop_on_first_rejected_request",
            "contiguous_reservation_tokens": args.max_sequence_length,
        },
        "results": [result.to_dict() for result in results],
        "environment": collect_environment(repo_root),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    report_path = args.output_dir / "report.md"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(make_report(payload, results), encoding="utf-8")
    print(make_report(payload, results))
    print(f"结果已写入：{result_path}")
    print(f"报告已写入：{report_path}")


if __name__ == "__main__":
    main()
