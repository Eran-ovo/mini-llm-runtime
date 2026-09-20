#!/usr/bin/env python3
"""在相同 Continuous Batching workload 下扫描 mixed Prefill token budget。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import BatchingPolicy
if __package__:
    from scripts.benchmark_batching_policy import (
        PROMPTS,
        RequestSpec,
        parse_positive_int_list,
        run_trial,
        summarize_trials,
        trial_tail_latency,
    )
else:
    # `python scripts/benchmark_mixed_prefill_budget.py` 会把 scripts/ 放在
    # sys.path[0]；此时使用同目录 import，保持与现有 benchmark 命令风格一致。
    from benchmark_batching_policy import (  # type: ignore[no-redef]
        PROMPTS,
        RequestSpec,
        parse_positive_int_list,
        run_trial,
        summarize_trials,
        trial_tail_latency,
    )


Budget = int | None


def parse_budgets(raw: str) -> tuple[Budget, ...]:
    """把 `none,4,8` 解析为互不重复的预算；None 表示不设额外限制。"""
    values: list[Budget] = []
    for part in raw.split(","):
        normalized = part.strip().lower()
        if normalized in {"none", "unbounded"}:
            value: Budget = None
        else:
            try:
                value = int(normalized)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "budgets 必须是逗号分隔的正整数或 none"
                ) from error
            if value <= 0:
                raise argparse.ArgumentTypeError("budgets 中的整数必须 > 0")
        if value in values:
            raise argparse.ArgumentTypeError("budgets 不能包含重复值")
        values.append(value)
    if len(values) < 2:
        raise argparse.ArgumentTypeError("至少需要两个 budget case")
    return tuple(values)


def budget_key(value: Budget) -> str:
    return "unbounded" if value is None else str(value)


def order_for_round(budgets: tuple[Budget, ...], round_index: int) -> tuple[Budget, ...]:
    """循环轮换 case，使每个 case 都能出现在不同执行位置。"""
    offset = round_index % len(budgets)
    return budgets[offset:] + budgets[:offset]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument(
        "--generation-lengths",
        type=parse_positive_int_list,
        default=(2, 4, 8, 12),
    )
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=64)
    parser.add_argument("--mixed-prefill-budgets", type=parse_budgets, default=(None, 8, 4))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Mixed Prefill Token Budget Benchmark",
        "",
        f"- model: `{result['model']}`",
        f"- requests: `{result['parameters']['request_count']}` (burst arrival)",
        f"- warmup / repeats: `{result['parameters']['warmup']} / {result['parameters']['repeats']}`",
        f"- git commit: `{result['environment_after'].get('git_commit')}` "
        f"(dirty={result['environment_after'].get('git_dirty')})",
        "",
        "| Mixed Prefill budget | Throughput median (tok/s) | TTFT median (ms) | TPOT median (ms) | E2E median (ms) | CUDA timeline median (ms) | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in result["case_keys"]:
        item = result["summary"][key]
        tpot = item["median_tpot_ms"]
        tpot_text = f"{tpot:.4f}" if tpot is not None else "N/A"
        lines.append(
            f"| {key} | {item['median_throughput_tokens_per_second']:.2f} | "
            f"{item['median_request_ttft_ms']:.4f} | {tpot_text} | "
            f"{item['median_request_e2e_ms']:.4f} | "
            f"{item['median_cuda_timeline_ms']:.4f} | "
            f"{item['max_peak_allocated_bytes'] / 2**20:.2f} |"
        )
    lines.extend(
        [
            "",
            "| Budget | TTFT p90* (ms) | TTFT p95* (ms) | TTFT max (ms) | TPOT p95* (ms) | E2E p90* (ms) | E2E p95* (ms) | E2E max (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in result["case_keys"]:
        item = result["summary"][key]
        tpot_p95 = item["median_trial_p95_tpot_ms"]
        tpot_p95_text = f"{tpot_p95:.4f}" if tpot_p95 is not None else "N/A"
        lines.append(
            f"| {key} | {item['median_trial_p90_request_ttft_ms']:.4f} | "
            f"{item['median_trial_p95_request_ttft_ms']:.4f} | "
            f"{item['max_request_ttft_ms']:.4f} | {tpot_p95_text} | "
            f"{item['median_trial_p90_request_e2e_ms']:.4f} | "
            f"{item['median_trial_p95_request_e2e_ms']:.4f} | "
            f"{item['max_request_e2e_ms']:.4f} |"
        )

    lines.extend(["", "## Arrival-position fairness", ""])
    header = "| Position | Request | " + " | ".join(
        f"{key} TTFT / E2E (ms)" for key in result["case_keys"]
    ) + " |"
    lines.append(header)
    lines.append("|---:|---|" + "---:|" * len(result["case_keys"]))
    first_summary = result["summary"][result["case_keys"][0]][
        "per_request_position"
    ]
    for position_item in first_summary:
        position = position_item["arrival_position"]
        request_id = position_item["request_id"]
        cells = []
        for key in result["case_keys"]:
            item = result["summary"][key]["per_request_position"][position]
            if item["request_id"] != request_id:
                raise ValueError("不同 case 的 request arrival position 不一致")
            cells.append(
                f"{item['median_ttft_ms']:.4f} / {item['median_e2e_ms']:.4f}"
            )
        lines.append(
            f"| {position} | {request_id} | " + " | ".join(cells) + " |"
        )

    lines.extend(
        [
            "",
            "`*` 表示每个 trial 先计算 percentile，再对 trial percentile 取 median；",
            "percentile 使用 linear interpolation，`rank=(N-1)*q`。max 是全部 measured raw samples 的最大值。",
            "",
            "所有 case 都使用 Continuous Batching，仅改变 mixed Prefill token budget。",
            "case 顺序逐轮循环轮换；表格是 measured samples 的中位数。",
            "完整 trial、request、inter-token、step CUDA Event 与显存原始样本见 `result.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def write_trial_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fields = (
        "round_index",
        "execution_position",
        "budget",
        "service_window_ms",
        "cuda_timeline_ms",
        "output_tokens_per_second",
        "peak_allocated_bytes",
        "peak_dynamic_allocated_bytes",
        "step_count",
        "ttft_p90_ms",
        "ttft_p95_ms",
        "ttft_max_ms",
        "tpot_p95_ms",
        "e2e_p90_ms",
        "e2e_p95_ms",
        "e2e_max_ms",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_data in rounds:
            for position, key in enumerate(round_data["execution_order"]):
                trial = round_data["trials"][key]
                tail = trial_tail_latency(trial)
                writer.writerow(
                    {
                        "round_index": round_data["round_index"],
                        "execution_position": position,
                        "budget": key,
                        "service_window_ms": trial["service_window_ns"] / 1e6,
                        "cuda_timeline_ms": trial["cuda_timeline_total_ms"],
                        "output_tokens_per_second": trial[
                            "output_tokens_per_second"
                        ],
                        "peak_allocated_bytes": trial["peak_allocated_bytes"],
                        "peak_dynamic_allocated_bytes": trial[
                            "peak_dynamic_allocated_bytes"
                        ],
                        "step_count": len(trial["steps"]),
                        **tail,
                    }
                )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 benchmark 要求 CUDA")
    if (
        args.request_count <= 0
        or args.max_running_requests <= 0
        or args.max_batch_tokens <= 0
        or args.block_size <= 0
        or args.warmup < 0
        or args.repeats <= 0
    ):
        raise SystemExit("count/budget/block/repeats 必须为正数，warmup 必须 >= 0")
    if args.max_running_requests > args.max_batch_tokens:
        raise SystemExit("max-running-requests 不能大于 max-batch-tokens")

    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    specs = tuple(
        RequestSpec(
            request_id=f"request_{index:03d}",
            prompt=PROMPTS[index % len(PROMPTS)],
            prompt_token_ids=tuple(
                tokenizer(PROMPTS[index % len(PROMPTS)])["input_ids"]
            ),
            max_new_tokens=args.generation_lengths[
                index % len(args.generation_lengths)
            ],
        )
        for index in range(args.request_count)
    )
    if any(len(spec.prompt_token_ids) > args.max_batch_tokens for spec in specs):
        raise SystemExit("至少一个 prompt 超过 max-batch-tokens")

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    if any(
        spec.max_cache_tokens > config.max_position_embeddings for spec in specs
    ):
        raise SystemExit("至少一个请求超过模型 max_position_embeddings")
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )

    budgets = args.mixed_prefill_budgets
    canonical_tokens: dict[str, tuple[int, ...]] | None = None
    for warmup_index in range(args.warmup):
        for budget in order_for_round(budgets, warmup_index):
            _, tokens = run_trial(
                policy=BatchingPolicy.CONTINUOUS,
                runner=runner,
                specs=specs,
                block_size=args.block_size,
                max_running_requests=args.max_running_requests,
                max_batch_tokens=args.max_batch_tokens,
                max_mixed_prefill_tokens=budget,
            )
            canonical_tokens = canonical_tokens or tokens
            if tokens != canonical_tokens:
                raise SystemExit("warmup 正确性失败：不同 budget 生成 token 不一致")
            gc.collect()

    case_keys = tuple(budget_key(value) for value in budgets)
    rounds = []
    case_trials: dict[str, list[dict[str, Any]]] = {
        key: [] for key in case_keys
    }
    for round_index in range(args.repeats):
        execution_order = order_for_round(budgets, round_index)
        trials: dict[str, dict[str, Any]] = {}
        for budget in execution_order:
            key = budget_key(budget)
            trial, tokens = run_trial(
                policy=BatchingPolicy.CONTINUOUS,
                runner=runner,
                specs=specs,
                block_size=args.block_size,
                max_running_requests=args.max_running_requests,
                max_batch_tokens=args.max_batch_tokens,
                max_mixed_prefill_tokens=budget,
            )
            canonical_tokens = canonical_tokens or tokens
            if tokens != canonical_tokens:
                raise SystemExit(
                    f"round {round_index} 正确性失败：budget={key} token 不一致"
                )
            trials[key] = trial
            case_trials[key].append(trial)
            gc.collect()
        rounds.append(
            {
                "round_index": round_index,
                "execution_order": tuple(budget_key(item) for item in execution_order),
                "trials": trials,
            }
        )

    result = {
        "schema_version": 2,
        "benchmark": "continuous_mixed_prefill_budget",
        "model": args.model,
        "case_keys": case_keys,
        "parameters": {
            "request_count": args.request_count,
            "generation_lengths": args.generation_lengths,
            "max_running_requests": args.max_running_requests,
            "max_batch_tokens": args.max_batch_tokens,
            "mixed_prefill_budgets": budgets,
            "block_size": args.block_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": str(weights.embedding.dtype),
            "arrival_pattern": "burst; all submitted before first schedule_step",
            "case_order": "cyclic rotation by round",
        },
        "request_specs": [
            {
                "request_id": spec.request_id,
                "prompt": spec.prompt,
                "prompt_token_ids": spec.prompt_token_ids,
                "max_new_tokens": spec.max_new_tokens,
            }
            for spec in specs
        ],
        "correctness": {
            "all_budget_token_sequences_equal": True,
            "generated_token_ids": canonical_tokens,
        },
        "measurement_definition": {
            "throughput": "total output tokens / (last token ready - first arrival)",
            "ttft_tpot_e2e": "CPU perf_counter_ns request timeline",
            "tail_latency": (
                "linear percentile rank=(N-1)*q within each trial, then median "
                "across measured trials; max spans all measured raw samples"
            ),
            "cuda_timeline": (
                "per-step CUDA Events on current default stream; includes device "
                "timeline gaps between recorded events; not sum of kernel durations"
            ),
            "peak_memory": (
                "torch.cuda max allocated/reserved; model and Paged KV pool are live; "
                "dynamic allocated subtracts pre-trial live allocation"
            ),
        },
        "rounds": rounds,
        "summary": {
            key: summarize_trials(trials) for key, trials in case_trials.items()
        },
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_trial_csv(args.output_dir / "trials.csv", rounds)
    (args.output_dir / "report.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(render_markdown(result))
    print(f"Raw JSON: {args.output_dir / 'result.json'}")
    print(f"Trial CSV: {args.output_dir / 'trials.csv'}")


if __name__ == "__main__":
    main()
