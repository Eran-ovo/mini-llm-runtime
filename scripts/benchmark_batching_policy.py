#!/usr/bin/env python3
"""用相同 burst workload 正式比较 Static 与 Continuous Batching。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.engine import ContinuousBatchEngine
from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import BatchingPolicy, RequestScheduler


PROMPTS = (
    "你好，GPU",
    "请简要解释 KV Cache。",
    "CUDA 是什么？",
    "Paged Attention 如何管理显存？",
)


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    prompt: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int

    @property
    def max_cache_tokens(self) -> int:
        return len(self.prompt_token_ids) + self.max_new_tokens - 1


def parse_positive_int_list(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("generation lengths 必须是逗号分隔整数") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("generation lengths 中每个值必须 > 0")
    return values


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
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def order_for_round(round_index: int) -> tuple[BatchingPolicy, BatchingPolicy]:
    policies = (BatchingPolicy.CONTINUOUS, BatchingPolicy.STATIC)
    return policies if round_index % 2 == 0 else tuple(reversed(policies))


def build_engine(
    *,
    policy: BatchingPolicy,
    runner: QwenPrefillRunner,
    specs: tuple[RequestSpec, ...],
    block_size: int,
    max_running_requests: int,
    max_batch_tokens: int,
) -> ContinuousBatchEngine:
    # 容量覆盖全部请求完整生命周期，排除 block OOM 对 admission policy 的干扰。
    total_blocks = sum(
        math.ceil(spec.max_cache_tokens / block_size) for spec in specs
    )
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=max_running_requests,
        max_batch_tokens=max_batch_tokens,
        admission_callback=admission.try_admit,
        batching_policy=policy,
    )
    return ContinuousBatchEngine(
        scheduler=scheduler,
        runner=runner,
        admission=admission,
    )


def run_trial(
    *,
    policy: BatchingPolicy,
    runner: QwenPrefillRunner,
    specs: tuple[RequestSpec, ...],
    block_size: int,
    max_running_requests: int,
    max_batch_tokens: int,
) -> tuple[dict[str, Any], dict[str, tuple[int, ...]]]:
    engine = build_engine(
        policy=policy,
        runner=runner,
        specs=specs,
        block_size=block_size,
        max_running_requests=max_running_requests,
        max_batch_tokens=max_batch_tokens,
    )
    device = runner.weights.embedding.device
    torch.cuda.synchronize(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    baseline_reserved = int(torch.cuda.memory_reserved(device))
    torch.cuda.reset_peak_memory_stats(device)

    host_started_ns = time.perf_counter_ns()
    # Burst workload：所有 arrival 都发生在第一个 schedule_step 之前。
    for spec in specs:
        engine.submit(
            spec.request_id,
            spec.prompt_token_ids,
            max_new_tokens=spec.max_new_tokens,
        )

    step_records: list[dict[str, Any]] = []
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    while engine.scheduler.has_unfinished_requests:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        step_host_started = time.perf_counter_ns()
        result = engine.step()
        step_host_ended = time.perf_counter_ns()
        end_event.record()
        if result is None:
            raise RuntimeError("存在未完成请求，但 Scheduler 无法取得进展")
        event_pairs.append((start_event, end_event))
        step_records.append(
            {
                "step_index": result.batch.step_index,
                "prefill_request_ids": result.batch.prefill_request_ids,
                "decode_request_ids": result.batch.decode_request_ids,
                "token_count": result.batch.token_count,
                "finished_request_ids": result.update.finished_request_ids,
                "host_step_ns": step_host_ended - step_host_started,
            }
        )

    torch.cuda.synchronize(device)
    host_completed_ns = time.perf_counter_ns()
    cuda_samples_ms = [
        float(start.elapsed_time(end)) for start, end in event_pairs
    ]
    for record, cuda_ms in zip(step_records, cuda_samples_ms, strict=True):
        record["cuda_timeline_ms"] = cuda_ms

    snapshots = engine.metrics.snapshots()
    first_arrival_ns = min(item.arrival_ns for item in snapshots)
    last_token_ns = max(item.token_events[-1].ready_ns for item in snapshots)
    service_window_ns = last_token_ns - first_arrival_ns
    total_output_tokens = sum(len(item.token_events) for item in snapshots)
    request_records = []
    token_signatures: dict[str, tuple[int, ...]] = {}
    for item in snapshots:
        token_signatures[item.request_id] = tuple(
            event.token_id for event in item.token_events
        )
        request_records.append(
            {
                "request_id": item.request_id,
                "arrival_offset_ns": item.arrival_ns - first_arrival_ns,
                "prefill_attempt_started_offset_ns": [
                    value - first_arrival_ns
                    for value in item.prefill_attempt_started_ns
                ],
                "token_events": [
                    {
                        "token_id": event.token_id,
                        "ready_offset_ns": event.ready_ns - first_arrival_ns,
                    }
                    for event in item.token_events
                ],
                "completion_offset_ns": (
                    item.completed_ns - first_arrival_ns
                    if item.completed_ns is not None
                    else None
                ),
                "queue_wait_ns": item.queue_wait_ns,
                "ttft_ns": item.ttft_ns,
                "inter_token_ns": item.inter_token_ns,
                "median_tpot_ns": item.median_tpot_ns,
                "e2e_ns": item.e2e_ns,
                "post_token_completion_ns": item.post_token_completion_ns,
            }
        )

    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    trial = {
        "policy": policy.value,
        "host_trial_ns": host_completed_ns - host_started_ns,
        "service_window_ns": service_window_ns,
        "cuda_timeline_samples_ms": cuda_samples_ms,
        "cuda_timeline_total_ms": sum(cuda_samples_ms),
        "total_output_tokens": total_output_tokens,
        "output_tokens_per_second": total_output_tokens * 1e9 / service_window_ns,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_dynamic_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "steps": step_records,
        "requests": request_records,
    }
    return trial, token_signatures


def summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    if not trials:
        raise ValueError("至少需要一个 measured trial")
    throughput = [float(item["output_tokens_per_second"]) for item in trials]
    service_ms = [float(item["service_window_ns"]) / 1e6 for item in trials]
    cuda_ms = [float(item["cuda_timeline_total_ms"]) for item in trials]
    peak_bytes = [int(item["peak_allocated_bytes"]) for item in trials]
    ttft_ms = [
        float(request["ttft_ns"]) / 1e6
        for trial in trials
        for request in trial["requests"]
    ]
    tpot_ms = [
        float(sample) / 1e6
        for trial in trials
        for request in trial["requests"]
        for sample in request["inter_token_ns"]
    ]
    e2e_ms = [
        float(request["e2e_ns"]) / 1e6
        for trial in trials
        for request in trial["requests"]
    ]
    return {
        "trial_throughput_samples_tokens_per_second": throughput,
        "median_throughput_tokens_per_second": float(statistics.median(throughput)),
        "trial_service_window_samples_ms": service_ms,
        "median_service_window_ms": float(statistics.median(service_ms)),
        "trial_cuda_timeline_samples_ms": cuda_ms,
        "median_cuda_timeline_ms": float(statistics.median(cuda_ms)),
        "peak_allocated_samples_bytes": peak_bytes,
        "max_peak_allocated_bytes": max(peak_bytes),
        "request_ttft_samples_ms": ttft_ms,
        "median_request_ttft_ms": float(statistics.median(ttft_ms)),
        "inter_token_samples_ms": tpot_ms,
        "median_tpot_ms": float(statistics.median(tpot_ms)) if tpot_ms else None,
        "request_e2e_samples_ms": e2e_ms,
        "median_request_e2e_ms": float(statistics.median(e2e_ms)),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Static vs Continuous Batching Benchmark",
        "",
        f"- model: `{result['model']}`",
        f"- requests: `{result['parameters']['request_count']}` (burst arrival)",
        f"- warmup / repeats: `{result['parameters']['warmup']} / {result['parameters']['repeats']}`",
        f"- git commit: `{result['environment_after'].get('git_commit')}` "
        f"(dirty={result['environment_after'].get('git_dirty')})",
        "",
        "| Policy | Throughput median (tok/s) | TTFT median (ms) | TPOT median (ms) | E2E median (ms) | CUDA timeline median (ms) | Peak allocated (MiB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("continuous", "static"):
        item = result["summary"][policy]
        tpot = item["median_tpot_ms"]
        tpot_text = f"{tpot:.4f}" if tpot is not None else "N/A"
        lines.append(
            f"| {policy} | {item['median_throughput_tokens_per_second']:.2f} | "
            f"{item['median_request_ttft_ms']:.4f} | {tpot_text} | "
            f"{item['median_request_e2e_ms']:.4f} | "
            f"{item['median_cuda_timeline_ms']:.4f} | "
            f"{item['max_peak_allocated_bytes'] / 2**20:.2f} |"
        )
    lines.extend(
        [
            "",
            "策略逐轮交错且奇偶轮反转顺序。表格来自 measured trials 的中位数；",
            "所有 trial、request、inter-token、step CUDA Event 与显存原始样本见 `result.json`。",
            "`trials.csv` 每行保存一个 policy trial，便于后续绘图。",
            "",
        ]
    )
    return "\n".join(lines)


def write_trial_csv(path: Path, rounds: list[dict[str, Any]]) -> None:
    fields = (
        "round_index",
        "execution_position",
        "policy",
        "service_window_ms",
        "cuda_timeline_ms",
        "output_tokens_per_second",
        "peak_allocated_bytes",
        "peak_dynamic_allocated_bytes",
        "step_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for round_data in rounds:
            for position, policy in enumerate(round_data["execution_order"]):
                trial = round_data["trials"][policy]
                writer.writerow(
                    {
                        "round_index": round_data["round_index"],
                        "execution_position": position,
                        "policy": policy,
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
                    }
                )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 benchmark 要求 CUDA")
    generation_lengths = args.generation_lengths
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
            max_new_tokens=generation_lengths[index % len(generation_lengths)],
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

    canonical_tokens: dict[str, tuple[int, ...]] | None = None
    for warmup_index in range(args.warmup):
        for policy in order_for_round(warmup_index):
            _, tokens = run_trial(
                policy=policy,
                runner=runner,
                specs=specs,
                block_size=args.block_size,
                max_running_requests=args.max_running_requests,
                max_batch_tokens=args.max_batch_tokens,
            )
            canonical_tokens = canonical_tokens or tokens
            if tokens != canonical_tokens:
                raise SystemExit("warmup 正确性失败：策略生成 token 不一致")
            gc.collect()

    rounds = []
    policy_trials: dict[str, list[dict[str, Any]]] = {
        policy.value: [] for policy in BatchingPolicy
    }
    for round_index in range(args.repeats):
        execution_order = order_for_round(round_index)
        trials: dict[str, dict[str, Any]] = {}
        for policy in execution_order:
            trial, tokens = run_trial(
                policy=policy,
                runner=runner,
                specs=specs,
                block_size=args.block_size,
                max_running_requests=args.max_running_requests,
                max_batch_tokens=args.max_batch_tokens,
            )
            canonical_tokens = canonical_tokens or tokens
            if tokens != canonical_tokens:
                raise SystemExit(
                    f"round {round_index} 正确性失败：{policy.value} token 不一致"
                )
            trials[policy.value] = trial
            policy_trials[policy.value].append(trial)
            gc.collect()
        rounds.append(
            {
                "round_index": round_index,
                "execution_order": tuple(item.value for item in execution_order),
                "trials": trials,
            }
        )

    result = {
        "schema_version": 1,
        "benchmark": "static_vs_continuous_burst_batching",
        "model": args.model,
        "parameters": {
            "request_count": args.request_count,
            "generation_lengths": generation_lengths,
            "max_running_requests": args.max_running_requests,
            "max_batch_tokens": args.max_batch_tokens,
            "block_size": args.block_size,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": str(weights.embedding.dtype),
            "arrival_pattern": "burst; all submitted before first schedule_step",
            "case_order": "interleaved; reversed on odd rounds",
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
            "all_policy_token_sequences_equal": True,
            "generated_token_ids": canonical_tokens,
        },
        "measurement_definition": {
            "throughput": "total output tokens / (last token ready - first arrival)",
            "ttft_tpot_e2e": "CPU perf_counter_ns request timeline",
            "cuda_timeline": (
                "per-step CUDA Events on current default stream; includes device "
                "timeline gaps (including stream idle caused by CPU orchestration) "
                "between recorded events; not the sum of kernel durations"
            ),
            "peak_memory": (
                "torch.cuda max allocated/reserved; model and Paged KV pool are live; "
                "dynamic allocated subtracts pre-trial live allocation"
            ),
        },
        "rounds": rounds,
        "summary": {
            policy: summarize_trials(trials)
            for policy, trials in policy_trials.items()
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
