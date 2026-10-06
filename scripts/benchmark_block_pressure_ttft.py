#!/usr/bin/env python3
"""固定晚到请求，交错比较 28/29-block pool 对 TTFT 的影响。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.engine import ContinuousBatchEngine
from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import RequestScheduler


PROMPTS = (
    "你好，GPU",
    "请简要解释 KV Cache。",
    "CUDA 是什么？",
    "Paged Attention 如何管理显存？",
)
LATE_PROMPT = "稍后加入的短请求"
INITIAL_LENGTHS = (128, 128, 128, 1)
LATE_LENGTH = 7
BLOCK_SIZE = 16
INITIAL_IDS = tuple(f"request-{index}" for index in range(4))
LATE_ID = "request-late"
NEW_TOKENS = {**{rid: (2 if index == 0 else 3) for index, rid in enumerate(INITIAL_IDS)}, LATE_ID: 2}
PRESSURE_BLOCKS = 28
SPARE_BLOCKS = 29


def order_for_round(index: int) -> tuple[str, str]:
    return ("pressure", "spare") if index % 2 == 0 else ("spare", "pressure")


def validate_workload_capacity(prompts: dict[str, tuple[int, ...]]) -> None:
    """锁定单变量实验：首批恰占 28 block，晚到请求恰需 1 block。"""
    if set(prompts) != {*INITIAL_IDS, LATE_ID}:
        raise ValueError("workload 请求 ID 不完整")
    initial = sum(
        math.ceil((len(prompts[rid]) + NEW_TOKENS[rid] - 1) / BLOCK_SIZE)
        for rid in INITIAL_IDS
    )
    late = math.ceil(
        (len(prompts[LATE_ID]) + NEW_TOKENS[LATE_ID] - 1) / BLOCK_SIZE
    )
    if (initial, late) != (PRESSURE_BLOCKS, SPARE_BLOCKS - PRESSURE_BLOCKS):
        raise ValueError(f"pool 容量不再构成单变量压力实验：initial={initial}, late={late}")


def validate_trial(trial: dict, reference: dict[str, tuple[int, ...]]) -> dict:
    """正式计时必须同时满足 HF token 与预期 admission/reuse 路径。"""
    case = trial["case"]
    steps = trial["steps"]
    expected_schedule = (
        (
            (INITIAL_IDS, (), (LATE_ID,)),
            ((), INITIAL_IDS, (LATE_ID,) if case == "pressure" else ()),
            ((LATE_ID,), INITIAL_IDS[1:], ()) if case == "pressure"
            else ((), (*INITIAL_IDS[1:], LATE_ID), ()),
            ((), (LATE_ID,), ()),
        )
        if case == "pressure"
        else (
            (INITIAL_IDS, (), (LATE_ID,)),
            ((LATE_ID,), INITIAL_IDS, ()),
            ((), (*INITIAL_IDS[1:], LATE_ID), ()),
        )
    )
    checks = {
        "tokens_match_hf": trial["tokens"] == reference,
        "pool_capacity_matches_case": trial["total_blocks"] == (
            PRESSURE_BLOCKS if case == "pressure" else SPARE_BLOCKS
        ),
        "expected_step_count": len(steps) == len(expected_schedule),
        "admission_schedule": True,
        "waiting_has_no_cache_or_reservation": True,
        "physical_reuse_after_release": case == "spare",
        "all_blocks_returned": trial["final_free_blocks"] == trial["total_blocks"]
        and not trial["final_waiting"] and not trial["final_running"]
        and not trial["final_cache_ids"] and not trial["final_reservations"],
        "late_request_timeline_complete": (
            trial["late_ttft_ns"] is not None
            and trial["late_queue_wait_ns"] is not None
            and trial["late_e2e_ns"] is not None
            and len(trial["late_token_ready_offsets_ns"]) == 2
        ),
    }
    for index, step in enumerate(steps):
        if index >= len(expected_schedule):
            checks["admission_schedule"] = False
            continue
        expected_prefill, expected_decode, expected_waiting = expected_schedule[index]
        if (
            step["prefill"] != expected_prefill
            or step["decode"] != expected_decode
            or step["waiting_after_step"] != expected_waiting
        ):
            checks["admission_schedule"] = False
        if LATE_ID in expected_waiting and (
            LATE_ID in step["active_cache"]
            or LATE_ID in step["reservations_after_step"]
        ):
            checks["waiting_has_no_cache_or_reservation"] = False
    if case == "pressure" and len(steps) == 4:
        released = set(steps[1]["released_blocks"].get(INITIAL_IDS[0], ()))
        late_blocks = set(steps[2]["active_cache"].get(LATE_ID, {}).get("block_ids", ()))
        checks["physical_reuse_after_release"] = bool(late_blocks) and late_blocks <= released
        checks["waiting_has_no_cache_or_reservation"] &= (
            steps[0]["free_blocks_after_step"] == 0
            and steps[1]["free_blocks_after_step"] == len(released)
        )
    return {"passed": all(checks.values()), "checks": checks}


def summarize_trials(trials: list[dict]) -> dict:
    """每个 case 先保存逐轮 raw sample，再计算跨 trial 中位数。"""
    if not trials:
        raise ValueError("至少需要一个 measured trial")
    keys = (
        "late_ttft_ns", "late_queue_wait_ns", "late_e2e_ns",
        "cuda_timeline_total_ms", "peak_allocated_bytes",
    )
    return {
        key: {
            "samples": [item[key] for item in trials],
            "median": float(statistics.median(item[key] for item in trials)),
        }
        for key in keys
    }


def expand_prompt(template: tuple[int, ...], length: int) -> tuple[int, ...]:
    if not template or length <= 0:
        raise ValueError("token 模板与目标长度必须非空且为正")
    return (template * math.ceil(length / len(template)))[:length]


@torch.inference_mode()
def hf_reference(model, prompts: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    """逐请求显式 Prefill/Decode，绝不使用 generate()。"""
    reference = {}
    for rid, prompt in prompts.items():
        input_ids = torch.tensor((prompt,), dtype=torch.long, device="cuda")
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        tokens = [int(token.item())]
        for _ in range(NEW_TOKENS[rid] - 1):
            output = model(
                input_ids=token, past_key_values=cache,
                use_cache=True, return_dict=True,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            tokens.append(int(token.item()))
        reference[rid] = tuple(tokens)
    return reference


def run_trial(
    *, case: str, runner: QwenPrefillRunner, prompts: dict[str, tuple[int, ...]]
) -> dict:
    """每轮新建 Engine/Cache；晚到请求总在首个 step 完成后提交。"""
    total_blocks = PRESSURE_BLOCKS if case == "pressure" else SPARE_BLOCKS
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=BLOCK_SIZE,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=5,
        max_batch_tokens=sum(INITIAL_LENGTHS),
        admission_callback=admission.try_admit,
    )
    engine = ContinuousBatchEngine(
        scheduler=scheduler, runner=runner, admission=admission,
        prefill_attention_backend="auto", prefill_kv_write_backend="vectorized",
    )
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    for rid in INITIAL_IDS:
        engine.submit(rid, prompts[rid], max_new_tokens=NEW_TOKENS[rid])

    event_pairs = []
    steps = []
    while scheduler.has_unfinished_requests:
        if len(steps) >= 5:
            raise RuntimeError("超过预期 Engine step 数")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = engine.step()
        end.record()
        if result is None:
            raise RuntimeError("请求仍在 waiting/running，但 Scheduler 无法进展")
        event_pairs.append((start, end))
        step = {
            "step": result.batch.step_index,
            "prefill": result.batch.prefill_request_ids,
            "decode": result.batch.decode_request_ids,
            "batch_token_count": result.batch.token_count,
            "finished": result.update.finished_request_ids,
            "released_blocks": dict(result.released_blocks),
            "active_cache": {
                rid: {
                    "token_count": manager.get_request(rid).token_count,
                    "block_ids": manager.get_request(rid).block_ids,
                }
                for rid in manager.request_ids
            },
        }
        steps.append(step)
        if len(steps) == 1:
            engine.submit(LATE_ID, prompts[LATE_ID], max_new_tokens=NEW_TOKENS[LATE_ID])
        step["waiting_after_step"] = scheduler.waiting_request_ids
        step["reservations_after_step"] = tuple(
            item.request_id for item in admission.reservations
        )
        step["free_blocks_after_step"] = manager.allocator.free_count

    torch.cuda.synchronize()
    for step, (start, end) in zip(steps, event_pairs, strict=True):
        step["cuda_timeline_ms"] = float(start.elapsed_time(end))
    late = engine.metrics.snapshot(LATE_ID)
    first_arrival_ns = min(engine.metrics.snapshot(rid).arrival_ns for rid in INITIAL_IDS)
    trial = {
        "case": case,
        "total_blocks": total_blocks,
        "steps": steps,
        "tokens": {
            rid: scheduler.get_request(rid).generated_token_ids for rid in prompts
        },
        "late_arrival_offset_ns": late.arrival_ns - first_arrival_ns,
        "late_prefill_started_offsets_ns": [
            value - first_arrival_ns for value in late.prefill_attempt_started_ns
        ],
        "late_token_ready_offsets_ns": [
            item.ready_ns - first_arrival_ns for item in late.token_events
        ],
        "late_queue_wait_ns": late.queue_wait_ns,
        "late_ttft_ns": late.ttft_ns,
        "late_e2e_ns": late.e2e_ns,
        "cuda_timeline_total_ms": sum(step["cuda_timeline_ms"] for step in steps),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "final_free_blocks": manager.allocator.free_count,
        "final_waiting": scheduler.waiting_request_ids,
        "final_running": scheduler.running_request_ids,
        "final_cache_ids": manager.request_ids,
        "final_reservations": tuple(item.request_id for item in admission.reservations),
    }
    return trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("此 benchmark 需要 CUDA")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("warmup 必须 >= 0，repeats 必须 > 0")
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    templates = tuple(
        tuple(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in PROMPTS
    )
    prompts = {
        rid: expand_prompt(template, length)
        for rid, template, length in zip(INITIAL_IDS, templates, INITIAL_LENGTHS, strict=True)
    }
    late_template = tuple(tokenizer(LATE_PROMPT, add_special_tokens=True)["input_ids"])
    prompts[LATE_ID] = expand_prompt(late_template, LATE_LENGTH)
    validate_workload_capacity(prompts)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).cuda().eval()
    reference = hf_reference(hf_model, prompts)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(model_dir, device="cuda", dtype=torch.float16)
    if max(len(prompts[rid]) + NEW_TOKENS[rid] - 1 for rid in prompts) > config.max_position_embeddings:
        raise SystemExit("请求超过模型 max_position_embeddings")
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")

    rounds = []
    trials_by_case: dict[str, list[dict]] = {"pressure": [], "spare": []}
    for round_index in range(args.warmup + args.repeats):
        order = order_for_round(round_index)
        measured = round_index >= args.warmup
        trials = {}
        for case in order:
            trial = run_trial(case=case, runner=runner, prompts=prompts)
            gate = validate_trial(trial, reference)
            if not gate["passed"]:
                raise RuntimeError(f"{case} 正确性/路径失败：{gate}")
            trial["gate"] = gate
            if measured:
                trials[case] = trial
                trials_by_case[case].append(trial)
            gc.collect()
        if measured:
            rounds.append({"round_index": round_index - args.warmup, "execution_order": order, "trials": trials})

    paired_ttft_difference_ns = [
        item["trials"]["pressure"]["late_ttft_ns"]
        - item["trials"]["spare"]["late_ttft_ns"]
        for item in rounds
    ]
    environment_after = collect_environment(repo_root)
    clean_same_commit = (
        not environment_before["git_dirty"] and not environment_after["git_dirty"]
        and environment_before["git_commit"] == environment_after["git_commit"]
    )
    result = {
        "schema_version": 1,
        "classification": "formal_clean_tree" if clean_same_commit else "exploratory_dirty_tree",
        "benchmark": "block_pressure_late_request_ttft",
        "model": args.model,
        "parameters": {
            "initial_prompt_lengths": INITIAL_LENGTHS,
            "late_prompt_length": LATE_LENGTH,
            "max_new_tokens": NEW_TOKENS,
            "block_size": BLOCK_SIZE,
            "pool_blocks_by_case": {"pressure": PRESSURE_BLOCKS, "spare": SPARE_BLOCKS},
            "arrival_pattern": "late request submitted immediately after step 0 returns",
            "warmup": args.warmup,
            "repeats": args.repeats,
            "case_order": "interleaved; reversed on odd rounds",
            "dtype": str(weights.embedding.dtype),
        },
        "prompt_token_ids": prompts,
        "hf_reference_tokens": reference,
        "measurement_definition": {
            "late_ttft_ns": "CPU perf_counter_ns: late submit to first token ready after D2H sync",
            "late_queue_wait_ns": "CPU perf_counter_ns: late submit to first Prefill attempt",
            "cuda_timeline_ms": "per-step CUDA Events; device timeline including host orchestration gaps, not kernel-only time",
            "peak_allocated_bytes": "torch.cuda max allocated; model and preallocated KV pool live",
            "comparison": "only block pool capacity changes (28 versus 29); arrival is step-relative, not fixed wall-clock",
        },
        "rounds": rounds,
        "summary": {case: summarize_trials(trials) for case, trials in trials_by_case.items()},
        "paired_pressure_minus_spare_ttft_ns": {
            "samples": paired_ttft_difference_ns,
            "median": float(statistics.median(paired_ttft_difference_ns)),
        },
        "environment_before": environment_before,
        "environment_after": environment_after,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "classification": result["classification"],
        "pressure_ttft_median_ms": result["summary"]["pressure"]["late_ttft_ns"]["median"] / 1e6,
        "spare_ttft_median_ms": result["summary"]["spare"]["late_ttft_ns"]["median"] / 1e6,
        "paired_difference_median_ms": result["paired_pressure_minus_spare_ttft_ns"]["median"] / 1e6,
        "result": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
