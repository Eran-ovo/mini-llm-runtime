#!/usr/bin/env python3
"""验证变长 packed Prefill 经 Engine/Scheduler 续写并释放 KV Cache。"""

from __future__ import annotations

import argparse
import gc
import json
import math
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
PROMPT_LENGTHS = (128, 128, 128, 1)
NEW_TOKENS = 3
BLOCK_SIZE = 16
LATE_PROMPT = "稍后加入的短请求"
LATE_PROMPT_LENGTH = 7
LATE_NEW_TOKENS = 2


def evaluate_engine_gate(
    *,
    request_ids: tuple[str, ...],
    prompt_lengths: tuple[int, ...],
    expected_tokens: dict[str, tuple[int, ...]],
    actual_tokens: dict[str, tuple[int, ...]],
    steps: list[dict],
    new_tokens: int,
    block_size: int,
    final_state: dict,
) -> dict:
    """区分已提交 token_count 与提前预留的 token_capacity。"""
    first = steps[0] if steps else None
    checks = {
        "tokens_match_hf": expected_tokens == actual_tokens
        and all(len(actual_tokens.get(rid, ())) == new_tokens for rid in request_ids),
        "step_emissions_match_requests": len(steps) == new_tokens,
        "first_step_packs_ragged_prefill": bool(first)
        and first["prefill"] == request_ids
        and not first["decode"]
        and first["prefill_attention_backend"] == "masked",
        "later_steps_batch_decode": len(steps) == new_tokens
        and all(
            not step["prefill"] and step["decode"] == request_ids
            for step in steps[1:]
        ),
        "committed_lengths_and_reserved_capacity": True,
        "reserved_block_tables_stable_and_distinct": True,
        "logical_block_boundary_crossed": False,
        "finished_blocks_released": False,
        "scheduler_and_cache_empty": (
            not final_state["waiting"]
            and not final_state["running"]
            and final_state["finished"] == request_ids
            and not final_state["active_cache_ids"]
            and not final_state["reservations"]
            and final_state["free_blocks"] == final_state["total_blocks"]
        ),
    }
    initial_blocks = first["active_cache"] if first else {}
    for step_index, step in enumerate(steps):
        if (
            step_index >= new_tokens
            or any(len(actual_tokens.get(rid, ())) <= step_index for rid in request_ids)
            or step["emitted_tokens"] != tuple(
                (rid, actual_tokens[rid][step_index]) for rid in request_ids
            )
        ):
            checks["step_emissions_match_requests"] = False
        active = step["active_cache"]
        if step_index == new_tokens - 1:
            # 最后一个生成 token 不再作为 Decode 输入；Engine 当轮释放全部请求。
            if active:
                checks["finished_blocks_released"] = False
            continue
        if set(active) != set(request_ids):
            checks["committed_lengths_and_reserved_capacity"] = False
            checks["reserved_block_tables_stable_and_distinct"] = False
            continue
        live_blocks = []
        for rid, prompt_length in zip(request_ids, prompt_lengths, strict=True):
            cache = active[rid]
            max_cache_tokens = prompt_length + new_tokens - 1
            expected_capacity = math.ceil(max_cache_tokens / block_size) * block_size
            if cache["token_count"] != prompt_length + step_index:
                checks["committed_lengths_and_reserved_capacity"] = False
            if cache["token_capacity"] != expected_capacity:
                checks["committed_lengths_and_reserved_capacity"] = False
            if len(cache["block_ids"]) != expected_capacity // block_size:
                checks["reserved_block_tables_stable_and_distinct"] = False
            if cache["block_ids"] != initial_blocks[rid]["block_ids"]:
                checks["reserved_block_tables_stable_and_distinct"] = False
            live_blocks.extend(cache["block_ids"])
            if step_index > 0 and (
                (prompt_length - 1) // block_size
                < (cache["token_count"] - 1) // block_size
            ):
                checks["logical_block_boundary_crossed"] = True
        if len(live_blocks) != len(set(live_blocks)):
            checks["reserved_block_tables_stable_and_distinct"] = False

    if steps and len(steps) == new_tokens:
        last = steps[-1]
        checks["finished_blocks_released"] = (
            not last["active_cache"]
            and last["finished"] == request_ids
            and all(not step["released_blocks"] for step in steps[:-1])
            and set(last["released_blocks"]) == set(request_ids)
            and all(
                last["released_blocks"][rid] == initial_blocks[rid]["block_ids"]
                for rid in request_ids
            )
        )
    return {"passed": all(checks.values()), "checks": checks}


def evaluate_late_arrival_gate(
    *,
    initial_ids: tuple[str, ...],
    late_id: str,
    prompt_lengths: dict[str, int],
    expected_tokens: dict[str, tuple[int, ...]],
    actual_tokens: dict[str, tuple[int, ...]],
    steps: list[dict],
    block_size: int,
    final_state: dict,
) -> dict:
    """验证晚到请求与旧请求共处一轮，且 token 行和 block 归属不串位。"""
    all_ids = (*initial_ids, late_id)
    expected_schedule = (
        (initial_ids, (), initial_ids, sum(prompt_lengths[rid] for rid in initial_ids)),
        ((late_id,), initial_ids, (*initial_ids, late_id), len(initial_ids) + prompt_lengths[late_id]),
        ((), all_ids, all_ids, len(all_ids)),
    )
    checks = {
        "tokens_match_hf": expected_tokens == actual_tokens
        and all(len(actual_tokens.get(rid, ())) == (2 if rid == late_id else 3) for rid in all_ids),
        "mixed_schedule_and_budget": len(steps) == 3,
        "step_emissions_in_scheduler_order": len(steps) == 3,
        "committed_lengths_and_reserved_capacity": True,
        "physical_blocks_distinct_and_stable": True,
        "logical_block_boundary_crossed": False,
        "finished_blocks_released": False,
        "scheduler_and_cache_empty": (
            not final_state["waiting"]
            and not final_state["running"]
            and final_state["finished"] == all_ids
            and not final_state["active_cache_ids"]
            and not final_state["reservations"]
            and final_state["free_blocks"] == final_state["total_blocks"]
        ),
    }
    # 第 0/1 步成功后 Cache 可见；最后一步全部完成并立即释放。
    expected_live_lengths = (
        {rid: prompt_lengths[rid] for rid in initial_ids},
        {**{rid: prompt_lengths[rid] + 1 for rid in initial_ids}, late_id: prompt_lengths[late_id]},
    )
    first_seen_blocks: dict[str, tuple[int, ...]] = {}
    for index, step in enumerate(steps):
        if index >= len(expected_schedule):
            checks["mixed_schedule_and_budget"] = False
            checks["step_emissions_in_scheduler_order"] = False
            continue
        prefill, decode, order, token_budget = expected_schedule[index]
        if (
            step["prefill"] != prefill
            or step["decode"] != decode
            or step["batch_token_count"] != token_budget
            or step["prefill_attention_backend"] != ("masked" if prefill else None)
        ):
            checks["mixed_schedule_and_budget"] = False
        expected_indices = {rid: index for rid in initial_ids}
        if index > 0:
            expected_indices[late_id] = index - 1
        expected_emissions = tuple(
            (rid, actual_tokens.get(rid, ())[expected_indices[rid]])
            for rid in order
        ) if all(
            len(actual_tokens.get(rid, ())) > expected_indices[rid] for rid in order
        ) else ()
        if step["emitted_tokens"] != expected_emissions:
            checks["step_emissions_in_scheduler_order"] = False

        active = step["active_cache"]
        if index < 2:
            expected_lengths = expected_live_lengths[index]
            if set(active) != set(expected_lengths):
                checks["committed_lengths_and_reserved_capacity"] = False
                checks["physical_blocks_distinct_and_stable"] = False
                continue
            live_blocks = []
            for rid, expected_length in expected_lengths.items():
                cache = active[rid]
                max_cache = prompt_lengths[rid] + (1 if rid == late_id else 2)
                expected_blocks = math.ceil(max_cache / block_size)
                if (
                    cache["token_count"] != expected_length
                    or cache["token_capacity"] != expected_blocks * block_size
                ):
                    checks["committed_lengths_and_reserved_capacity"] = False
                blocks = cache["block_ids"]
                if len(blocks) != expected_blocks:
                    checks["physical_blocks_distinct_and_stable"] = False
                if rid in first_seen_blocks and blocks != first_seen_blocks[rid]:
                    checks["physical_blocks_distinct_and_stable"] = False
                first_seen_blocks.setdefault(rid, blocks)
                live_blocks.extend(blocks)
                if index == 1 and (
                    (prompt_lengths[rid] - 1) // block_size
                    < (cache["token_count"] - 1) // block_size
                ):
                    checks["logical_block_boundary_crossed"] = True
            if len(live_blocks) != len(set(live_blocks)):
                checks["physical_blocks_distinct_and_stable"] = False

    if len(steps) == 3:
        last = steps[-1]
        checks["finished_blocks_released"] = (
            not last["active_cache"]
            and last["finished"] == all_ids
            and all(not step["released_blocks"] for step in steps[:-1])
            and set(last["released_blocks"]) == set(all_ids)
            and all(
                last["released_blocks"][rid] == first_seen_blocks.get(rid)
                for rid in all_ids
            )
        )
    return {"passed": all(checks.values()), "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--late-short-request", action="store_true",
        help="首轮 Prefill 后提交 7-token 请求，验证下一轮 mixed Prefill/Decode",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("此 correctness gate 需要 CUDA")
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    templates = tuple(
        tuple(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in PROMPTS
    )
    prompts = tuple(
        (template * math.ceil(length / len(template)))[:length]
        for template, length in zip(templates, PROMPT_LENGTHS, strict=True)
    )
    initial_ids = tuple(f"request-{index}" for index in range(len(prompts)))
    late_id = "request-late"
    prompt_by_id = dict(zip(initial_ids, prompts, strict=True))
    if args.late_short_request:
        late_template = tuple(
            tokenizer(LATE_PROMPT, add_special_tokens=True)["input_ids"]
        )
        prompt_by_id[late_id] = (
            late_template * math.ceil(LATE_PROMPT_LENGTH / len(late_template))
        )[:LATE_PROMPT_LENGTH]
    request_ids = tuple(prompt_by_id)
    prompt_lengths = {rid: len(prompt) for rid, prompt in prompt_by_id.items()}

    # HF reference 显式 Prefill/Decode；与 Runtime 分时加载，适配 6 GB GPU。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float16, attn_implementation="eager", local_files_only=True
    ).cuda().eval()
    expected_tokens: dict[str, tuple[int, ...]] = {}
    with torch.inference_mode():
        for rid, prompt in prompt_by_id.items():
            input_ids = torch.tensor((prompt,), dtype=torch.long, device="cuda")
            output = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            generated = [int(token.item())]
            target_tokens = LATE_NEW_TOKENS if rid == late_id else NEW_TOKENS
            for _ in range(target_tokens - 1):
                output = hf_model(
                    input_ids=token, past_key_values=cache,
                    use_cache=True, return_dict=True,
                )
                token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                generated.append(int(token.item()))
            expected_tokens[rid] = tuple(generated)
    del hf_model, output, cache, token, input_ids
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(model_dir, device="cuda", dtype=torch.float16)
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    total_blocks = sum(
        math.ceil(
            (length + (LATE_NEW_TOKENS if rid == late_id else NEW_TOKENS) - 1)
            / BLOCK_SIZE
        )
        for rid, length in prompt_lengths.items()
    )
    manager = PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=BLOCK_SIZE,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=len(request_ids),
        max_batch_tokens=sum(PROMPT_LENGTHS),
        admission_callback=admission.try_admit,
    )
    engine = ContinuousBatchEngine(
        scheduler=scheduler,
        runner=runner,
        admission=admission,
        prefill_attention_backend="auto",
        prefill_kv_write_backend="vectorized",
    )
    for rid in initial_ids:
        prompt = prompt_by_id[rid]
        engine.submit(rid, prompt, max_new_tokens=NEW_TOKENS)

    steps = []
    while scheduler.has_unfinished_requests:
        if len(steps) >= NEW_TOKENS + 1:
            raise RuntimeError("Engine 超过预期 step 数，可能未能取得进展")
        result = engine.step()
        if result is None:
            raise RuntimeError("仍有未完成请求，但 Scheduler 无法调度")
        steps.append({
            "step": result.batch.step_index,
            "batch_token_count": result.batch.token_count,
            "prefill": result.batch.prefill_request_ids,
            "decode": result.batch.decode_request_ids,
            "prefill_attention_backend": result.prefill_attention_backend,
            "emitted_tokens": result.update.emitted_tokens,
            "finished": result.update.finished_request_ids,
            "released_blocks": dict(result.released_blocks),
            "active_cache": {
                rid: {
                    "token_count": manager.get_request(rid).token_count,
                    "token_capacity": manager.get_request(rid).token_capacity,
                    "block_ids": manager.get_request(rid).block_ids,
                }
                for rid in manager.request_ids
            },
        })
        if args.late_short_request and len(steps) == 1:
            # GPU step 边界到达：新请求必须由下一轮 Scheduler 接纳。
            engine.submit(
                late_id, prompt_by_id[late_id], max_new_tokens=LATE_NEW_TOKENS
            )

    actual_tokens = {
        rid: scheduler.get_request(rid).generated_token_ids for rid in request_ids
    }
    final_state = {
        "waiting": scheduler.waiting_request_ids,
        "running": scheduler.running_request_ids,
        "finished": scheduler.finished_request_ids,
        "active_cache_ids": manager.request_ids,
        "reservations": tuple(item.request_id for item in admission.reservations),
        "free_blocks": manager.allocator.free_count,
        "total_blocks": manager.allocator.total_blocks,
    }
    final_state["all_resources_released"] = (
        not scheduler.has_unfinished_requests
        and not final_state["waiting"]
        and not final_state["running"]
        and not final_state["active_cache_ids"]
        and not final_state["reservations"]
        and final_state["free_blocks"] == final_state["total_blocks"]
    )
    if args.late_short_request:
        gate = evaluate_late_arrival_gate(
            initial_ids=initial_ids,
            late_id=late_id,
            prompt_lengths=prompt_lengths,
            expected_tokens=expected_tokens,
            actual_tokens=actual_tokens,
            steps=steps,
            block_size=BLOCK_SIZE,
            final_state=final_state,
        )
    else:
        gate = evaluate_engine_gate(
            request_ids=request_ids,
            prompt_lengths=PROMPT_LENGTHS,
            expected_tokens=expected_tokens,
            actual_tokens=actual_tokens,
            steps=steps,
            new_tokens=NEW_TOKENS,
            block_size=BLOCK_SIZE,
            final_state=final_state,
        )
    result = {
        "schema_version": 2 if args.late_short_request else 1,
        "artifact": (
            "engine_ragged_mixed_step_correctness" if args.late_short_request
            else "engine_ragged_prefill_decode_correctness"
        ),
        "scenario": "late_short_request" if args.late_short_request else "initial_cohort",
        "reference": "HF eager explicit Prefill/Decode; no generate()",
        "model": args.model,
        "dtype": str(weights.embedding.dtype),
        "prompt_lengths": prompt_lengths if args.late_short_request else PROMPT_LENGTHS,
        "prompt_token_ids": prompt_by_id,
        "new_tokens": NEW_TOKENS,
        "late_new_tokens": LATE_NEW_TOKENS if args.late_short_request else None,
        "max_new_tokens": {
            rid: LATE_NEW_TOKENS if rid == late_id else NEW_TOKENS
            for rid in request_ids
        },
        "block_size": BLOCK_SIZE,
        "total_blocks": total_blocks,
        "prefill_backend": "auto",
        "prefill_kv_write_backend": "vectorized",
        "decode_backend": "paged_cuda",
        "expected_tokens": expected_tokens,
        "actual_tokens": actual_tokens,
        "steps": steps,
        "final_state": final_state,
        "gate": gate,
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"gate": gate, "result": str(target)}, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit("Engine 变长续写与 HF 对拍失败；详见 result.json")


if __name__ == "__main__":
    main()
