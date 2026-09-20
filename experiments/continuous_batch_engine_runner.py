#!/usr/bin/env python3
"""真实 Qwen 最小 Continuous Batching Engine 动态加入/离开演示。"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
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
from mini_llm_runtime.scheduler import RequestScheduler


REQUESTS = {
    "A": ("你好，GPU", 3),
    "B": ("CUDA 是什么？", 1),
    "C": ("请简要解释 KV Cache。", 2),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="可选 correctness artifact 目录；不提供时仅打印结果",
    )
    return parser.parse_args()


def hf_greedy_tokens(model, input_ids: torch.Tensor, count: int) -> tuple[int, ...]:
    """显式分离 HF Prefill/Decode，避免用 generate 隐藏 reference 数据流。"""
    generated: list[int] = []
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True, return_dict=True)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        cache = output.past_key_values
        generated.append(int(token.item()))
        for _ in range(count - 1):
            output = model(
                input_ids=token,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            generated.append(int(token.item()))
    return tuple(generated)


def record_step(result, manager: PagedKVCacheManager) -> dict[str, Any]:
    """保存调度、释放和 step 结束后的物理 block table 快照。"""
    active_cache = {}
    for request_id in manager.request_ids:
        table = manager.get_request(request_id)
        active_cache[request_id] = {
            "block_ids": table.block_ids,
            "token_count": table.token_count,
            "token_capacity": table.token_capacity,
        }
    return {
        "step": result.batch.step_index,
        "token_count": result.batch.token_count,
        "prefill": result.batch.prefill_request_ids,
        "decode": result.batch.decode_request_ids,
        "finished": result.update.finished_request_ids,
        "released_blocks": {
            request_id: block_ids
            for request_id, block_ids in result.released_blocks
        },
        "active_cache_after_step": active_cache,
        "cache_stats_after_step": asdict(manager.stats()),
    }


def evaluate_gate(
    *,
    step_records: list[dict[str, Any]],
    token_comparisons: dict[str, dict[str, Any]],
    block_size: int,
    all_resources_released: bool,
) -> dict[str, Any]:
    """把“跑过”提升为机器可验证的端到端覆盖条件。"""
    released_before: set[int] = set()
    reused_block_ids: set[int] = set()
    for record in step_records:
        active_ids = {
            int(block_id)
            for cache in record["active_cache_after_step"].values()
            for block_id in cache["block_ids"]
        }
        reused_block_ids.update(active_ids & released_before)
        released_before.update(
            int(block_id)
            for block_ids in record["released_blocks"].values()
            for block_id in block_ids
        )

    checks = {
        "all_token_sequences_match_hf": all(
            item["match"] for item in token_comparisons.values()
        ),
        "has_late_prefill_admission": any(
            record["step"] > 0 and record["prefill"] for record in step_records
        ),
        "has_mixed_prefill_decode_step": any(
            record["prefill"] and record["decode"] for record in step_records
        ),
        "has_batched_decode": any(
            len(record["decode"]) >= 2 for record in step_records
        ),
        "observes_cache_sequence_crossing_block_boundary": any(
            cache["token_count"] > block_size
            for record in step_records
            for cache in record["active_cache_after_step"].values()
        ),
        "reuses_released_physical_block": bool(reused_block_ids),
        "all_kv_resources_released": all_resources_released,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "reused_block_ids": tuple(sorted(reused_block_ids)),
    }


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Continuous Batching End-to-End Correctness Gate",
        "",
        f"- model: `{result['model']}`",
        f"- block size: `{result['block_size']}`",
        f"- git commit: `{result['environment_after'].get('git_commit')}` "
        f"(dirty={result['environment_after'].get('git_dirty')})",
        f"- overall passed: `{result['gate']['passed']}`",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    for name, passed in result["gate"]["checks"].items():
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "| Request | HF tokens | Engine tokens | Match |", "|---|---|---|---:|"])
    for request_id, item in result["token_comparisons"].items():
        lines.append(
            f"| {request_id} | `{item['expected']}` | `{item['actual']}` | "
            f"{item['match']} |"
        )
    lines.extend(
        [
            "",
            "本报告只证明当前固定 workload 的 correctness 与路径覆盖；时间线数值不属于 benchmark。",
            "完整 step、物理 block table、请求指标与环境信息见 `result.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")

    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = {
        request_id: tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        for request_id, (prompt, _) in REQUESTS.items()
    }

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    reference = {
        request_id: hf_greedy_tokens(hf_model, encoded[request_id], max_new_tokens)
        for request_id, (_, max_new_tokens) in REQUESTS.items()
    }
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )
    manager = PagedKVCacheManager(
        total_blocks=16,
        block_size=args.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=64,
        admission_callback=admission.try_admit,
    )
    engine = ContinuousBatchEngine(
        scheduler=scheduler, runner=runner, admission=admission
    )

    def submit(request_id: str) -> None:
        _, max_new_tokens = REQUESTS[request_id]
        engine.submit(
            request_id,
            encoded[request_id][0].cpu().tolist(),
            max_new_tokens=max_new_tokens,
        )

    submit("A")
    submit("B")
    step_records: list[dict[str, Any]] = []
    first = engine.step()
    if first is None:
        raise SystemExit("首个 Engine step 不应为空")
    step_records.append(record_step(first, manager))

    # 模拟 B 完成后、GPU step 边界到达的新请求 C。
    submit("C")
    while scheduler.has_unfinished_requests:
        result = engine.step()
        if result is None:
            raise SystemExit("存在未完成请求，但 block budget 无法取得进展")
        step_records.append(record_step(result, manager))

    print("===== Continuous Batch Engine Steps =====")
    for record in step_records:
        print(
            f"step={record['step']} prefill={record['prefill']} "
            f"decode={record['decode']} finished={record['finished']}"
        )

    print("\n===== Generated Tokens vs Hugging Face =====")
    failed: list[str] = []
    token_comparisons: dict[str, dict[str, Any]] = {}
    for request_id in ("A", "B", "C"):
        actual = scheduler.get_request(request_id).generated_token_ids
        expected = reference[request_id]
        match = actual == expected
        token_comparisons[request_id] = {
            "expected": expected,
            "actual": actual,
            "match": match,
            "decoded_text": tokenizer.decode(actual),
        }
        print(
            f"{request_id}: match={match} ids={actual} "
            f"text={tokenizer.decode(actual)!r}"
        )
        if not match:
            failed.append(request_id)

    all_released = (
        manager.request_ids == ()
        and admission.reservations == ()
        and manager.allocator.free_count == manager.allocator.total_blocks
    )
    print(f"\nall KV blocks released = {all_released}")
    print("\n===== Request Timeline（correctness run，非 benchmark） =====")
    request_metrics: dict[str, dict[str, Any]] = {}
    for request_id in ("A", "B", "C"):
        metrics = engine.metrics.snapshot(request_id)
        request_metrics[request_id] = {
            "queue_wait_ns": metrics.queue_wait_ns,
            "ttft_ns": metrics.ttft_ns,
            "inter_token_ns": metrics.inter_token_ns,
            "median_tpot_ns": metrics.median_tpot_ns,
            "e2e_ns": metrics.e2e_ns,
            "post_token_completion_ns": metrics.post_token_completion_ns,
        }
        print(
            f"{request_id}: queue={metrics.queue_wait_ms:.3f} ms, "
            f"TTFT={metrics.ttft_ms:.3f} ms, "
            f"TPOT_median={metrics.median_tpot_ms}, "
            f"E2E={metrics.e2e_ms:.3f} ms, "
            f"post_token={metrics.post_token_completion_ms:.3f} ms, "
            f"raw_inter_token_ns={metrics.inter_token_ns}"
        )
    request_specs = {
        request_id: {
            "prompt": prompt,
            "prompt_token_ids": tuple(encoded[request_id][0].cpu().tolist()),
            "prompt_tokens": int(encoded[request_id].shape[1]),
            "max_new_tokens": max_new_tokens,
            "max_cache_tokens": int(encoded[request_id].shape[1])
            + max_new_tokens
            - 1,
        }
        for request_id, (prompt, max_new_tokens) in REQUESTS.items()
    }
    gate = evaluate_gate(
        step_records=step_records,
        token_comparisons=token_comparisons,
        block_size=args.block_size,
        all_resources_released=all_released,
    )
    result_artifact = {
        "schema_version": 1,
        "artifact": "continuous_batching_end_to_end_correctness",
        "model": args.model,
        "dtype": str(weights.embedding.dtype),
        "block_size": args.block_size,
        "reference": "Hugging Face explicit greedy Prefill/Decode; no generate()",
        "request_specs": request_specs,
        "steps": step_records,
        "token_comparisons": token_comparisons,
        "request_metrics_debug_only": request_metrics,
        "gate": gate,
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "result.json").write_text(
            json.dumps(result_artifact, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / "report.md").write_text(
            render_report(result_artifact), encoding="utf-8"
        )
        print(f"correctness artifact = {args.output_dir}")

    if failed or not gate["passed"]:
        raise SystemExit(
            f"Engine 对拍失败：token_mismatch={failed}, gate={gate}"
        )
    print(f"gate checks = {gate['checks']}")
    print(f"reused physical blocks = {gate['reused_block_ids']}")
    print("全部通过：HF token、动态组批、Paged KV 复用与资源生命周期正确。")


if __name__ == "__main__":
    main()
