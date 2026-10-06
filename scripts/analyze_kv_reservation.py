#!/usr/bin/env python3
"""从正式 block-pressure benchmark 原始轨迹推导 KV 预留与已提交量。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from mini_llm_runtime.environment import collect_environment


def analyze_step(step: dict, *, block_size: int, total_blocks: int) -> dict:
    """仅用 CPU 元数据分解未来预留与当前 block 的尾部空位。"""
    if block_size <= 0 or total_blocks <= 0:
        raise ValueError("block_size 与 total_blocks 必须为正")
    per_request = {}
    all_block_ids = []
    for rid, cache in step["active_cache"].items():
        committed = int(cache["token_count"])
        block_ids = tuple(int(value) for value in cache["block_ids"])
        if committed <= 0:
            raise ValueError(f"活动请求 {rid} 的 committed token 必须为正")
        minimal_blocks = math.ceil(committed / block_size)
        if len(block_ids) < minimal_blocks:
            raise ValueError(f"请求 {rid} 的 block table 装不下已提交 token")
        per_request[rid] = {
            "committed_tokens": committed,
            "allocated_blocks": len(block_ids),
            "minimal_blocks_for_committed_tokens": minimal_blocks,
            "future_reserved_blocks": len(block_ids) - minimal_blocks,
            "tail_slack_slots": minimal_blocks * block_size - committed,
            "uncommitted_slots": len(block_ids) * block_size - committed,
        }
        all_block_ids.extend(block_ids)
    if len(all_block_ids) != len(set(all_block_ids)):
        raise ValueError("活动请求的物理 block ID 重叠")
    if any(block_id < 0 or block_id >= total_blocks for block_id in all_block_ids):
        raise ValueError("block ID 越界")
    allocated = len(all_block_ids)
    if allocated + int(step["free_blocks_after_step"]) != total_blocks:
        raise ValueError("分配数 + free 数与 pool 大小不一致")
    committed = sum(item["committed_tokens"] for item in per_request.values())
    minimal = sum(item["minimal_blocks_for_committed_tokens"] for item in per_request.values())
    future = allocated - minimal
    tail = sum(item["tail_slack_slots"] for item in per_request.values())
    uncommitted = allocated * block_size - committed
    if uncommitted != future * block_size + tail:
        raise RuntimeError("未提交 slot 分解不守恒")
    return {
        "step": step["step"],
        "allocated_blocks": allocated,
        "free_blocks": total_blocks - allocated,
        "allocated_token_slots": allocated * block_size,
        "committed_tokens": committed,
        "minimal_blocks_for_committed_tokens": minimal,
        "future_reserved_blocks": future,
        "future_reserved_slots": future * block_size,
        "tail_slack_slots": tail,
        "uncommitted_slots": uncommitted,
        "per_request": per_request,
    }


def analyze_next_step_capacity(
    *, first_step: dict, next_step: dict, block_size: int, total_blocks: int,
    late_request_id: str, late_prompt_length: int,
) -> dict:
    """反事实：只按当前已提交长度分配时，下一 step 是否能同时运行。"""
    first = analyze_step(first_step, block_size=block_size, total_blocks=total_blocks)
    growth = {}
    for rid in next_step["decode"]:
        if rid not in first_step["active_cache"]:
            raise ValueError(f"下一步 Decode 请求 {rid} 在当前 Cache 中不存在")
        length = int(first_step["active_cache"][rid]["token_count"])
        growth[rid] = math.ceil((length + 1) / block_size) - math.ceil(length / block_size)
    late_prefill_blocks = math.ceil(late_prompt_length / block_size)
    if late_request_id in first_step["active_cache"]:
        raise ValueError("晚到请求不应已在首步 Cache 中")
    available_if_incremental = total_blocks - first["minimal_blocks_for_committed_tokens"]
    required = sum(growth.values()) + late_prefill_blocks
    return {
        "model": "same next step; existing Decode + late Prefill concurrent before any completion release",
        "available_blocks_if_only_current_tokens_allocated": available_if_incremental,
        "decode_growth_blocks_by_request": growth,
        "late_prefill_blocks": late_prefill_blocks,
        "total_new_blocks_required": required,
        "shortfall_blocks": max(0, required - available_if_incremental),
        "fits_without_preemption_or_step_reordering": required <= available_if_incremental,
    }


def analyze_source(source: dict) -> dict:
    """拒绝非正式或 gate 失败的输入；保留逐轮推导与共同结论。"""
    if source.get("benchmark") != "block_pressure_late_request_ttft":
        raise ValueError("仅接受 block-pressure TTFT benchmark")
    if source.get("classification") != "formal_clean_tree":
        raise ValueError("仅接受 formal_clean_tree 原始结果")
    before = source["environment_before"]
    after = source["environment_after"]
    if (
        before["git_dirty"] or after["git_dirty"]
        or before["git_commit"] != after["git_commit"]
    ):
        raise ValueError("原始结果 Git 状态不满足 clean-tree 约束")
    params = source["parameters"]
    if len(source["rounds"]) != int(params["repeats"]):
        raise ValueError("原始 measured rounds 数与 repeats 不一致")
    block_size = int(params["block_size"])
    late_request_id = "request-late"
    late_length = int(params["late_prompt_length"])
    raw = []
    canonical = {}
    for round_record in source["rounds"]:
        round_index = int(round_record["round_index"])
        for case in ("pressure", "spare"):
            trial = round_record["trials"][case]
            if not trial["gate"]["passed"]:
                raise ValueError(f"round={round_index} case={case} gate 失败")
            total_blocks = int(trial["total_blocks"])
            if total_blocks != int(params["pool_blocks_by_case"][case]):
                raise ValueError("trial 的 pool 大小与参数不一致")
            steps = [
                analyze_step(step, block_size=block_size, total_blocks=total_blocks)
                for step in trial["steps"]
            ]
            next_step = analyze_next_step_capacity(
                first_step=trial["steps"][0],
                next_step=trial["steps"][1],
                block_size=block_size,
                total_blocks=total_blocks,
                late_request_id=late_request_id,
                late_prompt_length=late_length,
            )
            values = {"steps": steps, "next_step_counterfactual": next_step}
            if case in canonical and values != canonical[case]:
                raise ValueError(f"case={case} 的容量轨迹跨轮不一致")
            canonical.setdefault(case, values)
            raw.append({"round_index": round_index, "case": case, **values})
    if not raw:
        raise ValueError("原始结果没有 measured trials")
    return {
        "source_commit": before["git_commit"],
        "source_repeats": int(params["repeats"]),
        "block_size": block_size,
        "canonical_by_case": canonical,
        "raw_trial_analyses": raw,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes)
    analysis = analyze_source(source)
    result = {
        "schema_version": 1,
        "artifact": "derived_kv_reservation_analysis_not_a_new_benchmark",
        "source_path": str(args.source.resolve()),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "analysis": analysis,
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }
    result["classification"] = (
        "derived_clean_tree"
        if not environment_before["git_dirty"]
        and not result["environment_after"]["git_dirty"]
        and environment_before["git_commit"] == result["environment_after"]["git_commit"]
        else "exploratory_dirty_tree"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    first = analysis["canonical_by_case"]["pressure"]
    print(json.dumps({
        "pressure_first_step": first["steps"][0],
        "pressure_next_step_counterfactual": first["next_step_counterfactual"],
        "result": str(target),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
