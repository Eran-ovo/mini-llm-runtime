from copy import deepcopy

import pytest

from scripts.analyze_kv_reservation import (
    analyze_next_step_capacity,
    analyze_source,
    analyze_step,
)


def first_step(free_blocks: int) -> dict:
    return {
        "step": 0,
        "active_cache": {
            "A": {"token_count": 4, "block_ids": (0, 1)},
            "B": {"token_count": 1, "block_ids": (2,)},
        },
        "free_blocks_after_step": free_blocks,
    }


def test_step_separates_future_reservation_from_tail_slack() -> None:
    result = analyze_step(first_step(0), block_size=4, total_blocks=3)
    assert result["allocated_blocks"] == 3
    assert result["committed_tokens"] == 5
    assert result["minimal_blocks_for_committed_tokens"] == 2
    assert result["future_reserved_blocks"] == 1
    assert result["future_reserved_slots"] == 4
    assert result["tail_slack_slots"] == 3
    assert result["uncommitted_slots"] == 7


def test_step_rejects_shared_block_or_inconsistent_free_count() -> None:
    duplicate = first_step(0)
    duplicate["active_cache"]["B"]["block_ids"] = (1,)
    with pytest.raises(ValueError, match="重叠"):
        analyze_step(duplicate, block_size=4, total_blocks=3)
    with pytest.raises(ValueError, match="pool 大小"):
        analyze_step(first_step(1), block_size=4, total_blocks=3)


def test_next_step_growth_plus_late_prefill_exceeds_pressure_pool() -> None:
    next_step = {"decode": ("A", "B")}
    pressure = analyze_next_step_capacity(
        first_step=first_step(0), next_step=next_step,
        block_size=4, total_blocks=3,
        late_request_id="C", late_prompt_length=3,
    )
    spare = analyze_next_step_capacity(
        first_step=first_step(1), next_step=next_step,
        block_size=4, total_blocks=4,
        late_request_id="C", late_prompt_length=3,
    )
    assert pressure["decode_growth_blocks_by_request"] == {"A": 1, "B": 0}
    assert pressure["total_new_blocks_required"] == 2
    assert pressure["available_blocks_if_only_current_tokens_allocated"] == 1
    assert pressure["shortfall_blocks"] == 1
    assert not pressure["fits_without_preemption_or_step_reordering"]
    assert spare["shortfall_blocks"] == 0
    assert spare["fits_without_preemption_or_step_reordering"]


def source_result() -> dict:
    return {
        "benchmark": "block_pressure_late_request_ttft",
        "classification": "formal_clean_tree",
        "environment_before": {"git_dirty": False, "git_commit": "commit-a"},
        "environment_after": {"git_dirty": False, "git_commit": "commit-a"},
        "parameters": {
            "block_size": 4, "late_prompt_length": 3, "repeats": 1,
            "pool_blocks_by_case": {"pressure": 3, "spare": 4},
        },
        "rounds": [
            {"round_index": 0, "trials": {
                case: {
                    "total_blocks": total,
                    "gate": {"passed": True},
                    "steps": [first_step(total - 3), {
                        "step": 1, "decode": ("A", "B"),
                        "active_cache": {}, "free_blocks_after_step": total,
                    }],
                }
                for case, total in (("pressure", 3), ("spare", 4))
            }}
        ],
    }


def test_source_analysis_preserves_each_trial_and_clean_origin() -> None:
    result = analyze_source(source_result())
    assert result["source_commit"] == "commit-a"
    assert len(result["raw_trial_analyses"]) == 2
    assert result["canonical_by_case"]["pressure"]["steps"][0]["future_reserved_blocks"] == 1


def test_source_rejects_dirty_or_failed_gate() -> None:
    dirty = deepcopy(source_result())
    dirty["environment_after"]["git_dirty"] = True
    with pytest.raises(ValueError, match="Git 状态"):
        analyze_source(dirty)
    failed = deepcopy(source_result())
    failed["rounds"][0]["trials"]["pressure"]["gate"]["passed"] = False
    with pytest.raises(ValueError, match="gate 失败"):
        analyze_source(failed)


def test_source_rejects_missing_measured_round() -> None:
    source = source_result()
    source["parameters"]["repeats"] = 2
    with pytest.raises(ValueError, match="rounds 数"):
        analyze_source(source)
