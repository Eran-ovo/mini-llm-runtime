from copy import deepcopy

from scripts.check_ragged_prefill_decode import (
    evaluate_continuation_gate,
    expand_prompts,
)


def valid_case() -> dict:
    return {
        "prompt_lengths": (4, 1),
        "request_ids": ("A", "B"),
        "expected_tokens": {"A": (10, 11, 12), "B": (20, 21, 22)},
        "actual_tokens": {"A": (10, 11, 12), "B": (20, 21, 22)},
        "snapshots": [
            {
                "phase": "prefill",
                "input_positions": {},
                "cache_lengths": {"A": 4, "B": 1},
                "block_ids": {"A": (0,), "B": (1,)},
            },
            {
                "phase": "decode",
                "input_positions": {"A": 4, "B": 1},
                "cache_lengths": {"A": 5, "B": 2},
                "block_ids": {"A": (0, 2), "B": (1,)},
            },
            {
                "phase": "decode",
                "input_positions": {"A": 5, "B": 2},
                "cache_lengths": {"A": 6, "B": 3},
                "block_ids": {"A": (0, 2), "B": (1,)},
            },
        ],
        "block_size": 4,
        "new_tokens": 3,
        "all_resources_released": True,
    }


def test_expand_prompts_preserves_per_request_boundaries() -> None:
    assert expand_prompts(((1, 2), (3, 4, 5)), (4, 1)) == (
        (1, 2, 1, 2),
        (3,),
    )


def test_gate_accepts_token_and_cache_continuation() -> None:
    result = evaluate_continuation_gate(**valid_case())
    assert result["passed"]
    assert all(result["checks"].values())


def test_gate_rejects_wrong_decode_token() -> None:
    case = deepcopy(valid_case())
    case["actual_tokens"]["B"] = (20, 99, 22)
    result = evaluate_continuation_gate(**case)
    assert not result["passed"]
    assert not result["checks"]["tokens_match_hf"]


def test_gate_rejects_wrong_decode_position() -> None:
    case = deepcopy(valid_case())
    case["snapshots"][1]["input_positions"]["B"] = 4
    result = evaluate_continuation_gate(**case)
    assert not result["passed"]
    assert not result["checks"]["cache_lengths_and_positions"]


def test_gate_rejects_missing_block_growth() -> None:
    case = deepcopy(valid_case())
    for snapshot in case["snapshots"][1:]:
        snapshot["block_ids"]["A"] = (0,)
    result = evaluate_continuation_gate(**case)
    assert not result["passed"]
    assert not result["checks"]["crosses_new_block_on_decode"]


def test_gate_rejects_reused_live_block() -> None:
    case = deepcopy(valid_case())
    case["snapshots"][1]["block_ids"]["A"] = (0, 1)
    result = evaluate_continuation_gate(**case)
    assert not result["passed"]
    assert not result["checks"]["distinct_physical_blocks"]


def test_gate_rejects_missing_token_or_unreleased_blocks() -> None:
    case = deepcopy(valid_case())
    case["actual_tokens"]["A"] = (10, 11)
    case["all_resources_released"] = False
    result = evaluate_continuation_gate(**case)
    assert not result["passed"]
    assert not result["checks"]["token_counts"]
    assert not result["checks"]["all_resources_released"]
