from copy import deepcopy

from scripts.check_engine_ragged_continuation import (
    evaluate_engine_gate,
    evaluate_late_arrival_gate,
)


def valid_case() -> dict:
    first_cache = {
        "A": {"token_count": 4, "token_capacity": 8, "block_ids": (0, 1)},
        "B": {"token_count": 1, "token_capacity": 4, "block_ids": (2,)},
    }
    second_cache = deepcopy(first_cache)
    second_cache["A"]["token_count"] = 5
    second_cache["B"]["token_count"] = 2
    return {
        "request_ids": ("A", "B"),
        "prompt_lengths": (4, 1),
        "expected_tokens": {"A": (10, 11, 12), "B": (20, 21, 22)},
        "actual_tokens": {"A": (10, 11, 12), "B": (20, 21, 22)},
        "steps": [
            {
                "prefill": ("A", "B"), "decode": (),
                "prefill_attention_backend": "masked",
                "emitted_tokens": (("A", 10), ("B", 20)),
                "finished": (), "released_blocks": {}, "active_cache": first_cache,
            },
            {
                "prefill": (), "decode": ("A", "B"),
                "prefill_attention_backend": None,
                "emitted_tokens": (("A", 11), ("B", 21)),
                "finished": (), "released_blocks": {}, "active_cache": second_cache,
            },
            {
                "prefill": (), "decode": ("A", "B"),
                "prefill_attention_backend": None,
                "emitted_tokens": (("A", 12), ("B", 22)),
                "finished": ("A", "B"),
                "released_blocks": {"A": (0, 1), "B": (2,)},
                "active_cache": {},
            },
        ],
        "new_tokens": 3,
        "block_size": 4,
        "final_state": {
            "waiting": (), "running": (), "finished": ("A", "B"),
            "active_cache_ids": (), "reservations": (),
            "free_blocks": 3, "total_blocks": 3,
        },
    }


def test_gate_accepts_engine_lifecycle() -> None:
    gate = evaluate_engine_gate(**valid_case())
    assert gate["passed"]
    assert all(gate["checks"].values())


def test_gate_rejects_wrong_hf_token() -> None:
    case = deepcopy(valid_case())
    case["actual_tokens"]["B"] = (20, 99, 22)
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["tokens_match_hf"]


def test_gate_rejects_wrong_committed_length_or_capacity() -> None:
    case = deepcopy(valid_case())
    case["steps"][1]["active_cache"]["A"]["token_count"] = 4
    case["steps"][0]["active_cache"]["B"]["token_capacity"] = 8
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["committed_lengths_and_reserved_capacity"]


def test_gate_rejects_changed_or_shared_block_table() -> None:
    case = deepcopy(valid_case())
    case["steps"][1]["active_cache"]["A"]["block_ids"] = (0, 2)
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["reserved_block_tables_stable_and_distinct"]


def test_gate_rejects_unreturned_blocks() -> None:
    case = deepcopy(valid_case())
    case["steps"][2]["released_blocks"]["A"] = (0,)
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["finished_blocks_released"]


def test_gate_rejects_missing_decode_batch() -> None:
    case = deepcopy(valid_case())
    case["steps"][1]["decode"] = ("A",)
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["later_steps_batch_decode"]


def test_gate_rejects_wrong_step_emission() -> None:
    case = deepcopy(valid_case())
    case["steps"][1]["emitted_tokens"] = (("A", 11), ("B", 99))
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["step_emissions_match_requests"]


def test_gate_rejects_leaked_reservation_even_if_summary_claims_success() -> None:
    case = deepcopy(valid_case())
    case["final_state"]["reservations"] = ("A",)
    case["final_state"]["all_resources_released"] = True
    gate = evaluate_engine_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["scheduler_and_cache_empty"]


def valid_late_case() -> dict:
    first = {
        "A": {"token_count": 4, "token_capacity": 8, "block_ids": (0, 1)},
        "B": {"token_count": 1, "token_capacity": 4, "block_ids": (2,)},
    }
    mixed = deepcopy(first)
    mixed["A"]["token_count"] = 5
    mixed["B"]["token_count"] = 2
    mixed["C"] = {"token_count": 3, "token_capacity": 4, "block_ids": (3,)}
    return {
        "initial_ids": ("A", "B"),
        "late_id": "C",
        "prompt_lengths": {"A": 4, "B": 1, "C": 3},
        "expected_tokens": {"A": (10, 11, 12), "B": (20, 21, 22), "C": (30, 31)},
        "actual_tokens": {"A": (10, 11, 12), "B": (20, 21, 22), "C": (30, 31)},
        "steps": [
            {
                "prefill": ("A", "B"), "decode": (), "batch_token_count": 5,
                "prefill_attention_backend": "masked",
                "emitted_tokens": (("A", 10), ("B", 20)),
                "finished": (), "released_blocks": {}, "active_cache": first,
            },
            {
                "prefill": ("C",), "decode": ("A", "B"),
                "batch_token_count": 5, "prefill_attention_backend": "masked",
                "emitted_tokens": (("A", 11), ("B", 21), ("C", 30)),
                "finished": (), "released_blocks": {}, "active_cache": mixed,
            },
            {
                "prefill": (), "decode": ("A", "B", "C"),
                "batch_token_count": 3, "prefill_attention_backend": None,
                "emitted_tokens": (("A", 12), ("B", 22), ("C", 31)),
                "finished": ("A", "B", "C"),
                "released_blocks": {"A": (0, 1), "B": (2,), "C": (3,)},
                "active_cache": {},
            },
        ],
        "block_size": 4,
        "final_state": {
            "waiting": (), "running": (), "finished": ("A", "B", "C"),
            "active_cache_ids": (), "reservations": (),
            "free_blocks": 4, "total_blocks": 4,
        },
    }


def test_late_gate_accepts_mixed_step_and_release() -> None:
    gate = evaluate_late_arrival_gate(**valid_late_case())
    assert gate["passed"]
    assert all(gate["checks"].values())


def test_late_gate_rejects_missing_mixed_prefill() -> None:
    case = deepcopy(valid_late_case())
    case["steps"][1]["prefill"] = ()
    gate = evaluate_late_arrival_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["mixed_schedule_and_budget"]


def test_late_gate_rejects_swapped_token_rows() -> None:
    case = deepcopy(valid_late_case())
    case["steps"][1]["emitted_tokens"] = (("C", 30), ("A", 11), ("B", 21))
    gate = evaluate_late_arrival_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["step_emissions_in_scheduler_order"]


def test_late_gate_rejects_shared_physical_block() -> None:
    case = deepcopy(valid_late_case())
    case["steps"][1]["active_cache"]["C"]["block_ids"] = (2,)
    gate = evaluate_late_arrival_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["physical_blocks_distinct_and_stable"]


def test_late_gate_rejects_unreleased_late_block() -> None:
    case = deepcopy(valid_late_case())
    case["steps"][2]["released_blocks"]["C"] = ()
    gate = evaluate_late_arrival_gate(**case)
    assert not gate["passed"]
    assert not gate["checks"]["finished_blocks_released"]
