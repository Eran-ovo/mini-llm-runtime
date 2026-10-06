from copy import deepcopy

import pytest

from scripts.benchmark_block_pressure_ttft import (
    INITIAL_IDS,
    LATE_ID,
    order_for_round,
    summarize_trials,
    validate_trial,
    validate_workload_capacity,
)


REFERENCE = {
    INITIAL_IDS[0]: (10, 11),
    INITIAL_IDS[1]: (20, 21, 22),
    INITIAL_IDS[2]: (30, 31, 32),
    INITIAL_IDS[3]: (40, 41, 42),
    LATE_ID: (50, 51),
}


def step(prefill=(), decode=(), waiting=(), released=None, late_blocks=None, free=0):
    active = {} if late_blocks is None else {LATE_ID: {"block_ids": late_blocks}}
    return {
        "prefill": prefill,
        "decode": decode,
        "waiting_after_step": waiting,
        "released_blocks": released or {},
        "active_cache": active,
        "reservations_after_step": (),
        "free_blocks_after_step": free,
    }


def trial(case: str) -> dict:
    if case == "pressure":
        steps = [
            step(prefill=INITIAL_IDS, waiting=(LATE_ID,), free=0),
            step(decode=INITIAL_IDS, waiting=(LATE_ID,),
                 released={INITIAL_IDS[0]: (0, 1)}, free=2),
            step(prefill=(LATE_ID,), decode=INITIAL_IDS[1:], late_blocks=(0,)),
            step(decode=(LATE_ID,)),
        ]
    else:
        steps = [
            step(prefill=INITIAL_IDS, waiting=(LATE_ID,)),
            step(prefill=(LATE_ID,), decode=INITIAL_IDS, late_blocks=(28,)),
            step(decode=(*INITIAL_IDS[1:], LATE_ID)),
        ]
    return {
        "case": case,
        "steps": steps,
        "tokens": REFERENCE,
        "late_ttft_ns": 100,
        "late_queue_wait_ns": 20,
        "late_e2e_ns": 200,
        "late_token_ready_offsets_ns": (100, 200),
        "total_blocks": 28 if case == "pressure" else 29,
        "final_free_blocks": 28 if case == "pressure" else 29,
        "final_waiting": (),
        "final_running": (),
        "final_cache_ids": (),
        "final_reservations": (),
        "cuda_timeline_total_ms": 1.5,
        "peak_allocated_bytes": 1000,
    }


@pytest.mark.parametrize("case", ("pressure", "spare"))
def test_gate_accepts_expected_schedule(case: str) -> None:
    result = validate_trial(trial(case), REFERENCE)
    assert result["passed"]
    assert all(result["checks"].values())


def test_gate_rejects_premature_admission_while_pool_is_full() -> None:
    sample = trial("pressure")
    sample["steps"][1]["waiting_after_step"] = ()
    sample["steps"][1]["reservations_after_step"] = (LATE_ID,)
    result = validate_trial(sample, REFERENCE)
    assert not result["passed"]
    assert not result["checks"]["waiting_has_no_cache_or_reservation"]


def test_gate_rejects_non_reused_block() -> None:
    sample = trial("pressure")
    sample["steps"][2]["active_cache"][LATE_ID]["block_ids"] = (9,)
    result = validate_trial(sample, REFERENCE)
    assert not result["passed"]
    assert not result["checks"]["physical_reuse_after_release"]


def test_gate_rejects_token_mismatch() -> None:
    sample = deepcopy(trial("spare"))
    sample["tokens"][LATE_ID] = (50, 99)
    result = validate_trial(sample, REFERENCE)
    assert not result["passed"]
    assert not result["checks"]["tokens_match_hf"]


def test_order_and_summary_keep_raw_samples() -> None:
    assert order_for_round(0) == ("pressure", "spare")
    assert order_for_round(1) == ("spare", "pressure")
    samples = [trial("pressure"), trial("pressure")]
    samples[1]["late_ttft_ns"] = 300
    result = summarize_trials(samples)
    assert result["late_ttft_ns"] == {"samples": [100, 300], "median": 200.0}


def test_workload_capacity_rejects_accidental_shape_change() -> None:
    prompts = {
        rid: (1,) * length
        for rid, length in zip(INITIAL_IDS, (128, 128, 128, 1), strict=True)
    }
    prompts[LATE_ID] = (2,) * 7
    validate_workload_capacity(prompts)
    prompts[INITIAL_IDS[0]] = (1,) * 112
    with pytest.raises(ValueError, match="单变量压力实验"):
        validate_workload_capacity(prompts)
