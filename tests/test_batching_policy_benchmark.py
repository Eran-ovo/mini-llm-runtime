import argparse

import pytest

from scripts.benchmark_batching_policy import (
    order_for_round,
    parse_positive_int_list,
    summarize_trials,
)
from mini_llm_runtime.scheduler import BatchingPolicy


def make_trial(throughput: float, service_ns: int, cuda_ms: float) -> dict:
    return {
        "output_tokens_per_second": throughput,
        "service_window_ns": service_ns,
        "cuda_timeline_total_ms": cuda_ms,
        "peak_allocated_bytes": 100,
        "requests": [
            {"ttft_ns": 2_000_000, "inter_token_ns": (3_000_000,), "e2e_ns": 5_000_000}
        ],
    }


def test_policy_order_reverses_on_odd_rounds() -> None:
    assert order_for_round(0) == (
        BatchingPolicy.CONTINUOUS,
        BatchingPolicy.STATIC,
    )
    assert order_for_round(1) == (
        BatchingPolicy.STATIC,
        BatchingPolicy.CONTINUOUS,
    )


def test_summary_keeps_raw_samples_and_uses_median() -> None:
    summary = summarize_trials(
        [make_trial(10.0, 6_000_000, 4.0), make_trial(30.0, 10_000_000, 8.0)]
    )
    assert summary["trial_throughput_samples_tokens_per_second"] == [10.0, 30.0]
    assert summary["median_throughput_tokens_per_second"] == 20.0
    assert summary["median_service_window_ms"] == 8.0
    assert summary["median_cuda_timeline_ms"] == 6.0
    assert summary["median_request_ttft_ms"] == 2.0
    assert summary["median_tpot_ms"] == 3.0
    assert summary["median_request_e2e_ms"] == 5.0


def test_benchmark_argument_helpers_reject_invalid_values() -> None:
    assert parse_positive_int_list("2, 4,8") == (2, 4, 8)
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        parse_positive_int_list("2,0")
    with pytest.raises(ValueError, match="至少需要"):
        summarize_trials([])
