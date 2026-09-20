import argparse

import pytest

from scripts.benchmark_batching_policy import (
    linear_percentile,
    order_for_round,
    parse_positive_int_list,
    summarize_trials,
)
from mini_llm_runtime.scheduler import BatchingPolicy


def make_trial(
    throughput: float,
    service_ns: int,
    cuda_ms: float,
    request_samples: tuple[tuple[str, int, int], ...] = (("A", 2, 5),),
) -> dict:
    return {
        "output_tokens_per_second": throughput,
        "service_window_ns": service_ns,
        "cuda_timeline_total_ms": cuda_ms,
        "peak_allocated_bytes": 100,
        "requests": [
            {
                "request_id": request_id,
                "ttft_ns": ttft_ms * 1_000_000,
                "inter_token_ns": (3_000_000,),
                "e2e_ns": e2e_ms * 1_000_000,
            }
            for request_id, ttft_ms, e2e_ms in request_samples
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


def test_summary_reports_trial_tails_and_request_positions() -> None:
    summary = summarize_trials(
        [
            make_trial(
                10.0,
                6_000_000,
                4.0,
                (("A", 1, 10), ("B", 9, 18)),
            ),
            make_trial(
                20.0,
                7_000_000,
                5.0,
                (("A", 3, 12), ("B", 7, 16)),
            ),
        ]
    )

    # 每个 trial 先算 percentile，再在 trial 之间取 median。
    assert summary["trial_p90_request_ttft_samples_ms"] == pytest.approx([8.2, 6.6])
    assert summary["median_trial_p90_request_ttft_ms"] == pytest.approx(7.4)
    assert summary["trial_p95_request_ttft_samples_ms"] == pytest.approx([8.6, 6.8])
    assert summary["median_trial_p95_request_ttft_ms"] == pytest.approx(7.7)
    assert summary["median_trial_max_request_ttft_ms"] == 8.0
    assert summary["max_request_ttft_ms"] == 9.0
    assert summary["median_trial_p95_request_e2e_ms"] == pytest.approx(16.7)
    assert summary["max_request_e2e_ms"] == 18.0
    assert summary["per_request_position"][0]["request_id"] == "A"
    assert summary["per_request_position"][0]["median_ttft_ms"] == 2.0
    assert summary["per_request_position"][1]["median_e2e_ms"] == 17.0


def test_linear_percentile_boundaries_and_interpolation() -> None:
    assert linear_percentile([4.0, 1.0, 3.0, 2.0], 0) == 1.0
    assert linear_percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.5
    assert linear_percentile([4.0, 1.0, 3.0, 2.0], 90) == pytest.approx(3.7)
    assert linear_percentile([4.0, 1.0, 3.0, 2.0], 100) == 4.0
    with pytest.raises(ValueError, match="至少需要"):
        linear_percentile([], 95)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        linear_percentile([1.0], 101)


def test_summary_rejects_request_order_mismatch() -> None:
    first = make_trial(10.0, 6_000_000, 4.0, (("A", 1, 2), ("B", 2, 3)))
    second = make_trial(10.0, 6_000_000, 4.0, (("B", 2, 3), ("A", 1, 2)))
    with pytest.raises(ValueError, match="request 顺序"):
        summarize_trials([first, second])


def test_benchmark_argument_helpers_reject_invalid_values() -> None:
    assert parse_positive_int_list("2, 4,8") == (2, 4, 8)
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        parse_positive_int_list("2,0")
    with pytest.raises(ValueError, match="至少需要"):
        summarize_trials([])
