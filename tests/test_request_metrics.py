import pytest

from mini_llm_runtime.request_metrics import RequestMetricsCollector


def test_request_metrics_preserve_raw_events_and_derive_latencies() -> None:
    metrics = RequestMetricsCollector()
    metrics.register_request("A", 1_000_000)
    metrics.record_prefill_started("A", 2_000_000)
    metrics.record_tokens((("A", 10),), 6_000_000)
    metrics.record_tokens((("A", 11),), 9_000_000)
    metrics.record_tokens((("A", 12),), 15_000_000)
    metrics.record_completed(("A",), 16_000_000)

    snapshot = metrics.snapshot("A")
    assert snapshot.queue_wait_ns == 1_000_000
    assert snapshot.ttft_ns == 5_000_000
    assert snapshot.inter_token_ns == (3_000_000, 6_000_000)
    assert snapshot.median_tpot_ns == 4_500_000
    assert snapshot.e2e_ns == 14_000_000
    assert snapshot.post_token_completion_ns == 1_000_000
    assert snapshot.queue_wait_ms == 1.0
    assert snapshot.ttft_ms == 5.0
    assert snapshot.median_tpot_ms == 4.5
    assert snapshot.e2e_ms == 14.0
    assert snapshot.post_token_completion_ms == 1.0
    assert [event.token_id for event in snapshot.token_events] == [10, 11, 12]


def test_single_token_has_no_tpot_and_failed_prefill_attempt_is_retained() -> None:
    metrics = RequestMetricsCollector()
    metrics.register_request("A", 100)
    metrics.record_prefill_started("A", 110)
    metrics.record_prefill_started("A", 130)
    metrics.record_tokens((("A", 7),), 180)
    metrics.record_completed(("A",), 190)

    snapshot = metrics.snapshot("A")
    assert snapshot.prefill_attempt_started_ns == (110, 130)
    assert snapshot.ttft_ns == 80
    assert snapshot.inter_token_ns == ()
    assert snapshot.median_tpot_ns is None


def test_metrics_reject_unknown_duplicate_or_time_travel_events() -> None:
    metrics = RequestMetricsCollector()
    metrics.register_request("A", 100)
    with pytest.raises(ValueError, match="已经注册"):
        metrics.register_request("A", 110)
    with pytest.raises(RuntimeError, match="engine.submit"):
        metrics.require_registered(("A", "missing"))
    with pytest.raises(ValueError, match="不能倒退"):
        metrics.record_prefill_started("A", 99)
    with pytest.raises(KeyError, match="尚未注册"):
        metrics.snapshot("missing")

    metrics.record_prefill_started("A", 110)
    metrics.record_tokens((("A", 1),), 120)
    with pytest.raises(RuntimeError, match="不能再次开始 Prefill"):
        metrics.record_prefill_started("A", 130)
    with pytest.raises(ValueError, match="同一请求"):
        metrics.record_tokens((("A", 2), ("A", 3)), 130)


def test_multi_request_token_record_is_atomic_on_invalid_input() -> None:
    metrics = RequestMetricsCollector()
    metrics.register_request("A", 100)
    metrics.register_request("B", 100)
    metrics.record_prefill_started("A", 110)
    metrics.record_prefill_started("B", 110)

    with pytest.raises(ValueError, match="token_id 必须是整数"):
        metrics.record_tokens((("A", 1), ("B", True)), 120)

    assert metrics.snapshot("A").token_events == ()
    assert metrics.snapshot("B").token_events == ()
