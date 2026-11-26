from argparse import Namespace

import pytest

from scripts import benchmark_vllm


@pytest.mark.parametrize("arrival_ms", [30, 90])
def test_timed_workload_measures_from_arrival_and_handles_idle_gaps(monkeypatch, arrival_ms):
    now = 0
    active = {}
    submissions = []

    def advance(seconds):
        nonlocal now
        now += int(seconds * 1e9)

    def submit(request_id, prompt, output_length):
        submissions.append((request_id, now))
        active[request_id] = [prompt[0], output_length]

    def step():
        assert active
        advance(0.020)
        emitted = []
        for request_id, (token, remaining) in list(active.items()):
            emitted.append((request_id, [token], remaining == 1))
            if remaining == 1:
                del active[request_id]
            else:
                active[request_id][1] -= 1
        return emitted

    monkeypatch.setattr(benchmark_vllm.time, "perf_counter_ns", lambda: now)
    monkeypatch.setattr(benchmark_vllm.time, "sleep", advance)
    workload = benchmark_vllm.Workload((
        benchmark_vllm.RequestSpec(1, 3),
        benchmark_vllm.RequestSpec(1, 2, arrival_ms),
    ))
    result = benchmark_vllm.measure(
        submit, step, workload, [[3], [7]], Namespace(
            ttft_slo_ms=25, itl_slo_ms=25, record_tokens=True
        ),
    )
    assert result["output_token_ids"] == [[3, 3, 3], [7, 7]]
    assert result["initial_itl_max_ms"] == 20
    assert submissions == [("0", 0), ("1", max(40, arrival_ms) * 1_000_000)]
    assert result["late_ttft_p95_ms"] == (30 if arrival_ms == 30 else 20)
    assert result["goodput_requests_per_second"] == pytest.approx(
        (1 if arrival_ms == 30 else 2) * 1e9 / now
    )
