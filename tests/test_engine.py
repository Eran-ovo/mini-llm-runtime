from dataclasses import replace

import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.engine import (
    ContinuousBatchEngine,
    _nvtx_range,
    select_prefill_attention_backend,
)
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import BatchingPolicy, RequestScheduler, WorkKind

from test_qwen_prefill_cache import make_runner


def test_nvtx_range_balances_push_pop_on_success_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_push",
        lambda message: events.append(("push", message)),
    )
    monkeypatch.setattr(
        torch.cuda.nvtx,
        "range_pop",
        lambda: events.append(("pop", None)),
    )

    with _nvtx_range(True, "success"):
        pass
    with pytest.raises(RuntimeError, match="injected"):
        with _nvtx_range(True, "failure"):
            raise RuntimeError("injected")
    with _nvtx_range(False, "disabled"):
        pass

    assert events == [
        ("push", "success"),
        ("pop", None),
        ("push", "failure"),
        ("pop", None),
    ]


def make_engine(
    monkeypatch: pytest.MonkeyPatch,
    *,
    batching_policy: BatchingPolicy = BatchingPolicy.CONTINUOUS,
    max_mixed_prefill_tokens: int | None = None,
    prefill_attention_backend: str = "auto",
    prefill_kv_write_backend: str = "vectorized",
) -> tuple[ContinuousBatchEngine, RequestScheduler, PagedBlockAdmissionController]:
    base, weights = make_runner()
    runner = QwenPrefillRunner(
        base.config, weights, decode_attention_backend="paged_cuda"
    )
    manager = PagedKVCacheManager(
        total_blocks=12,
        block_size=2,
        num_layers=runner.config.num_hidden_layers,
        num_kv_heads=runner.config.num_key_value_heads,
        head_dim=runner.config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=5,
        max_mixed_prefill_tokens=max_mixed_prefill_tokens,
        admission_callback=admission.try_admit,
        batching_policy=batching_policy,
    )

    def reference_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return paged_decode_attention_reference(
            query, key, value, table, lengths
        ).output

    monkeypatch.setattr(
        model_runner_module, "paged_decode_attention_cuda", reference_attention
    )
    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        reference_attention,
    )
    return (
        ContinuousBatchEngine(
            scheduler=scheduler,
            runner=runner,
            admission=admission,
            prefill_attention_backend=prefill_attention_backend,
            prefill_kv_write_backend=prefill_kv_write_backend,
        ),
        scheduler,
        admission,
    )


def test_engine_packs_same_step_prefill_into_one_runner_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, _, _ = make_engine(monkeypatch)
    calls: list[tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...], str, str]] = []
    original = engine.runner.prefill_batch

    def observed(input_ids: torch.Tensor, *, cache, attention_backend):
        calls.append(
            (
                cache.request_ids,
                cache.lengths,
                tuple(input_ids[0].tolist()),
                attention_backend,
                cache.write_backend,
            )
        )
        return original(input_ids, cache=cache, attention_backend=attention_backend)

    monkeypatch.setattr(engine.runner, "prefill_batch", observed)
    engine.submit("A", (1, 2, 3), max_new_tokens=2)
    engine.submit("B", (4, 0), max_new_tokens=1)
    result = engine.step()

    assert result is not None
    assert calls == [(('A', 'B'), (3, 2), (1, 2, 3, 4, 0), 'masked', 'vectorized')]
    assert result.batch.prefill_request_ids == ("A", "B")
    assert result.prefill_attention_backend == "masked"


@pytest.mark.parametrize(
    ("lengths", "expected"),
    [
        ((8, 8, 8, 8), "masked"),
        ((64, 64, 64, 64), "masked"),
        ((128, 128, 128, 128), "segmented_sdpa"),
        ((256, 256, 256, 256), "segmented_sdpa"),
        ((64,) * 8, "segmented_sdpa"),
        ((32,) * 16, "segmented_sdpa"),
        ((16,) * 32, "masked"),
        ((128,) + (13,) * 31, "segmented_sdpa"),
        ((8, 64, 128, 256), "masked"),
        ((1,) * 512, "masked"),
        ((128, 1), "masked"),
    ],
)
def test_auto_prefill_backend_uses_total_longest_and_average_prompt(
    lengths: tuple[int, ...], expected: str
) -> None:
    assert select_prefill_attention_backend("auto", lengths) == expected


def test_explicit_prefill_backend_overrides_auto_shape() -> None:
    assert select_prefill_attention_backend("segmented_sdpa", (1,)) == "segmented_sdpa"
    assert select_prefill_attention_backend("masked", (512,)) == "masked"
    with pytest.raises(ValueError, match="lengths"):
        select_prefill_attention_backend("auto", ())
    with pytest.raises(ValueError, match="lengths"):
        select_prefill_attention_backend("auto", (0, 512))


@pytest.mark.parametrize(
    ("request_count", "prompt_length", "expected_backend"),
    [
        (4, 128, "segmented_sdpa"),
        (8, 64, "segmented_sdpa"),
        (16, 32, "segmented_sdpa"),
        (32, 16, "masked"),
    ],
)
def test_engine_auto_dispatches_packed_prefill_by_shape(
    request_count: int, prompt_length: int, expected_backend: str
) -> None:
    base, weights = make_runner()
    config = replace(base.config, max_position_embeddings=prompt_length)
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    manager = PagedKVCacheManager(
        total_blocks=256,
        block_size=2,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=request_count,
        max_batch_tokens=512,
        admission_callback=admission.try_admit,
    )
    engine = ContinuousBatchEngine(
        scheduler=scheduler, runner=runner, admission=admission
    )
    for index in range(request_count):
        engine.submit(
            f"request-{index}", (1, 2, 3, 4) * (prompt_length // 4),
            max_new_tokens=1,
        )
    result = engine.step()
    assert result is not None
    assert result.prefill_attention_backend == expected_backend
    assert len(result.update.finished_request_ids) == request_count
    assert manager.request_ids == ()


def test_engine_segmented_prefill_matches_default_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = []
    for backend in ("auto", "masked", "segmented_sdpa"):
        engine, scheduler, _ = make_engine(
            monkeypatch, prefill_attention_backend=backend
        )
        engine.submit("A", (1, 2, 3), max_new_tokens=1)
        engine.submit("B", (4, 0), max_new_tokens=1)
        assert engine.step() is not None
        outputs.append(tuple(scheduler.get_request(rid).generated_token_ids for rid in ("A", "B")))
    assert outputs[0] == outputs[1] == outputs[2]


def test_engine_vectorized_kv_write_matches_scalar_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outputs = []
    for backend in ("scalar", "vectorized"):
        engine, scheduler, _ = make_engine(
            monkeypatch, prefill_kv_write_backend=backend
        )
        engine.submit("A", (1, 2, 3), max_new_tokens=1)
        engine.submit("B", (4, 0), max_new_tokens=1)
        assert engine.step() is not None
        outputs.append(tuple(scheduler.get_request(rid).generated_token_ids for rid in ("A", "B")))
    assert outputs[0] == outputs[1]


def test_engine_runs_prefill_mixed_decode_and_releases_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, scheduler, admission = make_engine(monkeypatch)
    engine.submit("A", (1, 2, 3), max_new_tokens=3)
    engine.submit("B", (4, 0), max_new_tokens=1)

    first = engine.step()
    assert first is not None
    assert first.batch.prefill_request_ids == ("A", "B")
    assert first.batch.decode_request_ids == ()
    assert first.update.finished_request_ids == ("B",)
    assert tuple(request_id for request_id, _ in first.released_blocks) == ("B",)
    assert admission.manager.request_ids == ("A",)
    assert admission.manager.get_request("A").token_count == 3

    # C 在下一轮加入；该 step 同时包含旧请求 A 的 Decode 和新请求 C 的 Prefill。
    engine.submit("C", (1,), max_new_tokens=2)
    second = engine.step()
    assert second is not None
    assert [(item.request_id, item.kind) for item in second.batch.items] == [
        ("A", WorkKind.DECODE),
        ("C", WorkKind.PREFILL),
    ]
    assert admission.manager.get_request("A").token_count == 4
    assert admission.manager.get_request("C").token_count == 1

    third = engine.step()
    assert third is not None
    assert third.batch.decode_request_ids == ("A", "C")
    assert third.update.finished_request_ids == ("A", "C")
    assert not scheduler.has_unfinished_requests
    assert admission.reservations == ()
    assert admission.manager.request_ids == ()
    assert (
        admission.manager.allocator.free_count
        == admission.manager.allocator.total_blocks
    )
    metrics_a = engine.metrics.snapshot("A")
    metrics_b = engine.metrics.snapshot("B")
    metrics_c = engine.metrics.snapshot("C")
    assert len(metrics_a.token_events) == 3
    assert len(metrics_b.token_events) == 1
    assert len(metrics_c.token_events) == 2
    assert len(metrics_a.inter_token_ns) == 2
    assert metrics_b.inter_token_ns == ()
    assert metrics_b.median_tpot_ns is None
    assert all(item.completed_ns is not None for item in (metrics_a, metrics_b, metrics_c))
    assert engine.step() is None


def test_engine_applies_mixed_prefill_budget_at_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, scheduler, _ = make_engine(
        monkeypatch, max_mixed_prefill_tokens=1
    )
    engine.submit("A", (1, 2, 3), max_new_tokens=3)
    engine.submit("B", (4, 0), max_new_tokens=1)
    first = engine.step()
    assert first is not None
    # 初始 5-token cohort 不受 1-token mixed budget 限制。
    assert first.batch.prefill_request_ids == ("A", "B")

    engine.submit("C", (1,), max_new_tokens=1)
    engine.submit("D", (2,), max_new_tokens=1)
    second = engine.step()
    assert second is not None
    assert second.batch.decode_request_ids == ("A",)
    assert second.batch.prefill_request_ids == ("C",)
    assert scheduler.waiting_request_ids == ("D",)

    third = engine.step()
    assert third is not None
    assert third.batch.decode_request_ids == ("A",)
    assert third.batch.prefill_request_ids == ("D",)


def test_mixed_step_decode_failure_restores_scheduler_and_new_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, scheduler, admission = make_engine(monkeypatch)
    engine.submit("A", (1, 2), max_new_tokens=3)
    assert engine.step() is not None
    old_a_length = admission.manager.get_request("A").token_count
    old_free_blocks = admission.manager.allocator.free_count

    engine.submit("B", (3,), max_new_tokens=2)

    def fail_decode(*_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        fail_decode,
    )
    with pytest.raises(RuntimeError, match="injected decode failure"):
        engine.step()

    # A 的 batched Decode 已 abort；B 虽完成过 Prefill，也被释放并退回 waiting。
    assert scheduler.outstanding_batch is None
    assert scheduler.running_request_ids == ("A",)
    assert scheduler.waiting_request_ids == ("B",)
    assert scheduler.get_request("B").generated_token_ids == ()
    assert not scheduler.get_request("B").prefilled
    assert admission.manager.request_ids == ("A",)
    assert admission.manager.get_request("A").token_count == old_a_length
    assert admission.manager.allocator.free_count == old_free_blocks
    assert tuple(item.request_id for item in admission.reservations) == ("A",)

    # abort 不消耗 step index；恢复 kernel 后，同一逻辑 step 可以重新调度。
    def reference_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return paged_decode_attention_reference(
            query, key, value, table, lengths
        ).output

    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        reference_attention,
    )
    retried = engine.step()
    assert retried is not None
    assert retried.batch.step_index == 1
    assert retried.batch.decode_request_ids == ("A",)
    assert retried.batch.prefill_request_ids == ("B",)
    assert admission.manager.get_request("A").token_count == old_a_length + 1
    assert admission.manager.get_request("B").token_count == 1
    # 失败尝试本身也是时间线事实；没有产生虚假的 token event。
    assert len(engine.metrics.snapshot("B").prefill_attempt_started_ns) == 2
    assert len(engine.metrics.snapshot("B").token_events) == 1


def test_static_engine_does_not_refill_until_cohort_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine, scheduler, admission = make_engine(
        monkeypatch, batching_policy=BatchingPolicy.STATIC
    )
    engine.submit("A", (1, 2, 3), max_new_tokens=3)
    engine.submit("B", (4, 0), max_new_tokens=1)
    first = engine.step()
    assert first is not None
    assert first.batch.prefill_request_ids == ("A", "B")
    assert first.update.finished_request_ids == ("B",)

    engine.submit("C", (1,), max_new_tokens=1)
    second = engine.step()
    assert second is not None
    assert second.batch.decode_request_ids == ("A",)
    assert second.batch.prefill_request_ids == ()
    assert scheduler.waiting_request_ids == ("C",)

    third = engine.step()
    assert third is not None
    assert third.batch.decode_request_ids == ("A",)
    assert third.update.finished_request_ids == ("A",)
    assert scheduler.waiting_request_ids == ("C",)

    fourth = engine.step()
    assert fourth is not None
    assert fourth.batch.prefill_request_ids == ("C",)
    assert fourth.update.finished_request_ids == ("C",)
    assert admission.manager.request_ids == ()
    assert not scheduler.has_unfinished_requests


def test_scheduler_abort_preserves_fifo_before_new_arrivals() -> None:
    scheduler = RequestScheduler(max_running_requests=2, max_batch_tokens=4)
    scheduler.submit("A", (1, 2), max_new_tokens=2)
    batch = scheduler.schedule_step()
    assert batch is not None
    scheduler.submit("B", (3,), max_new_tokens=1)

    aborted = scheduler.abort_step()

    assert aborted is batch
    assert scheduler.waiting_request_ids == ("A", "B")
    assert scheduler.running_request_ids == ()
    retry = scheduler.schedule_step()
    assert retry is not None
    assert retry.step_index == batch.step_index
    assert retry.prefill_request_ids == ("A", "B")
