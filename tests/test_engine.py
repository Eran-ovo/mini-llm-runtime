import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.engine import ContinuousBatchEngine, _nvtx_range
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
            scheduler=scheduler, runner=runner, admission=admission
        ),
        scheduler,
        admission,
    )


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
