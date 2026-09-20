import pytest
import torch

from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.scheduler import (
    BatchingPolicy,
    FinishReason,
    RequestScheduler,
    RequestStatus,
    WorkKind,
)


def test_decode_priority_fifo_admission_and_dynamic_join_leave() -> None:
    scheduler = RequestScheduler(max_running_requests=3, max_batch_tokens=6)
    request_a = scheduler.submit("A", (1, 2, 3, 4), max_new_tokens=2)
    request_b = scheduler.submit(
        "B", (5, 6), max_new_tokens=4, eos_token_ids={21}
    )
    request_c = scheduler.submit("C", (7,), max_new_tokens=1)

    first = scheduler.schedule_step()
    assert first is not None
    assert first.step_index == 0
    assert first.token_count == 6
    assert first.prefill_request_ids == ("A", "B")
    assert first.decode_request_ids == ()
    assert scheduler.waiting_request_ids == ("C",)
    assert scheduler.running_request_ids == ("A", "B")

    # GPU batch 尚未写回时新请求可以到达，但只能参加下一 step。
    scheduler.submit("D", (8,), max_new_tokens=1)
    first_update = scheduler.apply_step_results({"B": 20, "A": 10})
    assert first_update.emitted_tokens == (("A", 10), ("B", 20))
    assert first_update.finished_request_ids == ()

    second = scheduler.schedule_step()
    assert second is not None
    assert [(item.request_id, item.kind) for item in second.items] == [
        ("A", WorkKind.DECODE),
        ("B", WorkKind.DECODE),
        ("C", WorkKind.PREFILL),
    ]
    assert [item.input_token_ids for item in second.items] == [(10,), (20,), (7,)]
    assert second.token_count == 3
    assert scheduler.waiting_request_ids == ("D",)

    second_update = scheduler.apply_step_results({"A": 11, "B": 21, "C": 30})
    assert second_update.finished_request_ids == ("A", "B", "C")
    assert scheduler.running_request_ids == ()
    assert scheduler.finished_request_ids == ("A", "B", "C")
    assert request_a.finish_reason is FinishReason.MAX_TOKENS
    assert request_b.finish_reason is FinishReason.EOS
    assert request_c.finish_reason is FinishReason.MAX_TOKENS
    assert all(
        request.status is RequestStatus.FINISHED
        for request in (request_a, request_b, request_c)
    )

    third = scheduler.schedule_step()
    assert third is not None
    assert third.prefill_request_ids == ("D",)
    final_update = scheduler.apply_step_results({"D": 40})
    assert final_update.finished_request_ids == ("D",)
    assert not scheduler.has_unfinished_requests
    assert scheduler.schedule_step() is None


def test_strict_fifo_does_not_skip_large_head_prompt() -> None:
    scheduler = RequestScheduler(max_running_requests=3, max_batch_tokens=5)
    scheduler.submit("running", (1,), max_new_tokens=2)
    first = scheduler.schedule_step()
    assert first is not None
    scheduler.apply_step_results({"running": 10})

    scheduler.submit("large", (2, 3, 4, 5, 6), max_new_tokens=1)
    scheduler.submit("small", (7,), max_new_tokens=1)
    second = scheduler.schedule_step()
    assert second is not None
    # Decode 消耗 1 后只剩 4；large 放不下，不能越过它接纳 small。
    assert second.decode_request_ids == ("running",)
    assert second.prefill_request_ids == ()
    assert scheduler.waiting_request_ids == ("large", "small")
    scheduler.apply_step_results({"running": 11})

    third = scheduler.schedule_step()
    assert third is not None
    assert third.prefill_request_ids == ("large",)
    assert third.token_count == 5
    scheduler.apply_step_results({"large": 20})

    fourth = scheduler.schedule_step()
    assert fourth is not None
    assert fourth.prefill_request_ids == ("small",)


def test_static_policy_waits_for_running_cohort_to_drain() -> None:
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=6,
        batching_policy=BatchingPolicy.STATIC,
    )
    scheduler.submit("A", (1, 2, 3), max_new_tokens=3)
    scheduler.submit("B", (4, 5), max_new_tokens=1)

    first = scheduler.schedule_step()
    assert first is not None
    # cohort 初建时仍应一次接纳 A/B，而不是只接纳第一个请求。
    assert first.prefill_request_ids == ("A", "B")
    scheduler.apply_step_results({"A": 10, "B": 20})
    assert scheduler.finished_request_ids == ("B",)

    scheduler.submit("C", (6,), max_new_tokens=1)
    second = scheduler.schedule_step()
    assert second is not None
    assert second.decode_request_ids == ("A",)
    assert second.prefill_request_ids == ()
    assert scheduler.waiting_request_ids == ("C",)
    scheduler.apply_step_results({"A": 11})

    # 即使 A 将在本 step 完成，也不能提前假设结果并把 C 混入 cohort。
    third = scheduler.schedule_step()
    assert third is not None
    assert third.decode_request_ids == ("A",)
    assert third.prefill_request_ids == ()
    scheduler.apply_step_results({"A": 12})

    fourth = scheduler.schedule_step()
    assert fourth is not None
    assert fourth.decode_request_ids == ()
    assert fourth.prefill_request_ids == ("C",)


def test_mixed_prefill_budget_spreads_refill_without_limiting_initial_cohort() -> None:
    scheduler = RequestScheduler(
        max_running_requests=4,
        max_batch_tokens=20,
        max_mixed_prefill_tokens=3,
    )
    scheduler.submit("A", (1, 2, 3, 4), max_new_tokens=4)
    scheduler.submit("B", (5, 6, 7, 8), max_new_tokens=1)

    first = scheduler.schedule_step()
    assert first is not None
    # 初始 cohort 不受 mixed budget=3 限制，两个 4-token prompt 都能进入。
    assert first.prefill_request_ids == ("A", "B")
    assert first.token_count == 8
    scheduler.apply_step_results({"A": 10, "B": 20})

    scheduler.submit("C", (9, 10, 11), max_new_tokens=1)
    scheduler.submit("D", (12, 13), max_new_tokens=1)
    second = scheduler.schedule_step()
    assert second is not None
    # A 的 Decode 不计入 mixed-prefill budget；C 恰好消耗 3，D 留到下一步。
    assert second.decode_request_ids == ("A",)
    assert second.prefill_request_ids == ("C",)
    assert second.token_count == 4
    scheduler.apply_step_results({"A": 11, "C": 30})

    third = scheduler.schedule_step()
    assert third is not None
    # budget 每个 step 重新计算，因此 D 可在下一 mixed step 加入。
    assert third.decode_request_ids == ("A",)
    assert third.prefill_request_ids == ("D",)


def test_over_mixed_budget_head_waits_then_enters_after_cohort_drains() -> None:
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=10,
        max_mixed_prefill_tokens=3,
    )
    scheduler.submit("running", (1,), max_new_tokens=2)
    first = scheduler.schedule_step()
    assert first is not None
    scheduler.apply_step_results({"running": 10})

    scheduler.submit("large", (2, 3, 4, 5), max_new_tokens=1)
    scheduler.submit("small", (6,), max_new_tokens=1)
    second = scheduler.schedule_step()
    assert second is not None
    # 队首 large 超过 mixed budget；strict FIFO 禁止跳过它接纳 small。
    assert second.decode_request_ids == ("running",)
    assert second.prefill_request_ids == ()
    assert scheduler.waiting_request_ids == ("large", "small")
    scheduler.apply_step_results({"running": 11})

    third = scheduler.schedule_step()
    assert third is not None
    # 旧 cohort 已排空，这不再是 mixed step；large 不会永久饥饿。
    assert third.prefill_request_ids == ("large", "small")
    assert third.token_count == 5


def test_scheduler_rejects_invalid_batching_policy() -> None:
    with pytest.raises(ValueError, match="BatchingPolicy"):
        RequestScheduler(
            max_running_requests=1,
            max_batch_tokens=1,
            batching_policy="static",  # type: ignore[arg-type]
        )


def test_outstanding_batch_requires_exact_atomic_result_set() -> None:
    scheduler = RequestScheduler(max_running_requests=2, max_batch_tokens=4)
    request = scheduler.submit("A", (1, 2), max_new_tokens=2)
    batch = scheduler.schedule_step()
    assert batch is not None
    with pytest.raises(RuntimeError, match="尚未写回"):
        scheduler.schedule_step()

    # 新到达请求只进入 waiting，不改变 outstanding batch。
    scheduler.submit("B", (3,), max_new_tokens=1)
    with pytest.raises(ValueError, match=r"missing=\['A'\]"):
        scheduler.apply_step_results({})
    with pytest.raises(ValueError, match=r"extra=\['surprise'\]"):
        scheduler.apply_step_results({"A": 10, "surprise": 11})
    assert request.generated_token_ids == ()
    assert scheduler.outstanding_batch is batch

    update = scheduler.apply_step_results({"A": 10})
    assert update.emitted_tokens == (("A", 10),)
    assert request.generated_token_ids == (10,)
    assert scheduler.outstanding_batch is None


def test_scheduler_validates_configuration_and_submissions() -> None:
    with pytest.raises(ValueError, match="必须 > 0"):
        RequestScheduler(max_running_requests=0, max_batch_tokens=4)
    with pytest.raises(ValueError, match="不能大于"):
        RequestScheduler(max_running_requests=5, max_batch_tokens=4)
    for invalid_budget in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="正整数或 None"):
            RequestScheduler(
                max_running_requests=1,
                max_batch_tokens=4,
                max_mixed_prefill_tokens=invalid_budget,  # type: ignore[arg-type]
            )

    scheduler = RequestScheduler(max_running_requests=2, max_batch_tokens=4)
    with pytest.raises(ValueError, match="不能为空"):
        scheduler.submit("A", (), max_new_tokens=1)
    with pytest.raises(ValueError, match="chunked prefill"):
        scheduler.submit("A", (1, 2, 3, 4, 5), max_new_tokens=1)
    with pytest.raises(ValueError, match="正整数"):
        scheduler.submit("A", (1,), max_new_tokens=True)
    with pytest.raises(ValueError, match="非负整数"):
        scheduler.submit("A", (1, -1), max_new_tokens=1)

    scheduler.submit("A", (1,), max_new_tokens=1)
    with pytest.raises(ValueError, match="已经存在"):
        scheduler.submit("A", (2,), max_new_tokens=1)
    with pytest.raises(KeyError, match="未知"):
        scheduler.get_request("missing")


def test_finished_event_lets_engine_release_cache_blocks() -> None:
    """Scheduler 只发完成事件；Engine/KV Manager 才拥有并释放物理 block。"""
    manager = PagedKVCacheManager(
        total_blocks=2,
        block_size=2,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    table = manager.create_request("A")
    table.append_tokens(3)
    assert manager.allocator.allocated_count == 2

    scheduler = RequestScheduler(max_running_requests=1, max_batch_tokens=2)
    scheduler.submit("A", (1, 2), max_new_tokens=1)
    batch = scheduler.schedule_step()
    assert batch is not None
    update = scheduler.apply_step_results({"A": 9})

    # 调度完成不应偷偷修改 KV Manager；Engine 消费事件后显式释放。
    assert update.finished_request_ids == ("A",)
    assert manager.allocator.allocated_count == 2
    for request_id in update.finished_request_ids:
        manager.release_request(request_id)
    assert manager.allocator.allocated_count == 0
    assert manager.allocator.free_count == 2
