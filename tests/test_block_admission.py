import pytest
import torch

from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.paged_kv_cache import FixedBlockAllocator, RequestBlockTable
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.scheduler import RequestScheduler, WorkKind


def make_manager(total_blocks: int = 5, block_size: int = 2) -> PagedKVCacheManager:
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=block_size,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )


def apply_fake_model_inputs(manager, batch) -> None:
    """模拟 Prefill/Decode 把输入 token 写进已预留的逻辑 Cache。"""
    for item in batch.items:
        expected_count = len(item.input_token_ids)
        if item.kind is WorkKind.DECODE:
            assert expected_count == 1
        new_blocks = manager.get_request(item.request_id).append_tokens(expected_count)
        # 完整生命周期已在 admission 时预留，执行阶段不应再临时分配。
        assert new_blocks == ()


def test_request_table_reserve_capacity_does_not_advance_length() -> None:
    allocator = FixedBlockAllocator(total_blocks=4)
    table = RequestBlockTable(request_id="A", block_size=3, allocator=allocator)

    assert table.reserve_capacity(7) == (0, 1, 2)
    assert table.block_ids == (0, 1, 2)
    assert table.token_capacity == 9
    assert table.token_count == 0
    assert table.internal_fragmentation_tokens == 9

    # append 只推进可见长度，复用 admission 已经预留的物理块。
    assert table.append_tokens(4) == ()
    assert table.token_count == 4
    assert table.block_ids == (0, 1, 2)
    assert table.reserve_capacity(8) == ()
    with pytest.raises(ValueError, match="不能小于"):
        table.reserve_capacity(3)

    assert table.reserve_capacity(10) == (3,)
    before = table.block_ids
    with pytest.raises(RuntimeError, match="容量不足"):
        table.reserve_capacity(13)
    assert table.block_ids == before
    assert table.token_count == 4


def test_block_budget_waits_then_admits_after_finished_release() -> None:
    manager = make_manager(total_blocks=5, block_size=2)
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=5,
        admission_callback=admission.try_admit,
    )
    # A: (prompt 3 + 2 outputs - 1) = 4 tokens = 2 blocks。
    # B: (prompt 2 + 3 outputs - 1) = 4 tokens = 2 blocks。
    # C 同样需要 2 blocks；A/B 接纳后只剩 1 block，因此必须等待。
    scheduler.submit("A", (1, 2, 3), max_new_tokens=2)
    scheduler.submit("B", (4, 5), max_new_tokens=3)
    scheduler.submit("C", (6, 7, 8), max_new_tokens=2)

    first = scheduler.schedule_step()
    assert first is not None
    assert first.prefill_request_ids == ("A", "B")
    assert scheduler.waiting_request_ids == ("C",)
    assert manager.allocator.free_count == 1
    assert [item.request_id for item in admission.reservations] == ["A", "B"]
    apply_fake_model_inputs(manager, first)
    first_update = scheduler.apply_step_results({"A": 10, "B": 20})
    assert first_update.finished_request_ids == ()

    second = scheduler.schedule_step()
    assert second is not None
    assert second.decode_request_ids == ("A", "B")
    assert second.prefill_request_ids == ()
    assert scheduler.waiting_request_ids == ("C",)
    apply_fake_model_inputs(manager, second)
    second_update = scheduler.apply_step_results({"A": 11, "B": 21})
    assert second_update.finished_request_ids == ("A",)
    released = admission.release_finished(second_update.finished_request_ids)
    assert released == {"A": (0, 1)}
    assert manager.allocator.free_count == 3

    # A 释放后，下一 step 可在 Decode B 的同时接纳 C。
    third = scheduler.schedule_step()
    assert third is not None
    assert third.decode_request_ids == ("B",)
    assert third.prefill_request_ids == ("C",)
    assert manager.get_request("C").token_count == 0
    assert admission.reservation_for("C").block_count == 2
    apply_fake_model_inputs(manager, third)
    third_update = scheduler.apply_step_results({"B": 22, "C": 30})
    assert third_update.finished_request_ids == ("B",)
    admission.release_finished(third_update.finished_request_ids)

    fourth = scheduler.schedule_step()
    assert fourth is not None
    assert fourth.decode_request_ids == ("C",)
    apply_fake_model_inputs(manager, fourth)
    fourth_update = scheduler.apply_step_results({"C": 31})
    assert fourth_update.finished_request_ids == ("C",)
    admission.release_finished(fourth_update.finished_request_ids)

    assert manager.allocator.free_count == manager.allocator.total_blocks
    assert admission.reservations == ()
    assert manager.request_ids == ()
    assert not scheduler.has_unfinished_requests


def test_resource_blocked_waiting_batch_has_no_side_effects() -> None:
    manager = make_manager(total_blocks=1, block_size=2)
    external = manager.create_request("external")
    external.reserve_capacity(2)
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=1,
        max_batch_tokens=2,
        admission_callback=admission.try_admit,
    )
    scheduler.submit("A", (1, 2), max_new_tokens=1)

    assert scheduler.schedule_step() is None
    assert scheduler.waiting_request_ids == ("A",)
    assert scheduler.running_request_ids == ()
    assert manager.request_ids == ("external",)
    assert admission.reservations == ()

    manager.release_request("external")
    batch = scheduler.schedule_step()
    assert batch is not None
    assert batch.prefill_request_ids == ("A",)


def test_permanently_oversized_request_fails_without_manager_mutation() -> None:
    manager = make_manager(total_blocks=1, block_size=2)
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=1,
        max_batch_tokens=3,
        admission_callback=admission.try_admit,
    )
    scheduler.submit("too-large", (1, 2, 3), max_new_tokens=1)

    with pytest.raises(ValueError, match="永久无法接纳"):
        scheduler.schedule_step()
    assert scheduler.waiting_request_ids == ("too-large",)
    assert manager.request_ids == ()
    assert manager.allocator.free_count == 1
    assert admission.reservations == ()


def test_release_finished_prevalidates_all_ids() -> None:
    manager = make_manager(total_blocks=2, block_size=2)
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=1,
        max_batch_tokens=2,
        admission_callback=admission.try_admit,
    )
    scheduler.submit("A", (1,), max_new_tokens=1)
    assert scheduler.schedule_step() is not None

    with pytest.raises(KeyError, match="没有 block reservation"):
        admission.release_finished(("A", "missing"))
    # 预校验失败不能先释放 A。
    assert manager.request_ids == ("A",)
    assert admission.reservation_for("A").block_count == 1
