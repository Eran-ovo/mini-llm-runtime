import pytest
import torch

from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def make_manager(total_blocks: int = 4) -> PagedKVCacheManager:
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=4,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )


def make_kv(token_count: int, offset: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (2, 1, token_count, 2)
    key = torch.arange(2 * token_count * 2).reshape(shape).float() + offset
    return key, key + 1_000


def test_manager_rejects_duplicate_and_unknown_request_ids() -> None:
    manager = make_manager()
    manager.create_request("request")

    with pytest.raises(ValueError, match="已经存在"):
        manager.create_request("request")
    with pytest.raises(KeyError, match="未知"):
        manager.get_request("missing")
    with pytest.raises(KeyError, match="未知"):
        manager.release_request("missing")


def test_batch_metadata_preserves_order_padding_and_noncontiguous_blocks() -> None:
    manager = make_manager()
    manager.create_request("a")
    manager.create_request("b")
    a_first_key, a_first_value = make_kv(4)
    b_key, b_value = make_kv(4, offset=10_000)
    a_second_key, a_second_value = make_kv(2, offset=20_000)

    manager.append_all("a", a_first_key, a_first_value)   # block 0
    manager.append_all("b", b_key, b_value)               # block 1
    manager.append_all("a", a_second_key, a_second_value)  # block 2
    metadata = manager.build_batch_metadata(("b", "a"))

    assert metadata.request_ids == ("b", "a")
    assert metadata.batch_size == 2
    assert metadata.max_blocks_per_request == 2
    assert metadata.block_table.dtype == torch.int32
    assert metadata.sequence_lengths.dtype == torch.int32
    assert metadata.block_table.tolist() == [[1, -1], [0, 2]]
    assert metadata.sequence_lengths.tolist() == [4, 6]


def test_manager_stats_report_block_and_internal_fragmentation() -> None:
    manager = make_manager()
    manager.create_request("a")
    manager.create_request("b")
    a_key, a_value = make_kv(6)
    b_key, b_value = make_kv(4, offset=10_000)
    manager.append_all("a", a_key, a_value)
    manager.append_all("b", b_key, b_value)

    stats = manager.stats()
    assert stats.active_requests == 2
    assert stats.total_blocks == 4
    assert stats.allocated_blocks == 3
    assert stats.free_blocks == 1
    assert stats.total_token_slots == 16
    assert stats.allocated_token_slots == 12
    assert stats.committed_tokens == 10
    assert stats.pending_tokens == 0
    assert stats.internal_fragmentation_tokens == 2
    assert stats.block_utilization == pytest.approx(0.75)
    assert stats.slot_utilization == pytest.approx(10 / 12)
    assert stats.storage_nbytes == 512


def test_release_removes_registry_entry_and_blocks_can_be_reused() -> None:
    manager = make_manager()
    manager.create_request("a")
    manager.create_request("b")
    key_six, value_six = make_kv(6)
    key_four, value_four = make_kv(4, offset=10_000)
    manager.append_all("a", key_six, value_six)   # blocks 0, 1
    manager.append_all("b", key_four, value_four)  # block 2

    assert manager.release_request("a") == (0, 1)
    assert manager.request_ids == ("b",)
    manager.create_request("c")
    manager.append_all("c", key_six, value_six)

    assert manager.get_request("c").block_ids == (0, 1)
    assert manager.get_request("b").block_ids == (2,)


def test_pending_request_is_excluded_from_metadata_and_counted_as_reserved() -> None:
    manager = make_manager()
    table = manager.create_request("request")
    table.begin_append(2)

    stats = manager.stats()
    assert stats.committed_tokens == 0
    assert stats.pending_tokens == 2
    assert stats.allocated_token_slots == 4
    assert stats.internal_fragmentation_tokens == 2
    assert stats.slot_utilization == pytest.approx(0.5)
    with pytest.raises(RuntimeError, match="未提交 append"):
        manager.build_batch_metadata(("request",))
    with pytest.raises(RuntimeError, match="commit 或 abort"):
        manager.release_request("request")
    assert manager.request_ids == ("request",)

    table.abort_append()
    manager.release_request("request")
    assert manager.request_ids == ()


def test_batch_metadata_validates_batch_and_supports_empty_request_table() -> None:
    manager = make_manager()
    manager.create_request("empty")

    with pytest.raises(ValueError, match="不能为空"):
        manager.build_batch_metadata(())
    with pytest.raises(ValueError, match="重复"):
        manager.build_batch_metadata(("empty", "empty"))

    metadata = manager.build_batch_metadata(("empty",))
    assert metadata.block_table.shape == (1, 0)
    assert metadata.sequence_lengths.tolist() == [0]
