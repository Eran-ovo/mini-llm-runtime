import pytest

from mini_llm_runtime.paged_kv_cache import (
    BlockLocation,
    BlockPoolExhaustedError,
    FixedBlockAllocator,
    RequestBlockTable,
)


def test_allocator_allocates_releases_and_reuses_blocks() -> None:
    allocator = FixedBlockAllocator(total_blocks=4)

    first = allocator.allocate(2)
    second = allocator.allocate(1)
    assert first == (0, 1)
    assert second == (2,)
    assert allocator.free_block_ids == (3,)
    assert allocator.allocated_block_ids == (0, 1, 2)

    allocator.free(first)
    assert allocator.allocate(2) == (0, 1)
    assert allocator.allocated_count == 3
    assert allocator.free_count == 1


def test_allocator_oom_and_invalid_free_are_atomic() -> None:
    allocator = FixedBlockAllocator(total_blocks=3)
    allocated = allocator.allocate(2)
    before_free = allocator.free_block_ids
    before_allocated = allocator.allocated_block_ids

    with pytest.raises(BlockPoolExhaustedError, match="requested=2"):
        allocator.allocate(2)
    assert allocator.free_block_ids == before_free
    assert allocator.allocated_block_ids == before_allocated

    with pytest.raises(ValueError, match="重复"):
        allocator.free((allocated[0], allocated[0]))
    with pytest.raises(RuntimeError, match="未分配或已释放"):
        allocator.free((allocated[0], 2))
    # 第二个 free 请求虽然包含一个合法 ID，但整组验证失败后不能部分释放。
    assert allocator.allocated_block_ids == before_allocated


def test_request_table_maps_tokens_across_block_boundary() -> None:
    allocator = FixedBlockAllocator(total_blocks=4)
    table = RequestBlockTable(
        request_id="request-a", block_size=4, allocator=allocator
    )

    assert table.append_tokens(3) == (0,)
    # 先填满 block 0 的最后一个槽，再跨边界进入新 block。
    assert table.append_tokens(3) == (1,)
    assert table.block_ids == (0, 1)
    assert table.token_count == 6
    assert table.token_capacity == 8
    assert table.internal_fragmentation_tokens == 2
    assert table.locate(0) == BlockLocation(0, 0)
    assert table.locate(3) == BlockLocation(0, 3)
    assert table.locate(4) == BlockLocation(1, 0)
    assert table.locate(5) == BlockLocation(1, 1)
    with pytest.raises(IndexError, match="越界"):
        table.locate(6)


def test_multiple_requests_do_not_share_live_blocks_and_release_reuses_them() -> None:
    allocator = FixedBlockAllocator(total_blocks=3)
    request_a = RequestBlockTable(
        request_id="a", block_size=4, allocator=allocator
    )
    request_b = RequestBlockTable(
        request_id="b", block_size=4, allocator=allocator
    )
    request_a.append_tokens(5)
    request_b.append_tokens(4)

    assert set(request_a.block_ids).isdisjoint(request_b.block_ids)
    assert allocator.free_count == 0
    assert request_a.release() == (0, 1)

    request_c = RequestBlockTable(
        request_id="c", block_size=4, allocator=allocator
    )
    request_c.append_tokens(8)
    assert request_c.block_ids == (0, 1)
    assert request_b.block_ids == (2,)


def test_failed_request_growth_changes_neither_table_nor_pool() -> None:
    allocator = FixedBlockAllocator(total_blocks=1)
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    table.append_tokens(4)

    with pytest.raises(BlockPoolExhaustedError):
        table.append_tokens(1)

    assert table.token_count == 4
    assert table.block_ids == (0,)
    assert allocator.allocated_block_ids == (0,)


def test_pending_append_is_invisible_until_commit_and_abort_returns_new_blocks() -> None:
    allocator = FixedBlockAllocator(total_blocks=3)
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    table.append_tokens(3)

    pending = table.begin_append(3)
    assert pending.start == 3
    assert pending.end == 6
    assert pending.new_block_ids == (1,)
    assert table.token_count == 3
    assert table.locate(3, include_pending=True) == BlockLocation(0, 3)
    assert table.locate(4, include_pending=True) == BlockLocation(1, 0)
    with pytest.raises(IndexError):
        table.locate(3)
    with pytest.raises(RuntimeError, match="未提交 append"):
        table.release()

    table.abort_append()
    assert table.pending is None
    assert table.token_count == 3
    assert table.block_ids == (0,)
    assert allocator.free_block_ids == (1, 2)

    table.begin_append(1)
    table.commit_append()
    assert table.token_count == 4
    assert table.block_ids == (0,)


def test_released_table_rejects_use_and_double_free() -> None:
    allocator = FixedBlockAllocator(total_blocks=2)
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    table.append_tokens(1)
    table.release()

    assert table.released
    assert table.token_count == 0
    assert table.block_ids == ()
    assert allocator.free_count == 2
    with pytest.raises(RuntimeError, match="已释放"):
        table.append_tokens(1)
    with pytest.raises(RuntimeError, match="已释放"):
        table.locate(0)
    with pytest.raises(RuntimeError, match="已释放"):
        table.release()
