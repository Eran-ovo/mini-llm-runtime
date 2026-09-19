import pytest
import torch

from mini_llm_runtime.paged_kv_cache import (
    FixedBlockAllocator,
    PagedKVStorage,
    RequestBlockTable,
)


def make_storage(
    *, total_blocks: int = 4, block_size: int = 4
) -> tuple[FixedBlockAllocator, PagedKVStorage]:
    allocator = FixedBlockAllocator(total_blocks=total_blocks)
    storage = PagedKVStorage(
        allocator=allocator,
        num_layers=2,
        num_kv_heads=2,
        block_size=block_size,
        head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )
    return allocator, storage


def make_kv(token_count: int, offset: float = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (2, 2, token_count, 3)
    key = torch.arange(torch.tensor(shape).prod()).reshape(shape).float() + offset
    value = key + 10_000
    return key, value


def test_paged_storage_round_trip_across_block_boundary() -> None:
    allocator, storage = make_storage()
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    key, value = make_kv(6)

    storage.append_all(table, key, value)
    gathered_key, gathered_value = storage.gather(table)

    assert table.token_count == 6
    assert table.block_ids == (0, 1)
    assert torch.equal(gathered_key, key)
    assert torch.equal(gathered_value, value)
    # 逻辑 token 4 是第二个物理 block 的第 0 个槽。
    assert torch.equal(storage.key[:, 1, :, 0, :], key[:, :, 4, :])
    assert torch.equal(storage.value[:, 1, :, 0, :], value[:, :, 4, :])


def test_multiple_appends_fill_partial_block_before_allocating_new_one() -> None:
    allocator, storage = make_storage()
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    first_key, first_value = make_kv(3)
    second_key, second_value = make_kv(3, offset=1_000)

    storage.append_all(table, first_key, first_value)
    assert table.block_ids == (0,)
    storage.append_all(table, second_key, second_value)
    gathered_key, gathered_value = storage.gather(table)

    assert table.block_ids == (0, 1)
    assert torch.equal(gathered_key, torch.cat((first_key, second_key), dim=2))
    assert torch.equal(
        gathered_value, torch.cat((first_value, second_value), dim=2)
    )


def test_gather_follows_noncontiguous_physical_block_table() -> None:
    allocator, storage = make_storage(total_blocks=4, block_size=4)
    request_a = RequestBlockTable(
        request_id="a", block_size=4, allocator=allocator
    )
    request_b = RequestBlockTable(
        request_id="b", block_size=4, allocator=allocator
    )
    a_first_key, a_first_value = make_kv(4)
    b_key, b_value = make_kv(4, offset=10_000)
    a_second_key, a_second_value = make_kv(2, offset=20_000)

    storage.append_all(request_a, a_first_key, a_first_value)  # block 0
    storage.append_all(request_b, b_key, b_value)              # block 1
    storage.append_all(request_a, a_second_key, a_second_value)  # block 2

    gathered_key, gathered_value = storage.gather(request_a)
    assert request_a.block_ids == (0, 2)
    assert request_a.locate(4).block_id == 2
    assert torch.equal(
        gathered_key, torch.cat((a_first_key, a_second_key), dim=2)
    )
    assert torch.equal(
        gathered_value, torch.cat((a_first_value, a_second_value), dim=2)
    )


def test_storage_validation_fails_before_changing_table() -> None:
    allocator, storage = make_storage()
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    wrong_key = torch.ones((2, 2, 1, 4))
    wrong_value = torch.ones_like(wrong_key)

    with pytest.raises(ValueError, match="布局不匹配"):
        storage.append_all(table, wrong_key, wrong_value)

    assert table.token_count == 0
    assert table.pending is None
    assert table.block_ids == ()
    assert allocator.free_count == allocator.total_blocks


def test_storage_failure_aborts_metadata_transaction(monkeypatch: pytest.MonkeyPatch) -> None:
    allocator, storage = make_storage()
    table = RequestBlockTable(
        request_id="request", block_size=4, allocator=allocator
    )
    key, value = make_kv(5)

    def fail_commit() -> None:
        raise RuntimeError("injected commit failure")

    monkeypatch.setattr(table, "commit_append", fail_commit)
    with pytest.raises(RuntimeError, match="injected"):
        storage.append_all(table, key, value)

    assert table.token_count == 0
    assert table.pending is None
    assert table.block_ids == ()
    assert allocator.free_count == allocator.total_blocks


def test_storage_rejects_table_from_another_pool_or_wrong_block_size() -> None:
    allocator, storage = make_storage()
    other_allocator = FixedBlockAllocator(total_blocks=allocator.total_blocks)
    other_table = RequestBlockTable(
        request_id="other", block_size=4, allocator=other_allocator
    )
    wrong_size = RequestBlockTable(
        request_id="wrong-size", block_size=2, allocator=allocator
    )
    key, value = make_kv(1)

    with pytest.raises(ValueError, match="同一 allocator"):
        storage.append_all(other_table, key, value)
    with pytest.raises(ValueError, match="block_size"):
        storage.append_all(wrong_size, key, value)


def test_reused_physical_blocks_are_overwritten_for_new_request() -> None:
    allocator, storage = make_storage(total_blocks=2)
    request_a = RequestBlockTable(
        request_id="a", block_size=4, allocator=allocator
    )
    old_key, old_value = make_kv(4)
    storage.append_all(request_a, old_key, old_value)
    request_a.release()

    request_b = RequestBlockTable(
        request_id="b", block_size=4, allocator=allocator
    )
    new_key, new_value = make_kv(4, offset=50_000)
    storage.append_all(request_b, new_key, new_value)
    gathered_key, gathered_value = storage.gather(request_b)

    assert request_b.block_ids == (0,)
    assert torch.equal(gathered_key, new_key)
    assert torch.equal(gathered_value, new_value)
