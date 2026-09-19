import pytest

from mini_llm_runtime.cache_capacity import (
    KVCacheGeometry,
    generate_log_uniform_lengths,
    simulate_contiguous_reservation,
    simulate_paged_allocation,
)


def test_qwen_05b_fp16_cache_geometry() -> None:
    geometry = KVCacheGeometry(
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        bytes_per_element=2,
    )

    assert geometry.bytes_per_token == 12_288
    assert geometry.bytes_per_block(16) == 196_608


def test_contiguous_reservation_counts_unused_reserved_slots() -> None:
    # 每 token 需要 2 bytes；16 bytes 预算可容纳 8 个 token slot。
    geometry = KVCacheGeometry(1, 1, 1, 1)
    result = simulate_contiguous_reservation(
        [3, 1, 4],
        budget_bytes=16,
        reservation_tokens=4,
        geometry=geometry,
    )

    assert result.admitted_request_lengths == (3, 1)
    assert result.admitted_request_count == 2
    assert result.allocated_units == 2
    assert result.allocated_token_slots == 8
    assert result.used_tokens == 4
    assert result.internal_fragmentation_tokens == 4
    assert result.first_rejected_request_index == 2
    assert result.first_rejected_request_length == 4
    assert result.allocation_unit_utilization == pytest.approx(1.0)
    assert result.slot_utilization == pytest.approx(0.5)


def test_paged_allocation_counts_block_rounding_fragmentation() -> None:
    geometry = KVCacheGeometry(1, 1, 1, 1)
    result = simulate_paged_allocation(
        [3, 1, 4],
        budget_bytes=16,
        block_size=2,
        geometry=geometry,
    )

    assert result.admitted_request_lengths == (3, 1)
    assert result.allocated_units == 3
    assert result.free_allocation_units == 1
    assert result.allocated_token_slots == 6
    assert result.used_tokens == 4
    assert result.internal_fragmentation_tokens == 2
    assert result.first_rejected_request_index == 2
    assert result.allocation_unit_utilization == pytest.approx(0.75)
    assert result.slot_utilization == pytest.approx(2 / 3)


def test_block_size_one_has_no_internal_fragmentation() -> None:
    geometry = KVCacheGeometry(1, 1, 1, 1)
    result = simulate_paged_allocation(
        [3, 1, 4],
        budget_bytes=16,
        block_size=1,
        geometry=geometry,
    )

    assert result.admitted_request_count == 3
    assert result.internal_fragmentation_tokens == 0
    assert result.slot_utilization == pytest.approx(1.0)
    assert result.first_rejected_request_index is None


def test_simulation_reports_unusable_tail_bytes() -> None:
    geometry = KVCacheGeometry(1, 1, 1, 1)
    result = simulate_paged_allocation(
        [1],
        budget_bytes=17,
        block_size=2,
        geometry=geometry,
    )

    assert result.total_allocation_units == 4
    assert result.unusable_tail_bytes == 1


def test_contiguous_rejects_request_larger_than_reservation() -> None:
    geometry = KVCacheGeometry(1, 1, 1, 1)
    with pytest.raises(ValueError, match="不能超过"):
        simulate_contiguous_reservation(
            [5], budget_bytes=100, reservation_tokens=4, geometry=geometry
        )


def test_log_uniform_workload_is_seeded_and_bounded() -> None:
    first = generate_log_uniform_lengths(
        num_requests=20, max_sequence_length=128, seed=2027
    )
    second = generate_log_uniform_lengths(
        num_requests=20, max_sequence_length=128, seed=2027
    )
    different = generate_log_uniform_lengths(
        num_requests=20, max_sequence_length=128, seed=2028
    )

    assert first == second
    assert first != different
    assert all(1 <= length <= 128 for length in first)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (KVCacheGeometry, {"num_layers": 0, "num_kv_heads": 1, "head_dim": 1, "bytes_per_element": 1}),
        (generate_log_uniform_lengths, {"num_requests": 0, "max_sequence_length": 8, "seed": 1}),
    ],
)
def test_invalid_positive_dimensions_are_rejected(factory, kwargs) -> None:
    with pytest.raises(ValueError, match="正整数"):
        factory(**kwargs)
