import pytest

from experiments.one_step_block_demand import (
    ActiveRequest,
    PlannedPrefill,
    estimate_one_step_block_demand,
)


def estimate(*, total_blocks, free_blocks, active, decode=(), prefills=()):
    return estimate_one_step_block_demand(
        block_size=16,
        total_blocks=total_blocks,
        free_blocks=free_blocks,
        active_requests=active,
        decode_request_ids=decode,
        prefills=prefills,
    )


def current_only_snapshot():
    # 三条 128-token 请求都恰好处于 block 边界，第四条有 15 个尾部空位。
    return (
        ActiveRequest("A", 128, 8),
        ActiveRequest("B", 128, 8),
        ActiveRequest("C", 128, 8),
        ActiveRequest("D", 1, 1),
    )


def test_pressure_and_spare_pool_boundary() -> None:
    active = current_only_snapshot()
    plan = dict(
        active=active,
        decode=("A", "B", "C", "D"),
        prefills=(PlannedPrefill("late", 7),),
    )
    pressure = estimate(total_blocks=28, free_blocks=3, **plan)
    spare = estimate(total_blocks=29, free_blocks=4, **plan)
    assert pressure.decode_growth_by_request == (("A", 1), ("B", 1), ("C", 1), ("D", 0))
    assert pressure.prefill_blocks_by_request == (("late", 1),)
    assert pressure.required_new_blocks == 4
    assert pressure.shortfall_blocks == 1
    assert not pressure.fits_without_release
    assert spare.shortfall_blocks == 0
    assert spare.fits_without_release


def test_existing_reservation_does_not_become_free_capacity() -> None:
    reserved = (
        ActiveRequest("A", 128, 9),
        ActiveRequest("B", 128, 9),
        ActiveRequest("C", 128, 9),
        ActiveRequest("D", 1, 1),
    )
    result = estimate(
        total_blocks=28, free_blocks=0, active=reserved,
        decode=("A", "B", "C", "D"),
        prefills=(PlannedPrefill("late", 7),),
    )
    assert result.decode_growth_by_request == (("A", 0), ("B", 0), ("C", 0), ("D", 0))
    assert result.required_new_blocks == 1
    assert result.shortfall_blocks == 1


def test_multiple_prefills_count_each_physical_block() -> None:
    result = estimate(
        total_blocks=4, free_blocks=3,
        active=(ActiveRequest("A", 1, 1),),
        prefills=(PlannedPrefill("B", 16), PlannedPrefill("C", 17)),
    )
    assert result.prefill_blocks_by_request == (("B", 1), ("C", 2))
    assert result.required_new_blocks == 3
    assert result.fits_without_release


def test_no_early_credit_for_request_that_will_finish_this_step() -> None:
    result = estimate(
        total_blocks=1, free_blocks=0,
        active=(ActiveRequest("finishing", 1, 1),),
        decode=("finishing",),
        prefills=(PlannedPrefill("new", 1),),
    )
    assert result.shortfall_blocks == 1
    assert not result.fits_without_release


@pytest.mark.parametrize("changes, message", [
    ({"free_blocks": 1}, "必须等于"),
    ({"active": (ActiveRequest("A", 17, 1),)}, "装不下"),
    ({"active": (ActiveRequest("A", 1, 1), ActiveRequest("A", 1, 1)),
      "total_blocks": 2, "free_blocks": 0}, "唯一"),
    ({"decode": ("missing",)}, "不在活动快照"),
    ({"decode": ("A", "A")}, "两次"),
    ({"prefills": (PlannedPrefill("A", 1),)}, "已占用"),
    ({"prefills": (PlannedPrefill("B", 0),)}, "必须为正"),
])
def test_rejects_incomplete_or_invalid_plan(changes, message) -> None:
    arguments = dict(
        total_blocks=1, free_blocks=0,
        active=(ActiveRequest("A", 1, 1),),
        decode=(), prefills=(),
    )
    arguments.update(changes)
    with pytest.raises(ValueError, match=message):
        estimate(**arguments)


def test_inputs_remain_unchanged() -> None:
    active = [ActiveRequest("A", 16, 1)]
    decode = ["A"]
    prefills = [PlannedPrefill("B", 1)]
    estimate(total_blocks=3, free_blocks=2, active=active, decode=decode, prefills=prefills)
    assert active == [ActiveRequest("A", 16, 1)]
    assert decode == ["A"]
    assert prefills == [PlannedPrefill("B", 1)]
