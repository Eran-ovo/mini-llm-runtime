from collections import Counter
import argparse
import pytest

from experiments.sweep_split_kv import parse_splits, round_order


def test_rotating_order_balances_every_case_position():
    names = ["v1", "s1", "s2", "s4", "s8", "s16", "s32"]
    positions = {name: Counter() for name in names}
    for index in range(28):
        order = round_order(names, index)
        assert sorted(order) == sorted(names)
        for position, name in enumerate(order):
            positions[name][position] += 1
    assert all(counts == Counter({p: 4 for p in range(7)}) for counts in positions.values())
    assert names == ["v1", "s1", "s2", "s4", "s8", "s16", "s32"]


def test_three_candidates_are_balanced_over_thirty_rounds():
    names = ["split_16", "split_32", "split_64"]
    counts = Counter((name, position) for index in range(30)
                     for position, name in enumerate(round_order(names, index)))
    assert counts == Counter({(name, p): 10 for name in names for p in range(3)})


@pytest.mark.parametrize("value", ["", "16,16", "0,32", "32,65", "x,16", "16,"])
def test_rejects_invalid_candidate_list(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_splits(value)


def test_parses_candidate_list_in_requested_order():
    assert parse_splits("64, 16,32") == (64, 16, 32)
