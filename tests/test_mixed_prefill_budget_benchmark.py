import argparse

import pytest

from scripts.benchmark_mixed_prefill_budget import (
    budget_key,
    order_for_round,
    parse_budgets,
)


def test_budget_parser_and_stable_keys() -> None:
    assert parse_budgets("none, 8,4") == (None, 8, 4)
    assert parse_budgets("unbounded,3") == (None, 3)
    assert budget_key(None) == "unbounded"
    assert budget_key(8) == "8"


@pytest.mark.parametrize("raw", ["none", "none,unbounded", "0,4", "x,4"])
def test_budget_parser_rejects_invalid_cases(raw: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_budgets(raw)


def test_case_order_rotates_all_positions() -> None:
    budgets = (None, 8, 4)
    assert order_for_round(budgets, 0) == (None, 8, 4)
    assert order_for_round(budgets, 1) == (8, 4, None)
    assert order_for_round(budgets, 2) == (4, None, 8)
    assert order_for_round(budgets, 3) == (None, 8, 4)
