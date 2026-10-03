import pytest
import torch

from experiments.diagnose_ragged_kv import error_stats


def test_error_stats_reports_exact_match_and_norms() -> None:
    reference = torch.tensor([[3.0, 4.0]])
    assert error_stats(reference, reference.clone()) == {
        "exact": True,
        "max_abs": 0.0,
        "difference_l2": 0.0,
        "reference_l2": 5.0,
        "relative_l2": 0.0,
    }


def test_error_stats_preserves_absolute_and_relative_error() -> None:
    actual = error_stats(torch.tensor([3.0, 4.0]), torch.tensor([0.0, 4.0]))
    assert actual["exact"] is False
    assert actual["max_abs"] == 3.0
    assert actual["difference_l2"] == 3.0
    assert actual["reference_l2"] == 5.0
    assert actual["relative_l2"] == pytest.approx(0.6)


def test_error_stats_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="形状不同"):
        error_stats(torch.zeros(2), torch.zeros(3))
