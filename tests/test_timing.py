import pytest

from mini_llm_runtime.timing import summarize_samples


def test_summary_keeps_raw_samples_and_uses_median() -> None:
    result = summarize_samples([4.0, 1.0, 2.0], warmup=2)
    assert result.samples_ms == [4.0, 1.0, 2.0]
    assert result.median_ms == 2.0
    assert result.warmup == 2
    assert result.repeats == 3


def test_summary_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="至少需要"):
        summarize_samples([], warmup=0)

