import pytest

from scripts.benchmark_packed_prefill import construct_prompts


BASE = ((1, 2), (3, 4, 5))


def test_construct_prompts_preserves_original_without_length_override() -> None:
    assert construct_prompts(BASE) == BASE


def test_construct_prompts_supports_equal_and_ragged_lengths() -> None:
    assert construct_prompts(BASE, fixed_prompt_length=4) == (
        (1, 2, 1, 2),
        (3, 4, 5, 3),
    )
    # 请求数可以多于模板数；每条请求仍独立使用自己的指定长度。
    assert construct_prompts(BASE, prompt_lengths=(1, 4, 3)) == (
        (1,),
        (3, 4, 5, 3),
        (1, 2, 1),
    )


@pytest.mark.parametrize(
    ("base", "fixed", "lengths"),
    [
        ((), None, None),
        (((1,), ()), None, None),
        (BASE, 0, None),
        (BASE, None, ()),
        (BASE, None, (1, 0)),
        (BASE, 2, (2,)),
    ],
)
def test_construct_prompts_rejects_invalid_shapes(base, fixed, lengths) -> None:
    with pytest.raises(ValueError):
        construct_prompts(
            base, fixed_prompt_length=fixed, prompt_lengths=lengths
        )
