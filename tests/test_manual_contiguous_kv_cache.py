import pytest
import torch

from experiments.manual_contiguous_kv_cache import ContiguousKVCache


def make_cache(capacity: int = 4) -> ContiguousKVCache:
    return ContiguousKVCache(
        batch_size=1,
        num_kv_heads=2,
        capacity=capacity,
        head_dim=3,
        dtype=torch.float32,
        device="cpu",
    )


def test_prefill_then_decode_appends_without_changing_history() -> None:
    cache = make_cache()
    prompt_key = torch.arange(18, dtype=torch.float32).reshape(1, 2, 3, 3)
    prompt_value = prompt_key + 100
    cache.append(prompt_key, prompt_value)
    old_key = cache.view()[0].clone()

    new_key = torch.full((1, 2, 1, 3), 9.0)
    new_value = torch.full((1, 2, 1, 3), 19.0)
    cache.append(new_key, new_value)

    key, value = cache.view()
    assert cache.length == 4
    assert torch.equal(key[:, :, :3], old_key)
    assert torch.equal(key[:, :, 3:], new_key)
    assert torch.equal(value[:, :, 3:], new_value)


def test_capacity_error_does_not_change_cache() -> None:
    cache = make_cache(capacity=2)
    initial = torch.ones((1, 2, 2, 3))
    cache.append(initial, initial)
    before_key = cache.key.clone()

    with pytest.raises(RuntimeError, match="容量不足"):
        cache.append(torch.ones((1, 2, 1, 3)), torch.ones((1, 2, 1, 3)))

    assert cache.length == 2
    assert torch.equal(cache.key, before_key)


def test_invalid_value_shape_does_not_partially_write() -> None:
    cache = make_cache()
    # torch.empty 可能含 NaN，而 NaN != NaN；先填 sentinel 才能可靠检查未写入。
    cache.key.fill_(42.0)
    before_key = cache.key.clone()
    with pytest.raises(ValueError, match="shape 必须相同"):
        cache.append(torch.ones((1, 2, 1, 3)), torch.ones((1, 2, 2, 3)))
    assert cache.length == 0
    # 失败前后底层存储必须没有被写入。
    assert torch.equal(cache.key, before_key)
