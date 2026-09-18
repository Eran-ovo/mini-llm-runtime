import pytest
import torch

from mini_llm_runtime.kv_cache import ContiguousKVCache


def make_cache(*, layers: int = 3, capacity: int = 5) -> ContiguousKVCache:
    return ContiguousKVCache(
        num_layers=layers,
        batch_size=1,
        num_kv_heads=2,
        capacity=capacity,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def tensors(layers: int, tokens: int, offset: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(layers * 1 * 2 * tokens * 4, dtype=torch.float32)
    key = key.reshape(layers, 1, 2, tokens, 4) + offset
    return key, key + 1000


def test_prefill_then_decode_appends_all_layers() -> None:
    cache = make_cache()
    prompt_k, prompt_v = tensors(3, 3)
    cache.append_all(prompt_k, prompt_v)
    assert cache.length == 3
    old_prefix = cache.key[:, :, :, :3].clone()

    decode_k, decode_v = tensors(3, 1, offset=5000)
    cache.append_all(decode_k, decode_v)
    assert cache.length == 4
    assert torch.equal(cache.key[:, :, :, :3], old_prefix)
    for layer in range(3):
        key, value = cache.view_layer(layer)
        assert key.shape == (1, 2, 4, 4)
        assert torch.equal(key[:, :, 3:], decode_k[layer])
        assert torch.equal(value[:, :, 3:], decode_v[layer])


def test_pending_is_visible_only_to_written_layer() -> None:
    cache = make_cache(layers=2)
    cache.begin_append(1)
    key = torch.ones((1, 2, 1, 4))
    value = key + 10
    cache.write_layer(0, key, value)

    assert cache.length == 0
    assert cache.view_layer(0)[0].shape[2] == 0
    assert cache.view_layer(0, include_pending=True)[0].shape[2] == 1
    with pytest.raises(RuntimeError, match="尚未写入"):
        cache.view_layer(1, include_pending=True)


def test_commit_requires_every_layer_and_abort_keeps_length() -> None:
    cache = make_cache(layers=2)
    key = torch.ones((1, 2, 1, 4))
    cache.begin_append(1)
    cache.write_layer(0, key, key)
    with pytest.raises(RuntimeError, match="尚未写入的层"):
        cache.commit_append()
    assert cache.length == 0

    cache.abort_append()
    assert cache.pending is None
    # 下一事务覆盖未提交区域并可正常完成。
    cache.begin_append(1)
    cache.write_layer(0, key * 2, key * 3)
    cache.write_layer(1, key * 4, key * 5)
    cache.commit_append()
    assert cache.length == 1


def test_capacity_failure_and_duplicate_write_are_atomic() -> None:
    cache = make_cache(layers=2, capacity=1)
    key = torch.ones((1, 2, 1, 4))
    cache.begin_append(1)
    cache.write_layer(0, key, key)
    with pytest.raises(RuntimeError, match="已经写入"):
        cache.write_layer(0, key, key)
    cache.write_layer(1, key, key)
    cache.commit_append()

    with pytest.raises(RuntimeError, match="容量不足"):
        cache.begin_append(1)
    assert cache.length == 1
    assert cache.pending is None


def test_reset_changes_visibility_without_clearing_storage() -> None:
    cache = make_cache(layers=2)
    key, value = tensors(2, 2)
    cache.append_all(key, value)
    stored = cache.key.clone()
    cache.reset()
    assert cache.length == 0
    assert cache.view_layer(0)[0].shape[2] == 0
    assert torch.equal(cache.key, stored)


def test_storage_size_includes_key_and_value() -> None:
    cache = make_cache(layers=3, capacity=5)
    expected = 2 * 3 * 1 * 2 * 5 * 4 * torch.tensor([], dtype=torch.float32).element_size()
    assert cache.storage_nbytes == expected

