import pytest
import torch

from test_qwen_prefill_cache import make_cache, make_runner


def test_cached_decode_matches_full_recomputation() -> None:
    """带 Cache 的单 token Decode 应等价于重算完整序列的最后位置。"""
    runner, _ = make_runner()
    prompt_ids = torch.tensor([[1, 2, 3]])
    cache = make_cache(runner, capacity=5)

    prefill = runner.prefill(prompt_ids, cache=cache)
    next_token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    old_length = cache.length
    old_prefixes = [
        tuple(tensor.clone() for tensor in cache.view_layer(layer_index))
        for layer_index in range(runner.config.num_hidden_layers)
    ]

    cached = runner.decode_one(next_token, cache=cache)
    full_ids = torch.cat((prompt_ids, next_token), dim=1)
    recomputed = runner.prefill(full_ids)

    assert cached.logits.shape == (1, 1, runner.config.vocab_size)
    assert torch.allclose(cached.logits, recomputed.logits[:, -1:], atol=1e-5)
    assert cache.length == old_length + 1
    assert cache.pending is None
    for layer_index, (old_key, old_value) in enumerate(old_prefixes):
        key, value = cache.view_layer(layer_index)
        # append 只能写新槽位，已经完成 Prefill 的历史前缀必须保持不变。
        assert torch.equal(key[:, :, :old_length], old_key)
        assert torch.equal(value[:, :, :old_length], old_value)


def test_decode_returns_each_layer_output_when_requested() -> None:
    runner, _ = make_runner()
    cache = make_cache(runner)
    prefill = runner.prefill(torch.tensor([[1, 2]]), cache=cache)
    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)

    output = runner.decode_one(
        token, cache=cache, return_layer_outputs=True
    )

    assert output.layer_outputs is not None
    assert len(output.layer_outputs) == runner.config.num_hidden_layers
    assert all(tensor.shape == (1, 1, runner.config.hidden_size) for tensor in output.layer_outputs)


def test_decode_rejects_invalid_shape_empty_cache_and_full_cache() -> None:
    runner, _ = make_runner()
    cache = make_cache(runner, capacity=3)

    with pytest.raises(ValueError, match=r"\[batch, 1\]"):
        runner.decode_one(torch.tensor([[1, 2]]), cache=cache)
    with pytest.raises(ValueError, match="先用 Prefill"):
        runner.decode_one(torch.tensor([[1]]), cache=cache)

    runner.prefill(torch.tensor([[1, 2, 3]]), cache=cache)
    with pytest.raises(RuntimeError, match="capacity 已满"):
        runner.decode_one(torch.tensor([[1]]), cache=cache)

