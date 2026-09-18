import pytest
import torch

from mini_llm_runtime.kv_cache import ContiguousKVCache
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.qwen_weights import QwenWeights

from test_qwen_weights import tiny_config, tiny_state_dict


def make_runner() -> tuple[QwenPrefillRunner, QwenWeights]:
    config = tiny_config()
    weights = QwenWeights.from_state_dict(config, tiny_state_dict(config))
    return QwenPrefillRunner(config, weights), weights


def make_cache(runner: QwenPrefillRunner, capacity: int = 5) -> ContiguousKVCache:
    config = runner.config
    return ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        capacity=capacity,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )


def test_prefill_with_cache_matches_no_cache_logits() -> None:
    runner, _ = make_runner()
    input_ids = torch.tensor([[1, 2, 3]])
    expected = runner.prefill(input_ids).logits
    cache = make_cache(runner)
    actual = runner.prefill(input_ids, cache=cache).logits

    assert torch.equal(actual, expected)
    assert cache.length == 3
    assert cache.pending is None
    for layer in range(runner.config.num_hidden_layers):
        key, value = cache.view_layer(layer)
        assert key.shape == (1, 1, 3, 2)
        assert value.shape == (1, 1, 3, 2)


def test_prefill_rejects_nonempty_or_too_small_cache() -> None:
    runner, _ = make_runner()
    input_ids = torch.tensor([[1, 2, 3]])
    cache = make_cache(runner)
    runner.prefill(input_ids, cache=cache)
    with pytest.raises(ValueError, match="只接受空闲"):
        runner.prefill(input_ids, cache=cache)

    too_small = make_cache(runner, capacity=2)
    with pytest.raises(RuntimeError, match="prompt length"):
        runner.prefill(input_ids, cache=too_small)
    assert too_small.length == 0
    assert too_small.pending is None


def test_prefill_rejects_cache_layout_mismatch() -> None:
    runner, weights = make_runner()
    config = runner.config
    wrong = ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        capacity=5,
        head_dim=config.head_dim + 1,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    with pytest.raises(ValueError, match="布局不匹配"):
        runner.prefill(torch.tensor([[1, 2, 3]]), cache=wrong)

