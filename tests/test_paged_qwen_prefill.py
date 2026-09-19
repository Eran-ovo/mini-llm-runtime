import pytest
import torch

from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager

from test_qwen_prefill_cache import make_cache, make_runner


def make_manager(total_blocks: int = 4) -> PagedKVCacheManager:
    runner, _ = make_runner()
    config = runner.config
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=2,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )


def test_adapter_exposes_pending_only_after_each_layer_is_written() -> None:
    manager = make_manager()
    manager.create_request("request")
    cache = PagedRequestKVCache(manager, "request")
    cache.begin_append(3)
    layer_zero_key = torch.arange(6).reshape(1, 1, 3, 2).float()
    layer_zero_value = layer_zero_key + 100

    cache.write_layer(0, layer_zero_key, layer_zero_value)
    actual_key, actual_value = cache.view_layer(0, include_pending=True)
    assert torch.equal(actual_key, layer_zero_key)
    assert torch.equal(actual_value, layer_zero_value)
    layer_zero_inputs = cache.paged_attention_inputs(0)
    assert layer_zero_inputs.key_cache.shape == (4, 1, 2, 2)
    assert layer_zero_inputs.block_table.tolist() == [[0, 1]]
    assert layer_zero_inputs.sequence_lengths.tolist() == [3]
    with pytest.raises(RuntimeError, match="尚未写入"):
        cache.view_layer(1, include_pending=True)
    with pytest.raises(RuntimeError, match="尚未写入"):
        cache.paged_attention_inputs(1)
    with pytest.raises(RuntimeError, match="尚未写入的层"):
        cache.commit_append()

    layer_one_key = layer_zero_key + 1_000
    layer_one_value = layer_zero_value + 1_000
    cache.write_layer(1, layer_one_key, layer_one_value)
    layer_one_inputs = cache.paged_attention_inputs(1)
    # 同一次 append 的所有层复用同一份 GPU metadata，而 K/V view 随层变化。
    assert layer_one_inputs.block_table.data_ptr() == layer_zero_inputs.block_table.data_ptr()
    assert (
        layer_one_inputs.sequence_lengths.data_ptr()
        == layer_zero_inputs.sequence_lengths.data_ptr()
    )
    assert layer_one_inputs.key_cache.data_ptr() != layer_zero_inputs.key_cache.data_ptr()
    cache.commit_append()

    assert cache.length == 3
    assert cache.pending is None
    assert torch.equal(cache.view_layer(0)[0], layer_zero_key)
    assert torch.equal(cache.view_layer(1)[0], layer_one_key)


def test_adapter_abort_returns_new_blocks_and_keeps_length() -> None:
    manager = make_manager()
    manager.create_request("request")
    cache = PagedRequestKVCache(manager, "request")
    cache.begin_append(3)
    key = torch.ones((1, 1, 3, 2))
    cache.write_layer(0, key, key)

    cache.abort_append()

    assert cache.length == 0
    assert cache.pending is None
    assert cache.table.block_ids == ()
    assert manager.allocator.free_count == manager.allocator.total_blocks


def test_qwen_prefill_matches_contiguous_cache_with_noncontiguous_blocks() -> None:
    runner, _ = make_runner()
    input_ids = torch.tensor([[1, 2, 3]])

    contiguous = make_cache(runner, capacity=3)
    contiguous_output = runner.prefill(input_ids, cache=contiguous)

    manager = make_manager()
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    paged = PagedRequestKVCache(manager, "target")
    paged_output = runner.prefill(input_ids, cache=paged)

    assert paged.table.block_ids == (0, 2)
    assert paged.length == 3
    assert torch.equal(paged_output.logits, contiguous_output.logits)
    for layer_index in range(runner.config.num_hidden_layers):
        paged_key, paged_value = paged.view_layer(layer_index)
        contiguous_key, contiguous_value = contiguous.view_layer(layer_index)
        assert torch.equal(paged_key, contiguous_key)
        assert torch.equal(paged_value, contiguous_value)

    metadata = manager.build_batch_metadata(("target",))
    assert metadata.block_table.tolist() == [[0, 2]]
    assert metadata.sequence_lengths.tolist() == [3]


def test_qwen_prefill_rejects_insufficient_paged_capacity_without_mutation() -> None:
    runner, _ = make_runner()
    manager = make_manager(total_blocks=1)
    manager.create_request("request")
    paged = PagedRequestKVCache(manager, "request")

    with pytest.raises(RuntimeError, match="prompt length"):
        runner.prefill(torch.tensor([[1, 2, 3]]), cache=paged)

    assert paged.length == 0
    assert paged.pending is None
    assert paged.table.block_ids == ()
    assert manager.allocator.free_count == 1
