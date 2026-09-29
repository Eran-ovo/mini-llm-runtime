import pytest
import torch

from mini_llm_runtime.paged_batch import PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager

from test_qwen_prefill_cache import make_runner


PROMPTS = {
    "A": (1, 2, 3),
    "B": (4,),
    "C": (2, 3),
}


def make_manager(runner):
    return PagedKVCacheManager(
        total_blocks=12,
        block_size=2,
        num_layers=runner.config.num_hidden_layers,
        num_kv_heads=runner.config.num_key_value_heads,
        head_dim=runner.config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )


@pytest.mark.parametrize("order", [("A", "B", "C"), ("C", "A", "B")])
@pytest.mark.parametrize("attention_backend", ["masked", "segmented_sdpa"])
@pytest.mark.parametrize("write_backend", ["scalar", "vectorized"])
def test_packed_prefill_matches_independent_requests_and_cache(
    order, attention_backend, write_backend
):
    runner, _ = make_runner()
    packed_manager = make_manager(runner)
    reference_manager = make_manager(runner)
    for request_id in order:
        packed_manager.create_request(request_id)
        reference_manager.create_request(request_id)

    lengths = tuple(len(PROMPTS[request_id]) for request_id in order)
    adapter = PagedBatchPrefillAdapter(
        packed_manager, order, lengths, write_backend=write_backend
    )
    tokens = torch.tensor(
        [[token for request_id in order for token in PROMPTS[request_id]]]
    )
    packed = runner.prefill_batch(
        tokens, cache=adapter, attention_backend=attention_backend
    )
    assert packed.logits.shape == (len(order), 1, runner.config.vocab_size)
    assert adapter.offsets == tuple(
        sum(lengths[:index]) for index in range(len(lengths) + 1)
    )

    for index, request_id in enumerate(order):
        reference = runner.prefill(
            torch.tensor([PROMPTS[request_id]]),
            cache=PagedRequestKVCache(reference_manager, request_id),
        )
        assert torch.allclose(
            packed.logits[index, 0], reference.logits[0, -1], atol=1e-5
        )
        assert packed_manager.get_request(request_id).token_count == lengths[index]
        for layer_index in range(runner.config.num_hidden_layers):
            packed_kv = packed_manager.storage.gather_layer(
                packed_manager.get_request(request_id), layer_index
            )
            reference_kv = reference_manager.storage.gather_layer(
                reference_manager.get_request(request_id), layer_index
            )
            for actual, expected in zip(packed_kv, reference_kv, strict=True):
                assert torch.allclose(actual, expected, atol=1e-5)


def test_packed_prefill_rejects_invalid_metadata_without_cache_mutation():
    runner, _ = make_runner()
    manager = make_manager(runner)
    manager.create_request("A")
    adapter = PagedBatchPrefillAdapter(manager, ("A",), (2,))
    with pytest.raises(ValueError, match="lengths 之和"):
        runner.prefill_batch(torch.tensor([[1, 2, 3]]), cache=adapter)
    assert manager.get_request("A").pending is None
    assert manager.get_request("A").token_count == 0
    assert manager.allocator.free_count == manager.allocator.total_blocks


def test_packed_prefill_rejects_unknown_attention_backend_before_mutation():
    runner, _ = make_runner()
    manager = make_manager(runner)
    manager.create_request("A")
    adapter = PagedBatchPrefillAdapter(manager, ("A",), (2,))
    with pytest.raises(ValueError, match="attention_backend"):
        runner.prefill_batch(
            torch.tensor([[1, 2]]), cache=adapter, attention_backend="unknown"
        )
    assert manager.get_request("A").pending is None
    assert manager.allocator.free_count == manager.allocator.total_blocks


@pytest.mark.parametrize("write_backend", ["scalar", "vectorized"])
def test_packed_prefill_failure_aborts_all_requests(monkeypatch, write_backend):
    runner, _ = make_runner()
    manager = make_manager(runner)
    for request_id in ("A", "B"):
        manager.create_request(request_id)
    adapter = PagedBatchPrefillAdapter(
        manager, ("A", "B"), (3, 1), write_backend=write_backend
    )
    original_write = adapter.write_layer

    def fail_on_second_layer(layer_index, key, value):
        if layer_index == 1:
            raise RuntimeError("injected packed layer failure")
        original_write(layer_index, key, value)

    monkeypatch.setattr(adapter, "write_layer", fail_on_second_layer)
    with pytest.raises(RuntimeError, match="injected packed layer failure"):
        runner.prefill_batch(torch.tensor([[1, 2, 3, 4]]), cache=adapter)
    assert not adapter.active
    assert manager.allocator.free_count == manager.allocator.total_blocks
    for request_id in ("A", "B"):
        table = manager.get_request(request_id)
        assert table.pending is None
        assert table.token_count == 0
        assert table.block_ids == ()
