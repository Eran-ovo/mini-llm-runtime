from dataclasses import replace

import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.paged_batch import PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.qwen_weights import QwenWeights

from test_qwen_prefill_cache import make_runner
from test_qwen_weights import tiny_config, tiny_state_dict


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


def test_batched_sdpa_rejects_ragged_lengths_before_mutation():
    runner, _ = make_runner()
    manager = make_manager(runner)
    for request_id in ("A", "B"):
        manager.create_request(request_id)
    cache = PagedBatchPrefillAdapter(manager, ("A", "B"), (2, 1))
    with pytest.raises(ValueError, match="等长"):
        runner.prefill_batch(torch.tensor([[1, 2, 3]]), cache=cache, attention_backend="batched_sdpa")
    assert not cache.active
    assert manager.allocator.free_count == manager.allocator.total_blocks


@pytest.mark.parametrize("backend", ["varlen_sdpa", "varlen_cuda"])
def test_packed_varlen_prefill_matches_segmented_and_preserves_kv(backend):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires CUDA sm_80 or newer")
    if backend == "varlen_cuda":
        extension = pytest.importorskip("ai_infra_ops")
        if not hasattr(extension, "FlashAttentionVarlenPlan"):
            pytest.skip("requires ai_infra_ops with packed attention")

    config = replace(tiny_config(), hidden_size=128, intermediate_size=128)
    state = {
        name: (tensor * 0.1).to(device="cuda", dtype=torch.float16)
        for name, tensor in tiny_state_dict(config).items()
    }
    runner = QwenPrefillRunner(config, QwenWeights.from_state_dict(config, state))
    tokens = torch.tensor([[1, 2, 3, 4, 2, 1]], device="cuda")
    request_ids, lengths = ("A", "B", "C"), (3, 1, 2)
    managers = [make_manager(runner), make_manager(runner)]
    logits = []
    for manager, attention in zip(managers, ("segmented_sdpa", backend), strict=True):
        for request_id in request_ids:
            manager.create_request(request_id)
        logits.append(runner.prefill_batch(
            tokens, cache=PagedBatchPrefillAdapter(manager, request_ids, lengths),
            attention_backend=attention,
        ).logits)

    torch.testing.assert_close(logits[0], logits[1], atol=2e-3, rtol=2e-3)
    for request_id in request_ids:
        for layer in range(config.num_hidden_layers):
            expected, actual = [
                manager.storage.gather_layer(manager.get_request(request_id), layer)
                for manager in managers
            ]
            for reference_tensor, result_tensor in zip(expected, actual, strict=True):
                torch.testing.assert_close(reference_tensor, result_tensor, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_batched_prefill_graph_matches_eager_and_updates_physical_slots(monkeypatch):
    config = replace(tiny_config(), hidden_size=128, intermediate_size=128)
    state = {
        name: (tensor * 0.1).to(device="cuda", dtype=torch.float16)
        for name, tensor in tiny_state_dict(config).items()
    }
    weights = QwenWeights.from_state_dict(config, state)
    eager = QwenPrefillRunner(config, weights)
    graph = QwenPrefillRunner(config, weights, enable_prefill_cuda_graph=True)
    eager_manager, graph_manager = make_manager(eager), make_manager(graph)
    retained = None

    for trial in range(3):
        order = ("A", "B") if trial < 2 else ("B", "A")
        # 在每轮占用不同数量的页，使捕获与重放使用不同的物理 slot。
        for manager in (eager_manager, graph_manager):
            manager.create_request("guard")
            if trial:
                manager.reserve_request_capacity("guard", trial * manager.block_size)
            for request_id in order:
                manager.create_request(request_id)
        tokens = torch.tensor(
            [[(token + trial) % config.vocab_size for token in (1, 2, 3, 4, 0, 1)]],
            device="cuda",
        )
        with monkeypatch.context() as reference:
            reference.setattr(model_runner_module, "cuda_toolkit_available", lambda: False)
            expected = eager.prefill_batch(
                tokens,
                cache=PagedBatchPrefillAdapter(eager_manager, order, (3, 3), write_backend="scalar"),
                attention_backend="segmented_sdpa",
            )
        actual = graph.prefill_batch(
            tokens, cache=PagedBatchPrefillAdapter(graph_manager, order, (3, 3)),
            attention_backend="batched_sdpa",
        )
        torch.testing.assert_close(actual.logits, expected.logits, rtol=2e-3, atol=2e-3)
        for request_id in order:
            torch.testing.assert_close(
                graph_manager.gather(request_id),
                eager_manager.gather(request_id),
                rtol=2e-3, atol=2e-3,
            )
        if trial == 1:
            retained = (actual.logits, actual.logits.clone(), tokens, tokens.clone())
        for manager in (eager_manager, graph_manager):
            for request_id in (*order, "guard"):
                manager.release_request(request_id)

    assert retained is not None
    logits, original_logits, tokens, original_tokens = retained
    torch.testing.assert_close(logits, original_logits, rtol=0, atol=0)
    torch.testing.assert_close(tokens, original_tokens)
    assert any(item is not None for item in graph._prefill_graphs[graph_manager].values())


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
