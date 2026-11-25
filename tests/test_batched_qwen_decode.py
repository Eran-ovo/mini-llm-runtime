from dataclasses import replace

import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.cuda_extension import load_cuda_extension
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.qwen_weights import QwenWeights

from test_qwen_prefill_cache import make_runner
from test_qwen_weights import tiny_config, tiny_state_dict


REQUEST_ORDER = ("C", "A", "B")
PROMPTS = {
    "A": torch.tensor([[1, 2, 3]]),
    "B": torch.tensor([[2, 3]]),
    "C": torch.tensor([[4]]),
}


def make_paged_runner() -> QwenPrefillRunner:
    base, weights = make_runner()
    return QwenPrefillRunner(
        base.config, weights, decode_attention_backend="paged_cuda"
    )


def initialize_requests(
    runner: QwenPrefillRunner,
) -> tuple[PagedKVCacheManager, dict[str, torch.Tensor]]:
    """逐请求 Prefill；返回的 token 行尚未写入 Cache。"""
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=8,
        block_size=3,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )
    next_tokens: dict[str, torch.Tensor] = {}
    for request_id in ("A", "B", "C"):
        manager.create_request(request_id)
        cache = PagedRequestKVCache(manager, request_id)
        output = runner.prefill(PROMPTS[request_id], cache=cache)
        next_tokens[request_id] = output.logits[:, -1].argmax(
            dim=-1, keepdim=True
        )
    return manager, next_tokens


def install_reference_attention(
    monkeypatch: pytest.MonkeyPatch,
    calls: list[tuple[str, int, tuple[int, ...]]],
) -> None:
    def make_operation(entry: str):
        def operation(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            table: torch.Tensor,
            lengths: torch.Tensor,
            **_: object,
        ) -> torch.Tensor:
            calls.append(
                (entry, query.shape[0], tuple(int(x) for x in lengths.tolist()))
            )
            return paged_decode_attention_reference(
                query, key, value, table, lengths
            ).output

        return operation

    monkeypatch.setattr(
        model_runner_module,
        "paged_decode_attention_cuda",
        make_operation("checked"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        make_operation("unchecked"),
    )


def test_batched_decode_matches_per_request_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_paged_runner()
    calls: list[tuple[str, int, tuple[int, ...]]] = []
    install_reference_attention(monkeypatch, calls)

    batch_manager, batch_tokens = initialize_requests(runner)
    batch_adapter = PagedBatchDecodeAdapter(batch_manager, REQUEST_ORDER)
    token_ids = torch.cat([batch_tokens[rid] for rid in REQUEST_ORDER], dim=0)
    batch_output = runner.decode_batch(
        token_ids, cache=batch_adapter, return_layer_outputs=True
    )

    # 两层模型只发起两次 Attention：第 0 层 checked，后续层 unchecked。
    assert calls == [
        ("checked", 3, (2, 4, 3)),
        ("unchecked", 3, (2, 4, 3)),
    ]
    assert batch_output.logits.shape[:2] == (3, 1)
    assert batch_output.layer_outputs is not None

    calls.clear()
    single_manager, single_tokens = initialize_requests(runner)
    single_outputs = []
    single_layers: list[list[torch.Tensor]] = [[], []]
    for request_id in REQUEST_ORDER:
        output = runner.decode_one(
            single_tokens[request_id],
            cache=PagedRequestKVCache(single_manager, request_id),
            return_layer_outputs=True,
        )
        single_outputs.append(output.logits)
        assert output.layer_outputs is not None
        for layer_index, tensor in enumerate(output.layer_outputs):
            single_layers[layer_index].append(tensor)

    assert torch.allclose(batch_output.logits, torch.cat(single_outputs), atol=1e-5)
    for layer_index, batch_layer in enumerate(batch_output.layer_outputs):
        assert torch.allclose(
            batch_layer, torch.cat(single_layers[layer_index]), atol=1e-5
        )
    # 逐请求路径每层各调用三次；batched 路径每层只调用一次。
    assert len(calls) == len(REQUEST_ORDER) * runner.config.num_hidden_layers
    assert all(batch_size == 1 for _, batch_size, _ in calls)

    for request_id in REQUEST_ORDER:
        batch_cache = PagedRequestKVCache(batch_manager, request_id)
        single_cache = PagedRequestKVCache(single_manager, request_id)
        assert batch_cache.length == single_cache.length
        for layer_index in range(runner.config.num_hidden_layers):
            batch_key, batch_value = batch_cache.view_layer(layer_index)
            single_key, single_value = single_cache.view_layer(layer_index)
            assert torch.allclose(batch_key, single_key, atol=1e-6)
            assert torch.allclose(batch_value, single_value, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available() or CUDA_HOME is None, reason="需要 CUDA")
@pytest.mark.parametrize("sequence,width", [(1, 64), (1, 896), (1, 1025), (13, 896)])
def test_decode_cuda_norm_matches_fp16_reference(sequence: int, width: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(9)
    x = torch.randn(8, sequence, width, generator=generator, device="cuda", dtype=torch.float16)
    weight = torch.randn(width, generator=generator, device="cuda", dtype=torch.float16)
    actual = load_cuda_extension().decode_norm(x, weight, 1e-6)
    expected = model_runner_module._rms_norm(x, weight, 1e-6)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available() or CUDA_HOME is None, reason="需要 CUDA")
def test_decode_cuda_rope_preserves_fp16_composition() -> None:
    generator = torch.Generator(device="cuda").manual_seed(9)
    query = torch.randn(3, 14, 1, 64, generator=generator, device="cuda", dtype=torch.float16)
    key = torch.randn(3, 2, 1, 64, generator=generator, device="cuda", dtype=torch.float16)
    positions = torch.tensor([[2], [256], [2049]], device="cuda")
    inv_freq = torch.linspace(0.003, 1.0, 32, device="cuda")
    cos, sin = model_runner_module._rope_factors(positions, inv_freq, torch.float16)
    expected = tuple(x * cos + model_runner_module._rotate_half(x) * sin for x in (query, key))
    actual = load_cuda_extension().decode_rope(query, key, cos, sin)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    value = torch.randn(key.shape, generator=generator, device="cuda", dtype=key.dtype)
    blocks = torch.tensor([2, 0, 2], device="cuda")
    offsets = torch.tensor([0, 3, 2], device="cuda")
    storage_shape = (3, 2, 4, 64)
    key_pages = torch.zeros(storage_shape, dtype=key.dtype, device="cuda")
    value_pages = torch.zeros_like(key_pages)
    expected_keys, expected_values = key_pages.clone(), value_pages.clone()
    expected_keys[blocks, :, offsets, :] = expected[1].squeeze(2)
    expected_values[blocks, :, offsets, :] = value.squeeze(2)
    rotated = load_cuda_extension().decode_rope_and_write(
        query, key, value, cos, sin, key_pages, value_pages, blocks, offsets,
    )
    torch.testing.assert_close(rotated, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(key_pages, expected_keys, rtol=0, atol=0)
    torch.testing.assert_close(value_pages, expected_values, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or CUDA_HOME is None, reason="需要 CUDA")
def test_cuda_swiglu_matches_separate_ops_with_partial_cta() -> None:
    generator = torch.Generator(device="cuda").manual_seed(21)
    gate = torch.randn(8, 7, 89, device="cuda", dtype=torch.float16, generator=generator)
    up = torch.randn(gate.shape, device="cuda", dtype=gate.dtype, generator=generator)
    torch.testing.assert_close(
        load_cuda_extension().swiglu(gate, up), torch.nn.functional.silu(gate) * up,
        rtol=1e-3, atol=2e-3,
    )


@pytest.mark.skipif(not torch.cuda.is_available() or CUDA_HOME is None, reason="需要 CUDA")
@pytest.mark.parametrize("greedy", [False, True])
def test_decode_graph_replays_with_new_slots_lengths_and_request_order(greedy: bool) -> None:
    config = replace(tiny_config(), hidden_size=128, intermediate_size=128)
    state_dict = {
        name: (weight * 0.1).to(device="cuda", dtype=torch.float16)
        for name, weight in tiny_state_dict(config).items()
    }
    weights = QwenWeights.from_state_dict(config, state_dict)
    eager = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    graph = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda", enable_decode_cuda_graph=True
    )

    def make_cache() -> PagedKVCacheManager:
        manager = PagedKVCacheManager(
            total_blocks=8,
            block_size=2,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=weights.embedding.dtype,
            device=weights.embedding.device,
        )
        for request_id in ("A", "B"):
            manager.create_request(request_id)
            manager.reserve_request_capacity(request_id, 8)
            eager.prefill(
                PROMPTS[request_id].cuda(), cache=PagedRequestKVCache(manager, request_id)
            )
        return manager

    graph_cache, eager_cache = make_cache(), make_cache()
    captured_output = None
    captured_inputs = None
    for step in range(4):
        order = ("A", "B") if step < 2 else ("B", "A")
        inputs = torch.tensor([[step], [step + 1]], device="cuda")
        expected = eager.decode_batch(inputs, cache=PagedBatchDecodeAdapter(eager_cache, order)).logits
        cache = PagedBatchDecodeAdapter(graph_cache, order)
        if greedy:
            actual = graph.decode_batch_greedy(inputs, cache=cache)
            expected = expected[:, 0].argmax(dim=-1)
        else:
            actual = graph.decode_batch(inputs, cache=cache).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if step == 1:
            captured_output = (actual, actual.clone())
            captured_inputs = (inputs, inputs.clone())

    assert captured_output is not None and captured_inputs is not None
    torch.testing.assert_close(*captured_output, rtol=0, atol=0)
    torch.testing.assert_close(*captured_inputs, rtol=0, atol=0)
    assert any(item is not None for item in graph._decode_graphs[graph_cache].values())
    for request_id in ("A", "B"):
        torch.testing.assert_close(
            graph_cache.gather(request_id), eager_cache.gather(request_id), rtol=0, atol=0
        )


def test_batched_decode_layer_failure_rolls_back_all_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = make_paged_runner()
    manager, next_tokens = initialize_requests(runner)
    adapter = PagedBatchDecodeAdapter(manager, REQUEST_ORDER)
    token_ids = torch.cat([next_tokens[rid] for rid in REQUEST_ORDER], dim=0)
    old_tables = {
        rid: (manager.get_request(rid).token_count, manager.get_request(rid).block_ids)
        for rid in REQUEST_ORDER
    }
    old_free_blocks = manager.allocator.free_count

    def first_layer(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        table: torch.Tensor,
        lengths: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        return paged_decode_attention_reference(
            query, key, value, table, lengths
        ).output

    def fail_second_layer(*_: object, **__: object) -> torch.Tensor:
        raise RuntimeError("injected layer failure")

    monkeypatch.setattr(
        model_runner_module, "paged_decode_attention_cuda", first_layer
    )
    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        fail_second_layer,
    )

    with pytest.raises(RuntimeError, match="injected layer failure"):
        runner.decode_batch(token_ids, cache=adapter)

    assert not adapter.active
    assert manager.allocator.free_count == old_free_blocks
    for request_id in REQUEST_ORDER:
        table = manager.get_request(request_id)
        assert table.pending is None
        assert (table.token_count, table.block_ids) == old_tables[request_id]

