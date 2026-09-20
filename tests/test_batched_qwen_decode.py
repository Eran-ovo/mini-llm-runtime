import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner

from test_qwen_prefill_cache import make_runner


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

