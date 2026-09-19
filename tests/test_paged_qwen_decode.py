import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner

from test_qwen_prefill_cache import make_cache, make_runner


def make_manager(total_blocks: int) -> PagedKVCacheManager:
    runner, _ = make_runner()
    config = runner.config
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=3,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )


def make_fragmented_target() -> tuple[PagedKVCacheManager, PagedRequestKVCache]:
    manager = make_manager(total_blocks=4)
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    return manager, PagedRequestKVCache(manager, "target")


def test_paged_decode_allocates_noncontiguous_block_and_matches_contiguous() -> None:
    runner, _ = make_runner()
    prompt_ids = torch.tensor([[1, 2, 3]])

    contiguous = make_cache(runner, capacity=4)
    contiguous_prefill = runner.prefill(prompt_ids, cache=contiguous)
    token = contiguous_prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    contiguous_decode = runner.decode_one(token, cache=contiguous)

    manager, paged = make_fragmented_target()
    paged_prefill = runner.prefill(prompt_ids, cache=paged)
    assert paged.table.block_ids == (0,)
    old_length = paged.length
    old_prefixes = [
        tuple(tensor.clone() for tensor in paged.view_layer(layer_index))
        for layer_index in range(runner.config.num_hidden_layers)
    ]
    paged_decode = runner.decode_one(token, cache=paged)

    assert torch.equal(paged_prefill.logits, contiguous_prefill.logits)
    assert torch.allclose(paged_decode.logits, contiguous_decode.logits, atol=1e-5)
    assert paged.table.block_ids == (0, 2)
    assert paged.length == 4
    assert paged.pending is None
    for layer_index, (old_key, old_value) in enumerate(old_prefixes):
        paged_key, paged_value = paged.view_layer(layer_index)
        contiguous_key, contiguous_value = contiguous.view_layer(layer_index)
        assert torch.equal(paged_key, contiguous_key)
        assert torch.equal(paged_value, contiguous_value)
        assert torch.equal(paged_key[:, :, :old_length], old_key)
        assert torch.equal(paged_value[:, :, :old_length], old_value)

    metadata = manager.build_batch_metadata(("target",))
    assert metadata.block_table.tolist() == [[0, 2]]
    assert metadata.sequence_lengths.tolist() == [4]


def test_paged_decode_capacity_failure_keeps_prefill_state() -> None:
    runner, _ = make_runner()
    manager = make_manager(total_blocks=2)
    blocker = manager.create_request("blocker")
    blocker.append_tokens(1)  # block 0
    manager.create_request("target")
    paged = PagedRequestKVCache(manager, "target")
    prefill = runner.prefill(torch.tensor([[1, 2, 3]]), cache=paged)
    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    old_prefixes = [
        tuple(tensor.clone() for tensor in paged.view_layer(layer_index))
        for layer_index in range(runner.config.num_hidden_layers)
    ]

    with pytest.raises(RuntimeError, match="capacity 已满"):
        runner.decode_one(token, cache=paged)

    assert paged.table.block_ids == (1,)
    assert paged.length == 3
    assert paged.pending is None
    for layer_index, expected in enumerate(old_prefixes):
        actual = paged.view_layer(layer_index)
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


def test_model_runner_paged_backend_uses_physical_cache_without_gather(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用通用 reference 替身验证 ModelRunner 的 backend 路由和事务顺序。"""
    base_runner, weights = make_runner()
    runner = QwenPrefillRunner(
        base_runner.config,
        weights,
        decode_attention_backend="paged_cuda",
    )
    manager, paged = make_fragmented_target()
    prompt_ids = torch.tensor([[1, 2, 3]])
    prefill = runner.prefill(prompt_ids, cache=paged)
    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
    expected = runner.prefill(torch.cat((prompt_ids, token), dim=1)).logits[:, -1:]

    calls: list[tuple[str, int, tuple[int, ...]]] = []

    def fake_paged_attention(entry: str):
        def operation(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            table: torch.Tensor,
            lengths: torch.Tensor,
            **_: object,
        ) -> torch.Tensor:
            calls.append((entry, int(lengths.item()), tuple(table[0].tolist())))
            return paged_decode_attention_reference(
                query, key, value, table, lengths
            ).output

        return operation

    monkeypatch.setattr(
        model_runner_module,
        "paged_decode_attention_cuda",
        fake_paged_attention("checked"),
    )
    monkeypatch.setattr(
        model_runner_module,
        "_paged_decode_attention_cuda_unchecked",
        fake_paged_attention("unchecked"),
    )

    # 如果 paged_cuda 分支意外退回 correctness gather，测试必须立即失败。
    def reject_gather(*_: object, **__: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("paged_cuda Decode 不应调用 view_layer/gather")

    monkeypatch.setattr(paged, "view_layer", reject_gather)
    actual = runner.decode_one(token, cache=paged)

    assert torch.allclose(actual.logits, expected, atol=1e-5)
    assert calls == [
        ("checked", 4, (0, 2)),
        ("unchecked", 4, (0, 2)),
    ]
    assert paged.length == 4
    assert paged.pending is None


def test_paged_backend_rejects_contiguous_cache_before_append() -> None:
    base_runner, weights = make_runner()
    runner = QwenPrefillRunner(
        base_runner.config,
        weights,
        decode_attention_backend="paged_cuda",
    )
    cache = make_cache(runner, capacity=4)
    prefill = runner.prefill(torch.tensor([[1, 2, 3]]), cache=cache)
    token = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)

    with pytest.raises(TypeError, match="PagedRequestKVCache"):
        runner.decode_one(token, cache=cache)

    assert cache.length == 3
    assert cache.pending is None
