import pytest
import torch

import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.generation import greedy_generate
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner

from test_qwen_prefill_cache import make_cache, make_runner


def naive_greedy_tokens(
    prompt_ids: torch.Tensor, max_new_tokens: int
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """测试 oracle：每一步重算完整序列，完全不使用 KV Cache。"""
    runner, _ = make_runner()
    sequence = prompt_ids.clone()
    generated: list[torch.Tensor] = []
    logits_trace: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits = runner.prefill(sequence).logits[:, -1, :]
        logits_trace.append(logits.clone())
        token = logits.argmax(dim=-1, keepdim=True)
        generated.append(token)
        sequence = torch.cat((sequence, token), dim=1)
    return torch.cat(generated, dim=1), tuple(logits_trace)


def test_cached_greedy_matches_full_recomputation() -> None:
    runner, _ = make_runner()
    prompt_ids = torch.tensor([[1, 2, 3]])
    expected_ids, expected_logits = naive_greedy_tokens(prompt_ids, 3)

    output = greedy_generate(
        runner,
        prompt_ids,
        max_new_tokens=3,
        return_step_logits=True,
    )

    assert torch.equal(output.generated_token_ids, expected_ids)
    assert output.step_logits is not None
    assert len(output.step_logits) == 3
    for actual, expected in zip(output.step_logits, expected_logits):
        assert torch.allclose(actual, expected, atol=1e-5)
    assert output.prefill_tokens == 3
    assert output.decode_steps == 2
    assert output.cache_length == 5
    assert output.cache_capacity == 5
    assert not output.stopped_by_eos


def test_greedy_stops_at_eos_without_extra_decode() -> None:
    runner, _ = make_runner()
    prompt_ids = torch.tensor([[1, 2, 3]])
    first_token = int(runner.prefill(prompt_ids).logits[:, -1].argmax().item())

    output = greedy_generate(
        runner,
        prompt_ids,
        max_new_tokens=5,
        eos_token_ids={first_token},
    )

    assert output.generated_token_ids.tolist() == [[first_token]]
    assert output.stopped_by_eos
    assert output.decode_steps == 0
    # EOS 来自 Prefill logits，尚未作为 Decode 输入写入 Cache。
    assert output.cache_length == prompt_ids.shape[1]


def test_generation_preflight_rejects_insufficient_capacity() -> None:
    runner, _ = make_runner()
    prompt_ids = torch.tensor([[1, 2, 3]])
    cache = make_cache(runner, capacity=4)

    with pytest.raises(RuntimeError, match="required_capacity=5"):
        greedy_generate(
            runner,
            prompt_ids,
            max_new_tokens=3,
            cache=cache,
        )

    assert cache.length == 0
    assert cache.pending is None


def test_generation_validates_request_limits() -> None:
    runner, _ = make_runner()
    with pytest.raises(ValueError, match="batch=1"):
        greedy_generate(runner, torch.tensor([[1], [2]]), max_new_tokens=1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_generate(runner, torch.tensor([[1]]), max_new_tokens=0)
    with pytest.raises(ValueError, match="max_position_embeddings"):
        greedy_generate(
            runner,
            torch.tensor([[1, 2, 3, 4, 1, 2, 3]]),
            max_new_tokens=3,
        )
    with pytest.raises(ValueError, match="词表范围"):
        greedy_generate(
            runner,
            torch.tensor([[1]]),
            max_new_tokens=1,
            eos_token_ids={runner.config.vocab_size},
        )


def test_paged_cuda_greedy_tracks_in_block_and_cross_block_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """连续 Decode 应复用未满 block，并只在跨边界时扩展 block table。"""
    base_runner, weights = make_runner()
    runner = QwenPrefillRunner(
        base_runner.config,
        weights,
        decode_attention_backend="paged_cuda",
    )
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=4,
        block_size=3,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    cache = PagedRequestKVCache(manager, "target")

    calls: list[tuple[str, int, tuple[int, ...]]] = []

    def fake_paged_attention(entry: str):
        def operation(query, key, value, table, lengths, **_):
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

    prompt_ids = torch.tensor([[1, 2, 3]])
    expected_ids, expected_logits = naive_greedy_tokens(prompt_ids, 5)
    output = greedy_generate(
        runner,
        prompt_ids,
        max_new_tokens=5,
        cache=cache,
        return_step_logits=True,
    )

    assert torch.equal(output.generated_token_ids, expected_ids)
    assert output.step_logits is not None
    for actual, expected in zip(output.step_logits, expected_logits, strict=True):
        assert torch.allclose(actual, expected, atol=1e-5)
    # 生成 5 个 token 只执行 4 次 Decode；最后一个输出 token 不写入 Cache。
    assert output.decode_steps == 4
    assert output.cache_length == 7
    assert output.cache_capacity == 9
    assert cache.table.block_ids == (0, 2, 3)
    assert cache.pending is None

    # 两层模型每一步分别走 checked/unchecked。length 4 跨块，5/6 复用
    # block 2，length 7 才再次分配 block 3。
    expected_metadata = [
        (4, (0, 2)),
        (5, (0, 2)),
        (6, (0, 2)),
        (7, (0, 2, 3)),
    ]
    assert calls == [
        (entry, length, table)
        for length, table in expected_metadata
        for entry in ("checked", "unchecked")
    ]


def test_paged_generation_preflight_rejects_small_pool_without_mutation() -> None:
    base_runner, weights = make_runner()
    runner = QwenPrefillRunner(
        base_runner.config,
        weights,
        decode_attention_backend="paged_cuda",
    )
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=2,
        block_size=3,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    manager.create_request("target")
    cache = PagedRequestKVCache(manager, "target")

    with pytest.raises(RuntimeError, match="capacity=6.*required_capacity=7"):
        greedy_generate(
            runner,
            torch.tensor([[1, 2, 3]]),
            max_new_tokens=5,
            cache=cache,
        )

    assert cache.length == 0
    assert cache.pending is None
    assert cache.table.block_ids == ()
    assert manager.allocator.free_count == 2


def test_paged_generation_eos_from_prefill_skips_decode() -> None:
    base_runner, weights = make_runner()
    runner = QwenPrefillRunner(
        base_runner.config,
        weights,
        decode_attention_backend="paged_cuda",
    )
    config = runner.config
    manager = PagedKVCacheManager(
        total_blocks=3,
        block_size=3,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    manager.create_request("target")
    cache = PagedRequestKVCache(manager, "target")
    prompt_ids = torch.tensor([[1, 2, 3]])
    first_token = int(runner.prefill(prompt_ids).logits[:, -1].argmax().item())

    output = greedy_generate(
        runner,
        prompt_ids,
        max_new_tokens=5,
        eos_token_ids=first_token,
        cache=cache,
    )

    assert output.generated_token_ids.tolist() == [[first_token]]
    assert output.stopped_by_eos
    assert output.decode_steps == 0
    assert output.cache_length == 3
    assert cache.table.block_ids == (0,)
    assert cache.pending is None
