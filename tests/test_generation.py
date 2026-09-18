import pytest
import torch

from mini_llm_runtime.generation import greedy_generate

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

