from types import SimpleNamespace

import pytest
import torch

from mini_llm_runtime.hf_baseline import HuggingFaceBaseline


class FakeTokenizer:
    eos_token_id = 99

    def __call__(self, prompt: str, return_tensors: str):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def decode(self, token_ids: list[int], skip_special_tokens: bool) -> str:
        return " ".join(map(str, token_ids))


class FakeCausalLM:
    def __init__(self) -> None:
        self.seen_input_lengths: list[int] = []

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, use_cache, return_dict, past_key_values=None):
        self.seen_input_lengths.append(input_ids.shape[1])
        vocab_size = 8
        logits = torch.zeros((*input_ids.shape, vocab_size))
        # 每一步稳定选择 token 4，足以验证数据流而非真实模型语义。
        logits[..., 4] = 1.0
        cache_length = attention_mask.shape[1]
        return SimpleNamespace(logits=logits, past_key_values=(cache_length,))


def test_greedy_path_is_one_prefill_then_single_token_decode() -> None:
    model = FakeCausalLM()
    runner = HuggingFaceBaseline(model, FakeTokenizer(), "cpu")
    result = runner.greedy_generate("hello", max_new_tokens=3)

    assert model.seen_input_lengths == [3, 1, 1]
    assert result.prompt_token_ids == [1, 2, 3]
    assert result.generated_token_ids == [4, 4, 4]
    assert len(result.next_token_logits) == 3


def test_decode_rejects_more_than_one_token() -> None:
    runner = HuggingFaceBaseline(FakeCausalLM(), FakeTokenizer(), "cpu")
    with pytest.raises(ValueError, match=r"\[batch, 1\]"):
        runner.decode_one(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2)), (1,))

