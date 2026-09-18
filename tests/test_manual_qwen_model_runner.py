import pytest
import torch

from experiments.manual_qwen_model_runner import embedding_lookup, tied_lm_head


def test_embedding_and_lm_head_use_the_same_weight_in_opposite_directions() -> None:
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    token_ids = torch.tensor([[2, 0]])
    hidden = embedding_lookup(token_ids, weight)
    assert torch.equal(hidden, torch.tensor([[[5.0, 6.0], [1.0, 2.0]]]))

    logits = tied_lm_head(hidden, weight)
    expected = hidden @ weight.T
    assert torch.equal(logits, expected)
    assert logits.shape == (1, 2, 3)


def test_embedding_rejects_non_batched_token_ids() -> None:
    with pytest.raises(ValueError, match=r"\[batch, sequence\]"):
        embedding_lookup(torch.tensor([1, 2]), torch.ones((3, 4)))
