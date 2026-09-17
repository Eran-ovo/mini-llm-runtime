import pytest
import torch

from experiments.attention_walkthrough import query_to_kv_head, to_head_layout


def test_projection_is_reshaped_to_attention_layout() -> None:
    raw = torch.arange(2 * 3 * 8).reshape(2, 3, 8)
    result = to_head_layout(raw, num_heads=2, head_dim=4)
    assert result.shape == (2, 2, 3, 4)
    # batch=0、head=1、token=2 对应 raw[0, 2, 4:8]。
    assert torch.equal(result[0, 1, 2], raw[0, 2, 4:8])


def test_qwen_gqa_maps_seven_query_heads_to_each_kv_head() -> None:
    mapping = [query_to_kv_head(head, 14, 2) for head in range(14)]
    assert mapping == [0] * 7 + [1] * 7


def test_invalid_gqa_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="整除"):
        query_to_kv_head(0, num_query_heads=14, num_kv_heads=3)

