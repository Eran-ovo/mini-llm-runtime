import torch

from experiments.manual_qwen_attention import (
    build_default_rope,
    build_prefill_causal_mask,
    repeat_kv,
    rotate_half,
)


def test_rotate_half_uses_qwen_half_split_layout() -> None:
    value = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
    assert torch.equal(rotate_half(value), expected)


def test_repeat_kv_preserves_gqa_group_order() -> None:
    value = torch.tensor([[[[10.0]], [[20.0]]]])  # [B=1,Hkv=2,S=1,D=1]
    repeated = repeat_kv(value, repeats=3)
    assert repeated.shape == (1, 6, 1, 1)
    assert repeated.flatten().tolist() == [10.0, 10.0, 10.0, 20.0, 20.0, 20.0]


def test_prefill_mask_blocks_only_future_positions() -> None:
    mask = build_prefill_causal_mask(
        1, 3, dtype=torch.float32, device=torch.device("cpu")
    )[0, 0]
    minimum = torch.finfo(torch.float32).min
    expected = torch.tensor(
        [[0.0, minimum, minimum], [0.0, 0.0, minimum], [0.0, 0.0, 0.0]]
    )
    assert torch.equal(mask, expected)


def test_position_zero_has_identity_rope() -> None:
    positions = torch.tensor([[0, 1]])
    cos, sin = build_default_rope(
        positions, head_dim=4, theta=10_000.0, dtype=torch.float32
    )
    assert torch.equal(cos[:, 0], torch.ones((1, 4)))
    assert torch.equal(sin[:, 0], torch.zeros((1, 4)))
    assert not torch.equal(cos[:, 1], torch.ones((1, 4)))

