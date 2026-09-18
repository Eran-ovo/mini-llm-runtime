from types import SimpleNamespace

import torch

from experiments.manual_qwen_decoder_layer import manual_rms_norm, manual_swiglu_mlp


def test_rms_norm_uses_root_mean_square_without_centering() -> None:
    hidden = torch.tensor([[3.0, 4.0]])
    weight = torch.tensor([2.0, 0.5])
    result = manual_rms_norm(hidden, weight, epsilon=0.0)
    rms = torch.sqrt(torch.tensor((9.0 + 16.0) / 2.0))
    expected = torch.tensor([[2.0 * 3.0 / rms, 0.5 * 4.0 / rms]])
    assert torch.allclose(result, expected)

    # 如果错误实现成 LayerNorm，[1,1] 减均值后会变成零；RMSNorm 不会。
    ones = manual_rms_norm(torch.ones((1, 2)), torch.ones(2), epsilon=0.0)
    assert torch.equal(ones, torch.ones((1, 2)))


def test_swiglu_multiplies_silu_gate_and_up_branch() -> None:
    gate = torch.nn.Linear(2, 2, bias=False)
    up = torch.nn.Linear(2, 2, bias=False)
    down = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        gate.weight.copy_(torch.eye(2))
        up.weight.copy_(2 * torch.eye(2))
        down.weight.copy_(torch.eye(2))
    module = SimpleNamespace(gate_proj=gate, up_proj=up, down_proj=down)
    value = torch.tensor([[1.0, -1.0]])

    result = manual_swiglu_mlp(value, module)
    expected = torch.nn.functional.silu(value) * (2 * value)
    assert torch.allclose(result.gate_raw, value)
    assert torch.allclose(result.up_raw, 2 * value)
    assert torch.allclose(result.output, expected)

