#!/usr/bin/env python3
"""使用真实权重手写 Qwen 第 0 个完整 Decoder Layer，并逐检查点对拍。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from experiments.manual_qwen_attention import compare, manual_prefill_attention


def manual_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Qwen RMSNorm：平方均值使用 FP32，输出恢复输入 dtype。"""
    input_dtype = hidden_states.dtype
    hidden_fp32 = hidden_states.to(torch.float32)
    variance = hidden_fp32.pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden_fp32 * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


@dataclass(frozen=True)
class MLPResult:
    output: torch.Tensor
    gate_raw: torch.Tensor
    gate_activated: torch.Tensor
    up_raw: torch.Tensor
    gated_product: torch.Tensor


def manual_swiglu_mlp(hidden_states: torch.Tensor, mlp_module: Any) -> MLPResult:
    """Qwen SwiGLU：down_proj(SiLU(gate_proj(x)) * up_proj(x))。"""
    gate_raw = F.linear(
        hidden_states, mlp_module.gate_proj.weight, mlp_module.gate_proj.bias
    )
    up_raw = F.linear(
        hidden_states, mlp_module.up_proj.weight, mlp_module.up_proj.bias
    )
    gate_activated = F.silu(gate_raw)
    gated_product = gate_activated * up_raw
    output = F.linear(
        gated_product, mlp_module.down_proj.weight, mlp_module.down_proj.bias
    )
    return MLPResult(output, gate_raw, gate_activated, up_raw, gated_product)


@dataclass(frozen=True)
class DecoderLayerResult:
    output: torch.Tensor
    checkpoints: dict[str, torch.Tensor]


def manual_decoder_layer_prefill(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    layer: Any,
    config: Any,
) -> DecoderLayerResult:
    """手写一个 Pre-Norm Qwen Decoder Layer 的 Prefill forward。"""
    first_residual = hidden_states
    input_norm = manual_rms_norm(
        hidden_states,
        layer.input_layernorm.weight,
        layer.input_layernorm.variance_epsilon,
    )
    attention_output, probabilities, _ = manual_prefill_attention(
        input_norm,
        position_ids,
        layer.self_attn,
        num_query_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        rope_theta=config.rope_theta,
    )
    after_attention = first_residual + attention_output

    # 第二条 residual 必须保存第一次 residual add 之后的值，而不是原始 layer input。
    second_residual = after_attention
    post_attention_norm = manual_rms_norm(
        after_attention,
        layer.post_attention_layernorm.weight,
        layer.post_attention_layernorm.variance_epsilon,
    )
    mlp = manual_swiglu_mlp(post_attention_norm, layer.mlp)
    output = second_residual + mlp.output

    checkpoints = {
        "input_norm": input_norm,
        "attention_probability": probabilities,
        "attention_output": attention_output,
        "after_attention_residual": after_attention,
        "post_attention_norm": post_attention_norm,
        "gate_raw": mlp.gate_raw,
        "gate_activated": mlp.gate_activated,
        "up_raw": mlp.up_raw,
        "gated_product": mlp.gated_product,
        "mlp_output": mlp.output,
    }
    return DecoderLayerResult(output, checkpoints)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).cuda().eval()
    layer = model.model.layers[0]
    captured: dict[str, torch.Tensor] = {}

    def capture_layer_input(_module: Any, args_: Any, kwargs: dict[str, Any]) -> None:
        hidden = kwargs.get("hidden_states", args_[0] if args_ else None)
        if hidden is None:
            raise RuntimeError("无法捕获 Decoder Layer hidden_states")
        captured["layer_input"] = hidden.detach().clone()
        captured["position_ids"] = kwargs["position_ids"].detach().clone()

    def capture_tensor(name: str):
        def hook(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            captured[name] = output.detach().clone()

        return hook

    def capture_attention(
        _module: Any, _inputs: Any, _kwargs: Any, output: tuple[torch.Tensor, Any]
    ) -> None:
        captured["attention_output"] = output[0].detach().clone()
        captured["attention_probability"] = output[1].detach().clone()

    def capture_layer_output(
        _module: Any, _inputs: Any, _kwargs: Any, output: torch.Tensor
    ) -> None:
        captured["layer_output"] = output.detach().clone()

    handles = [
        layer.register_forward_pre_hook(capture_layer_input, with_kwargs=True),
        layer.input_layernorm.register_forward_hook(capture_tensor("input_norm")),
        layer.self_attn.register_forward_hook(capture_attention, with_kwargs=True),
        layer.post_attention_layernorm.register_forward_hook(
            capture_tensor("post_attention_norm")
        ),
        layer.mlp.gate_proj.register_forward_hook(capture_tensor("gate_raw")),
        layer.mlp.up_proj.register_forward_hook(capture_tensor("up_raw")),
        layer.mlp.down_proj.register_forward_hook(capture_tensor("mlp_output")),
        layer.register_forward_hook(capture_layer_output, with_kwargs=True),
    ]
    try:
        encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            model(
                **encoded,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    # Reference 模块边界之间没有 hook，使用捕获结果按同一 residual 语义重建参考值。
    captured["after_attention_residual"] = (
        captured["layer_input"] + captured["attention_output"]
    )
    captured["gate_activated"] = F.silu(captured["gate_raw"])
    captured["gated_product"] = captured["gate_activated"] * captured["up_raw"]

    with torch.inference_mode():
        manual = manual_decoder_layer_prefill(
            captured["layer_input"], captured["position_ids"], layer, model.config
        )

    checkpoint_order = [
        "input_norm",
        "attention_probability",
        "attention_output",
        "after_attention_residual",
        "post_attention_norm",
        "gate_raw",
        "gate_activated",
        "up_raw",
        "gated_product",
        "mlp_output",
    ]
    reports = {
        name: compare(manual.checkpoints[name], captured[name])
        for name in checkpoint_order
    }
    reports["layer_output"] = compare(manual.output, captured["layer_output"])

    print("===== 手写 Qwen 第 0 个完整 Decoder Layer =====")
    print(f"prompt token ids          = {encoded['input_ids'][0].tolist()}")
    print(f"layer input               = {list(captured['layer_input'].shape)}")
    print(f"attention probability     = {list(manual.checkpoints['attention_probability'].shape)}")
    print(f"gate/up projection        = {list(manual.checkpoints['gate_raw'].shape)}")
    print(f"layer output              = {list(manual.output.shape)}")
    print("\n===== 分层检查点 =====")
    first_mismatch: str | None = None
    for name, report in reports.items():
        print(
            f"{name:28s} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = name

    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部检查点通过：RMSNorm、Attention、Residual 和 SwiGLU 组合正确。")


if __name__ == "__main__":
    main()

