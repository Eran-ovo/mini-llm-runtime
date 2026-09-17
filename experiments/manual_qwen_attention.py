#!/usr/bin/env python3
"""用真实 Qwen 权重手写第 0 层 Prefill Attention，并与 Hugging Face 对拍。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


def to_head_layout(raw: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """把 [B,S,H*D] 拆成 [B,H,S,D]；放在本文件中保证实验可独立运行。"""
    if raw.ndim != 3 or raw.shape[-1] != num_heads * head_dim:
        raise ValueError("Projection 输出无法按给定 num_heads/head_dim 拆分")
    batch, sequence, _ = raw.shape
    return raw.reshape(batch, sequence, num_heads, head_dim).transpose(1, 2)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen RoPE 的 rotate-half：[-后半部分, 前半部分]。"""
    first_half, second_half = x.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def build_default_rope(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """生成默认 RoPE 的 cos/sin；中间频率计算固定使用 FP32。"""
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim 必须是偶数")
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32)
            / head_dim
        )
    )
    # [B, S, 1] * [D/2] -> [B, S, D/2]
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    # Qwen 的 rotate_half 拆前后两半，因此频率也按 [freqs, freqs] 排列。
    embeddings = torch.cat((frequencies, frequencies), dim=-1)
    return embeddings.cos().to(dtype), embeddings.sin().to(dtype)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把 [B,S,D] 的 cos/sin 广播到 [B,H,S,D]，只旋转 Q/K。"""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden: torch.Tensor, repeats: int) -> torch.Tensor:
    """正确性版本 GQA：把 [B,Hkv,S,D] 逻辑展开为 [B,Hq,S,D]。"""
    if repeats <= 0:
        raise ValueError("repeats 必须 > 0")
    if repeats == 1:
        return hidden
    batch, kv_heads, sequence, head_dim = hidden.shape
    return (
        hidden[:, :, None, :, :]
        .expand(batch, kv_heads, repeats, sequence, head_dim)
        .reshape(batch, kv_heads * repeats, sequence, head_dim)
    )


def build_prefill_causal_mask(
    batch_size: int,
    sequence_length: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """构造 [B,1,S,S] additive mask：合法位置为 0，未来位置为 dtype 最小值。"""
    future = torch.triu(
        torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool),
        diagonal=1,
    )
    mask = torch.zeros((sequence_length, sequence_length), device=device, dtype=dtype)
    mask.masked_fill_(future, torch.finfo(dtype).min)
    return mask[None, None, :, :].expand(batch_size, 1, -1, -1)


@dataclass(frozen=True)
class ErrorReport:
    max_abs: float
    mean_abs: float
    allclose: bool


def compare(actual: torch.Tensor, expected: torch.Tensor) -> ErrorReport:
    difference = (actual.float() - expected.float()).abs()
    return ErrorReport(
        max_abs=difference.max().item(),
        mean_abs=difference.mean().item(),
        # FP16 Attention 包含 matmul/softmax，使用明确但不过宽的初始容差。
        allclose=torch.allclose(actual.float(), expected.float(), atol=2e-3, rtol=2e-3),
    )


def manual_prefill_attention(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    attention_module: Any,
    *,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """只用基本 PyTorch 算子重建 Qwen eager Prefill Attention。"""
    batch_size, sequence_length, hidden_size = hidden_states.shape

    raw_q = F.linear(
        hidden_states, attention_module.q_proj.weight, attention_module.q_proj.bias
    )
    raw_k = F.linear(
        hidden_states, attention_module.k_proj.weight, attention_module.k_proj.bias
    )
    raw_v = F.linear(
        hidden_states, attention_module.v_proj.weight, attention_module.v_proj.bias
    )
    query = to_head_layout(raw_q, num_query_heads, head_dim)
    key = to_head_layout(raw_k, num_kv_heads, head_dim)
    value = to_head_layout(raw_v, num_kv_heads, head_dim)

    cos, sin = build_default_rope(
        position_ids, head_dim=head_dim, theta=rope_theta, dtype=hidden_states.dtype
    )
    query, key = apply_rope(query, key, cos, sin)

    group_size = num_query_heads // num_kv_heads
    repeated_key = repeat_kv(key, group_size)
    repeated_value = repeat_kv(value, group_size)

    scores = torch.matmul(query, repeated_key.transpose(-2, -1)) / math.sqrt(head_dim)
    causal_mask = build_prefill_causal_mask(
        batch_size,
        sequence_length,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    scores = scores + causal_mask
    # 与 Qwen eager reference 一致：Softmax 累积用 FP32，结果转回 Query dtype。
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    per_head_output = torch.matmul(probabilities, repeated_value)

    merged = (
        per_head_output.transpose(1, 2)
        .contiguous()
        .reshape(batch_size, sequence_length, hidden_size)
    )
    output = F.linear(merged, attention_module.o_proj.weight, attention_module.o_proj.bias)
    debug = {
        "raw_q": raw_q,
        "raw_k": raw_k,
        "raw_v": raw_v,
        "query_after_rope": query,
        "key_after_rope": key,
        "cos": cos,
        "sin": sin,
        "causal_mask": causal_mask,
    }
    return output, probabilities, debug


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
    attention_module = model.model.layers[0].self_attn
    captured: dict[str, Any] = {}

    def capture_input(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        captured["hidden_states"] = kwargs["hidden_states"].detach()
        captured["position_ids"] = kwargs["position_ids"].detach()
        captured["reference_mask"] = kwargs["attention_mask"].detach()
        cos, sin = kwargs["position_embeddings"]
        captured["reference_cos"] = cos.detach()
        captured["reference_sin"] = sin.detach()

    def capture_output(
        _module: Any, _args: Any, _kwargs: Any, output: tuple[torch.Tensor, Any]
    ) -> None:
        captured["reference_output"] = output[0].detach()
        captured["reference_probabilities"] = output[1].detach()

    pre_handle = attention_module.register_forward_pre_hook(
        capture_input, with_kwargs=True
    )
    post_handle = attention_module.register_forward_hook(
        capture_output, with_kwargs=True
    )
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
        pre_handle.remove()
        post_handle.remove()

    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    with torch.inference_mode():
        manual_output, manual_probabilities, debug = manual_prefill_attention(
            captured["hidden_states"],
            captured["position_ids"],
            attention_module,
            num_query_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=head_dim,
            rope_theta=config.rope_theta,
        )

    reports = {
        "RoPE cos": compare(debug["cos"], captured["reference_cos"]),
        "RoPE sin": compare(debug["sin"], captured["reference_sin"]),
        "Causal mask": compare(debug["causal_mask"], captured["reference_mask"]),
        "Attention probability": compare(
            manual_probabilities, captured["reference_probabilities"]
        ),
        "Attention output": compare(manual_output, captured["reference_output"]),
    }

    print("===== 手写 Qwen 第 0 层 Prefill Attention =====")
    print(f"prompt token ids       = {encoded['input_ids'][0].tolist()}")
    print(f"attention input        = {list(captured['hidden_states'].shape)}")
    print(f"raw Q/K/V              = {list(debug['raw_q'].shape)} / "
          f"{list(debug['raw_k'].shape)} / {list(debug['raw_v'].shape)}")
    print(f"Q after RoPE           = {list(debug['query_after_rope'].shape)}")
    print(f"K after RoPE           = {list(debug['key_after_rope'].shape)}")
    print(f"attention probability  = {list(manual_probabilities.shape)}")
    print(f"attention output       = {list(manual_output.shape)}")
    print("\n===== 与 Hugging Face Reference 对拍 =====")
    for name, report in reports.items():
        print(
            f"{name:24s} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )

    print("\n手写 Head-0 Attention probability：")
    for row in manual_probabilities[0, 0].float().cpu().tolist():
        print("  " + " ".join(f"{value:8.5f}" for value in row))

    if not all(report.allclose for report in reports.values()):
        raise SystemExit("对拍失败：请从上面第一个 allclose=False 的检查点开始定位")
    print("\n全部检查点通过：手写 Attention 与 Hugging Face reference 一致。")


if __name__ == "__main__":
    main()
