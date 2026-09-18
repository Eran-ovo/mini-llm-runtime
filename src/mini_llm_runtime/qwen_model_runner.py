"""不依赖 Hugging Face 模块对象的最小 Qwen2 Prefill Runner。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .qwen_config import QwenConfig
from .qwen_weights import AttentionWeights, DecoderLayerWeights, QwenWeights


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    dtype = x.dtype
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(dtype)


def _heads(x: torch.Tensor, count: int, head_dim: int) -> torch.Tensor:
    batch, sequence, _ = x.shape
    return x.reshape(batch, sequence, count, head_dim).transpose(1, 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope(
    query: torch.Tensor,
    key: torch.Tensor,
    position_ids: torch.Tensor,
    config: QwenConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        config.rope_theta
        ** (
            torch.arange(
                0, config.head_dim, 2, device=query.device, dtype=torch.float32
            )
            / config.head_dim
        )
    )
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    cos = embedding.cos().to(query.dtype).unsqueeze(1)
    sin = embedding.sin().to(query.dtype).unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, heads, sequence, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(batch, heads, repeats, sequence, head_dim)
        .reshape(batch, heads * repeats, sequence, head_dim)
    )


def _attention(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    weights: AttentionWeights,
    config: QwenConfig,
) -> torch.Tensor:
    query = _heads(
        F.linear(x, weights.q_proj.weight, weights.q_proj.bias),
        config.num_attention_heads,
        config.head_dim,
    )
    key = _heads(
        F.linear(x, weights.k_proj.weight, weights.k_proj.bias),
        config.num_key_value_heads,
        config.head_dim,
    )
    value = _heads(
        F.linear(x, weights.v_proj.weight, weights.v_proj.bias),
        config.num_key_value_heads,
        config.head_dim,
    )
    query, key = _rope(query, key, position_ids, config)
    key = _repeat_kv(key, config.gqa_group_size)
    value = _repeat_kv(value, config.gqa_group_size)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config.head_dim)
    sequence = x.shape[1]
    future = torch.triu(
        torch.ones((sequence, sequence), device=x.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(future[None, None], torch.finfo(x.dtype).min)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
    per_head = torch.matmul(probabilities, value)
    merged = (
        per_head.transpose(1, 2)
        .contiguous()
        .reshape(x.shape[0], sequence, config.hidden_size)
    )
    return F.linear(merged, weights.o_proj.weight, weights.o_proj.bias)


def _decoder_layer(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    weights: DecoderLayerWeights,
    config: QwenConfig,
) -> torch.Tensor:
    residual = x
    x = _rms_norm(x, weights.input_norm, config.rms_norm_eps)
    x = residual + _attention(x, position_ids, weights.attention, config)
    residual = x
    x = _rms_norm(x, weights.post_attention_norm, config.rms_norm_eps)
    gate = F.silu(F.linear(x, weights.mlp.gate_proj.weight))
    up = F.linear(x, weights.mlp.up_proj.weight)
    x = F.linear(gate * up, weights.mlp.down_proj.weight)
    return residual + x


@dataclass(frozen=True)
class QwenPrefillOutput:
    logits: torch.Tensor
    layer_outputs: tuple[torch.Tensor, ...] | None = None


class QwenPrefillRunner:
    """当前仅支持等长、无 padding、无 KV Cache 的 Qwen2 Prefill。"""

    def __init__(self, config: QwenConfig, weights: QwenWeights) -> None:
        if len(weights.layers) != config.num_hidden_layers:
            raise ValueError("权重层数与配置不一致")
        self.config = config
        self.weights = weights

    @torch.inference_mode()
    def prefill(
        self, input_ids: torch.Tensor, *, return_layer_outputs: bool = False
    ) -> QwenPrefillOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空 [batch, sequence]")
        if input_ids.device != self.weights.embedding.device:
            raise ValueError("input_ids 与权重必须位于同一 device")
        batch, sequence = input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError("序列长度超过 max_position_embeddings")

        x = F.embedding(input_ids, self.weights.embedding)
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch, -1)
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None
        for layer in self.weights.layers:
            x = _decoder_layer(x, positions, layer, self.config)
            if captured is not None:
                captured.append(x)
        x = _rms_norm(x, self.weights.final_norm, self.config.rms_norm_eps)
        logits = F.linear(x, self.weights.embedding)
        return QwenPrefillOutput(
            logits=logits,
            layer_outputs=tuple(captured) if captured is not None else None,
        )

