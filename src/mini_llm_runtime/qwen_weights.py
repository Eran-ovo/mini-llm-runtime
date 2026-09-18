"""Qwen2 权重 schema 与严格 state-dict 映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .qwen_config import QwenConfig


@dataclass(frozen=True)
class LinearWeights:
    weight: torch.Tensor
    bias: torch.Tensor | None = None


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: LinearWeights
    k_proj: LinearWeights
    v_proj: LinearWeights
    o_proj: LinearWeights


@dataclass(frozen=True)
class MLPWeights:
    gate_proj: LinearWeights
    up_proj: LinearWeights
    down_proj: LinearWeights


@dataclass(frozen=True)
class DecoderLayerWeights:
    input_norm: torch.Tensor
    attention: AttentionWeights
    post_attention_norm: torch.Tensor
    mlp: MLPWeights


@dataclass(frozen=True)
class QwenWeights:
    embedding: torch.Tensor
    layers: tuple[DecoderLayerWeights, ...]
    final_norm: torch.Tensor

    @classmethod
    def from_state_dict(
        cls, config: QwenConfig, state_dict: Mapping[str, torch.Tensor]
    ) -> "QwenWeights":
        """严格映射 HF Qwen2 state dict；Tensor 只读共享，不复制 storage。"""
        consumed: set[str] = set()

        def take(name: str, shape: tuple[int, ...]) -> torch.Tensor:
            if name not in state_dict:
                raise KeyError(f"缺少权重：{name}")
            tensor = state_dict[name]
            if tuple(tensor.shape) != shape:
                raise ValueError(
                    f"{name} shape={tuple(tensor.shape)}，预期={shape}"
                )
            consumed.add(name)
            return tensor

        hidden = config.hidden_size
        kv_width = config.num_key_value_heads * config.head_dim
        intermediate = config.intermediate_size
        embedding = take("model.embed_tokens.weight", (config.vocab_size, hidden))
        layers: list[DecoderLayerWeights] = []

        for index in range(config.num_hidden_layers):
            prefix = f"model.layers.{index}"
            attention = AttentionWeights(
                q_proj=LinearWeights(
                    take(f"{prefix}.self_attn.q_proj.weight", (hidden, hidden)),
                    take(f"{prefix}.self_attn.q_proj.bias", (hidden,)),
                ),
                k_proj=LinearWeights(
                    take(f"{prefix}.self_attn.k_proj.weight", (kv_width, hidden)),
                    take(f"{prefix}.self_attn.k_proj.bias", (kv_width,)),
                ),
                v_proj=LinearWeights(
                    take(f"{prefix}.self_attn.v_proj.weight", (kv_width, hidden)),
                    take(f"{prefix}.self_attn.v_proj.bias", (kv_width,)),
                ),
                o_proj=LinearWeights(
                    take(f"{prefix}.self_attn.o_proj.weight", (hidden, hidden))
                ),
            )
            mlp = MLPWeights(
                gate_proj=LinearWeights(
                    take(f"{prefix}.mlp.gate_proj.weight", (intermediate, hidden))
                ),
                up_proj=LinearWeights(
                    take(f"{prefix}.mlp.up_proj.weight", (intermediate, hidden))
                ),
                down_proj=LinearWeights(
                    take(f"{prefix}.mlp.down_proj.weight", (hidden, intermediate))
                ),
            )
            layers.append(
                DecoderLayerWeights(
                    input_norm=take(f"{prefix}.input_layernorm.weight", (hidden,)),
                    attention=attention,
                    post_attention_norm=take(
                        f"{prefix}.post_attention_layernorm.weight", (hidden,)
                    ),
                    mlp=mlp,
                )
            )

        final_norm = take("model.norm.weight", (hidden,))
        if "lm_head.weight" in state_dict:
            lm_head = take("lm_head.weight", (config.vocab_size, hidden))
            if lm_head.data_ptr() != embedding.data_ptr() and not torch.equal(
                lm_head, embedding
            ):
                raise ValueError("tied lm_head.weight 与 embedding 数值不一致")

        unexpected = sorted(set(state_dict) - consumed)
        if unexpected:
            preview = ", ".join(unexpected[:5])
            raise ValueError(f"存在未识别权重（前 5 项）：{preview}")

        tensors = [embedding, final_norm]
        for layer in layers:
            tensors.extend(
                [
                    layer.input_norm,
                    layer.post_attention_norm,
                    layer.attention.q_proj.weight,
                    layer.attention.q_proj.bias,
                    layer.attention.k_proj.weight,
                    layer.attention.k_proj.bias,
                    layer.attention.v_proj.weight,
                    layer.attention.v_proj.bias,
                    layer.attention.o_proj.weight,
                    layer.mlp.gate_proj.weight,
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                ]
            )
        concrete = [tensor for tensor in tensors if tensor is not None]
        dtype = concrete[0].dtype
        device = concrete[0].device
        if any(tensor.dtype != dtype for tensor in concrete):
            raise ValueError("权重 dtype 不一致")
        if any(tensor.device != device for tensor in concrete):
            raise ValueError("权重 device 不一致")
        return cls(embedding, tuple(layers), final_norm)

