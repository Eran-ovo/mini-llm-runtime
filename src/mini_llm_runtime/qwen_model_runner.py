"""不依赖 Hugging Face 模块对象的最小 Qwen2 ModelRunner。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .kv_cache import LayerKVCache
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
    *,
    apply_causal_mask: bool,
    cache: LayerKVCache | None = None,
    layer_index: int | None = None,
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

    if cache is not None:
        if layer_index is None:
            raise ValueError("使用 KV Cache 时必须提供 layer_index")
        # Cache 持久化 RoPE 后的 K 和原始 V；当前层随后即可看见 pending prompt。
        cache.write_layer(layer_index, key, value)
        key, value = cache.view_layer(layer_index, include_pending=True)
    key = _repeat_kv(key, config.gqa_group_size)
    value = _repeat_kv(value, config.gqa_group_size)

    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config.head_dim)
    query_length = x.shape[1]
    if apply_causal_mask:
        key_length = key.shape[2]
        if query_length != key_length:
            raise ValueError("当前 causal mask 只支持 Prefill 的方形 attention")
        future = torch.triu(
            torch.ones(
                (query_length, key_length), device=x.device, dtype=torch.bool
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(future[None, None], torch.finfo(x.dtype).min)
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
    per_head = torch.matmul(probabilities, value)
    merged = (
        per_head.transpose(1, 2)
        .contiguous()
        .reshape(x.shape[0], query_length, config.hidden_size)
    )
    return F.linear(merged, weights.o_proj.weight, weights.o_proj.bias)


def _decoder_layer(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    weights: DecoderLayerWeights,
    config: QwenConfig,
    *,
    apply_causal_mask: bool,
    cache: LayerKVCache | None = None,
    layer_index: int | None = None,
) -> torch.Tensor:
    residual = x
    x = _rms_norm(x, weights.input_norm, config.rms_norm_eps)
    x = residual + _attention(
        x,
        position_ids,
        weights.attention,
        config,
        apply_causal_mask=apply_causal_mask,
        cache=cache,
        layer_index=layer_index,
    )
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
    """支持等长、无 padding 的 Qwen2 Prefill 与单 token Decode。"""

    def __init__(self, config: QwenConfig, weights: QwenWeights) -> None:
        if len(weights.layers) != config.num_hidden_layers:
            raise ValueError("权重层数与配置不一致")
        self.config = config
        self.weights = weights

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        *,
        cache: LayerKVCache | None = None,
        return_layer_outputs: bool = False,
    ) -> QwenPrefillOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空 [batch, sequence]")
        if input_ids.device != self.weights.embedding.device:
            raise ValueError("input_ids 与权重必须位于同一 device")
        batch, sequence = input_ids.shape
        if sequence > self.config.max_position_embeddings:
            raise ValueError("序列长度超过 max_position_embeddings")
        if cache is not None:
            self._validate_empty_prefill_cache(cache, batch, sequence)

        x = F.embedding(input_ids, self.weights.embedding)
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch, -1)
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None
        if cache is not None:
            cache.begin_append(sequence)
        try:
            for layer_index, layer in enumerate(self.weights.layers):
                x = _decoder_layer(
                    x,
                    positions,
                    layer,
                    self.config,
                    apply_causal_mask=True,
                    cache=cache,
                    layer_index=layer_index if cache is not None else None,
                )
                if captured is not None:
                    captured.append(x)
            if cache is not None:
                cache.commit_append()
        except Exception:
            # 某一层失败时，新 prompt 对所有层保持不可见。
            if cache is not None and cache.pending is not None:
                cache.abort_append()
            raise
        x = _rms_norm(x, self.weights.final_norm, self.config.rms_norm_eps)
        logits = F.linear(x, self.weights.embedding)
        return QwenPrefillOutput(
            logits=logits,
            layer_outputs=tuple(captured) if captured is not None else None,
        )

    @torch.inference_mode()
    def decode_one(
        self,
        token_ids: torch.Tensor,
        *,
        cache: LayerKVCache,
        return_layer_outputs: bool = False,
    ) -> QwenPrefillOutput:
        """只计算一个新 token，并把每层的新 K/V 追加到 Cache。

        `token_ids` 是上一步 logits 选出的 token；本方法返回的 logits 用于预测
        再下一个 token。当前实现要求整个 batch 的历史长度相同且没有 padding。
        """
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("Decode token_ids 必须是 [batch, 1]")
        if token_ids.device != self.weights.embedding.device:
            raise ValueError("token_ids 与权重必须位于同一 device")

        batch = token_ids.shape[0]
        self._validate_cache_layout(cache, batch)
        if cache.pending is not None:
            raise ValueError("Decode 开始前 KV Cache 不能存在未提交 append")
        if cache.length == 0:
            raise ValueError("decode_one 需要先用 Prefill 初始化非空 KV Cache")
        if cache.available_token_capacity < 1:
            raise RuntimeError("KV Cache capacity 已满，无法追加 Decode token")
        if cache.length >= self.config.max_position_embeddings:
            raise ValueError("Decode 后序列长度将超过 max_position_embeddings")

        # 新 token 的绝对位置等于追加前的历史长度，而不是重新从 0 开始。
        position_ids = torch.full(
            (batch, 1), cache.length, dtype=torch.long, device=token_ids.device
        )
        x = F.embedding(token_ids, self.weights.embedding)
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None

        cache.begin_append(1)
        try:
            for layer_index, layer in enumerate(self.weights.layers):
                x = _decoder_layer(
                    x,
                    position_ids,
                    layer,
                    self.config,
                    # 单个 query 位于序列末尾，可以读取包括自己在内的全部 Cache。
                    apply_causal_mask=False,
                    cache=cache,
                    layer_index=layer_index,
                )
                if captured is not None:
                    captured.append(x)
            cache.commit_append()
        except Exception:
            # 保证某层失败时，全局 length 不会暴露只写了一部分层的新 token。
            if cache.pending is not None:
                cache.abort_append()
            raise

        x = _rms_norm(x, self.weights.final_norm, self.config.rms_norm_eps)
        logits = F.linear(x, self.weights.embedding)
        return QwenPrefillOutput(
            logits=logits,
            layer_outputs=tuple(captured) if captured is not None else None,
        )

    def _validate_empty_prefill_cache(
        self, cache: LayerKVCache, batch_size: int, sequence_length: int
    ) -> None:
        self._validate_cache_layout(cache, batch_size)
        if cache.length != 0 or cache.pending is not None:
            raise ValueError("当前 Prefill 初始化只接受空闲且 length=0 的 KV Cache")
        if sequence_length > cache.available_token_capacity:
            raise RuntimeError("prompt length 超过 KV Cache capacity")

    def _validate_cache_layout(
        self, cache: LayerKVCache, batch_size: int
    ) -> None:
        """校验模型与 Cache 的静态布局；生命周期规则由具体入口负责。"""
        expected = {
            "num_layers": self.config.num_hidden_layers,
            "batch_size": batch_size,
            "num_kv_heads": self.config.num_key_value_heads,
            "head_dim": self.config.head_dim,
        }
        actual = {
            "num_layers": cache.num_layers,
            "batch_size": cache.batch_size,
            "num_kv_heads": cache.num_kv_heads,
            "head_dim": cache.head_dim,
        }
        if actual != expected:
            raise ValueError(f"KV Cache 布局不匹配：actual={actual}, expected={expected}")
        if cache.dtype != self.weights.embedding.dtype:
            raise ValueError("KV Cache dtype 与模型权重不一致")
        if cache.device != self.weights.embedding.device:
            raise ValueError("KV Cache device 与模型权重不一致")
