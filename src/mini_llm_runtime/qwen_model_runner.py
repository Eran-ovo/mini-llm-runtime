"""不依赖 Hugging Face 模块对象的最小 Qwen2 ModelRunner。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F

from .cuda_extension import cuda_toolkit_available, load_cuda_extension
from .kv_cache import LayerKVCache
from .paged_attention_cuda import (
    _paged_decode_attention_cuda_unchecked,
    _paged_decode_attention_split_cuda,
    paged_decode_attention_cuda,
)
from .paged_batch import PagedBatchDecodeAdapter, PagedBatchPrefillAdapter
from .paged_kv_adapter import PagedRequestKVCache
from .paged_kv_manager import PagedKVCacheManager
from .qwen_config import QwenConfig
from .qwen_weights import AttentionWeights, DecoderLayerWeights, QwenWeights


VARLEN_PREFILL_ATTENTION_BACKENDS = frozenset({"varlen_sdpa", "varlen_cuda"})
PACKED_PREFILL_ATTENTION_BACKENDS = (
    frozenset({"masked", "segmented_sdpa", "batched_sdpa"}) | VARLEN_PREFILL_ATTENTION_BACKENDS
)
PackedAttention = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _prepare_varlen_attention(
    backend: str, offsets: tuple[int, ...], max_length: int, device: torch.device,
) -> PackedAttention:
    if backend == "varlen_cuda":
        try:
            from ai_infra_ops import FlashAttentionVarlenPlan, flashattention_varlen
        except ImportError as error:
            raise RuntimeError("varlen_cuda 需要安装带 flashattention_varlen 的 ai_infra_ops") from error
        plan = FlashAttentionVarlenPlan(offsets, device)
        return lambda q, k, v: flashattention_varlen(q, k, v, plan)

    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=device)
    # PyTorch 2.6 exposes packed Flash Attention through ATen. The public
    # SDPA entry uses dense inputs or NestedTensor, which cannot express this GQA layout.
    return lambda q, k, v: torch.ops.aten._flash_attention_forward(
        q, k, v, cu_seqlens, cu_seqlens, max_length, max_length, 0.0, True, False,
    )[0]


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    dtype = x.dtype
    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + epsilon)
    return weight * normalized.to(dtype)


def _cuda_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    if (
        x.is_cuda and x.dtype == weight.dtype == torch.float16
        and x.numel() > 0 and x.is_contiguous() and weight.is_contiguous()
    ):
        return load_cuda_extension().decode_norm(x, weight, epsilon)
    return _rms_norm(x, weight, epsilon)


def _heads(x: torch.Tensor, count: int, head_dim: int) -> torch.Tensor:
    batch, sequence, _ = x.shape
    return x.reshape(batch, sequence, count, head_dim).transpose(1, 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope_factors(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = position_ids.float().unsqueeze(-1) * inv_freq
    embedding = torch.cat((frequencies, frequencies), dim=-1)
    return embedding.cos().to(dtype).unsqueeze(1), embedding.sin().to(dtype).unsqueeze(1)


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
    rotary: tuple[torch.Tensor, torch.Tensor],
    weights: AttentionWeights,
    config: QwenConfig,
    *,
    apply_causal_mask: bool,
    cache: LayerKVCache | PagedBatchDecodeAdapter | PagedBatchPrefillAdapter | None = None,
    layer_index: int | None = None,
    use_paged_decode_attention: bool = False,
    decode_attention_splits: int = 1,
    validate_decode_metadata: bool = True,
    packed_causal_mask: torch.Tensor | None = None,
    packed_segments: tuple[tuple[int, int], ...] | None = None,
    uniform_prompt_length: int | None = None,
    varlen_attention: PackedAttention | None = None,
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
    cos, sin = rotary
    if cache is not None and layer_index is None:
        raise ValueError("使用 KV Cache 时必须提供 layer_index")
    cuda_rope = config.head_dim == 64 and query.is_cuda and query.dtype == cos.dtype == torch.float16
    if (
        cuda_rope and use_paged_decode_attention and isinstance(cache, PagedBatchDecodeAdapter)
        and all(tensor.is_contiguous() for tensor in (query, key, value))
    ):
        query = cache.rotate_and_write_layer(layer_index, query, key, value, cos, sin)
    elif (
        cuda_rope and isinstance(cache, PagedBatchPrefillAdapter)
        and cache.write_backend == "vectorized" and cuda_toolkit_available()
        and all(tensor.transpose(1, 2).is_contiguous() for tensor in (query, key, value))
    ):
        query, key = cache.rotate_and_write_layer(layer_index, query, key, value, cos, sin)
    else:
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin
        if cache is not None:
            cache.write_layer(layer_index, key, value)
    query_length = x.shape[1]
    if use_paged_decode_attention:
        if not isinstance(cache, (PagedRequestKVCache, PagedBatchDecodeAdapter)):
            raise TypeError(
                "paged_cuda Decode 需要 PagedRequestKVCache 或 "
                "PagedBatchDecodeAdapter"
            )
        if query_length != 1 or apply_causal_mask:
            raise ValueError("Paged Attention v1 只支持无显式 mask 的单 token Decode")
        inputs = cache.paged_attention_inputs(layer_index)
        query_decode = query[:, :, 0, :].contiguous()
        # 地址元数据对所有层相同：第 0 层走安全入口完成一次同步验证，后续层
        # 只跳过 metadata CPU copy，其余 shape/dtype/device 检查仍在 C++ 中执行。
        parameters = {}
        if decode_attention_splits > 1 or (isinstance(cache, PagedBatchDecodeAdapter) and query.is_cuda):
            operation = _paged_decode_attention_split_cuda
            parameters = {
                "num_splits": decode_attention_splits,
                "validate_metadata": validate_decode_metadata and layer_index == 0,
            }
        elif validate_decode_metadata and layer_index == 0:
            operation = paged_decode_attention_cuda
        else:
            operation = _paged_decode_attention_cuda_unchecked
        per_head = operation(
            query_decode,
            inputs.key_cache,
            inputs.value_cache,
            inputs.block_table,
            inputs.sequence_lengths,
            **parameters,
        ).unsqueeze(2)
    else:
        if cache is not None and not isinstance(cache, PagedBatchPrefillAdapter):
            key, value = cache.view_layer(layer_index, include_pending=True)

        if varlen_attention is not None:
            packed_qkv = (tensor[0].transpose(0, 1).contiguous() for tensor in (query, key, value))
            merged = varlen_attention(*packed_qkv).reshape(1, query_length, config.hidden_size)
            return F.linear(merged, weights.o_proj.weight, weights.o_proj.bias)

        if uniform_prompt_length is not None:
            requests = query_length // uniform_prompt_length
            # [1,H,B*L,D] → [B,H,L,D]。各请求使用独立 batch 行，共享一次 SDPA。
            shape = (requests, uniform_prompt_length)
            per_head = F.scaled_dot_product_attention(
                query.unflatten(2, shape).squeeze(0).transpose(0, 1),
                key.unflatten(2, shape).squeeze(0).transpose(0, 1),
                value.unflatten(2, shape).squeeze(0).transpose(0, 1),
                is_causal=True,
                enable_gqa=config.gqa_group_size > 1,
            )
            merged = per_head.transpose(1, 2).reshape(1, query_length, config.hidden_size)
            return F.linear(merged, weights.o_proj.weight, weights.o_proj.bias)
        if packed_segments is not None:
            # Q/K/V Projection 已对所有请求合批；Attention 按边界分别执行，
            # 从而只处理 sum(length_i²) 个因果关系，不生成 [T,T] 跨请求矩阵。
            per_head = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        query[:, :, start:end],
                        key[:, :, start:end],
                        value[:, :, start:end],
                        is_causal=True,
                        enable_gqa=config.gqa_group_size > 1,
                    )
                    for start, end in packed_segments
                ],
                dim=2,
            )
        else:
            key = _repeat_kv(key, config.gqa_group_size)
            value = _repeat_kv(value, config.gqa_group_size)
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(config.head_dim)
            if packed_causal_mask is not None:
                # 同一请求只能看见本请求过去和当前 token；不同请求的分数必须遮住。
                scores = scores.masked_fill(
                    ~packed_causal_mask[None, None], torch.finfo(x.dtype).min
                )
            elif apply_causal_mask:
                key_length = key.shape[2]
                if query_length != key_length:
                    raise ValueError("当前 causal mask 只支持 Prefill 的方形 attention")
                future = torch.triu(
                    torch.ones(
                        (query_length, key_length), device=x.device, dtype=torch.bool
                    ),
                    diagonal=1,
                )
                scores = scores.masked_fill(
                    future[None, None], torch.finfo(x.dtype).min
                )
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
    rotary: tuple[torch.Tensor, torch.Tensor],
    weights: DecoderLayerWeights,
    config: QwenConfig,
    *,
    apply_causal_mask: bool,
    cache: LayerKVCache | PagedBatchDecodeAdapter | PagedBatchPrefillAdapter | None = None,
    layer_index: int | None = None,
    use_paged_decode_attention: bool = False,
    decode_attention_splits: int = 1,
    validate_decode_metadata: bool = True,
    packed_causal_mask: torch.Tensor | None = None,
    packed_segments: tuple[tuple[int, int], ...] | None = None,
    uniform_prompt_length: int | None = None,
    varlen_attention: PackedAttention | None = None,
) -> torch.Tensor:
    use_cuda_ops = isinstance(cache, PagedBatchDecodeAdapter) or (
        isinstance(cache, PagedBatchPrefillAdapter) and x.is_cuda and cuda_toolkit_available()
    )
    normalize = _cuda_rms_norm if use_cuda_ops else _rms_norm
    residual = x
    x = normalize(x, weights.input_norm, config.rms_norm_eps)
    x = residual + _attention(
        x,
        rotary,
        weights.attention,
        config,
        apply_causal_mask=apply_causal_mask,
        cache=cache,
        layer_index=layer_index,
        use_paged_decode_attention=use_paged_decode_attention,
        decode_attention_splits=decode_attention_splits,
        validate_decode_metadata=validate_decode_metadata,
        packed_causal_mask=packed_causal_mask,
        packed_segments=packed_segments,
        uniform_prompt_length=uniform_prompt_length,
        varlen_attention=varlen_attention,
    )
    residual = x
    x = normalize(x, weights.post_attention_norm, config.rms_norm_eps)
    gate = F.linear(x, weights.mlp.gate_proj.weight)
    up = F.linear(x, weights.mlp.up_proj.weight)
    if use_cuda_ops and gate.is_cuda and gate.dtype == up.dtype == torch.float16:
        activated = load_cuda_extension().swiglu(gate, up)
    else:
        activated = F.silu(gate) * up
    x = F.linear(activated, weights.mlp.down_proj.weight)
    return residual + x


@dataclass(frozen=True)
class QwenPrefillOutput:
    logits: torch.Tensor
    layer_outputs: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class _CapturedForward:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor

    def replay(self, inputs: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        if inputs is not None:
            for target, source in zip(self.inputs, inputs, strict=True):
                target.copy_(source)
        self.graph.replay()
        # 保留调用方持有的结果，下一次执行可以覆盖图内缓冲。
        return self.output.clone()


class QwenPrefillRunner:
    """支持 Qwen2 Prefill、单请求 Decode 与多请求 Paged Decode。"""

    def __init__(
        self,
        config: QwenConfig,
        weights: QwenWeights,
        *,
        decode_attention_backend: str = "torch",
        enable_decode_cuda_graph: bool = False,
        enable_prefill_cuda_graph: bool = False,
    ) -> None:
        if len(weights.layers) != config.num_hidden_layers:
            raise ValueError("权重层数与配置不一致")
        if decode_attention_backend not in {"torch", "paged_cuda"}:
            raise ValueError("decode_attention_backend 必须是 'torch' 或 'paged_cuda'")
        if enable_decode_cuda_graph and (
            decode_attention_backend != "paged_cuda" or weights.embedding.device.type != "cuda"
        ):
            raise ValueError("Decode CUDA Graph 需要 CUDA 上的 paged_cuda backend")
        if enable_prefill_cuda_graph and weights.embedding.device.type != "cuda":
            raise ValueError("Prefill CUDA Graph 需要 CUDA 权重")
        self.config = config
        self.weights = weights
        self.decode_attention_backend = decode_attention_backend
        self.enable_decode_cuda_graph = enable_decode_cuda_graph
        self.enable_prefill_cuda_graph = enable_prefill_cuda_graph
        self._decode_graphs: WeakKeyDictionary[
            PagedKVCacheManager, dict[tuple[int, int, int, torch.dtype, bool], _CapturedForward | None]
        ] = WeakKeyDictionary()
        self._prefill_graphs: WeakKeyDictionary[
            PagedKVCacheManager, dict[tuple[int, int, torch.dtype], _CapturedForward | None]
        ] = WeakKeyDictionary()
        self._capture_stream = (
            torch.cuda.Stream(device=weights.embedding.device)
            if enable_decode_cuda_graph or enable_prefill_cuda_graph
            else None
        )
        self._rope_inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(
                    0, config.head_dim, 2, device=weights.embedding.device, dtype=torch.float32
                )
                / config.head_dim
            )
        )

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
        rotary = _rope_factors(positions, self._rope_inv_freq, x.dtype)
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None
        if cache is not None:
            cache.begin_append(sequence)
        try:
            for layer_index, layer in enumerate(self.weights.layers):
                x = _decoder_layer(
                    x,
                    rotary,
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
    def prefill_batch(
        self,
        input_ids: torch.Tensor,
        *,
        cache: PagedBatchPrefillAdapter,
        attention_backend: str = "masked",
    ) -> QwenPrefillOutput:
        """一次模型前向处理多个无 padding prompt，返回每条请求末位 logits。

        `input_ids` 为 `[1,total_tokens]`，依照 `cache.request_ids` 顺序拼接。
        `masked` 使用 dense mask，`segmented_sdpa` 逐请求调用 SDPA，
        `batched_sdpa` 将等长请求组合成一次 causal SDPA，`varlen_sdpa`
        和 `varlen_cuda` 将变长请求组合为一次支持 GQA 的调用。
        """
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Packed Prefill input_ids 必须是 [1,total_tokens]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Packed Prefill input_ids 必须是 int32 或 int64")
        if input_ids.device != self.weights.embedding.device:
            raise ValueError("input_ids 与权重必须位于同一 device")
        self._validate_paged_batch_layout(cache, cache.batch_size)
        offsets = cache.offsets
        if input_ids.shape[1] != offsets[-1]:
            raise ValueError("packed token 数与各请求 lengths 之和不一致")
        max_length = max(cache.lengths)
        if max_length > self.config.max_position_embeddings:
            raise ValueError("至少一个 prompt 超过 max_position_embeddings")
        if cache.active:
            raise ValueError("Prefill 开始前不能存在 active transaction")
        if attention_backend not in PACKED_PREFILL_ATTENTION_BACKENDS:
            raise ValueError(f"attention_backend 必须是 {', '.join(sorted(PACKED_PREFILL_ATTENTION_BACKENDS))}")

        uniform_length = None
        causal_mask = None
        segments = None
        varlen_attention = None
        if attention_backend == "batched_sdpa":
            uniform_length = cache.lengths[0]
            if any(length != uniform_length for length in cache.lengths):
                raise ValueError("batched_sdpa 要求等长 prompt")
            positions = torch.arange(uniform_length, device=input_ids.device)
            positions = positions.repeat(cache.batch_size).unsqueeze(0)
        else:
            # position 每遇到新请求都归零。segment_id 用于阻止跨请求 Attention。
            positions_cpu = [position for length in cache.lengths for position in range(length)]
            positions = torch.tensor(
                (positions_cpu,), dtype=torch.long, device=input_ids.device
            )
            if attention_backend == "masked":
                segment_cpu = [
                    index for index, length in enumerate(cache.lengths) for _ in range(length)
                ]
                segment_ids = torch.tensor(segment_cpu, device=input_ids.device)
                indices = torch.arange(offsets[-1], device=input_ids.device)
                causal_mask = (segment_ids[:, None] == segment_ids[None, :]) & (
                    indices[:, None] >= indices[None, :]
                )
            elif attention_backend == "segmented_sdpa":
                segments = tuple(zip(offsets[:-1], offsets[1:], strict=True))
            else:
                if not (
                    input_ids.is_cuda and self.weights.embedding.dtype == torch.float16
                    and self.config.head_dim == 64 and torch.version.cuda is not None
                    and torch.cuda.get_device_capability(input_ids.device)[0] >= 8
                ):
                    raise ValueError("varlen Prefill 需要 CUDA sm_80+、FP16 和 head_dim=64")
                varlen_attention = _prepare_varlen_attention(
                    attention_backend, offsets, max_length, input_ids.device
                )

        cache.begin_prefill()
        try:
            if (
                self.enable_prefill_cuda_graph
                and uniform_length is not None
                and cache.write_backend == "vectorized"
                and offsets[-1] <= 512
            ):
                logits = self._prefill_with_graph(input_ids, positions, cache, uniform_length)
            else:
                logits = self._prefill_layers(
                    input_ids, positions, cache,
                    causal_mask=causal_mask, segments=segments, uniform_length=uniform_length,
                    varlen_attention=varlen_attention,
                )
            cache.commit_prefill()
        except Exception:
            if cache.active:
                cache.abort_prefill()
            raise
        return QwenPrefillOutput(logits=logits)

    def _prefill_layers(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedBatchPrefillAdapter,
        *,
        causal_mask: torch.Tensor | None = None,
        segments: tuple[tuple[int, int], ...] | None = None,
        uniform_length: int | None = None,
        varlen_attention: PackedAttention | None = None,
    ) -> torch.Tensor:
        x = F.embedding(input_ids, self.weights.embedding)
        rotary = _rope_factors(positions, self._rope_inv_freq, x.dtype)
        for layer_index, layer in enumerate(self.weights.layers):
            x = _decoder_layer(
                x,
                rotary,
                layer,
                self.config,
                apply_causal_mask=False,
                cache=cache,
                layer_index=layer_index,
                packed_causal_mask=causal_mask,
                packed_segments=segments,
                uniform_prompt_length=uniform_length,
                varlen_attention=varlen_attention,
            )
        # 只对每个 prompt 的末位执行 LM Head，避免分配 [T,vocab] logits。
        if uniform_length is not None:
            last_hidden = x[:, uniform_length - 1 :: uniform_length]
        else:
            last_indices = torch.tensor(
                [end - 1 for end in cache.offsets[1:]],
                dtype=torch.long,
                device=input_ids.device,
            )
            last_hidden = x[:, last_indices]
        last_hidden = _rms_norm(last_hidden, self.weights.final_norm, self.config.rms_norm_eps)
        return F.linear(last_hidden, self.weights.embedding).transpose(0, 1)

    def _prefill_with_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        cache: PagedBatchPrefillAdapter,
        prompt_length: int,
    ) -> torch.Tensor:
        signature = (cache.batch_size, prompt_length, input_ids.dtype)
        captured = self._prefill_graphs.setdefault(cache.manager, {})
        if signature not in captured:
            if len(captured) < 4:
                captured[signature] = None
            return self._prefill_layers(input_ids, positions, cache, uniform_length=prompt_length)

        inputs = (input_ids, positions, *cache._graph_inputs())
        replay = captured[signature]
        if replay is None:
            def forward(static_ids, static_positions, blocks, offsets):
                with cache._use_graph_inputs((blocks, offsets)):
                    return self._prefill_layers(
                        static_ids, static_positions, cache, uniform_length=prompt_length
                    )

            replay = self._capture_forward(
                inputs, forward, lambda: cache._set_all_layers_written(False)
            )
            captured[signature] = replay
            logits = replay.replay()
        else:
            logits = replay.replay(inputs)
        cache._set_all_layers_written(True)
        return logits

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
        use_paged_decode_attention = self.decode_attention_backend == "paged_cuda"
        if use_paged_decode_attention and not isinstance(cache, PagedRequestKVCache):
            raise TypeError("paged_cuda Decode 需要 PagedRequestKVCache")
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
        rotary = _rope_factors(position_ids, self._rope_inv_freq, x.dtype)
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None

        cache.begin_append(1)
        try:
            for layer_index, layer in enumerate(self.weights.layers):
                x = _decoder_layer(
                    x,
                    rotary,
                    layer,
                    self.config,
                    # 单个 query 位于序列末尾，可以读取包括自己在内的全部 Cache。
                    apply_causal_mask=False,
                    cache=cache,
                    layer_index=layer_index,
                    use_paged_decode_attention=use_paged_decode_attention,
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

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids: torch.Tensor,
        *,
        cache: PagedBatchDecodeAdapter,
        return_layer_outputs: bool = False,
        defer_commit: bool = False,
    ) -> QwenPrefillOutput:
        """对多个变长请求各推进一个 token，并共享每层的一次 Paged Attention。

        `token_ids` 的第 i 行必须属于 `cache.request_ids[i]`。本入口只负责
        Decode；每个请求仍需先用单请求或 packed Prefill 填满自己的 Cache。
        设置 `defer_commit` 时，由调用方在取回 GPU 结果后提交或回滚。
        """
        logits, layers = self._decode_batch(
            token_ids, cache, return_layer_outputs=return_layer_outputs,
            defer_commit=defer_commit, greedy=False,
        )
        return QwenPrefillOutput(logits=logits, layer_outputs=layers)

    @torch.inference_mode()
    def decode_batch_greedy(
        self,
        token_ids: torch.Tensor,
        *,
        cache: PagedBatchDecodeAdapter,
        defer_commit: bool = False,
    ) -> torch.Tensor:
        """每条请求返回一个 token ID，CUDA Graph 可连同 LM Head 重放 argmax。"""
        tokens, _ = self._decode_batch(
            token_ids, cache, return_layer_outputs=False,
            defer_commit=defer_commit, greedy=True,
        )
        return tokens

    def _decode_batch(
        self,
        token_ids: torch.Tensor,
        cache: PagedBatchDecodeAdapter,
        *,
        return_layer_outputs: bool,
        defer_commit: bool,
        greedy: bool,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if self.decode_attention_backend != "paged_cuda":
            raise ValueError("decode_batch 只支持 paged_cuda backend")
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("Batched Decode token_ids 必须是 [batch, 1]")
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Batched Decode token_ids 必须是 int32 或 int64")
        if token_ids.device != self.weights.embedding.device:
            raise ValueError("token_ids 与权重必须位于同一 device")
        if cache.active:
            raise ValueError("Decode 开始前不能存在 active batch transaction")

        batch = token_ids.shape[0]
        self._validate_paged_batch_layout(cache, batch)
        # 每一行的 position 都是该请求 append 前的长度，变长 batch 不能共用标量。
        positions = cache.current_positions
        # 长度本来就在 CPU request table，不能用 GPU tensor.item() 制造热路径同步。
        if max(positions) >= self.config.max_position_embeddings:
            raise ValueError("Decode 后至少一个请求将超过 max_position_embeddings")
        decode_attention_splits = 1
        # 分区数来自 Qwen2.5-0.5B 的端到端 Decode 形状扫描。
        if batch == 1 and positions[0] + 1 >= 2048:
            decode_attention_splits = 16
        elif batch == 8 and min(positions) + 1 >= 128:
            decode_attention_splits = 4
        captured: list[torch.Tensor] | None = [] if return_layer_outputs else None
        cache.begin_decode()
        try:
            position_ids = cache.build_position_ids()
            if self.enable_decode_cuda_graph and captured is None:
                output = self._decode_with_graph(
                    token_ids, position_ids, cache, decode_attention_splits, greedy=greedy,
                )
            else:
                output = self._decode_layers(
                    token_ids, position_ids, cache, decode_attention_splits, captured,
                    greedy=greedy,
                )
            # LM Head 也成功后才提交，使整个 decode_batch 调用保持原子语义。
            if not defer_commit:
                cache.commit_decode()
        except Exception:
            # kernel、projection 或某一层写入失败，均不暴露半完成 token。
            if cache.active:
                cache.abort_decode()
            raise

        return output, tuple(captured) if captured is not None else None

    def _decode_layers(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: PagedBatchDecodeAdapter,
        num_splits: int,
        layer_outputs: list[torch.Tensor] | None = None,
        *,
        validate_metadata: bool = True,
        greedy: bool = False,
    ) -> torch.Tensor:
        x = F.embedding(token_ids, self.weights.embedding)
        rotary = _rope_factors(position_ids, self._rope_inv_freq, x.dtype)
        for layer_index, layer in enumerate(self.weights.layers):
            x = _decoder_layer(
                x,
                rotary,
                layer,
                self.config,
                apply_causal_mask=False,
                cache=cache,
                layer_index=layer_index,
                use_paged_decode_attention=True,
                decode_attention_splits=num_splits,
                validate_decode_metadata=validate_metadata,
            )
            if layer_outputs is not None:
                layer_outputs.append(x)
        x = _cuda_rms_norm(x, self.weights.final_norm, self.config.rms_norm_eps)
        logits = F.linear(x, self.weights.embedding)
        return logits[:, 0].argmax(dim=-1) if greedy else logits

    def _decode_with_graph(
        self,
        token_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache: PagedBatchDecodeAdapter,
        num_splits: int,
        *,
        greedy: bool,
    ) -> torch.Tensor:
        # 图中的 KV 指针绑定到 storage 实例。slot、block table、长度和 token 每轮更新。
        graph_inputs = (token_ids, cache._graph_inputs())
        block_table, _ = cache._attention_metadata()
        signature = (*block_table.shape, num_splits, token_ids.dtype, greedy)
        captured = self._decode_graphs.setdefault(cache.manager, {})
        if signature not in captured:
            # 首次仍走带校验的 eager 路径，且限制每个池的图私有显存占用。
            if len(captured) < 8:
                captured[signature] = None
            return self._decode_layers(token_ids, position_ids, cache, num_splits, greedy=greedy)

        replay = captured[signature]
        if replay is None:
            def forward(static_tokens, static_metadata):
                with cache._use_graph_inputs(static_metadata):
                    return self._decode_layers(
                        static_tokens,
                        cache.build_position_ids(),
                        cache,
                        num_splits,
                        validate_metadata=False,
                        greedy=greedy,
                    )
            replay = self._capture_forward(
                graph_inputs, forward, lambda: cache._set_all_layers_written(False)
            )
            captured[signature] = replay
            output = replay.replay()
        else:
            output = replay.replay(graph_inputs)
        cache._set_all_layers_written(True)
        return output

    def _capture_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        forward: Callable[..., torch.Tensor],
        reset_writes: Callable[[], None],
    ) -> _CapturedForward:
        stream = self._capture_stream
        assert stream is not None
        stream.wait_stream(torch.cuda.current_stream(inputs[0].device))
        with torch.cuda.stream(stream):
            forward(*inputs)
        reset_writes()
        graph = torch.cuda.CUDAGraph()
        static_inputs = tuple(tensor.clone() for tensor in inputs)
        with torch.cuda.graph(graph, stream=stream):
            output = forward(*static_inputs)
        return _CapturedForward(graph, static_inputs, output)

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

    def _validate_paged_batch_layout(
        self, cache: PagedBatchDecodeAdapter | PagedBatchPrefillAdapter, batch_size: int
    ) -> None:
        """校验 batch Adapter 与模型的静态布局，不读取或 gather 物理 K/V。"""
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
            raise ValueError(
                f"Paged batch KV Cache 布局不匹配：actual={actual}, "
                f"expected={expected}"
            )
        if cache.dtype != self.weights.embedding.dtype:
            raise ValueError("Paged batch KV Cache dtype 与模型权重不一致")
        if cache.device != self.weights.embedding.device:
            raise ValueError("Paged batch KV Cache device 与模型权重不一致")
