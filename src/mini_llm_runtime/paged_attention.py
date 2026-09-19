"""Decode-only Paged Attention 的 PyTorch correctness reference。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PagedAttentionReferenceOutput:
    """Reference 输出；probabilities 允许 batch 中每个请求具有不同长度。"""

    output: torch.Tensor
    probabilities: tuple[torch.Tensor, ...]


def _validate_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    """验证未来 CUDA kernel 也必须遵守的 shape、dtype 与 device 契约。"""
    if query.ndim != 3:
        raise ValueError("query 必须是 [batch,query_head,head_dim]")
    if key_cache.ndim != 4:
        raise ValueError(
            "key_cache 必须是 [physical_block,kv_head,block_offset,head_dim]"
        )
    if value_cache.shape != key_cache.shape:
        raise ValueError("key_cache 与 value_cache shape 必须相同")
    if block_table.ndim != 2:
        raise ValueError("block_table 必须是 [batch,max_blocks]")
    if sequence_lengths.ndim != 1:
        raise ValueError("sequence_lengths 必须是 [batch]")

    batch_size, num_query_heads, head_dim = query.shape
    total_blocks, num_kv_heads, block_size, cache_head_dim = key_cache.shape
    if batch_size <= 0:
        raise ValueError("batch 必须 > 0")
    if min(total_blocks, num_query_heads, num_kv_heads, block_size, head_dim) <= 0:
        raise ValueError("Attention 与 Cache 各维度必须 > 0")
    if cache_head_dim != head_dim:
        raise ValueError("query 与 K/V Cache 的 head_dim 必须相同")
    if block_table.shape[0] != batch_size:
        raise ValueError("block_table 的 batch 维度与 query 不一致")
    if sequence_lengths.shape[0] != batch_size:
        raise ValueError("sequence_lengths 的 batch 维度与 query 不一致")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_query_heads 必须能被 num_kv_heads 整除")

    supported_float_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if query.dtype not in supported_float_dtypes:
        raise ValueError("query dtype 必须是 float16、bfloat16 或 float32")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise ValueError("query、key_cache 和 value_cache dtype 必须相同")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("block_table dtype 必须是 int32 或 int64")
    if sequence_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("sequence_lengths dtype 必须是 int32 或 int64")

    tensors = (key_cache, value_cache, block_table, sequence_lengths)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("query、K/V Cache 和 metadata 必须位于同一 device")
    return batch_size, num_query_heads, num_kv_heads, block_size, head_dim


@torch.inference_mode()
def paged_decode_attention_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float | None = None,
) -> PagedAttentionReferenceOutput:
    """从非连续物理 block 直接计算 query length=1 的 GQA/MQA Attention。

    参数布局：
      query:          [batch, query_head, head_dim]
      key/value_cache:[physical_block, kv_head, block_offset, head_dim]
      block_table:    [batch, max_logical_blocks]，未使用项允许为 -1
      sequence_lengths:[batch]，表示每个请求可见的有效 K/V token 数

    这是清晰性优先的数值 oracle，不是性能实现。它会发生 Python loop、GPU 同步，
    并物化完整 score/probability；未来 CUDA kernel 不应照搬其执行方式。
    """
    (
        batch_size,
        num_query_heads,
        num_kv_heads,
        block_size,
        head_dim,
    ) = _validate_inputs(
        query, key_cache, value_cache, block_table, sequence_lengths
    )
    attention_scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("scale 必须是有限正数")

    # GQA 映射保持相邻 Query Heads 共享同一个 KV Head。
    group_size = num_query_heads // num_kv_heads
    kv_head_for_query = torch.arange(
        num_query_heads, dtype=torch.long, device=query.device
    ) // group_size

    outputs: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        # `.item()` 在 CUDA 上会同步；reference 接受此成本，benchmark 禁止使用本函数。
        sequence_length = int(sequence_lengths[batch_index].item())
        if sequence_length <= 0:
            raise ValueError("Decode Attention 的 sequence length 必须 > 0")
        required_blocks = (sequence_length + block_size - 1) // block_size
        if required_blocks > block_table.shape[1]:
            raise ValueError(
                f"batch {batch_index} 需要 {required_blocks} 个 block，"
                f"但 block_table 只有 {block_table.shape[1]} 列"
            )

        physical_blocks: list[tuple[int, int]] = []
        score_chunks: list[torch.Tensor] = []
        remaining_tokens = sequence_length
        for logical_block in range(required_blocks):
            physical_block = int(block_table[batch_index, logical_block].item())
            if not 0 <= physical_block < key_cache.shape[0]:
                raise ValueError(
                    f"batch {batch_index} 的有效 block_table[{logical_block}]="
                    f"{physical_block} 越界"
                )
            valid_tokens = min(block_size, remaining_tokens)
            physical_blocks.append((physical_block, valid_tokens))

            # 只读取尾块中的有效 token；未初始化 padding 必须保持不可见。
            key_block = key_cache[
                physical_block, :, :valid_tokens, :
            ].index_select(0, kv_head_for_query)
            scores = torch.einsum(
                "hd,htd->ht", query[batch_index].float(), key_block.float()
            )
            score_chunks.append(scores * attention_scale)
            remaining_tokens -= valid_tokens

        # Softmax 必须跨所有 logical blocks 统一归一化，不能逐 block 独立计算。
        joined_scores = torch.cat(score_chunks, dim=-1)
        probabilities = torch.softmax(joined_scores, dim=-1, dtype=torch.float32)
        all_probabilities.append(probabilities)

        # 按与 score 相同的逻辑顺序消费 V，并用 FP32 累加降低舍入误差。
        per_head_output = torch.zeros(
            (num_query_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        probability_offset = 0
        for physical_block, valid_tokens in physical_blocks:
            value_block = value_cache[
                physical_block, :, :valid_tokens, :
            ].index_select(0, kv_head_for_query)
            block_probability = probabilities[
                :, probability_offset : probability_offset + valid_tokens
            ]
            per_head_output.add_(
                torch.einsum(
                    "ht,htd->hd", block_probability, value_block.float()
                )
            )
            probability_offset += valid_tokens
        outputs.append(per_head_output.to(query.dtype))

    return PagedAttentionReferenceOutput(
        output=torch.stack(outputs, dim=0),
        probabilities=tuple(all_probabilities),
    )
