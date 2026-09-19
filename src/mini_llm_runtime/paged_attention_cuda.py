"""Decode Paged Attention v1 CUDA extension 的 Python 稳定入口。"""

from __future__ import annotations

import torch

from .cuda_extension import load_cuda_extension


def paged_decode_attention_cuda(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """运行 FP16、head_dim=64 的第一版 CUDA correctness kernel。"""
    attention_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    return _paged_decode_attention_cuda(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        scale=attention_scale,
        validate_metadata=True,
    )


def _paged_decode_attention_cuda_unchecked(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """仅供已预先验证 metadata 的内部热路径使用。"""
    attention_scale = query.shape[-1] ** -0.5 if scale is None else float(scale)
    return _paged_decode_attention_cuda(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        scale=attention_scale,
        validate_metadata=False,
    )


def _paged_decode_attention_cuda(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    scale: float,
    validate_metadata: bool,
) -> torch.Tensor:
    extension = load_cuda_extension()
    return extension.paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        scale,
        validate_metadata,
    )
