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
    extension = load_cuda_extension()
    return extension.paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        sequence_lengths,
        attention_scale,
    )
