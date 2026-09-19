"""split-KV 实验入口；验证完成前不加入稳定 runtime dispatch。"""

import torch
from mini_llm_runtime.cuda_extension import load_cuda_extension


def split_kv_attention(query, key, value, table, lengths, *, num_splits=8,
                       scale=None, validate_metadata=True):
    """FP16/D64；unchecked 调用只允许使用已经验证且未改变的 metadata。"""
    if isinstance(num_splits, bool) or not isinstance(num_splits, int):
        raise ValueError("num_splits 必须是整数")
    if query.ndim != 3:
        raise ValueError("query 必须是 [batch,query_head,head_dim]")
    with torch.inference_mode():
        return load_cuda_extension().paged_decode_attention_split(
            query, key, value, table, lengths,
            query.shape[-1] ** -0.5 if scale is None else float(scale),
            num_splits, validate_metadata,
        )
