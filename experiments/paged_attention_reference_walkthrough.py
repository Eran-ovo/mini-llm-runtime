#!/usr/bin/env python3
"""用小型 GQA batch 观察 Decode Paged Attention 的逐 block 寻址。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def contiguous_attention(
    query: torch.Tensor,
    manager: PagedKVCacheManager,
    request_ids: tuple[str, ...],
) -> torch.Tensor:
    """gather 后调用 PyTorch SDPA，作为独立于 Paged 遍历的对照。"""
    num_query_heads = query.shape[1]
    num_kv_heads = manager.storage.num_kv_heads
    group_size = num_query_heads // num_kv_heads
    kv_head_for_query = torch.arange(num_query_heads, device=query.device) // group_size
    outputs = []
    for batch_index, request_id in enumerate(request_ids):
        key, value = manager.gather(request_id)
        key = key[0].index_select(0, kv_head_for_query)
        value = value[0].index_select(0, kv_head_for_query)
        output = F.scaled_dot_product_attention(
            query[batch_index][None, :, None, :].float(),
            key[None].float(),
            value[None].float(),
            dropout_p=0.0,
            is_causal=False,
        )[0, :, 0, :]
        outputs.append(output)
    return torch.stack(outputs).to(query.dtype)


def main() -> None:
    torch.manual_seed(2027)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = PagedKVCacheManager(
        total_blocks=8,
        block_size=3,
        num_layers=1,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device=device,
    )
    manager.create_request("request-a")
    manager.create_request("request-b")
    manager.storage.key.fill_(torch.nan)
    manager.storage.value.fill_(torch.nan)

    # Hq=4、Hkv=2：Q heads [0,1] 共用 KV head 0，[2,3] 共用 KV head 1。
    a_key = torch.randn(1, 2, 5, 8, device=device)
    a_value = torch.randn_like(a_key)
    b_key = torch.randn(1, 2, 7, 8, device=device)
    b_value = torch.randn_like(b_key)
    manager.append_all("request-a", a_key[:, :, :3], a_value[:, :, :3])
    manager.append_all("request-b", b_key[:, :, :3], b_value[:, :, :3])
    manager.append_all("request-a", a_key[:, :, 3:], a_value[:, :, 3:])
    manager.append_all("request-b", b_key[:, :, 3:], b_value[:, :, 3:])

    request_ids = ("request-b", "request-a")
    metadata = manager.build_batch_metadata(request_ids)
    query = torch.randn(2, 4, 8, device=device)
    paged = paged_decode_attention_reference(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    )
    contiguous = contiguous_attention(query, manager, request_ids)
    torch.testing.assert_close(paged.output, contiguous)

    print("===== Decode Paged Attention Reference =====")
    print(f"device           = {device}")
    print(f"query shape      = {list(query.shape)}")
    print(f"K/V pool shape   = {list(manager.storage.key[0].shape)}")
    print(f"request order    = {request_ids}")
    print(f"block table      = {metadata.block_table.cpu().tolist()}")
    print(f"sequence lengths = {metadata.sequence_lengths.cpu().tolist()}")
    print("GQA mapping      = Q[0,1]->KV[0], Q[2,3]->KV[1]")
    print("\n逻辑 token -> (physical block, block offset)")
    for request_id in request_ids:
        table = manager.get_request(request_id)
        mapping = [
            (token, table.locate(token).block_id, table.locate(token).block_offset)
            for token in range(table.token_count)
        ]
        print(f"{request_id}: {mapping}")
    print("\nProbability sums per query head")
    for request_id, probability in zip(
        request_ids, paged.probabilities, strict=True
    ):
        print(f"{request_id}: {probability.sum(dim=-1).cpu().tolist()}")
    max_abs = (paged.output.float() - contiguous.float()).abs().max().item()
    print(f"\nPaged vs contiguous max_abs = {max_abs:.8f}")
    print("对拍通过：非连续 block、尾块 padding、GQA 与变长 batch 均正确。")


if __name__ == "__main__":
    main()
