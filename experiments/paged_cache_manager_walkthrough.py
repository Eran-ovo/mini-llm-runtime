#!/usr/bin/env python3
"""观察多请求 Paged KV Cache manager 的 GPU metadata、统计与复用。"""

from __future__ import annotations

import torch

from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def make_values(token_count: int, *, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (24, 2, token_count, 64)
    element_count = 24 * 2 * token_count * 64
    key = (
        torch.arange(element_count, device="cuda", dtype=torch.int64)
        .remainder(2_000)
        .to(torch.float16)
        .reshape(shape)
        + offset
    )
    return key, key + 4_000


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("该 walkthrough 需要 CUDA")

    manager = PagedKVCacheManager(
        total_blocks=6,
        block_size=16,
        num_layers=24,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    manager.create_request("A")
    manager.create_request("B")

    a_first = make_values(16, offset=0)
    b_values = make_values(8, offset=8_000)
    a_second = make_values(4, offset=16_000)
    manager.append_all("A", *a_first)   # block 0
    manager.append_all("B", *b_values)  # block 1
    manager.append_all("A", *a_second)  # block 2，A 变为非连续 [0, 2]

    metadata = manager.build_batch_metadata(("B", "A"))
    stats = manager.stats()
    gathered_a = manager.gather("A")
    expected_a = tuple(
        torch.cat((first, second), dim=2)
        for first, second in zip(a_first, a_second)
    )

    print("===== Scheduler batch order: [B, A] =====")
    print(f"metadata device   = {metadata.block_table.device}")
    print(f"metadata dtype    = {metadata.block_table.dtype}")
    print(f"block table       = {metadata.block_table.cpu().tolist()}")
    print(f"sequence lengths  = {metadata.sequence_lengths.cpu().tolist()}")
    print(f"A gather exact    = {all(torch.equal(a, b) for a, b in zip(gathered_a, expected_a))}")
    print("\n===== Pool statistics =====")
    print(f"active requests   = {stats.active_requests}")
    print(f"allocated/free    = {stats.allocated_blocks}/{stats.free_blocks}")
    print(f"block utilization = {stats.block_utilization:.4f}")
    print(f"slot utilization  = {stats.slot_utilization:.4f}")
    print(f"internal fragments= {stats.internal_fragmentation_tokens}")
    print(f"storage           = {stats.storage_nbytes / 2**20:.4f} MiB")

    # B 释放 block 1 后，C 应复用它；A 的 [0,2] 必须保持不变。
    released = manager.release_request("B")
    manager.create_request("C")
    manager.append_all("C", *make_values(5, offset=24_000))
    print("\n===== Release B, admit C =====")
    print(f"B released        = {list(released)}")
    print(f"A blocks          = {list(manager.get_request('A').block_ids)}")
    print(f"C blocks          = {list(manager.get_request('C').block_ids)}")

    if metadata.block_table.cpu().tolist() != [[1, -1], [0, 2]]:
        raise SystemExit("GPU batch block table 不符合预期")
    if manager.get_request("C").block_ids != (1,):
        raise SystemExit("C 没有复用 B 释放的 block")


if __name__ == "__main__":
    main()
