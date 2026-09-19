#!/usr/bin/env python3
"""用三个请求观察 Paged KV Cache 的逻辑地址映射、释放和复用。"""

from __future__ import annotations

from mini_llm_runtime.paged_kv_cache import (
    FixedBlockAllocator,
    RequestBlockTable,
)


def print_request_mapping(table: RequestBlockTable) -> None:
    print(
        f"request={table.request_id!r} tokens={table.token_count} "
        f"blocks={list(table.block_ids)} capacity={table.token_capacity} "
        f"internal_fragmentation={table.internal_fragmentation_tokens}"
    )
    for token_index in range(table.token_count):
        location = table.locate(token_index)
        print(
            f"  token {token_index:2d} -> physical block "
            f"{location.block_id}, offset {location.block_offset}"
        )


def main() -> None:
    allocator = FixedBlockAllocator(total_blocks=5)
    request_a = RequestBlockTable(
        request_id="A", block_size=4, allocator=allocator
    )
    request_b = RequestBlockTable(
        request_id="B", block_size=4, allocator=allocator
    )

    # A 的 6 个 token 跨越两个 block；B 的 3 个 token 只占一个 block。
    request_a.append_tokens(6)
    request_b.append_tokens(3)
    print("===== A/B 分配后 =====")
    print_request_mapping(request_a)
    print_request_mapping(request_b)
    print(f"free blocks = {list(allocator.free_block_ids)}")

    released = request_a.release()
    print("\n===== A 完成并释放 =====")
    print(f"released blocks = {list(released)}")
    print(f"free blocks     = {list(allocator.free_block_ids)}")

    # C 不需要连续地址；它复用 A 归还的 block 0/1，B 仍独占 block 2。
    request_c = RequestBlockTable(
        request_id="C", block_size=4, allocator=allocator
    )
    request_c.append_tokens(8)
    print("\n===== C 复用物理块 =====")
    print_request_mapping(request_c)
    print(f"B still owns    = {list(request_b.block_ids)}")
    print(f"free blocks     = {list(allocator.free_block_ids)}")


if __name__ == "__main__":
    main()
