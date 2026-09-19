#!/usr/bin/env python3
"""用 Qwen 形状在 GPU 上验证 paged K/V 物理写入与逻辑 gather。"""

from __future__ import annotations

import torch

from mini_llm_runtime.paged_kv_cache import (
    FixedBlockAllocator,
    PagedKVStorage,
    RequestBlockTable,
)


def make_values(token_count: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (24, 2, token_count, 64)
    element_count = 24 * 2 * token_count * 64
    # 限制数值范围，避免 FP16 arange 在 65504 以上变成 inf。
    key = (
        torch.arange(element_count, device="cuda", dtype=torch.int64)
        .remainder(2_000)
        .to(torch.float16)
        .reshape(shape)
    )
    key = key + offset
    return key, key + 4_000


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("该 walkthrough 需要 CUDA")

    allocator = FixedBlockAllocator(total_blocks=4)
    storage = PagedKVStorage(
        allocator=allocator,
        num_layers=24,
        num_kv_heads=2,
        block_size=16,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    request = RequestBlockTable(
        request_id="request-a", block_size=16, allocator=allocator
    )
    blocker = RequestBlockTable(
        request_id="request-b", block_size=16, allocator=allocator
    )

    prompt_key, prompt_value = make_values(19)
    blocker_key, blocker_value = make_values(16, offset=16_000)
    decode_key, decode_value = make_values(1, offset=8_000)
    # A 先拿 block 0；B 插入拿 block 1；A 增长后只能拿 block 2。
    storage.append_all(request, prompt_key[:, :, :16], prompt_value[:, :, :16])
    storage.append_all(blocker, blocker_key, blocker_value)
    storage.append_all(request, prompt_key[:, :, 16:], prompt_value[:, :, 16:])
    storage.append_all(request, decode_key, decode_value)
    gathered_key, gathered_value = storage.gather(request)
    expected_key = torch.cat((prompt_key, decode_key), dim=2)
    expected_value = torch.cat((prompt_value, decode_value), dim=2)

    print("===== Qwen-shaped Paged K/V Storage =====")
    print(f"physical K shape    = {list(storage.key.shape)}")
    print(f"storage             = {storage.storage_nbytes / 2**20:.4f} MiB")
    print(f"request block table = {list(request.block_ids)}")
    print(f"blocker owns        = {list(blocker.block_ids)}")
    print(f"logical tokens      = {request.token_count}")
    print(f"token capacity      = {request.token_capacity}")
    print(f"internal fragments  = {request.internal_fragmentation_tokens}")
    for token_index in (0, 15, 16, 19):
        location = request.locate(token_index)
        print(
            f"token {token_index:2d} -> block {location.block_id}, "
            f"offset {location.block_offset}"
        )
    print(f"K round-trip exact  = {torch.equal(gathered_key, expected_key)}")
    print(f"V round-trip exact  = {torch.equal(gathered_value, expected_value)}")

    if not torch.equal(gathered_key, expected_key) or not torch.equal(
        gathered_value, expected_value
    ):
        raise SystemExit("paged K/V round-trip 对拍失败")


if __name__ == "__main__":
    main()
