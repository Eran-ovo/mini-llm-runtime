#!/usr/bin/env python3
"""用人工 Q/K/V 演示变长多请求 Paged Decode batch。"""

import torch

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_attention_cuda import paged_decode_attention_cuda
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    generator = torch.Generator(device="cuda").manual_seed(2027)
    manager = PagedKVCacheManager(
        total_blocks=8,
        block_size=3,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    for request_id in ("A", "B", "C"):
        manager.create_request(request_id)

    histories = {}
    for request_id, length in (("A", 3), ("B", 5), ("C", 6)):
        key = torch.randn(
            1, 2, length, 64, device="cuda", dtype=torch.float16, generator=generator
        )
        value = torch.randn(
            key.shape, device="cuda", dtype=torch.float16, generator=generator
        )
        histories[request_id] = (key, value)
    manager.append_all("A", *histories["A"])
    manager.append_all("B", histories["B"][0][:, :, :3], histories["B"][1][:, :, :3])
    manager.append_all("C", histories["C"][0][:, :, :3], histories["C"][1][:, :, :3])
    manager.append_all("B", histories["B"][0][:, :, 3:], histories["B"][1][:, :, 3:])
    manager.append_all("C", histories["C"][0][:, :, 3:], histories["C"][1][:, :, 3:])

    order = ("C", "A", "B")
    adapter = PagedBatchDecodeAdapter(manager, order)
    positions = adapter.build_position_ids()
    adapter.begin_decode()
    query = torch.randn(
        3, 14, 64, device="cuda", dtype=torch.float16, generator=generator
    )
    key = torch.randn(
        3, 2, 1, 64, device="cuda", dtype=torch.float16, generator=generator
    )
    value = torch.randn(
        key.shape, device="cuda", dtype=torch.float16, generator=generator
    )
    adapter.write_layer(0, key, value)
    inputs = adapter.paged_attention_inputs(0)

    reference = paged_decode_attention_reference(
        query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_table,
        inputs.sequence_lengths,
    ).output
    batched = paged_decode_attention_cuda(
        query,
        inputs.key_cache,
        inputs.value_cache,
        inputs.block_table,
        inputs.sequence_lengths,
    )
    individual = torch.cat(
        [
            paged_decode_attention_cuda(
                query[index : index + 1],
                inputs.key_cache,
                inputs.value_cache,
                inputs.block_table[index : index + 1],
                inputs.sequence_lengths[index : index + 1],
            )
            for index in range(3)
        ]
    )
    reference_error = (batched.float() - reference.float()).abs()
    individual_error = (batched.float() - individual.float()).abs()
    torch.testing.assert_close(batched, reference, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(batched, individual, rtol=0, atol=0)
    adapter.commit_decode()

    print("===== Multi-request Paged Decode batch =====")
    print(f"request order          = {list(order)}")
    print(f"positions before append= {positions[:, 0].tolist()}")
    print(f"sequence lengths       = {inputs.sequence_lengths.tolist()}")
    print(f"block table            = {inputs.block_table.tolist()}")
    print(f"output shape           = {list(batched.shape)}")
    print(f"max_abs vs Python ref  = {reference_error.max().item():.8f}")
    print(f"max_abs vs individual  = {individual_error.max().item():.8f}")
    print(
        "committed lengths      = "
        f"{[manager.get_request(request_id).token_count for request_id in order]}"
    )


if __name__ == "__main__":
    main()
