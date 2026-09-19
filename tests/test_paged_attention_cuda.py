import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_attention_cuda import (
    _paged_decode_attention_cuda_unchecked,
    paged_decode_attention_cuda,
)
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or CUDA_HOME is None,
    reason="Paged Attention CUDA tests require CUDA and nvcc",
)


def make_fragmented_manager(
    *, num_kv_heads: int = 2
) -> tuple[PagedKVCacheManager, tuple[str, ...]]:
    torch.manual_seed(2027)
    manager = PagedKVCacheManager(
        total_blocks=7,
        block_size=3,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    manager.create_request("a")
    manager.create_request("b")
    manager.storage.key.fill_(torch.nan)
    manager.storage.value.fill_(torch.nan)
    a_key = torch.randn(1, num_kv_heads, 5, 64, device="cuda", dtype=torch.float16)
    a_value = torch.randn_like(a_key)
    b_key = torch.randn(1, num_kv_heads, 7, 64, device="cuda", dtype=torch.float16)
    b_value = torch.randn_like(b_key)
    manager.append_all("a", a_key[:, :, :3], a_value[:, :, :3])
    manager.append_all("b", b_key[:, :, :3], b_value[:, :, :3])
    manager.append_all("a", a_key[:, :, 3:], a_value[:, :, 3:])
    manager.append_all("b", b_key[:, :, 3:], b_value[:, :, 3:])
    return manager, ("b", "a")


def test_cuda_kernel_matches_reference_for_qwen_gqa_shape() -> None:
    manager, request_ids = make_fragmented_manager(num_kv_heads=2)
    metadata = manager.build_batch_metadata(request_ids)
    query = torch.randn(2, 14, 64, device="cuda", dtype=torch.float16)

    expected = paged_decode_attention_reference(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    ).output
    actual = paged_decode_attention_cuda(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    )

    assert metadata.block_table.tolist() == [[1, 3, 4], [0, 2, -1]]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    # benchmark hot path 只跳过 metadata D2H 验证，不能改变 kernel 数值结果。
    unchecked = _paged_decode_attention_cuda_unchecked(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    )
    torch.testing.assert_close(unchecked, actual, rtol=0, atol=0)


def test_cuda_kernel_supports_mqa_and_custom_scale() -> None:
    manager, request_ids = make_fragmented_manager(num_kv_heads=1)
    metadata = manager.build_batch_metadata(request_ids)
    query = torch.randn(2, 4, 64, device="cuda", dtype=torch.float16)
    scale = 0.2

    expected = paged_decode_attention_reference(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
        scale=scale,
    ).output
    actual = paged_decode_attention_cuda(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
        scale=scale,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_cuda_kernel_block_boundary_suite() -> None:
    torch.manual_seed(17)
    lengths = (1, 2, 3, 4, 5, 6, 7)
    manager = PagedKVCacheManager(
        total_blocks=sum((length + 2) // 3 for length in lengths),
        block_size=3,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    request_ids = tuple(f"request-{length}" for length in lengths)
    for request_id, length in zip(request_ids, lengths, strict=True):
        manager.create_request(request_id)
        key = torch.randn(1, 2, length, 64, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        manager.append_all(request_id, key, value)
    metadata = manager.build_batch_metadata(request_ids)
    query = torch.randn(len(lengths), 14, 64, device="cuda", dtype=torch.float16)

    expected = paged_decode_attention_reference(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    ).output
    actual = paged_decode_attention_cuda(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    )

    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


def test_cuda_v1_rejects_used_padding_entry() -> None:
    query = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 2, 64, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.tensor([[-1]], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([1], device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="used block_table entry"):
        paged_decode_attention_cuda(
            query, key, value, block_table, sequence_lengths
        )


def test_cuda_v1_rejects_unsupported_head_dim() -> None:
    query = torch.randn(1, 2, 32, device="cuda", dtype=torch.float16)
    key = torch.randn(1, 1, 2, 32, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    block_table = torch.tensor([[0]], device="cuda", dtype=torch.int32)
    sequence_lengths = torch.tensor([1], device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="head_dim=64"):
        paged_decode_attention_cuda(
            query, key, value, block_table, sequence_lengths
        )
