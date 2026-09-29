import pytest
import torch

from mini_llm_runtime.paged_batch import PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Packed KV GPU write test requires CUDA",
)


def make_fragmented_manager():
    manager = PagedKVCacheManager(
        total_blocks=6,
        block_size=3,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        dtype=torch.float16,
        device="cuda",
    )
    manager.create_request("A")
    manager.create_request("B")
    # 交错预留使 A=(0,2)、B=(1,3)，覆盖非连续 block table。
    manager.reserve_request_capacity("A", 3)
    manager.reserve_request_capacity("B", 3)
    manager.reserve_request_capacity("A", 5)
    manager.reserve_request_capacity("B", 4)
    assert manager.get_request("A").block_ids == (0, 2)
    assert manager.get_request("B").block_ids == (1, 3)
    manager.storage.key.fill_(torch.nan)
    manager.storage.value.fill_(torch.nan)
    return manager


def test_vectorized_write_matches_scalar_for_fragmented_blocks_and_strided_input():
    scalar_manager = make_fragmented_manager()
    vector_manager = make_fragmented_manager()
    scalar = PagedBatchPrefillAdapter(
        scalar_manager, ("A", "B"), (5, 4), write_backend="scalar"
    )
    vectorized = PagedBatchPrefillAdapter(
        vector_manager, ("A", "B"), (5, 4), write_backend="vectorized"
    )
    scalar.begin_prefill()
    vectorized.begin_prefill()

    # 模拟 _heads() 的 transpose：输入 [1,H,T,D] 不连续。
    key = torch.arange(1 * 9 * 2 * 64, device="cuda").to(torch.float16)
    key = key.reshape(1, 9, 2, 64).transpose(1, 2)
    value = (torch.arange(1 * 9 * 2 * 64, device="cuda") + 1000)
    value = value.to(torch.float16).reshape(1, 9, 2, 64).transpose(1, 2)
    assert not key.is_contiguous()
    assert not value.is_contiguous()
    for layer_index in range(2):
        scalar.write_layer(layer_index, key + layer_index, value + layer_index)
        vectorized.write_layer(layer_index, key + layer_index, value + layer_index)
    scalar.commit_prefill()
    vectorized.commit_prefill()

    for request_id in ("A", "B"):
        scalar_kv = scalar_manager.gather(request_id)
        vector_kv = vector_manager.gather(request_id)
        for actual, expected in zip(vector_kv, scalar_kv, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in (
        (vector_manager.storage.key, scalar_manager.storage.key),
        (vector_manager.storage.value, scalar_manager.storage.value),
    ):
        assert torch.equal(torch.isnan(actual), torch.isnan(expected))
        torch.testing.assert_close(
            torch.nan_to_num(actual), torch.nan_to_num(expected), rtol=0, atol=0
        )
