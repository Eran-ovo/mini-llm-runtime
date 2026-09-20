import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_attention_cuda import paged_decode_attention_cuda
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def make_cpu_manager(*, total_blocks: int = 5) -> PagedKVCacheManager:
    return PagedKVCacheManager(
        total_blocks=total_blocks,
        block_size=2,
        num_layers=2,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )


def cpu_values(length: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(2 * length * 2, dtype=torch.float32).reshape(2, 1, length, 2)
    key = key + offset
    return key, key + 1_000


def test_batch_begin_is_atomic_when_later_request_cannot_allocate() -> None:
    manager = make_cpu_manager(total_blocks=3)
    manager.create_request("A")
    manager.create_request("B")
    manager.append_all("A", *cpu_values(2, 0))       # block 0，正好填满
    manager.append_all("B", *cpu_values(2, 100))     # block 1，正好填满
    adapter = PagedBatchDecodeAdapter(manager, ("A", "B"))

    with pytest.raises(RuntimeError, match="容量不足"):
        adapter.begin_decode()

    # A 曾短暂申请 block 2；B 随后失败时必须把 A 一起回滚。
    assert not adapter.active
    assert manager.get_request("A").block_ids == (0,)
    assert manager.get_request("B").block_ids == (1,)
    assert manager.get_request("A").pending is None
    assert manager.get_request("B").pending is None
    assert manager.allocator.free_block_ids == (2,)


def test_batch_metadata_order_write_commit_and_abort() -> None:
    manager = make_cpu_manager()
    manager.create_request("A")
    manager.create_request("B")
    manager.append_all("A", *cpu_values(2, 0))
    manager.append_all("B", *cpu_values(1, 100))
    old_a = tuple(tensor.clone() for tensor in manager.gather("A"))
    old_b = tuple(tensor.clone() for tensor in manager.gather("B"))
    adapter = PagedBatchDecodeAdapter(manager, ("B", "A"))

    assert adapter.build_position_ids().tolist() == [[1], [2]]
    adapter.begin_decode()
    assert adapter.build_position_ids().tolist() == [[1], [2]]
    layer_zero_key = torch.tensor([[[[200.0, 201.0]]], [[[300.0, 301.0]]]])
    layer_zero_value = layer_zero_key + 1_000
    adapter.write_layer(0, layer_zero_key, layer_zero_value)
    layer_zero_inputs = adapter.paged_attention_inputs(0)

    assert layer_zero_inputs.block_table.tolist() == [[1, -1], [0, 2]]
    assert layer_zero_inputs.sequence_lengths.tolist() == [2, 3]
    with pytest.raises(RuntimeError, match="尚未写入的层"):
        adapter.commit_decode()
    adapter.abort_decode()
    assert manager.get_request("A").block_ids == (0,)
    assert manager.get_request("B").block_ids == (1,)
    assert manager.get_request("A").token_count == 2
    assert manager.get_request("B").token_count == 1

    # 重开事务并写完两层；metadata tensor 在层间复用。
    adapter.begin_decode()
    new_keys = []
    new_values = []
    metadata_ptrs = None
    for layer_index in range(2):
        key = layer_zero_key + layer_index * 10
        value = layer_zero_value + layer_index * 10
        new_keys.append(key)
        new_values.append(value)
        adapter.write_layer(layer_index, key, value)
        inputs = adapter.paged_attention_inputs(layer_index)
        pointers = (inputs.block_table.data_ptr(), inputs.sequence_lengths.data_ptr())
        metadata_ptrs = pointers if metadata_ptrs is None else metadata_ptrs
        assert pointers == metadata_ptrs
    adapter.commit_decode()

    assert manager.get_request("B").token_count == 2
    assert manager.get_request("A").token_count == 3
    actual_a = manager.gather("A")
    actual_b = manager.gather("B")
    assert torch.equal(actual_a[0][:, :, :2], old_a[0])
    assert torch.equal(actual_a[1][:, :, :2], old_a[1])
    assert torch.equal(actual_b[0][:, :, :1], old_b[0])
    assert torch.equal(actual_b[1][:, :, :1], old_b[1])
    # batch row 0 是 B，row 1 是 A。
    for layer_index in range(2):
        assert torch.equal(actual_b[0][layer_index, :, -1], new_keys[layer_index][0, :, 0])
        assert torch.equal(actual_a[0][layer_index, :, -1], new_keys[layer_index][1, :, 0])
        assert torch.equal(actual_b[1][layer_index, :, -1], new_values[layer_index][0, :, 0])
        assert torch.equal(actual_a[1][layer_index, :, -1], new_values[layer_index][1, :, 0])


@pytest.mark.skipif(
    not torch.cuda.is_available() or CUDA_HOME is None,
    reason="batched Paged Attention test requires CUDA and nvcc",
)
def test_batched_cuda_matches_per_request_cuda_and_python_reference() -> None:
    generator = torch.Generator(device="cuda").manual_seed(2027)
    manager = PagedKVCacheManager(
        total_blocks=8,
        block_size=3,
        num_layers=2,
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
            2, 2, length, 64, device="cuda", dtype=torch.float16, generator=generator
        )
        value = torch.randn(
            key.shape, device="cuda", dtype=torch.float16, generator=generator
        )
        histories[request_id] = (key, value)
    # 交错增长制造非连续物理布局：A=[0], B=[1,3], C=[2,4]。
    manager.append_all("A", histories["A"][0], histories["A"][1])
    manager.append_all("B", histories["B"][0][:, :, :3], histories["B"][1][:, :, :3])
    manager.append_all("C", histories["C"][0][:, :, :3], histories["C"][1][:, :, :3])
    manager.append_all("B", histories["B"][0][:, :, 3:], histories["B"][1][:, :, 3:])
    manager.append_all("C", histories["C"][0][:, :, 3:], histories["C"][1][:, :, 3:])
    old = {
        request_id: tuple(tensor.clone() for tensor in manager.gather(request_id))
        for request_id in ("A", "B", "C")
    }

    order = ("C", "A", "B")
    adapter = PagedBatchDecodeAdapter(manager, order)
    assert adapter.build_position_ids().tolist() == [[6], [3], [5]]
    adapter.begin_decode()
    queries = []
    new_keys = []
    new_values = []
    first_metadata_ptrs = None

    for layer_index in range(2):
        query = torch.randn(
            3, 14, 64, device="cuda", dtype=torch.float16, generator=generator
        )
        key = torch.randn(
            3, 2, 1, 64, device="cuda", dtype=torch.float16, generator=generator
        )
        value = torch.randn(
            key.shape, device="cuda", dtype=torch.float16, generator=generator
        )
        queries.append(query)
        new_keys.append(key)
        new_values.append(value)
        adapter.write_layer(layer_index, key, value)
        inputs = adapter.paged_attention_inputs(layer_index)

        assert inputs.block_table.tolist() == [
            [2, 4, 5],
            [0, 6, -1],
            [1, 3, -1],
        ]
        assert inputs.sequence_lengths.tolist() == [7, 4, 6]
        pointers = (inputs.block_table.data_ptr(), inputs.sequence_lengths.data_ptr())
        first_metadata_ptrs = pointers if first_metadata_ptrs is None else first_metadata_ptrs
        assert pointers == first_metadata_ptrs

        python_reference = paged_decode_attention_reference(
            query,
            inputs.key_cache,
            inputs.value_cache,
            inputs.block_table,
            inputs.sequence_lengths,
        ).output
        batched_cuda = paged_decode_attention_cuda(
            query,
            inputs.key_cache,
            inputs.value_cache,
            inputs.block_table,
            inputs.sequence_lengths,
        )
        per_request_cuda = torch.cat(
            [
                paged_decode_attention_cuda(
                    query[index : index + 1],
                    inputs.key_cache,
                    inputs.value_cache,
                    inputs.block_table[index : index + 1],
                    inputs.sequence_lengths[index : index + 1],
                )
                for index in range(3)
            ],
            dim=0,
        )
        torch.testing.assert_close(
            batched_cuda, python_reference, rtol=2e-3, atol=2e-3
        )
        torch.testing.assert_close(batched_cuda, per_request_cuda, rtol=0, atol=0)

    adapter.commit_decode()
    assert adapter.build_position_ids().tolist() == [[7], [4], [6]]
    for batch_index, request_id in enumerate(order):
        actual_key, actual_value = manager.gather(request_id)
        old_key, old_value = old[request_id]
        assert torch.equal(actual_key[:, :, :-1], old_key)
        assert torch.equal(actual_value[:, :, :-1], old_value)
        for layer_index in range(2):
            assert torch.equal(
                actual_key[layer_index, :, -1],
                new_keys[layer_index][batch_index, :, 0],
            )
            assert torch.equal(
                actual_value[layer_index, :, -1],
                new_values[layer_index][batch_index, :, 0],
            )
