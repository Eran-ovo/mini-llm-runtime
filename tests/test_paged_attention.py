import pytest
import torch
import torch.nn.functional as F

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager


def make_fragmented_batch() -> tuple[
    PagedKVCacheManager,
    torch.Tensor,
    tuple[str, ...],
]:
    """交错增长两个请求，使每个请求的物理 block 都不连续。"""
    torch.manual_seed(2027)
    manager = PagedKVCacheManager(
        total_blocks=7,
        block_size=3,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    manager.create_request("a")
    manager.create_request("b")

    # 未写入的 block slot 使用 NaN；错误读取 padding 会立刻污染结果。
    manager.storage.key.fill_(torch.nan)
    manager.storage.value.fill_(torch.nan)
    a_key = torch.randn(1, 2, 5, 4)
    a_value = torch.randn_like(a_key)
    b_key = torch.randn(1, 2, 7, 4)
    b_value = torch.randn_like(b_key)

    manager.append_all("a", a_key[:, :, :3], a_value[:, :, :3])   # block 0
    manager.append_all("b", b_key[:, :, :3], b_value[:, :, :3])   # block 1
    manager.append_all("a", a_key[:, :, 3:], a_value[:, :, 3:])   # block 2
    manager.append_all("b", b_key[:, :, 3:], b_value[:, :, 3:])   # blocks 3, 4

    assert manager.get_request("a").block_ids == (0, 2)
    assert manager.get_request("b").block_ids == (1, 3, 4)
    query = torch.randn(2, 4, 4)
    return manager, query, ("b", "a")


def contiguous_reference(
    query: torch.Tensor,
    manager: PagedKVCacheManager,
    request_ids: tuple[str, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """先 gather 连续 K/V，再用 PyTorch SDPA 计算独立 reference。"""
    kv_head_for_query = torch.arange(query.shape[1]) // 2
    outputs = []
    probabilities = []
    for batch_index, request_id in enumerate(request_ids):
        key, value = manager.gather(request_id)
        key = key[0].index_select(0, kv_head_for_query)
        value = value[0].index_select(0, kv_head_for_query)
        scores = torch.einsum(
            "hd,htd->ht", query[batch_index].float(), key.float()
        ) / (query.shape[-1] ** 0.5)
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32)
        output = F.scaled_dot_product_attention(
            query[batch_index][None, :, None, :].float(),
            key[None].float(),
            value[None].float(),
            dropout_p=0.0,
            is_causal=False,
        )[0, :, 0, :]
        outputs.append(output.to(query.dtype))
        probabilities.append(probability)
    return torch.stack(outputs), tuple(probabilities)


def test_paged_attention_matches_contiguous_reference_for_gqa_batch() -> None:
    manager, query, request_ids = make_fragmented_batch()
    metadata = manager.build_batch_metadata(request_ids)

    actual = paged_decode_attention_reference(
        query,
        manager.storage.key[0],
        manager.storage.value[0],
        metadata.block_table,
        metadata.sequence_lengths,
    )
    expected_output, expected_probabilities = contiguous_reference(
        query, manager, request_ids
    )

    assert metadata.block_table.tolist() == [[1, 3, 4], [0, 2, -1]]
    assert metadata.sequence_lengths.tolist() == [7, 5]
    torch.testing.assert_close(actual.output, expected_output)
    for actual_probability, expected_probability in zip(
        actual.probabilities, expected_probabilities, strict=True
    ):
        torch.testing.assert_close(actual_probability, expected_probability)
        torch.testing.assert_close(
            actual_probability.sum(dim=-1),
            torch.ones(query.shape[1]),
        )
        assert torch.isfinite(actual_probability).all()


def test_custom_scale_changes_probabilities_but_preserves_shape_and_dtype() -> None:
    manager, query, request_ids = make_fragmented_batch()
    metadata = manager.build_batch_metadata(request_ids)
    half_query = query.half()
    half_key = manager.storage.key[0].half()
    half_value = manager.storage.value[0].half()

    result = paged_decode_attention_reference(
        half_query,
        half_key,
        half_value,
        metadata.block_table,
        metadata.sequence_lengths,
        scale=0.25,
    )

    assert result.output.shape == half_query.shape
    assert result.output.dtype == torch.float16
    assert all(probability.dtype == torch.float32 for probability in result.probabilities)


def test_mqa_maps_every_query_head_to_the_only_kv_head() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 3, 4)
    key_cache = torch.full((3, 1, 2, 4), torch.nan)
    value_cache = torch.full_like(key_cache, torch.nan)
    # 逻辑顺序先读 physical block 2，再读 physical block 0 的第一个 slot。
    key_cache[2] = torch.randn(1, 2, 4)
    value_cache[2] = torch.randn(1, 2, 4)
    key_cache[0, :, :1] = torch.randn(1, 1, 4)
    value_cache[0, :, :1] = torch.randn(1, 1, 4)
    block_table = torch.tensor([[2, 0]], dtype=torch.int32)
    sequence_lengths = torch.tensor([3], dtype=torch.int32)

    actual = paged_decode_attention_reference(
        query, key_cache, value_cache, block_table, sequence_lengths
    )
    logical_key = torch.cat((key_cache[2], key_cache[0, :, :1]), dim=1)
    logical_value = torch.cat((value_cache[2], value_cache[0, :, :1]), dim=1)
    repeated_key = logical_key.expand(3, -1, -1)
    repeated_value = logical_value.expand(3, -1, -1)
    expected = F.scaled_dot_product_attention(
        query[:, :, None, :].float(),
        repeated_key[None].float(),
        repeated_value[None].float(),
        dropout_p=0.0,
        is_causal=False,
    )[:, :, 0, :]

    torch.testing.assert_close(actual.output, expected)
    assert torch.isfinite(actual.output).all()


def test_used_negative_block_id_is_rejected_instead_of_indexing_last_block() -> None:
    manager, query, request_ids = make_fragmented_batch()
    metadata = manager.build_batch_metadata(request_ids)
    invalid_table = metadata.block_table.clone()
    invalid_table[0, 1] = -1

    with pytest.raises(ValueError, match="有效 block_table"):
        paged_decode_attention_reference(
            query,
            manager.storage.key[0],
            manager.storage.value[0],
            invalid_table,
            metadata.sequence_lengths,
        )


def test_sequence_length_must_fit_block_table_width() -> None:
    manager, query, request_ids = make_fragmented_batch()
    metadata = manager.build_batch_metadata(request_ids)
    too_long = metadata.sequence_lengths.clone()
    too_long[0] = 10

    with pytest.raises(ValueError, match="需要 4 个 block"):
        paged_decode_attention_reference(
            query,
            manager.storage.key[0],
            manager.storage.value[0],
            metadata.block_table,
            too_long,
        )


def test_query_heads_must_be_divisible_by_kv_heads() -> None:
    manager, query, request_ids = make_fragmented_batch()
    metadata = manager.build_batch_metadata(request_ids)

    with pytest.raises(ValueError, match="num_query_heads"):
        paged_decode_attention_reference(
            query[:, :3],
            manager.storage.key[0],
            manager.storage.value[0],
            metadata.block_table,
            metadata.sequence_lengths,
        )


def test_decode_reference_rejects_empty_sequence() -> None:
    query = torch.ones(1, 2, 4)
    key = torch.ones(1, 1, 2, 4)
    value = torch.ones_like(key)
    block_table = torch.empty((1, 0), dtype=torch.int32)
    sequence_lengths = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(ValueError, match="sequence length 必须 > 0"):
        paged_decode_attention_reference(
            query, key, value, block_table, sequence_lengths
        )
