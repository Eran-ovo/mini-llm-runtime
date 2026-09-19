"""验证分区覆盖、空分区和跨分区 softmax 的数值语义。"""
import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from experiments.split_kv_attention import split_kv_attention
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_attention_cuda import paged_decode_attention_cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or CUDA_HOME is None, reason="需要 CUDA toolkit/GPU"
)


def make_case(lengths, hq=14, hkv=2, block_size=16):
    torch.manual_seed(2027)
    counts = [(n + block_size - 1) // block_size for n in lengths]
    total = sum(counts)
    permutation = torch.randperm(total).tolist()
    k = torch.full((total, hkv, block_size, 64), torch.nan,
                   dtype=torch.float16, device="cuda")
    v = torch.full_like(k, torch.nan)
    table = torch.full((len(lengths), max(counts)), -1, dtype=torch.int32)
    cursor = 0
    for b, n in enumerate(lengths):
        for j in range(counts[b]):
            physical = permutation[cursor]
            cursor += 1
            table[b, j] = physical
            valid = min(block_size, n - j * block_size)
            k[physical, :, :valid] = torch.randn(hkv, valid, 64, device="cuda")
            v[physical, :, :valid] = torch.randn(hkv, valid, 64, device="cuda")
    q = torch.randn(len(lengths), hq, 64, device="cuda", dtype=torch.float16)
    return q, k, v, table.cuda(), torch.tensor(lengths, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("splits", [1, 2, 3, 8, 64])
@pytest.mark.parametrize("hq,hkv", [(14, 2), (4, 1), (2, 2)])
def test_split_matches_reference_with_empty_and_partial_blocks(splits, hq, hkv):
    args = make_case([1, 15, 16, 17, 65], hq, hkv)
    expected = paged_decode_attention_reference(*args, scale=0.2).output
    actual = split_kv_attention(*args, num_splits=splits, scale=0.2)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_split_long_context_and_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        args = make_case([2048, 2049])
        expected = paged_decode_attention_reference(*args).output
        checked = split_kv_attention(*args, num_splits=8)
        actual = split_kv_attention(*args, num_splits=8, validate_metadata=False)
        # 对照调用也验证 v1 共享 validator 重构后的行为。
        v1 = paged_decode_attention_cuda(*args)
    stream.synchronize()
    torch.testing.assert_close(actual, checked, atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(v1, expected, atol=2e-3, rtol=2e-3)


def test_merge_uses_global_weights_not_average_of_partition_outputs():
    args = make_case([2], hq=1, hkv=1, block_size=1)
    q, k, v, table, _ = args
    q.fill_(1)
    low, high = table[0].tolist()
    k[low].fill_(-10)
    k[high].fill_(10)
    v[low].fill_(-3)
    v[high].fill_(7)
    actual = split_kv_attention(*args, num_splits=8)
    # 两段 score 差异巨大，全局输出应接近 7，而非局部输出平均值 2。
    torch.testing.assert_close(actual, torch.full_like(q, 7), atol=0, rtol=0)


@pytest.mark.parametrize("splits", [0, -1, 65])
def test_rejects_invalid_split_count(splits):
    with pytest.raises(RuntimeError, match="num_splits"):
        split_kv_attention(*make_case([1]), num_splits=splits)


def test_rejects_invalid_metadata():
    args = make_case([17])
    args[3][0, 1] = -1
    with pytest.raises(RuntimeError, match="used block_table"):
        split_kv_attention(*args)
