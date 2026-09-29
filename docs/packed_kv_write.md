# Packed Prefill 的 KV 批量写入

## 为什么需要这一步

旧的 `PagedKVStorage.write_layer()` 对每个新 token 调用 `table.locate()`，再向
GPU 分别发起一次 K 和一次 V 的 `copy_()`。一批请求总共有 `T` 个 prompt token、
模型有 `N` 层时，写入约需 `2 × T × N` 次小操作。长 prompt 下，CPU 发射和
GPU 小 kernel 的固定开销可能超过实际复制的 K/V 字节。

Packed Prefill 已经持有请求顺序和每条请求的 `offsets`。批量写入在
`begin_prefill()` 后，把每个 token 的逻辑位置映射成两个 GPU 索引：

```text
logical_token → logical_block = logical_token // block_size
              → physical_block = block_table[logical_block]
              → block_offset = logical_token % block_size
```

此映射在本轮所有 Decoder Layer 中不变，因此只创建一次。

## 物理布局与写入

Cache 形状仍是 `[layer, physical_block, kv_head, block_offset, head_dim]`。
当前层的 K/V 形状是 `[1, kv_head, T, head_dim]`，转成
`[T, kv_head, head_dim]` 后，使用配对的 `(physical_block, block_offset)`
索引，分别批量写 K 和 V。不能直接把原 Cache reshape 为
`[physical_slot, kv_head, head_dim]`：`kv_head` 位于 `block_offset` 前，
这样的 reshape 不保证与原存储共享内存，也会改变地址含义。

GPU 索引保证同一批次中的物理 slot 唯一。Cache 长度仍在全层和 LM Head
成功后提交；失败时撤销 pending 元数据，已写入但不可见的 K/V 字节由后续
请求覆盖。`scalar` 路径保留为 correctness 与单变量 benchmark 基线。

## 验证入口

```bash
python -m pytest -q tests/test_packed_kv_write_cuda.py tests/test_packed_prefill.py

python scripts/benchmark_packed_prefill.py \
  --local-files-only --fixed-prompt-length 256 \
  --compare-vectorized-kv-write --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/packed_kv_write_256
```

Benchmark 的 `packed` 和 `packed_vectorized` 使用同一模型、Attention backend、
token 输入、物理池容量及请求顺序。准备 Cache 在 CUDA Event 区间外；
Prefill ModelRunner 和 KV 写入在计时区间内。结果保存每轮原始延迟、median、
显存峰值、环境与 Git commit；它不是端到端 TTFT。

这一步减少的是小规模 GPU 写入次数。若后续 profiler 显示批量索引仍是瓶颈，
可以在保持同一 slot mapping 和测试不变的前提下换成专用 CUDA kernel。
