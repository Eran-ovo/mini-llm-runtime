# 变长 Prefill 的 KV 数值差异诊断

## 问题与边界

在 Qwen2.5-0.5B、RTX 3060 Laptop、FP16、请求长度 `(128,128,128,1)` 下，
packed Prefill 与四次逐请求 Prefill 的最终 greedy token 相同，但第 4 个请求的
跨层 V Cache 相对 L2 为 `0.0137077`，超过 benchmark 原有的 `0.01` 门槛。
该形状仍不能进入正式性能计时；本页是 correctness 诊断，不是 benchmark。

`relative L2 = ||reference - actual||₂ / ||reference||₂`。它反映整个张量的相对
偏差，但数值大小会受 reference 范数影响；单看百分比不能判断物理 block 写错了。

## 定位顺序与结果

1. 同一 packed 前向只切换 `scalar` / `vectorized` KV 写入：所有请求、所有层的
   已提交 K/V 和最终 logits **逐元素完全相同**。这排除了本形状下批量物理
   `(block, offset)` 索引造成当前差异的可能。
2. 第 4 个 1-token 请求在第 0 层的 embedding、input RMSNorm、Q/K/V projection、
   Attention 输出、Attention 残差和 post-attention RMSNorm 完全一致。
   第一次分叉在 MLP 的 `gate_proj` / `up_proj`：逐请求的 1 行计算与 packed
   的 385 行计算，FP16 相对 L2 分别为 `2.16e-4` / `2.85e-4`。
3. 固定同一批 FP16 输入与权重，仅把这两个 projection 转为 FP32 计算，
   相应差异降至 `2.41e-7` / `3.25e-7`。因此证据指向不同矩阵形状下的
   低精度算术/舍入路径，而不是输入顺序或 KV 地址映射。
4. 误差经残差逐层传播。第 4 个请求的 hidden reference L2 从第 20 层的
   `1579.36` 降到第 21 层的 `62.98`，小的绝对偏差此时表现为更大的相对偏差。
   V Cache 的最大单层相对 L2 是第 22 层 `0.0289932`；第 22/23 层贡献了
   该请求跨层 V 差异平方和的约 99.98%。
5. 完整 FP32 前向是独立佐证：第 4 个请求跨层 V 相对 L2 从 FP16 的
   `0.0137077` 降到 `1.03e-5`，最坏单层从 `0.0289932` 降到 `2.00e-5`。
   这个对照同时改变了模型激活/计算精度，不能单独用于证明某个 CUDA kernel
   的具体舍入机制；第 3 点的相同输入/权重投影对照更有针对性。

前三个 128-token 请求的首层 Attention 输出也会出现微小差异，因此不能把
“所有请求的数值差异”都归因于 MLP；以上定位针对触发 1% KV 门槛的第 4 个请求。
它证明了这次失败不是 vectorized KV write 引起，但不代表任意变长输入都安全。

## 复现与后续判据

诊断脚本只在 `experiments/` 中 hook 私有层函数，不改变稳定 ModelRunner：

```bash
python -m experiments.diagnose_ragged_kv --local-files-only --dtype float16 \
  --output-dir benchmarks/results/ragged_kv_fp16
python -m experiments.diagnose_ragged_kv --local-files-only --dtype float32 \
  --output-dir benchmarks/results/ragged_kv_fp32
```

clean-tree commit `fafddb4` 的完整逐请求、逐层原始诊断位于本地
`benchmarks/results/v2_6_ragged_kv_fp{16,32}_clean_fafddb4/result.json`，
包括环境与 Git commit。该目录默认不进 Git。诊断运行不使用 CUDA Event、
warmup 或多轮计时，也不能被引用为性能结果。

本次**不放宽 1% 门槛**。下一步应把 benchmark 的两种问题分开验证：
物理 KV 写入是否逐元素正确，和不同 FP16 模型执行形状是否在明确的数值容差内。
在新判据经过独立测试之前，`(128,128,128,1)` 继续被当前 benchmark 拦截。
