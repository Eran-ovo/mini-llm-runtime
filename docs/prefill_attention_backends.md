# Packed Prefill 的两种 Attention 路径

v2.1 把同一步多个请求的 token 拼成 `[1, total_tokens]`，共用 Q/K/V Projection、
MLP 和 LM Head 调用。每个请求仍须只能读取自己的历史 token。

## 地址和计算关系

对于长度 `L₀, L₁, ...`，`offsets = [0, L₀, L₀+L₁, ...]`。
第 `i` 个请求使用 `Q/K/V[:, :, offsets[i]:offsets[i+1]]`，RoPE position
在每段从零开始。每层 K/V 按同样的 offsets 写入对应请求的 Paged KV block table；
只有整批所有层和 LM Head 成功后才提交 Cache 长度。

`masked` 构造总 token 数 `T` 的块对角 causal mask，
随后对 `[T,T]` scores 执行 Softmax。它每层只做一次 Attention 计算，短 prompt
表现较好，但 scores 和 probability 的内存随 `T²` 增长，且会计算不同请求间
随后被 mask 掉的分数。

`segmented_sdpa` 对每个请求的 Q/K/V 切片调用一次 PyTorch SDPA，并传入
`is_causal=True`。它不生成跨请求的 `[T,T]` scores；计算规模随
`sum(Lᵢ²)` 增长。该实现每层仍有多次 Attention 调用，不是 fused varlen kernel。

Engine 默认的 `auto` 在本轮总 token 数至少 512、最长 prompt 至少 128 时选
`segmented_sdpa`，其他形状选 `masked`。这两个条件来自下方 RTX 3060 Laptop 的
有限长度扫描：最长 prompt 条件避免大量极短请求仅因总数大就触发很多 SDPA 调用。
它是保守的单 GPU 启发式，不是通用最优调度器。显式指定 backend 始终覆盖 `auto`；
每个 Engine step 的实际选择记录在 correctness artifact 的 `steps` 中。

## 使用与验证

```bash
python -m experiments.continuous_batch_engine_runner \
  --local-files-only \
  --prefill-attention-backend segmented_sdpa

python scripts/benchmark_packed_prefill.py \
  --local-files-only \
  --fixed-prompt-length 512 \
  --compare-segmented-sdpa --compare-vectorized-kv-write \
  --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/prefill_attention_512
```

Benchmark 在同一进程中交错执行 serial、scalar KV + masked、vectorized KV +
masked、vectorized KV + segmented，奇偶轮反转顺序。比较 Attention 时只看后两项，
确保 KV 写入方式相同。
它用 CUDA Event 测量 Prefill ModelRunner 与 Paged KV 写入，并保存原始样本、
中位数、显存峰值、GPU/CUDA/PyTorch 与 Git commit。输入构造和 Scheduler
不在计时区间内，因此这些数据不是端到端 TTFT。

FP16 GEMM 随 batch 形状变化会有舍入差异；benchmark 同时检查 greedy token
一致、logits 与 KV 的相对 L2 误差，并保留最大绝对差。真实模型的生成正确性
另用 Hugging Face 门禁验证。

v2.2 的旧扫描仍使用标量 KV 写入，写入耗时掩盖了 Attention 的差距。
v2.3 批量写入后，在 clean-tree commit `3e3597e` 上对 4 个等长 prompt 重测：

| 每请求 token | masked + vectorized KV | segmented + vectorized KV | segmented 峰值显存 |
|---:|---:|---:|---:|
| 8 | 27.31 ms | 27.52 ms | 961.43 MiB |
| 64 | 27.49 ms | 27.59 ms | 975.00 MiB |
| 128 | 51.69 ms | 37.62 ms | 983.97 MiB |
| 256 | 116.63 ms | 67.86 ms | 1007.64 MiB |
| 512 | 365.33 ms | 133.84 ms | 1055.99 MiB |

512-token 组的 masked 峰值是 1453.53 MiB。8/64-token 的差异很小且样本波动
较大，不据此声称短 prompt 有收益。原始 JSON 在本地
`benchmarks/results/v2_4_attention_probe_len{8,64,128,256,512}_3e3597e/`；
benchmark 目录不进 Git，复现时须重新运行并核对 commit、GPU 时钟和原始样本。
本机 WSL 下的算子级 profiler 未给出可靠的逐 CUDA kernel 时间，因此只把上面
CUDA Event 的多轮正式样本用于性能结论；不能据此声称某个具体 kernel 的占比。

自动选择接入 Engine 后，在 clean-tree commit `7f4c844` 上复测：4×8 token 的
masked 为 26.46 ms（`auto` 选它）；4×256 token 的 masked/segmented 分别为
116.61/68.77 ms；4×512 token 分别为 372.13/136.47 ms，峰值显存分别为
1453.53/1055.99 MiB。三组均通过 logits、greedy token 和 KV 对拍，原始样本位于
`benchmarks/results/v2_4_attention_len{8,256,512}_clean_7f4c844/`。
8-token 组的路径差异与测量波动接近，不把它解释为短 prompt 的收益。
