# Packed Prefill 的两种 Attention 路径

v2.1 把同一步多个请求的 token 拼成 `[1, total_tokens]`，共用 Q/K/V Projection、
MLP 和 LM Head 调用。每个请求仍须只能读取自己的历史 token。

## 地址和计算关系

对于长度 `L₀, L₁, ...`，`offsets = [0, L₀, L₀+L₁, ...]`。
第 `i` 个请求使用 `Q/K/V[:, :, offsets[i]:offsets[i+1]]`，RoPE position
在每段从零开始。每层 K/V 按同样的 offsets 写入对应请求的 Paged KV block table；
只有整批所有层和 LM Head 成功后才提交 Cache 长度。

`masked` 是 Engine 默认路径：构造总 token 数 `T` 的块对角 causal mask，
随后对 `[T,T]` scores 执行 Softmax。它每层只做一次 Attention 计算，短 prompt
表现较好，但 scores 和 probability 的内存随 `T²` 增长，且会计算不同请求间
随后被 mask 掉的分数。

`segmented_sdpa` 对每个请求的 Q/K/V 切片调用一次 PyTorch SDPA，并传入
`is_causal=True`。它不生成跨请求的 `[T,T]` scores；计算规模随
`sum(Lᵢ²)` 增长。该实现每层仍有多次 Attention 调用，不是 fused varlen kernel。

## 使用与验证

```bash
python -m experiments.continuous_batch_engine_runner \
  --local-files-only \
  --prefill-attention-backend segmented_sdpa

python scripts/benchmark_packed_prefill.py \
  --local-files-only \
  --fixed-prompt-length 512 \
  --compare-segmented-sdpa \
  --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/prefill_attention_512
```

Benchmark 在同一进程中交错执行 serial、masked 和 segmented，奇偶轮反转顺序。
它用 CUDA Event 测量 Prefill ModelRunner 与 Paged KV 写入，并保存原始样本、
中位数、显存峰值、GPU/CUDA/PyTorch 与 Git commit。输入构造和 Scheduler
不在计时区间内，因此这些数据不是端到端 TTFT。

FP16 GEMM 随 batch 形状变化会有舍入差异；benchmark 同时检查 greedy token
一致、logits 与 KV 的相对 L2 误差，并保留最大绝对差。真实模型的生成正确性
另用 Hugging Face 门禁验证。

长度扫描显示分段 SDPA 在本机的短、中、长 prompt 档均未获得可靠延迟收益；
每请求 512 token、4 请求时观察到明显峰值显存下降。故默认继续用 `masked`，
`segmented_sdpa` 作为显式低显存选项。下一项延迟优化应单独测量并改进
Paged KV 的逐 token 写入，而不把 Attention 与 KV 写入同时改动。
