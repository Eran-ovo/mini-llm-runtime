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

Engine 默认的 `auto` 在本轮总 token 数至少 512，且满足“最长 prompt 至少 128”
或“平均每请求至少 32 token”任一条件时选 `segmented_sdpa`，其他形状选
`masked`。保留最长 prompt 条件，并用平均长度条件覆盖一批中等长度的请求；
两者都不满足时，避免大量短请求触发过多 SDPA 调用。
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

## v2.5 变长请求校准

在 clean-tree commit `af3280d` 上增加 `--prompt-lengths`，按四条已 tokenize 的
短句循环构造任意请求数与长度。下表两条被计时的路径都使用 vectorized KV write；
每组均先与逐请求 reference 对拍 logits、greedy token 和 KV，且每组包含 warmup、
交错 CUDA Event 多轮原始样本。表内负数代表 segmented 更慢。

| 请求长度 | 总 token | masked | segmented | segmented 降幅 | 旧 auto |
|---|---:|---:|---:|---:|---|
| 256,8,8,8 | 280 | 27.78 ms | 27.19 ms | 2.1% | masked |
| 8,64,128,256 | 456 | 42.58 ms | 35.57 ms | 16.5% | masked |
| 8×64 | 512 | 50.47 ms | 39.82 ms | 21.1% | masked |
| 16×32 | 512 | 48.64 ms | 40.48 ms | 16.8% | masked |
| 32×16 | 512 | 44.36 ms | 46.74 ms | -5.4% | masked |
| 128 + 31×13 | 531 | 55.18 ms | 52.07 ms | 5.6% | segmented |

8×64 的独立重复实验仍显示 segmented 快 16.5%；16×32 的独立重复实验为
8.8%，但笔记本频率波动较明显，因此不把中位数差异当作通用保证。新规则只扩展
`T≥512` 内的选择，8,64,128,256 这组当前仍会选 masked；后续需单独验证是否
降低总 token 阈值，不在同一次优化里一起改变。原始 JSON 位于本地
`benchmarks/results/v2_5_ragged_*_af3280d/`。

失败实验：`(128,128,128,1)` 在计时前被 correctness gate 拦截。虽然 logits
相对 L2 为 0.00668 且 greedy token 相同，逐请求 reference 与 packed KV 的最大
相对 L2 为 0.01371，超过当前 0.01 门槛；因此没有正式 timing JSON，不能拿它
调阈值。尚未确认这是 FP16 GEMM 形状舍入还是实现问题，不能为得到性能数字而放宽门槛。

规则提交 `57ff869` 后又从 clean tree 复测三档；`auto` 选择、同写入 backend
的 CUDA Event 中位数与显存峰值如下。每档 warmup 3、正式样本 10，所有
logits/token/KV 门禁通过：

| 形状 | auto 选择 | masked | segmented | segmented 相对变化 | masked / segmented 峰值显存 |
|---|---|---:|---:|---:|---:|
| 8×64 | segmented | 45.77 ms | 40.08 ms | -12.4% | 1000.01 / 985.13 MiB |
| 16×32 | segmented | 45.84 ms | 43.49 ms | -5.1% | 1002.33 / 987.45 MiB |
| 32×16 | masked | 45.14 ms | 47.54 ms | +5.3% | 1006.98 / 992.10 MiB |

原始 JSON 位于本地 `benchmarks/results/v2_5_dispatch_*_clean_57ff869/`。
16×32 的差距接近本机样本波动，应视为边界启发式而非稳定性能保证。
真实 Qwen Engine 的 8×64 单步检查还验证了 `auto` 实际走 segmented，
与显式 masked 的 8 个 greedy token 完全一致，且全部 KV block 释放；
该单步检查不是正式性能 benchmark。
