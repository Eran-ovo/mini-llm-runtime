# 首页性能数据与可复核样本

这里保留首页实验的原始样本摘录、配置和计算口径，便于直接核对结果。

## 来源

| 文件 | 原始产物 | Source commit |
|---|---|---|
| [static-continuous.json](static-continuous.json) | `release_v1_d31ded2/static_vs_continuous/result.json` | `d31ded234702d9f51d419d8c2d200e51c5994950` |
| [prefill-4x512.json](prefill-4x512.json) | `v2_4_attention_len512_clean_7f4c844/result.json` | `7f4c8444c34b78732775f8df86582e5c7f4075ce` |

这两份历史摘录保留测试参数、correctness、原始 latency/memory 样本、统计结果、
GPU/CUDA/PyTorch/编译器环境和 `git_dirty=false`。
`provenance` 保存原始完整文件的相对路径、SHA-256 和保留字段，
明确省略逐 step 轨迹或 prompt token buffer；没有更改测量值。

完整 v1.0.0 时间线、CSV 与 manifest 位于
[Release evidence asset](https://github.com/Eran-ovo/mini-llm-runtime/releases/download/v1.0.0/mini-llm-runtime-v1.0.0-release-evidence.tar.gz)。
Prefill 完整原始 JSON 当前保存在作者本地结果目录；本页公开其计算结论所需样本，
复现命令见下文。

## 与 vLLM 同机交错

[vllm-interleaved.json](vllm-interleaved.json) 保留 2026-11-25 的全部逐轮样本。
采样使用随本页发布的优化实现，运行时源码位于本地工作区。旧 mini、优化版和 vLLM
在一张 RTX 3060 Laptop 6 GB 上依次运行，每组先各预热 3 轮，再正式交错测量 12 轮。
归档包括每轮执行顺序、输入和输出 token 的 SHA-256，以及每轮开始前的 GPU 温度、
SM 频率、功耗与显存占用快照。

- 三组形状为 `4×32`、`8×256`、`1×2048`，每请求固定输出 32 token。输入通过
  `random.Random(0)` 在每组内生成，ID 范围为 3–999。两边使用同一 checkpoint、
  FP16 和 greedy 解码。
- max running 为 8，单步 token budget 为 4096，block size 为 16。mini 的 KV 池为
  144 块，vLLM 0.8.5 V1 使用 `gpu_memory_utilization=0.40`，启动后可用 KV 容量
  为 18,240 token。vLLM 的前缀缓存关闭，两边启用 CUDA Graph。
- 吞吐以输出 token 数除以从提交该轮请求到最后一个 token 返回的墙钟时间。
  首页增幅为两个后端的 12 轮吞吐中位数之比减一。配对胜场统计同一轮中 mini
  的吞吐更高的次数。TTFT 为每轮请求中位数，再对 12 轮取中位数。
- 采样量的是本地推理引擎的固定 burst。权重加载、预热、tokenizer、HTTP
  和在线流量均在测量窗口之外。三组 mini 优化前后输出 token 一致，
  `8×256` 组的 vLLM 输出 token 序列与 mini 有差异。

[复现命令](../../README.md#复现-vllm-交错对照) 使用当前 mini 和 vLLM 双后端交错。
归档中的 `prior-mini` 是当时的本地实现，用于观察优化前后的变化。

## 错峰变长 Prefill 对照

RTX 3060 Laptop 6 GB、PyTorch 2.6.0+cu124、Qwen2.5-0.5B FP16。
四个后端各预热 3 轮，按轮换顺序交错测量 12 轮。两引擎均使用
160 个 16-token KV blocks、8 个运行请求上限、每步 4096 token 预算和 CUDA Graph。
每轮先提交 4 条 64-token prompt，各生成 48 token。计划到达时间 60 ms
的三条 prompt 长度为 128、512、1024，各生成 32 token。
所有测量轮次都在首批请求仍在 Decode 时提交了三条后到请求。
[复现命令](../../README.md#变长-prefill-与错峰请求) · [逐轮样本](vllm-varlen-mixed.json)

| 后端 | 输出 tok/s | 后到请求 P95 TTFT | 首批 P95 ITL | 首批最大 ITL | 达标 req/s |
|---|---:|---:|---:|---:|---:|
| mini `auto` | 465.8 | 125.2 ms | 17.1 ms | 121.9 ms | 4.85 |
| mini `auto_varlen_sdpa` | 466.2 | 124.2 ms | 17.0 ms | 120.1 ms | 4.86 |
| mini `auto_varlen_cuda` | 467.7 | 123.4 ms | 16.9 ms | 119.0 ms | 4.87 |
| vLLM 0.8.5 V1 | 549.2 | 111.5 ms | 14.3 ms | 109.5 ms | 5.72 |

表中各列取 12 轮的中位数。TTFT 从各请求的计划到达时刻开始计时。
达标请求同时满足 TTFT ≤ 200 ms 和该请求的每个 token 间隔 ≤ 50 ms。
自研路径在 8/12 个配对轮次改善后到请求 TTFT，配对降幅中位数为 2.8 ms。
首批请求在长 Prefill 到达后仍有约 119 ms 的单次生成间隔。
三组固定 burst 也收录在同一份归档中，用于复核等长路径。

两条 packed 路径在所有四组、12 轮中均与 mini 原分发输出相同，
并通过独立的 Hugging Face eager 变长 Prefill → Paged Decode 门禁。
`8×256` 中 mini 与 vLLM 的首个分歧出现在第 6 条请求的第 2 个生成 token，
12 轮一致。HF FP16 eager 的 top-2 是 mini 选出的 token 13 和 vLLM 选出的 token 83，
对应 logits 10.4140625 与 10.3984375。归档保存首个分歧、验证状态及 token 哈希。

## 分块 Prefill 对照

在同一 RTX 3060 Laptop 上复用上节的错峰请求和 Qwen2.5-0.5B FP16，
交错测量未分块的 `auto_varlen_cuda`、384-token 和 512-token 分块预算以及
vLLM 0.8.5 V1。两引擎每步最多 4096 token，KV 池均为 160 个 16-token blocks。
各路径预热 3 轮，测量 12 轮，下表各列取中位数。

| 后端 | 输出 tok/s | 后到请求 P95 TTFT | 首批 P95 ITL | 首批最大 ITL | 达标 req/s |
|---|---:|---:|---:|---:|---:|
| mini 整段 Prefill | 502.8 | 111.9 ms | 15.5 ms | 107.7 ms | 5.24 |
| mini 分块 384 | 474.2 | 162.8 ms | 35.5 ms | 39.7 ms | 11.52 |
| mini 分块 512 | 486.4 | 150.6 ms | 38.2 ms | 46.2 ms | 11.82 |
| vLLM 0.8.5 V1 | 570.7 | 107.5 ms | 13.2 ms | 103.3 ms | 5.95 |

512-token 分块把一次较长的生成停顿拆成多次较短的停顿，12/12 配对轮次均降低了
首批最大 ITL。该预算以较高的后到请求 TTFT 和较低的原始输出吞吐换取更高的达标吞吐。
达标请求须同时满足 TTFT ≤ 200 ms、每个 token 间隔 ≤ 50 ms。
两条分块路径在固定 burst 和错峰 trace 的全部 12 轮中都与未分块路径生成相同的 token。
[复现命令](../../README.md#分块-prefill) · [逐轮样本和输出哈希](vllm-chunked-prefill.json)

## 数字如何计算

Static/Continuous 的吞吐是各 measured trial 吞吐的 median。
首页 TTFT、TPOT、E2E 分别对应原报告聚合的 request TTFT、
inter-token intervals、request E2E 样本的 median，
不是将各 trial median 再取 median。

```text
吞吐变化 = (continuous / static - 1) × 100%
延迟变化 = (continuous / static - 1) × 100%

Prefill latency 降幅 = (1 - segmented / masked_vectorized) × 100%
Prefill memory 降幅  = (1 - segmented_bytes / masked_vectorized_bytes) × 100%
MiB = bytes / 2^20
```

不能把 pooled median 代替尾延迟。
原始 report 另外按 trial 计算 p90/p95 再取 median，
见 [Tail 指标定义](../tail_latency_metrics.md)。

## 测量范围

### Static / Continuous

8 请求 burst、生成长度循环 `2,4,8,12`、max running `4`、
token budget `64`、block size `16`；warmup `3`、measured `10`；
case 逐轮交错并反转顺序。

TTFT/TPOT/E2E 来自 CPU `perf_counter_ns` 请求事件，
throughput 为 output tokens / service window。
CUDA Event 记录当前 stream step timeline，可能含 host 提交间隙，
不是单个 kernel duration 之和。

动态接纳带来 throughput/TTFT/E2E 改善，同时 TPOT 变慢；
原始 sampler 的 Engine 尚未包含后续 packed Prefill 优化。
这组数字不自动代表当前 HEAD 的性能。

### 4×512 Prefill

两个被比较 case 都使用 vectorized KV write，只改变 Attention backend：
`packed_vectorized` 是 dense block-diagonal masked attention，
`segmented` 是逐请求 SDPA；warmup `2`、measured `6`。

计时只含 Prefill ModelRunner + Paged KV write。
不含输入构造、tokenizer 和 Scheduler，因此不是 TTFT。
Peak allocated memory 包含模型和存活的 KV storage。

摘录还保留 `serial` 与 scalar `packed` case 的数据以便审查，
首页的 63.3% 降幅只比较 `packed_vectorized` 与 `segmented`。

## 复现

从仓库根目录执行；先按 [README](../../README.md#快速开始) 安装环境。
精确复核历史版本时使用上述 source commit 的独立 worktree。

```bash
python scripts/benchmark_batching_policy.py \
  --request-count 8 --generation-lengths 2,4,8,12 \
  --max-running-requests 4 --max-batch-tokens 64 \
  --block-size 16 --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/batching_local

python scripts/benchmark_packed_prefill.py \
  --fixed-prompt-length 512 \
  --compare-segmented-sdpa --compare-vectorized-kv-write \
  --warmup 2 --repeats 6 \
  --output-dir benchmarks/results/prefill_local
```

Laptop GPU 的频率、温度与功耗会改变绝对数字。复核应比较多轮分布、环境和
correctness，不能要求任意机器复现完全相同的毫秒数。
