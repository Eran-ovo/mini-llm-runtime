# 首页性能数据与可复核样本

这里提交轻量的正式 benchmark 摘录，让 GitHub 读者无需下载模型即可核对首页数字。
它们来自已有原始 JSON，不是这次文档整理期间新运行的 benchmark。

## 来源

| 文件 | 原始产物 | Source commit |
|---|---|---|
| [static-continuous.json](static-continuous.json) | `release_v1_d31ded2/static_vs_continuous/result.json` | `d31ded234702d9f51d419d8c2d200e51c5994950` |
| [prefill-4x512.json](prefill-4x512.json) | `v2_4_attention_len512_clean_7f4c844/result.json` | `7f4c8444c34b78732775f8df86582e5c7f4075ce` |

摘录保留测试参数、correctness、原始 latency/memory 样本、统计结果、
GPU/CUDA/PyTorch/编译器环境和 `git_dirty=false`。
`provenance` 保存原始完整文件的相对路径、SHA-256 和保留字段，
明确省略逐 step 轨迹或 prompt token buffer；没有更改测量值。

完整 v1.0.0 时间线、CSV 与 manifest 位于
[Release evidence asset](https://github.com/Eran-ovo/mini-llm-runtime/releases/download/v1.0.0/mini-llm-runtime-v1.0.0-release-evidence.tar.gz)。
Prefill 完整原始 JSON 当前保存在作者本地结果目录；本页公开其计算结论所需样本，
复现命令见下文。

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
