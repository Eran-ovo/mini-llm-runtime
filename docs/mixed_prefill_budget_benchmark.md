# Mixed Prefill Token Budget 受控 Benchmark

## 实验问题

此前的 Static/Continuous 对比发现：Continuous 降低 TTFT、提高吞吐，但 TPOT 略有上升；
NVTX 时间线进一步确认 mixed step 中 Prefill 位于 Decode 之前。当前实验只回答一个问题：

> 限制 mixed step 新接纳的 Prefill token 数，能否用可接受的 TTFT/吞吐代价改善 TPOT？

本实验不改 CUDA kernel、不改 Paged KV Cache 布局、不做 Chunked Prefill，也不改变请求流。

## 自变量与控制变量

唯一主要自变量是 `max_mixed_prefill_tokens`：

| Case | 含义 |
|---|---|
| `unbounded` | Continuous Batching 原始行为 |
| `8` | 最多在一个 mixed step 加入 8 个完整 prompt token |
| `4` | 更强限制，用来暴露 strict-FIFO 队头阻塞 |

固定项包括：模型权重、8 个 burst 请求、prompt token、generation length、greedy decoding、
`max_running_requests=4`、`max_batch_tokens=64`、KV block pool、Paged Attention、计时代码、
warmup 和 measured repeats。

三个 case 每轮循环换位：

```text
round 0: unbounded -> 8 -> 4
round 1: 8 -> 4 -> unbounded
round 2: 4 -> unbounded -> 8
```

因此 repeats 默认使用 12，使每个 case 在三个执行位置各出现 4 次，减轻 Laptop GPU 温度、
功耗和频率漂移带来的固定顺序偏差。

## 为什么选 8 和 4

当前 prompt 长度依次是：

```text
3, 7, 4, 9, 3, 7, 4, 9
```

前四个请求组成相同的初始 cohort，预算只影响后四个 refill 请求。

- `unbounded` 可以在空出两个 slot 时把 7 和 4 一起加入；
- `8` 可以加入 7，但不能在同一 step 再加入 4；遇到 9 时必须等待；
- `4` 遇到队首 7 时不能接纳它，也不能越过它接纳后面的 4。

所以 4 并不是候选“最佳值”，而是用于验证 whole-prefill + strict-FIFO 的阈值效应。

## 正确性门禁

不同 admission 时刻不应改变单请求的 greedy 生成结果。每个 warmup 和 measured trial 都会
保存每个请求的 token IDs，并与第一个 case 的 canonical token sequence 比较。任一 case
不同，benchmark 立即失败，不输出伪性能结论。

逐 step 还保存：

- Prefill/Decode request IDs；
- token count；
- 完成的 request IDs；
- host step 时间；
- CUDA Event 时间。

它们用于确认预算实际改变了预期 step，避免出现“参数传入但没有生效”的无效实验。

## 指标口径

- throughput：总输出 token 数除以“第一请求 arrival 到最后 token ready”的 service window；
- TTFT：单请求 arrival 到第一个 token ready；
- TPOT：同一请求相邻输出 token 的 ready 时间差；
- E2E：arrival 到请求完成；
- CUDA timeline：每个 Engine step 前后 default-stream CUDA Event 的 elapsed time；
- peak memory：PyTorch allocator 的 peak allocated/reserved bytes。

CUDA Event 区间可能包含 CPU 编排导致的 stream idle，不等于 kernel duration 之和。若需要把
kernel、memcpy 和 launch gap 分开，仍需 Nsight Systems；本实验只做调度策略比较。

## 必须警惕的解释错误

1. 一次 smoke run 只验证 harness，不能写入 README 或简历。
2. TPOT 改善不能单独视为胜利；若 TTFT/吞吐显著退化，策略可能得不偿失。
3. budget 小于队首 prompt 时会出现离散的 head-of-line blocking，不能假设性能随 budget
   单调变化。
4. 当前是 burst workload，结论不能直接外推到 Poisson/在线到达流量。
5. 环境快照不是持续 GPU telemetry；循环换位只能减轻、不能消除频率与功耗波动。
6. `git_dirty=true` 时必须说明原因，不能把结果描述为 clean-tree reproducible run。

## 正式运行

```bash
cd /home/eran/mini-llm-runtime
source /home/eran/venvs/torch/bin/activate

TORCH_CUDA_ARCH_LIST=8.6 python scripts/benchmark_mixed_prefill_budget.py \
  --local-files-only \
  --request-count 8 \
  --generation-lengths 2,4,8,12 \
  --max-running-requests 4 \
  --max-batch-tokens 64 \
  --mixed-prefill-budgets none,8,4 \
  --block-size 16 \
  --warmup 3 \
  --repeats 12 \
  --output-dir benchmarks/results/mixed_prefill_budget_v06
```

输出包含 `result.json`、`trials.csv` 和 `report.md`。正式结论还应检查逐轮 raw samples、
执行位置偏差、step trace 和至少一次独立重复实验。
