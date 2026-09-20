# Tail Latency 与 Request Fairness：为什么 Median 不够

## 问题来源

LLM Runtime 同时服务多个请求时，调度策略可能让一部分请求更快、另一部分请求更慢。若只把
所有请求的 TTFT/E2E 混在一起取 median，前半请求的改善可能掩盖后半 waiting 请求的退化。

这不是统计公式出错，而是 median 回答的问题有限：

> 随机取一个中间位置的请求，它大约有多慢？

它不回答：

- 最慢的 10% 或 5% 请求有多慢；
- 最坏请求有多慢；
- 哪个 arrival position 获益，哪个 position 受损；
- tail 变化是否在多轮实验中稳定。

## TTFT、TPOT 和 E2E 的 tail 含义

- TTFT tail 高：请求在 waiting queue、Prefill 或前序 GPU 工作上等待过久；
- TPOT tail 高：生成过程中某些 token 被 mixed Prefill、同步或调度抖动阻塞；
- E2E tail 高：上述等待在完整生命周期上的综合结果。

三者不能相互替代。一个策略可能降低 running 请求的 TPOT，却推迟新请求 admission，使 TTFT
tail 和整体吞吐恶化。

## 百分位数定义

实现显式使用 linear interpolation，避免依赖 NumPy 或其他库可能变化的默认 method：

```text
排序后的 N 个样本：x[0] <= ... <= x[N-1]
rank = (N - 1) * q
结果 = rank 两侧样本的线性插值
```

其中 p90 的 `q=0.90`，p95 的 `q=0.95`。

当前每个 trial 只有 8 个请求，所以 p95 位于第 7、8 个有序样本之间，并且非常靠近最大值。
此时 p95 不是精确的生产 SLA 估计，只是一个稳定、明确的 tail 描述量。报告必须同时展示 max
和 arrival-position 明细，不能给 p95 过度解释。

## 为什么采用“两层聚合”

12 个 trial、每个 8 个请求，共有 96 个 TTFT 样本。直接把 96 个值混起来算 p95 会把它们
视为同一层级的数据，并隐藏 trial 间 GPU 温度、频率和功耗变化。

当前正式口径是：

```text
trial 0 的 8 个请求 -> p95_0
trial 1 的 8 个请求 -> p95_1
...
trial 11 的 8 个请求 -> p95_11

报告值 = median(p95_0, ..., p95_11)
```

这样保留 trial 作为独立测量单位，同时用 median 降低偶发抖动的影响。JSON 仍保存：

- 每个 trial 的 p90/p95/max；
- 所有 request raw samples；
- trial percentile 的 median；
- 整次运行的真实最大值。

## Arrival-position fairness

对于 burst workload，`request_000` 到 `request_007` 同时提交，但 strict FIFO 仍确定它们进入
cohort 的先后位置。按 request ID/arrival position 汇总多轮 median，可以直接看到调度收益
如何分配：

```text
position 0-3：初始 cohort
position 4-7：waiting / refill 请求
```

如果前四个请求 E2E 大幅改善、后四个请求 TTFT 大幅恶化，单一 pooled median 无法表达这种
不公平。位置表也比自创一个“fairness score”更透明：读者可以看到原始业务含义。

## Max 的作用与风险

max 能显示最坏观测值，但只要一次 OS 调度、温度降频或 Python pause 就可能把它推高。因此：

- max 不能单独用于比较策略；
- 必须配合每个 trial 的 max samples；
- 应检查异常值发生在哪个 request、step 和执行位置；
- 正式结论优先看多轮 percentile 的 median，再用 max 做风险边界。

## 当前报告字段

`result.json` schema version 2 新增：

- trial TTFT p90/p95/max samples；
- trial TPOT p95 samples；
- trial E2E p90/p95/max samples；
- 上述 trial samples 的 median；
- 全部 measured samples 的 TTFT/E2E max；
- 每个 request arrival position 的 TTFT、TPOT、E2E raw samples 与 median。

`trials.csv` 每行保存该 trial 自己的 tail 指标；`report.md` 展示两层聚合结果和按 arrival
position 的 TTFT/E2E 表。

## 仍然不能声称什么

- 8-request burst 的 p95 不能直接代表生产流量 SLA；
- 当前结果不能外推到长 prompt、高并发或真实到达过程；
- position fairness 不是租户级公平性或优先级调度；
- CPU request timeline 与 CUDA Event 含义不同，不能混用；
- percentile 改善必须与 throughput、显存和 correctness 一起判断。
