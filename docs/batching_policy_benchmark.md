# Static vs Continuous Burst Benchmark 设计

本 benchmark 让所有请求在第一个 `schedule_step()` 前到达。Burst workload 不是生产流量
模型，但能在不引入线程和到达时间模拟误差的情况下，稳定比较 no-refill Static 与
Continuous admission。

## 控制变量

两种策略共享模型权重、请求 prompt、generation length、KV block budget、ModelRunner、
Paged Attention、greedy decoding 和指标代码。Paged KV pool 容量覆盖全部请求完整生命周期，
避免某种策略恰好因 block 不足改变 admission；唯一主要变量是 Scheduler policy。

每个 trial 重建 Scheduler、Paged KV Manager 和 request metrics，但不重载模型。策略逐轮
交错，奇数轮反转执行顺序，以减轻 RTX Laptop 温度、功耗和频率漂移造成的顺序偏差。

## 两套计时

- CPU `perf_counter_ns`：请求 arrival 到 token ready，用于 throughput、TTFT、TPOT、E2E；
- CUDA Event：每个 Engine step 的 default-stream device timeline，最后保存逐 step 原始值。

吞吐量定义为 `总输出 token / (最后 token ready - 第一请求 arrival)`。CUDA Event 不覆盖
CPU waiting queue，不能拿 CUDA 时间计算“用户吞吐”。Engine 每 step 本来就需要把 token
同步回 CPU，因此 Event 不额外改变其核心同步语义。

这里的 Event elapsed 是两个 event 在 device timeline 上的时间差；区间内若 CPU 编排导致
default stream 暂时空闲，该 gap 也会被包含。它不是 kernel duration 之和，若要分解 kernel、
memcpy 和 launch gap，必须另用 Nsight Systems/Compute。

## 输出与正确性门禁

warmup 不进入统计；measured rounds 保存完整 raw samples，summary 才计算 median。每个
policy/trial 的生成 token IDs 必须完全一致，否则 benchmark 立即失败。输出包括：

- `result.json`：环境、配置、请求、round/trial/request/step 原始数据和 summary；
- `trials.csv`：每个 policy trial 一行；
- `report.md`：便于阅读的中位数表格。

正式运行示例：

```bash
TORCH_CUDA_ARCH_LIST=8.6 python scripts/benchmark_batching_policy.py \
  --local-files-only \
  --request-count 8 \
  --generation-lengths 2,4,8,12 \
  --max-running-requests 4 \
  --max-batch-tokens 64 \
  --block-size 16 \
  --warmup 3 \
  --repeats 10 \
  --output-dir benchmarks/results/batching_policy_v06
```

正式报告前应保证目标 commit 已提交，并检查 JSON 中的 `git_dirty`、GPU telemetry 与所有
raw samples。一次 smoke run 只能验证 harness，不能写成项目性能结论。
