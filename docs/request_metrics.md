# 请求级时间线：TTFT、TPOT 与 E2E

这一阶段只建立指标语义和原始事件记录，不进行 Static/Continuous 性能比较。

## 为什么不能只记录一次 forward latency

推理服务的用户延迟包含排队和多轮执行：

```text
arrival → waiting → Prefill → first token → Decode steps → completion
```

一次 CUDA kernel 或一次 ModelRunner latency 都不能代表这条完整路径。因此使用
`time.perf_counter_ns()` 记录 CPU 可观察的单调时间：

- `arrival_ns`：`engine.submit()` 接到请求的时间；
- `prefill_attempt_started_ns`：每次 Prefill 尝试开始；失败重试不会被隐藏；
- `TokenEvent.ready_ns`：GPU 结果同步到 CPU、Scheduler 可以看到 token 的时间；
- `completed_ns`：完成状态已应用且 KV blocks 已释放的时间，用于分析收尾开销。

这里不使用 wall clock，因为系统时间可能被 NTP 或人工调整；也不把 CUDA Event 用作
端到端时钟，因为 CUDA Event 无法覆盖 waiting queue 和 CPU Scheduler。

## 指标定义

```text
queue wait = first prefill start - arrival
TTFT       = first token ready - arrival
TPOT[i]    = token[i+1] ready - token[i] ready
E2E        = last token ready - arrival
post-token = completion - last token ready
```

TPOT 保留每个原始 interval，再派生 median。只有一个输出 token 时没有相邻 token，TPOT
应为 `None`，而不是 0；写成 0 会让短请求虚假拉低平均延迟。

E2E 在最后 token 对用户可见时结束。Scheduler 写回和 KV block 释放属于 Runtime 的
post-token completion overhead，单独记录，不能因为某种 allocator 清理较慢而污染用户
可见延迟定义。

同一 Engine step 中的请求共享一次 GPU→CPU token 同步，因此它们的 `ready_ns` 可以相同。
这表示服务层真正能观察到这些 token 的时间，不声称是每个请求独立 kernel 的完成时间。

当前 mixed step 先执行新请求 Prefill，再执行已有请求 Decode，因此较长 Prefill 会直接反映
为旧请求某个 inter-token interval 增大。这是 Prefill/Decode interference，也称一种
head-of-line blocking；它是后续 chunked prefill 和调度策略实验要解决的问题，而不应从
单次 correctness run 判断其稳定幅度。

## 同步开销

Engine 先在 GPU 上完成所有 argmax，再按 Scheduler batch 顺序拼接，最后只执行一次
`.cpu().tolist()`。逐请求 `.item()` 会产生多次 CPU/GPU synchronization，不适合作为
Continuous Batching 热路径。

## 正确使用边界

- 请求必须通过 `engine.submit()`，直接调用 `scheduler.submit()` 会缺少 arrival event；
- correctness 演示打印的单次时间不是 benchmark 数据；
- 正式 benchmark 仍需 warmup、多轮 raw samples、median、CUDA Event、环境与 Git commit；
- RTX 3060 Laptop 需要同时记录功耗/频率波动，不能根据一次运行下结论。

测试命令：

```bash
python -m pytest -q tests/test_request_metrics.py tests/test_engine.py
```
