# Static Batching 与 Continuous Batching 策略基线

本阶段只改变 Scheduler 的 admission policy，Engine、ModelRunner、Paged KV Cache、
Paged Attention 和请求指标完全共用。

## 策略定义

Continuous Batching 在每个 step 先保证所有 running 请求各 Decode 一次，再使用剩余的
request slot 和 token budget，按 FIFO 接纳新 Prefill：

```text
step 0: Prefill A, B
step 1: Decode A + Prefill C   # B 已完成，C refill 空位
step 2: Decode A, C
```

Static Batching 在 running cohort 非空时禁止 refill。已完成请求会离开，batch 可以缩小，
但 waiting 请求必须等 cohort 完全排空后才能组成下一批：

```text
step 0: Prefill A, B
step 1: Decode A               # C 等待
step 2: Decode A               # A 完成
step 3: Prefill C
```

这里没有让已完成请求继续做 padding compute，因此它是“no-refill static baseline”，而不是
最朴素的固定矩形 batch。这样对比主要隔离动态 admission 的影响，不把无意义 padding
计算同时混进实验变量。

## 边界细节

Static cohort 初建时可以一次接纳多个请求。`may_admit_prefill` 必须根据 step 开始时的
running 状态固定一次；如果在接纳第一个请求后重新检查 `running`，会错误地退化成 batch
size 恒为 1。

Scheduler 在执行前不知道某个 Decode 是否会生成 EOS。因此即使 running 请求很可能在
当前 step 完成，Static 策略也必须等 `apply_step_results()` 后的下一 step 才能建立新
cohort，不能偷看未来结果。

## 公平性能对比仍需满足

- 两种策略使用完全相同的请求 arrival trace；
- prompt、`max_new_tokens`、模型权重和 greedy token 必须相同；
- 相同 warmup、测试轮数、GPU 状态和 KV Cache budget；
- 保存每轮原始 TTFT、TPOT、E2E 和吞吐样本，再比较 median；
- 单次 walkthrough 只证明调度语义，不能证明哪种策略更快。

验证命令：

```bash
python -m pytest -q tests/test_scheduler.py tests/test_engine.py
python -m experiments.batching_policy_walkthrough
```
