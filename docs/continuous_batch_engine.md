# 最小 Continuous Batching Engine

本阶段把已有的 Scheduler、Paged KV Cache 和 Qwen ModelRunner 接成同步执行闭环。它已经
能在每个 step 动态加入和移除请求，但还不是异步生产级 serving engine。

## 控制平面与执行平面

- Scheduler 是控制平面：保存 waiting/running/finished 状态，按 token budget 选请求。
- ModelRunner 是执行平面：接收 tensor 和 Cache，只负责模型计算。
- Engine 是编排层：把 request ID 转成 tensor，保持 batch row 顺序，写回 token，并根据
  完成事件释放物理 blocks。

Scheduler 不能直接操作 GPU tensor，否则调度策略会与模型实现耦合；ModelRunner 也不能
自行决定请求完成，因为 EOS、`max_new_tokens` 和公平性属于请求语义。

## 一个 step 的数据流

```text
schedule_step()
      │
      ├─ 新请求：逐请求 Prefill ─┐
      │                           ├─ greedy argmax（GPU）
      └─ 旧请求：batched Decode ─┘
                                  │ 按 Scheduler 原顺序重排
                                  ▼
                         一次 GPU→CPU token 同步
                                  │
                                  ▼
                       apply_step_results()
                                  │
                                  ▼
                        释放 finished blocks
```

执行顺序与 Scheduler `items` 顺序可以不同，但结果归属不能不同。当前先逐请求执行 Prefill，
再把全部 Decode 请求组成一个 Paged batch；最后必须按原始 `items` 顺序拼接 token。

## 为什么 Prefill 必须先于 Decode

一个 mixed step 可能同时包含旧请求 Decode 和新请求 Prefill。当前 ModelRunner 尚不支持把
两种形状放进同一次 forward，因此 Engine 要发起多个模型调用。为获得可重试语义：

1. 新请求 Prefill 先执行；成功结果暂不写回 Scheduler。
2. 旧请求 Decode 最后执行，它自身具有跨请求、跨层事务。
3. 任一操作失败时，Decode 回滚；本轮 Prefill reservation 全部释放。
4. Scheduler 把本轮 Prefill 请求退回 waiting 队首，逻辑 step index 不前进。

如果反过来先提交 Decode，后续 Prefill 失败时，旧请求 Cache 已经推进，就无法安全重试。
已经写入但随后释放的 KV 字节只是不可见垃圾，不影响正确性。

## 当前同步点

生成 token 必须回到 CPU，Scheduler 才能检查 EOS 并更新队列。逐请求调用 `.item()` 会产生
多次 synchronization；当前实现先在 GPU 上收集所有 argmax，再按 Scheduler 顺序拼接，
最后统一 `.cpu().tolist()`，每个 step 只有一次结果同步。未来异步引擎还会进一步用 CUDA
stream、event 和双缓冲隐藏这段开销，本阶段不提前实现。

请求现在统一通过 `engine.submit()` 记录 arrival，并保存 first-token、逐 token 和 completion
事件；指标定义与同步边界见 [请求级时间线](request_metrics.md)。

同一个 Engine 也可以使用 no-refill Static policy 作为公平基线；两种 admission 轨迹及其
实验边界见 [Static 与 Continuous Batching 策略](batching_policy.md)。

## 当前边界

- 只支持 greedy decoding，不支持 temperature、top-k/top-p。
- Prefill 仍逐请求执行，不支持 chunked/continuous prefill。
- 一个 `step()` 是同步调用，没有 CPU/GPU overlap。
- 尚未记录 TTFT、TPOT 或吞吐量；不能从功能测试推断性能提升。

验证命令：

```bash
python -m pytest -q tests/test_engine.py
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.continuous_batch_engine_runner \
  --local-files-only
```
