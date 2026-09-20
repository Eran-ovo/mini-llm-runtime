# Mixed Prefill Token Budget：限制 Decode 被 Prefill 阻塞的时间

## 这一步解决什么问题

Continuous Batching 允许已有请求 Decode 时，把新请求的 Prefill 加入同一 engine step。
这能降低新请求的排队时间，但当前 Engine 在一个 mixed step 内按顺序执行：

```text
新请求 Prefill ──> 已有请求 Decode ──> token 回传
```

因此，加入的 prompt 越多、越长，已有请求越晚开始本轮 Decode，TPOT 越容易上升。
此前 NVTX 时间线已经观察到这种串行关系；本步骤把它变成一个最小、可检验的调度变量。

`max_mixed_prefill_tokens=P` 的含义是：**若 step 开始时已经有 running 请求，本 step
最多新接纳总计 P 个 prompt token。**

它不是 GPU 显存容量，也不是总 batch token 上限：

```text
total_tokens = decode_request_count + admitted_prefill_tokens
total_tokens <= max_batch_tokens
admitted_prefill_tokens <= max_mixed_prefill_tokens  # 只在 mixed step 生效
```

两个约束必须同时满足。

## 为什么单独设置这个预算

原有 `max_batch_tokens` 同时限制 Decode token 和 Prefill token，也限制初始 cohort。直接调小它
会一次改变多个因素：初始 batch size、refill 数量和 mixed step 工作量。这样即使性能变化，也
难以判断原因。

新预算只改变 mixed step 的 refill 强度：

- 没有旧 running 请求时，仍按 `max_batch_tokens` 建立初始 cohort；
- 已有 running 请求时，Decode 仍然优先且每个请求恰好占 1 token；
- 只限制本轮新加入的完整 prompt token 总数；
- 每个 step 重新计数，不跨 step 累积。

这符合性能实验中的“只改变一个主要变量”。

## 数据流示例

假设 A 正在 Decode，C、D 的 prompt 长度分别为 3、2：

```text
无限制： step 1 = Decode A + Prefill C(3) + Prefill D(2)
预算 3： step 1 = Decode A + Prefill C(3)
         step 2 = Decode A + Prefill D(2)
```

预算把一次较长的阻塞分散成两次较短的阻塞。这里存在真实 trade-off：已有请求的 TPOT
可能改善，但 D 的 TTFT 可能变差；总吞吐也可能因 batch 变小而下降。结论必须由正式
benchmark 给出，不能仅凭调度图判断。

## 为什么暂时不做 Chunked Prefill

本实现仍是 **whole-prefill**。长度为 4 的 prompt 在预算为 3 时不能拆成 `3 + 1`：

```text
错误理解：本 step 处理前三个 token，下个 step 处理最后一个
当前语义：整个 prompt 留在 waiting queue
```

Chunked Prefill 需要额外状态：已处理的 prompt offset、分段 RoPE position、分段 KV 写入、
最后一段何时产出首 token，以及失败回滚。现在提前加入会把“调度预算实验”与“Prefill
分块正确性”两个变量混在一起，偏离当前 Continuous Batching 主线。

## FIFO、饥饿与边界条件

Scheduler 保持 strict FIFO。若队首 prompt 长度 4、预算为 3，它不能越过队首去接纳后面
长度为 1 的请求。这样行为可预测，但会产生 head-of-line blocking。

队首不会永久饥饿：预算只在 step 开始时已有 running 请求时生效；旧 cohort 最终完成后，
下一步是新 cohort 的初始 Prefill，此时只受 `max_batch_tokens` 限制。所有请求的
`max_new_tokens` 都有限，因此在当前同步模型中旧 cohort 最终会排空。

实现时还要注意：必须在 admission 之前保存 `running_at_step_start`。如果接纳第一个 Prefill
后再检查 `self._running`，初始 cohort 会被错误识别成 mixed step。

## 现在能够证明与尚不能证明的事

单元测试与 walkthrough 能证明：

- 初始 cohort 不受 mixed budget 影响；
- mixed step 的 Prefill token 总量不越界；
- budget 每步重置；
- strict FIFO 不被破坏；
- 超预算队首在旧 cohort 排空后仍可被接纳；
- Engine 与 Scheduler 的集成使用了同一语义。

它们不能证明预算改善了 TPOT、TTFT 或吞吐。下一小步应固定同一 arrival trace，在真实 GPU
上只扫描 `max_mixed_prefill_tokens`，保存 warmup、多轮 CUDA Event/时钟样本和环境信息。

运行：

```bash
source /home/eran/venvs/torch/bin/activate
python -m experiments.mixed_prefill_budget_walkthrough
python -m pytest -q tests/test_scheduler.py tests/test_engine.py
```
