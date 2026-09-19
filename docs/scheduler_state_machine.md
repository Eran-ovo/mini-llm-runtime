# Continuous Batching：请求状态机与基础 Scheduler

2026-09-19；这是 v0.6 的第一小步。当前实现完全运行在 CPU，不调用 ModelRunner，也不
分配 GPU Cache。目标是先固定请求状态、batch 边界和 token budget 语义。

## Scheduler、Engine 和 KV Cache Manager 的职责

三个组件不能混成一个对象：

```text
Scheduler
  决定本 step 做哪些 Prefill/Decode，并维护请求状态
        │ SchedulerBatch
        ▼
Engine / ModelRunner
  真正执行模型，返回每个请求生成的一个 token
        │ generated_tokens
        ▼
Scheduler
  写回 token，判断 EOS/max_new_tokens，发出 finished IDs
        │ finished_request_ids
        ▼
KV Cache Manager
  由 Engine 调用 release_request()，归还物理 blocks
```

Scheduler 不拥有 GPU tensor，也不应该偷偷释放 block。这样调度策略可以用纯 CPU
确定性测试覆盖，而 KV 生命周期仍由唯一的 Cache Manager 管理。

## 请求状态机

当前只有三个稳定状态：

```text
submit
  │
  ▼
WAITING ──admit/prefill scheduled──> RUNNING
                                      │
                                      ├─普通 token──> RUNNING
                                      │
                                      └─EOS 或达到上限──> FINISHED
```

`RUNNING` 表示请求已经占用一个并发 slot。刚从 waiting 接纳、Prefill 尚未返回的请求也
属于 running；outstanding batch 保证它在结果写回前不会再次被调度。

每个请求保存：

- `prompt_token_ids`；
- `generated_token_ids`；
- `max_new_tokens`；
- `eos_token_ids`；
- `arrival_index`；
- `status/prefilled/finish_reason`。

EOS 与长度上限同时满足时，EOS 优先成为 finish reason，因为它是模型显式给出的停止原因。

## 一个 step 的 token budget

当前使用 vLLM 类系统中常见的“本轮输入 token 数”作为简化 budget：

```text
Prefill cost = prompt_length
Decode cost  = 1

batch token count
  = Σ admitted prompt lengths + Σ running decode requests
```

它不是 FLOPs 的精确估计。一个 Decode token 会读取整段历史 KV，context 越长成本越高；
一个 Prefill token 也存在 causal attention 的长度差异。本阶段 token budget 只定义批次
上限和确定性语义，不能被解释成不同 batch 计算量完全相同。

两个硬不变量是：

```text
running_count ≤ max_running_requests
batch.token_count ≤ max_batch_tokens
```

并要求 `max_running_requests ≤ max_batch_tokens`，否则无法保证每个 running 请求在同一
step 都获得一个 Decode slot。

## 当前调度策略

### 1. Decode priority

先为所有 running 请求安排一个 Decode，再考虑新 Prefill。这能降低已有请求的 TPOT
抖动，并避免大 prompt 不断插入导致 Decode 饥饿。

### 2. Whole prefill

Prompt 必须整体放进剩余 budget。若 prompt 本身超过 `max_batch_tokens`，submit 立即失败。
当前没有 chunked prefill，也没有跨 step 保存部分 Prefill 进度。

### 3. Strict FIFO admission

按 arrival order 查看 waiting 队首。队首 prompt 当前放不下时，本 step 停止接纳，不越过
它选择后面的短请求。

这保证顺序简单、可复现，但会产生 head-of-line blocking：一个长 prompt 可能让剩余
budget 闲置，即使后面的短 prompt 能放下。本阶段保留这个缺点作为 baseline；以后比较
调度策略时才能量化“跳过长请求”对吞吐、公平性和 TTFT 的影响。

## Outstanding batch 为什么必不可少

`schedule_step()` 会返回一个 batch。在 `apply_step_results()` 前再次 schedule 会报错：

```text
schedule step k
    ↓
GPU 正在执行
    ├─允许新请求 submit 到 waiting
    └─禁止把 running 请求再次放进 step k+1
    ↓
apply results of step k
    ↓
schedule step k+1
```

如果没有这个约束，同一个 request 可能同时存在两个 Decode，二者使用相同 position 和
Cache length，最终造成重复写入、block table 竞争和错误 token 顺序。

结果写回要求 request ID 集合与 batch 完全一致。missing、extra 或非法 token 会在修改
任何请求之前失败，outstanding batch 保留，调用者可以修正结果后重试。

## 动态加入和移除示例

配置：`max_running_requests=3`、`max_batch_tokens=6`。

```text
step 0:
  Prefill A(4) + Prefill B(2) = 6
  GPU 执行期间 D 到达 waiting

step 1:
  Decode A(1) + Decode B(1) + Prefill C(1) = 3
  B 因 EOS 完成，C 因 max_new_tokens=1 完成

step 2:
  Decode A(1) + Prefill D(1) = 2
  A、D 完成
```

完成顺序为 `B,C,A,D`，不同于提交顺序。这是 Continuous Batching 的正常现象：请求可以
在每个 step 加入和离开，batch shape 也随之变化。

## Cache 释放事件

`apply_step_results()` 返回：

```text
SchedulerStepUpdate.finished_request_ids
```

测试使用 CPU PagedKVCacheManager 验证：Scheduler 标记请求 finished 后，物理 block
仍保持 allocated；模拟 Engine 对这些 ID 调用 `release_request()` 后，free list 才恢复。
这同时验证了“完成事件不会丢失”和“内存所有权没有越界”。

## 正确性覆盖

当前测试包括：

- Decode priority、FIFO admission 和 token budget；
- GPU batch outstanding 期间的新请求到达；
- waiting/running/finished 动态转换；
- EOS 与 max token 两种完成原因；
- strict FIFO 的 head-of-line blocking；
- missing/extra result 的原子失败；
- 非法配置、重复 request ID、非法 token；
- finished event 驱动 KV block 释放。

运行：

```bash
python -m pytest -q tests/test_scheduler.py
python -m experiments.scheduler_walkthrough
```

## 当前没有实现什么

- 没有真实 ModelRunner batch 执行；
- 没有根据 free blocks 做准入；`max_running_requests` 只是并发 slot 上限；
- 没有 chunked prefill、preemption、priority 或 cancellation；
- 没有异步 CUDA stream/event；
- 没有 TTFT、TPOT 或吞吐 benchmark。

后续已完成保守的 block-aware admission：接纳时按请求最大生命周期预留物理 block，
完成事件释放全部 reservation。原理、利用率代价与确定性测试见
[Block-aware Admission](block_aware_admission.md)。真正的 batched Qwen execution 仍未实现。
