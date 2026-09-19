# Continuous Batching：Block-aware Admission

2026-09-20；这是 v0.6 的第二小步。Scheduler 仍使用 fake ModelRunner，但 token budget
之外已经接入 Paged KV block pool，验证请求准入、完整生命周期预留和完成释放。

## 为什么只检查 Prompt blocks 不够

假设 pool 只剩 2 blocks，一个请求的 prompt 需要 1 block，但后续生成还可能需要 2 个。
若只检查 Prefill：

```text
Prefill 成功
→ 请求进入 running
→ 另一个请求拿走剩余 block
→ 当前请求跨 block Decode
→ 中途 OOM
```

此时请求已经向用户输出部分 token。若没有 preemption、swap 或 recompute，就无法干净恢复。

成熟引擎通常按需增长，并在 block 不足时使用更复杂的抢占策略。本阶段还没有这些机制，
因此选择保守但可证明安全的 baseline：接纳时预留请求可能写入 Cache 的最大 token 数。

## 为什么是 prompt + max_new_tokens - 1

第一个生成 token 来自 Prefill logits；生成的最后一个 token 不需要再次作为 Decode 输入。
因此最多写入 Cache 的 token 数为：

```text
max_cache_tokens = prompt_length + max_new_tokens - 1
required_blocks  = ceil(max_cache_tokens / block_size)
```

例如 prompt=3、max_new_tokens=2、block_size=2：

```text
max_cache_tokens = 3 + 2 - 1 = 4
required_blocks  = 2
```

即使请求因 EOS 提前完成，多预留的 block 也会在 release 时一起归还。

## 预留容量不等于提交 token

`RequestBlockTable.reserve_capacity()` 只扩展：

```text
block_ids
token_capacity = block_count * block_size
```

不会改变：

```text
token_count
sequence length
pending range
```

真正的 Prefill/Decode 仍通过 `begin_append → commit_append` 推进可见长度。因为 block 已经
预留，执行阶段的 append 只填写既有地址，不再访问 free list。

另外，`PagedKVStorage` 的大 K/V tensor 在 Manager 初始化时已经整体分配。block allocator
管理的是物理 block ID 的所有权，不是每次准入都调用 `cudaMalloc`。

## 原子准入

`PagedBlockAdmissionController.try_admit()` 的顺序是：

```text
计算 required_blocks
  ├─ required > total：永久无法接纳，明确报错
  ├─ required > free：暂时不足，无副作用返回 False
  └─ 足够：
       create_request
       reserve_request_capacity
       写入 reservation registry
       返回 True
```

若 create 后的预留步骤异常，会立即 `release_request()`，避免 Manager registry 中残留一个
没有完整 reservation 的请求。

Scheduler 只在 token budget 和 running slot 都允许后调用 admission callback。返回 False
时，队首请求继续留在 waiting，并保持 strict FIFO，不跳过它接纳后续请求。

若当前没有 running Decode，且 waiting 队首因外部 block 占用无法接纳，
`schedule_step()` 返回 `None`，但 `has_unfinished_requests` 仍为 True。这表示资源阻塞，
不同于所有请求已经完成。

## 完成与释放

Scheduler 的 `apply_step_results()` 只发出 `finished_request_ids`。Engine 随后调用：

```text
admission.release_finished(finished_request_ids)
```

controller 会先验证所有 ID 都有 reservation，再逐个调用 KV Manager 的
`release_request()`。预校验可以防止列表中后一个 ID 非法时，前面的请求已经被部分释放。

## 确定性示例

配置：5 个 blocks、block size=2、token budget=5、running slots=3。

```text
A: prompt=3, max_new=2 → reserve 2 blocks
B: prompt=2, max_new=3 → reserve 2 blocks
C: prompt=3, max_new=2 → reserve 2 blocks
```

执行过程：

```text
step 0:
  Prefill A + B
  A/B 共预留 4 blocks，只剩 1；C 留在 waiting

step 1:
  Decode A + B
  A 完成，释放 blocks (0,1)，free 变为 3

step 2:
  Decode B + Prefill C
  C 获得刚释放的 2 blocks
  B 完成并释放

step 3:
  Decode C
  C 完成并释放

final:
  manager requests = []
  free blocks = 5/5
```

从 admission 到完成，fake Prefill/Decode 的每次 append 都断言没有新分配 block，说明完整
生命周期容量确实已经预留。

## 这种 baseline 的代价

完整生命周期预留不会中途 OOM，但会降低 Paged KV Cache 的优势：

- 请求可能提前 EOS，预留空间长期未使用；
- `max_new_tokens` 很大时，一个请求会占用大量未来 blocks；
- waiting 数量可能比按需增长策略更多；
- Manager 当前的 unused/fragmentation 统计会包含预留但尚未写入的 slots。

它的作用是建立正确性下界，而不是最终性能策略。未来按需分配必须同时设计：

- Decode step 的原子 block planning；
- block 不足时 stall 哪些请求；
- preemption/recompute/swap 中至少一种恢复策略；
- 防止某些请求长期饥饿的公平性规则。

## 复现

```bash
python -m pytest -q tests/test_block_admission.py
python -m experiments.block_aware_scheduler_walkthrough
```

当前仍未执行真实 GPU batch，也没有性能数据。

下一步只应实现“多请求 Paged Decode batch adapter”：按 Scheduler 给出的 request 顺序，
为多个不同长度请求写入当前层 K/V，并生成共享物理 storage view、GPU block table 和
sequence lengths。先与逐请求执行对拍，再让 Qwen ModelRunner 使用它。
