# 多 token Paged CUDA Greedy Decode

2026-09-19；本里程碑完成单请求从 Prefill 到多次 Decode 的生命周期闭环。它仍然不是
Continuous Batching：任一时刻只有一个请求，且每一步只处理一个 query token。

## 自回归循环中的两个“当前 token”

生成循环中最容易混淆的是：`step.logits` 预测出的 token 尚未进入模型。

```text
Prefill(prompt) → logits_0 → 选择 token_0
Decode(token_0) → logits_1 → 选择 token_1
Decode(token_1) → logits_2 → 选择 token_2
```

若要求生成 N 个 token，最后一个 `token_(N-1)` 已经是输出，但不需要再调用 Decode，
因为我们不需要预测第 N+1 个 token。因此在没有提前 EOS 时：

```text
decode_steps = max_new_tokens - 1
cache_length = prompt_length + decode_steps
```

本次真实实验生成 8 个 token，只执行 7 次 Decode；prompt 长度为 3，所以最终 Cache
长度是 10，而不是 11。这不是少写了一个 token，而是避免了无用计算。

## EOS 为什么要在 Decode 前检查

若 Prefill logits 已经选出 EOS，EOS 应出现在返回 token 序列中，但不应再作为
`decode_one()` 输入。否则会发生：

- 多执行一次模型前向；
- Cache 长度错误增加 1；
- 可能无意义地申请一个新物理 block；
- TTFT/TPOT 与显存统计被污染。

单元测试强制把 Prefill 的第一个预测 token 当作 EOS，验证 `decode_steps=0`、Cache
只包含 prompt，且没有调用 Paged Attention。

## Block 增长条件

逻辑位置 `t` 的地址为：

```text
logical_block = t // block_size
block_offset  = t % block_size
physical      = block_table[logical_block]
```

追加后的长度为 L 时，需要的 block 数是：

```text
ceil(L / block_size)
```

只有这个值比当前 `block_count` 大时才向 free list 申请新块。固定 block size=3、prompt
长度=3，本次记录为：

```text
length 4  → [0,2]       申请 block 2，写 offset 0
length 5  → [0,2]       复用 block 2，写 offset 1
length 6  → [0,2]       复用 block 2，写 offset 2
length 7  → [0,2,3]     申请 block 3
length 8  → [0,2,3]     复用 block 3
length 9  → [0,2,3]     复用 block 3
length 10 → [0,2,3,4]   申请 block 4
```

物理 block 1 始终属于 blocker 请求，说明 CUDA kernel 确实依赖 block table 间接寻址，
而不是偶然假设物理块连续。

## 每一步 metadata 的生命周期

同一个 Decode step 的 24 层共享 block table 和 sequence length，所以只创建一次 GPU
metadata，并由第 0 层 checked、后续层 unchecked。进入下一 Decode step 后，length
至少变化，block table 也可能增长，因此必须创建新一代 metadata，不能复用上一步的
sequence length。

可以把生命周期理解为：

```text
step k: begin_append → metadata_k → 24 layers → commit
step k+1: begin_append → metadata_(k+1) → 24 layers → commit
```

不能在 commit 前让 Scheduler 把该请求加入下一批，也不能让下一步读到上一步尚未完成的
pending length。当前单线程 Python 编排天然串行；未来 Scheduler 必须显式维持这一状态机。

## Paged Cache 的容量预检

`greedy_generate()` 不再依赖连续 Cache 特有的 `.capacity`，而是使用统一接口：

```text
cache.length + cache.available_token_capacity
```

对 Paged Cache，`available_token_capacity` 包括：

- 当前请求最后一个已分配 block 的剩余 slot；
- 当前时刻 free list 中所有空闲 block 的 slot。

该值是生成开始时的容量快照，不是永久预留。当前只有一个活动生成请求，因此预检后不会
被其他请求抢走 block。进入 Continuous Batching 后，仅做预检不够：Scheduler 必须在每步
分配时处理竞争、准入失败和抢占策略。

容量不足测试会在 Prefill 前失败，并验证 request table、free list 和 Cache length 完全
未改变，避免生成到中途才暴露 OOM。

## 真实 Qwen 结果

配置：Qwen2.5-0.5B、FP16、prompt `你好，GPU`、block size=3、生成 8 tokens。

```text
HF/Candidate tokens:
[9370, 102111, 33108, 111390, 9370, 92032, 104139, 100145]

文本：的性能和内存的大小有什么关系
decode steps: 7
final cache length/capacity snapshot: 10/12
final block table: [0,2,3,4]
```

7 次 Decode、每次 24 层，共 168 次 CUDA Attention。所有调用均与同输入 Paged Python
reference 在 `atol=rtol=2e-3` 下通过，最大的单调用 max absolute error 为
0.00097656。

逐 step logits 均与 HF 产生相同 argmax token，并通过 provisional FP16 累积预算
`atol=3e-2, rtol=3e-3`。8 个 step 的 max absolute error 范围为 0 到 0.03027344。
该预算仍只是 smoke-test 标准；需要更多 prompt、长度和生成步数后才能确定 release 阈值。

## 复现

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.paged_greedy_generation_runner \
  --max-new-tokens 8 \
  --block-size 3 \
  --local-files-only
```

脚本包含 HF reference、逐层 Python Attention 和大量同步，只能用于 correctness，不能用来
报告 TTFT、TPOT 或吞吐量。

## 下一步边界

单请求 ModelRunner、Paged KV Cache 与 CUDA Attention 已经贯通。下一步进入 v0.6 的
第一小步：只建立请求状态机与 waiting/running/finished 队列，定义 token budget 和准入
规则，并用纯 CPU 确定性测试验证调度顺序。暂不同时实现 batched ModelRunner 或性能测试，
避免把调度语义和 GPU batch 执行错误混在一起。
