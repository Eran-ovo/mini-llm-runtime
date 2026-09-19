# ModelRunner 接入 CUDA Paged Attention

2026-09-19；本里程碑只接通单请求、单 token Decode。Prefill 继续使用 PyTorch
correctness 路径，split-KV 仍留在实验入口，不参与 runtime dispatch。

## 为什么这一步是推理引擎主线

此前 `PagedRequestKVCache` 虽然把 K/V 写入了非连续物理 block，但 ModelRunner 随后会
调用 `view_layer()`，按逻辑 token 顺序重新 gather 成连续 K/V，再用 PyTorch 计算
Attention。它验证了地址管理，却绕开了 Paged Attention 的核心价值。

现在的单层 Decode 数据流为：

```text
hidden [B,1,H]
  ├─ Q projection → RoPE → query [B,Hq,1,D]
  ├─ K projection → RoPE → key   [B,Hkv,1,D]
  └─ V projection        → value [B,Hkv,1,D]
                              │
                              ▼
                 write current K/V to physical block
                              │
                              ▼
query [B,Hq,D] + block_table [B,max_blocks] + sequence_lengths [B]
                              │
                              ▼
          CUDA Paged Attention directly reads layer storage
 [physical_block,Hkv,block_offset,D] → output [B,Hq,D]
                              │
                              ▼
                    reshape + O projection
```

## 为什么必须先写当前 K/V

假设 prompt 长度为 3，当前正在处理第 4 个 token。当前 query 的合法 key 范围是
`[0,1,2,3]`，其中逻辑位置 3 就是当前 token 自己。若先执行 Attention 再写 Cache，
sequence length 只能是 3，结果会错误地漏掉对自己的注意力。

本实现先调用 `begin_append(1)` 预留地址，然后每层依次：

1. 生成并写入该层当前 K/V；
2. 用 `pending.end` 作为 CUDA 可读长度；
3. 执行该层 Paged Attention；
4. 所有层成功后统一 `commit_append()`。

对其他请求而言，`table.token_count` 在 commit 前仍是旧长度。某层失败时
`abort_append()` 不推进长度，并归还本次新申请的 block；已经写入的字节只是不可见垃圾。

## Layer 维为什么不能交给 kernel

完整 storage 是：

```text
[layer, physical_block, kv_head, block_offset, head_dim]
```

Paged Attention v1 的单次调用只处理一层，要求：

```text
[physical_block, kv_head, block_offset, head_dim]
```

因此 adapter 按 `layer_index` 返回 `storage.key[layer_index]` 和
`storage.value[layer_index]`。切掉最外层后仍是 contiguous view，不发生 K/V copy。
这和 gather 的根本区别是：逻辑顺序只存在 block table 中，K/V 本身仍留在物理池。

## Metadata 生命周期

一次单 token append 中，24 层使用相同的 block table 和 sequence length。adapter 在
第一次 Attention 时创建一次 GPU int32 metadata，并缓存到事务结束：

```text
block_table      = [[0, 2]]
sequence_lengths = [4]
```

第 0 层调用 checked CUDA 入口，把 metadata 同步到 CPU 验证一次有效 block ID；后续
23 层使用 unchecked hot path，只跳过这次 D2H metadata 验证。C++ 中的 tensor shape、
dtype、device、contiguous 和 GQA 整除检查仍然每层执行。

这是安全性和热路径成本的折中。未来 Continuous Batching 会由 Scheduler 每个 step
统一构造 batch metadata，届时不应继续由单请求 adapter 分别创建。

## 两种 Decode backend

`QwenPrefillRunner` 现在显式接受：

- `torch`：gather 连续 K/V 后执行矩阵化 Attention，作为 CPU 测试和 correctness reference；
- `paged_cuda`：要求 `PagedRequestKVCache`，直接调用 FP16/D=64 CUDA v1。

不进行隐式 fallback。用户选择 `paged_cuda` 却传入连续 Cache 时，会在
`begin_append()` 前报错，避免 Cache 进入半完成事务。

## 正确性为什么分三层验收

真实 Qwen 第一次运行出现过一个有学习价值的“严格对拍失败”：CUDA 路径最终 logits
相对 HF 的 max absolute error 为 0.0234375，超过原先单算子使用的 0.002 阈值；但
argmax token 相同。不能直接放宽阈值，也不能据此认定地址错误，所以进一步按层定位。

### 1. 单层、同输入算子对拍

实验在每层实际 CUDA 调用外包装独立 Paged Python reference。两者使用完全相同的
query、物理 K/V、block table 和 length：

- 24/24 层通过 `atol=rtol=2e-3`；
- 所有层中最大的 max absolute error 为 0.00024414；
- 第 0 层走 checked，后续 23 层走 unchecked。

这一级最适合发现 scale、GQA head 映射、block address 和 length 错误。

### 2. Cache 生命周期

测试强制第一次 Decode 跨 block：`[0] → [0,2]`，中间的物理 block 1 属于另一个请求。
结果显示历史 K/V 前缀逐元素不变、所有 Cache 有限、第 0 层新增 K/V 与 HF 严格一致。

从第 1 层起，CUDA 和 HF 的上一层 hidden state 已有微小舍入差异，新的 Q/K/V 输入也
随之不同，因此不能再把后续 Cache 的差异误判成地址错误。

### 3. 24 层端到端结果

固定 prompt `你好，GPU` 的结果：

- Prefill last logits 与 HF 严格对拍通过；
- Decode logits max/mean absolute error 为 0.02343750 / 0.00415673；
- provisional FP16 累积预算为 `atol=3e-2, rtol=3e-3`，本例通过；
- CUDA 与 HF 的下一 token 均为 `性能`。

该累计预算只用于真实模型 smoke test，不替代严格的 kernel 对拍，也不能据单个 prompt
宣称普遍精度。后续需要多 prompt、多长度和多 decode step 的回归集再决定正式阈值。

## 复现

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.paged_qwen_decode_runner \
  --local-files-only
```

该脚本包含 HF 模型加载、逐层 Python reference 和 CUDA 同步，不是 benchmark。
其中任何时间都不能用于 TTFT、TPOT 或吞吐量结论。

## 当前边界

- 只接通单请求、单 token Decode；
- Paged Attention v1 只支持 CUDA FP16、head_dim=64；
- Prefill 仍使用 PyTorch Attention，但会把 RoPE 后的 K/V 写入 Paged Cache；
- 没有 Continuous Batching，也没有变长 batch；
- 没有采用 split-KV 自动 dispatch；
- 还没有正式测量端到端 TPOT。

下一步只应验证多 token greedy Decode：让同一请求连续执行若干步，覆盖块内追加和再次
跨块增长，并与 HF 的生成 token 序列对拍。完成这个生命周期闭环后，再进入 Scheduler
和 Continuous Batching。
