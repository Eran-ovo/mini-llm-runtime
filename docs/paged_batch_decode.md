# 多请求 Paged Decode Batch Adapter

2026-09-20；本步骤只打通多请求 Cache transaction 与 CUDA Attention 输入，不修改完整
Qwen ModelRunner。输入使用人工 Q/K/V，便于把地址错误与模型投影/RoPE 错误分离。

## Batch row 是最重要的契约

Scheduler 给出的请求顺序假设为：

```text
[C, A, B]
```

那么所有 tensor 的 batch row 必须一致：

```text
row 0: token_C, position_C, query_C, key_C, value_C, table_C, length_C
row 1: token_A, position_A, query_A, key_A, value_A, table_A, length_A
row 2: token_B, position_B, query_B, key_B, value_B, table_B, length_B
```

若某处偷偷按 request 创建顺序 `[A,B,C]` 排列，shape、dtype 和 block ID 都可能合法，
kernel 不会报错，却会把不同用户的上下文混合。这类错误必须用非默认顺序测试，而不能只测
`[A,B,C]`。

## Position 与 Sequence Length 相差 1

Decode 当前 token 的绝对位置等于 append 前历史长度：

```text
position = old token_count
```

写入当前 K/V 后，Attention 可读长度为：

```text
sequence_length = pending.end = old token_count + 1
```

本次请求顺序 `[C,A,B]`：

```text
positions        = [6,3,5]
sequence_lengths = [7,4,6]
```

RoPE 必须使用 positions，而 Paged Attention 必须使用 sequence_lengths。两者若误用同一
数组，会出现典型 off-by-one：要么 RoPE 多旋转一位，要么 Attention 看不到当前 token。

## 物理 K/V 与 layer view

Manager 的完整 storage 为：

```text
[layer, physical_block, kv_head, block_offset, head_dim]
```

单层 Paged Attention 收到：

```text
[physical_block, kv_head, block_offset, head_dim]
```

Adapter 通过 `storage.key[layer_index]` 取得 view，不复制或 gather K/V。三个请求共享同一
物理 pool，只有 block table 行不同。

## Padded Block Table

三个请求 append 后的物理布局为：

```text
C: [2,4,5], length=7
A: [0,6],   length=4
B: [1,3],   length=6
```

GPU batch table 必须是矩形，所以补成：

```text
[[2,4,5],
 [0,6,-1],
 [1,3,-1]]
```

`-1` 不是有效 block。kernel 根据每行 length 计算：

```text
required_blocks = ceil(sequence_length / block_size)
```

只访问前 `required_blocks` 项。若 length 错大一位，kernel 可能读到 `-1`；安全入口会拒绝，
unchecked 路径则依赖上层 metadata 已经验证。

完整生命周期预留模式下，一行还可能包含当前 length 暂时用不到、但属于该请求的合法未来
blocks。kernel 同样必须以 length 为访问边界，不能以 table width 为实际上下文长度。

## Batch transaction

一次 `begin_decode()` 要为所有请求开启一个 token 的 pending append：

```text
begin C
begin A
begin B
```

若 B 因 block 不足失败，必须反向：

```text
abort A
abort C
```

恢复所有请求原来的 length、block table 和 free list。测试专门构造了“第一个请求成功申请
最后一个空闲 block，第二个请求随后 OOM”的情况，确认 batch 级原子回滚。

每层 `write_layer()` 接受：

```text
key/value [B,Hkv,1,D]
```

只有该层所有请求都写完，才能生成 Attention inputs。所有模型层写完后才能
`commit_decode()`；否则必须 abort。已写入但 abort 的字节是不可见垃圾，之后会被覆盖。

## Metadata 只在层间复用，不跨 Decode step

同一次 Decode 的所有层共享 block table 和 sequence lengths，因此 Adapter 创建一次 GPU
metadata，并在层间复用同一 tensor。下一 step 的 length 已改变，block table 也可能增长，
所以 commit/abort 会清除 metadata，禁止跨 step 复用。

## 三路正确性验证

固定 FP16、Hq/Hkv=14/2、D=64、block size=3，对每一层比较：

1. 一次 batched CUDA Paged Attention；
2. 按 batch row 拆成三个逐请求 CUDA 调用再拼接；
3. Python Paged Attention reference。

要求 batched 与逐请求 CUDA 逐元素一致，并与 Python reference 满足
`atol=rtol=2e-3`。commit 后还逐请求 gather，验证历史前缀未改变，最后一个 K/V 与输入
batch 的对应 row 完全一致。

运行：

```bash
python -m pytest -q tests/test_paged_batch.py
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.paged_batch_decode_walkthrough
```

## 当前边界

- 只支持所有请求各追加一个 token 的 Decode batch；
- 不处理 Prefill，Scheduler 的 mixed batch 需要由 Engine 拆成 Prefill/Decode 子批；
- 不计算 embedding、QKV projection 或 RoPE；
- 不支持请求在 kernel 执行中被取消；
- correctness 脚本不是 benchmark。

下一步只应把该 Adapter 接入 Qwen ModelRunner 的 batched `decode_one` 路径：使用每个请求
不同的 position 做 RoPE，矩阵投影按 batch 执行，每层只 launch 一次 Paged Attention，
并与逐请求 ModelRunner/HF 结果对拍。Prefill 仍暂时单独执行。
