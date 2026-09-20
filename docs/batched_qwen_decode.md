# Qwen 多请求 Batched Decode

这一阶段把 `PagedBatchDecodeAdapter` 接入完整 Qwen Decoder。Prefill 仍逐请求执行，
因此本次实验只改变一个主要变量：Decode 从每请求一次 ModelRunner 调用，变成一次
`[B, 1]` 调用。

## 每层数据流

```text
Scheduler request_ids
        │ 保持顺序
        ▼
token_ids [B,1] + position_ids [B,1]
        │
        ▼
Embedding → RMSNorm → Q/K/V projection → RoPE
        │                         │
        │                         └─ K/V 写入各请求 pending slot
        ▼
Paged Attention(query, block_table, sequence_lengths)
        │
        ▼
O projection → residual → RMSNorm → MLP → residual
        │
        └─ 下一层；全部层成功后统一 commit
```

不同请求的 prompt 长度可以不同，因为 Decode 每行都只有一个 query，不需要把历史
K/V padding 成矩形。`block_table[b]` 指出第 `b` 个请求的逻辑块位于哪些物理块，
`sequence_lengths[b]` 限制 kernel 只读取有效 token。表尾的 `-1` 只是 metadata
padding，kernel 不应访问它。

## 三个容易混淆的长度

- `position_ids[b,0]`：append 前的历史长度，用于当前 token 的 RoPE。
- `pending.end`：append 后的长度，Paged Attention 必须读到当前 token 的 K/V。
- committed `token_count`：事务成功前保持旧值，防止其他组件看到半成品。

因此正确顺序是：先由旧长度生成 position，再开启 append，逐层写 K/V 并计算
Attention，LM Head 成功生成 logits 后统一 commit。任一阶段失败时，所有请求都 abort；已经写入 storage 的字节
可以暂留，但由于逻辑长度和 block table 已回滚，它们不可见，之后会被覆盖。

position 上界检查直接读取 CPU request table。若先在 GPU 上构造 position tensor，再调用
`.max().item()` 检查，会在每个 Decode step 引入一次 CPU/GPU synchronization，应避免。

## 为什么 Prefill 暂不 batching

变长 Prefill 需要 padding mask、packed sequence 或专门的 variable-length kernel，容易同时
引入多个变量。本里程碑只验证 Decode batching：QKV、MLP 和 Paged Attention 都在 batch
维度上执行，而每个请求已有独立 Paged KV Cache。这样出现误差时，能明确定位到 batch
顺序、position、地址映射或事务边界。

## 正确性边界

纯 CPU 测试把 Paged CUDA 调用替换成独立 reference，验证：

1. 非创建顺序 `C, A, B` 不会打乱 token 与 Cache；
2. batched logits、逐层输出和最终 K/V 与逐请求 Decode 一致；
3. 两层模型从六次 Attention 调用降为两次；
4. 后续层注入失败时，请求长度、block table 和 free list 全部回滚。

真实 Qwen 实验还会逐层比较 CUDA Paged Attention 和 PyTorch reference，并把 batched
结果同时与逐请求 ModelRunner、Hugging Face eager Attention 对拍。FP16 端到端比较允许
不同 GEMM/归约路径造成的小量累计误差，但 token argmax 必须一致。这是 correctness
阈值，不是性能结论。

```bash
python -m pytest -q tests/test_batched_qwen_decode.py
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.batched_qwen_decode_runner \
  --local-files-only
```

当前仍未把 Scheduler 自动接到 ModelRunner，也没有测 Continuous Batching 的 TTFT、TPOT
或吞吐量；这些属于后续小步，不能由本次 correctness 实验推断。
