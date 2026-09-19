# 架构与里程碑验收标准

## 数据结构与数据流

稳定入口最终只保留四个核心对象：

- `Request`：prompt token、生成上限、状态、已生成 token 和逻辑 block table。
- `Scheduler`：维护 waiting/running queue，按 token budget 与空闲 block 选择本轮 batch。
- `KVCacheManager`：拥有 GPU block pool 与 free list，负责 allocate/append/free。
- `ModelRunner`：接受本轮 token、position 和 block table，执行各层并返回 logits。

Prefill batch 可以包含多个 prompt token；执行后，K/V 按层写入属于请求的 block。
Decode batch 中每个运行请求通常贡献一个 query token；Paged Attention 根据
`block_table[request, logical_block]` 找到物理 K/V。一个请求完成后，Scheduler
通知 KVCacheManager 归还全部 block，并允许 waiting request 在下一 step 加入。

Qwen2.5-0.5B 使用 GQA：query head 数量可大于 KV head 数量。query head `h` 通过
`kv_h = h // (num_query_heads / num_kv_heads)` 共享对应的 K/V head；后续 CUDA kernel
必须显式验证这一地址映射。

## 为什么按此顺序实现

Reference 固定数值语义；ModelRunner 固定权重映射与 layer 数据流；连续 KV Cache
先验证“不重算历史”；paged allocator 再改变存储布局；Paged Attention 最后消费
该布局。若越过前一层直接优化 kernel，错误会同时混入模型语义、cache 生命周期和
CUDA 地址计算，无法有效定位。

## 里程碑验收

### v0.1 正确性基线

- 环境信息可导出；不使用 `generate()`，Prefill 与单 token Decode 有独立 API。
- 固定 prompt 的 token、最后位置 logits 和 greedy 输出可保存并复现。
- benchmark 有 warmup、CUDA Event、raw samples、median、peak memory 和版本元数据。

### v0.2 最小 ModelRunner

- 显式映射 Qwen 权重并组织每层 RMSNorm、RoPE、QKV、attention、MLP。
- 固定输入下逐层关键 tensor 与 HF 对拍；最终 logits 满足约定误差。
- GEMM 等仍可调用 PyTorch/cuBLAS，但上层不得调用完整 HF model forward。

### v0.3 连续 KV Cache

- 每层拥有连续 K/V buffer；Prefill 批量写入，Decode 仅追加一个位置。
- token-by-token logits 与无 cache reference 对拍。
- 同一配置测量有/无 cache 的 Decode 延迟、重复计算量和 peak memory。

### v0.4 Paged KV Cache

- 固定 block size、GPU block pool、CPU/GPU block table、free list 生命周期完整。
- 跨 block 边界、增长、释放、复用和 OOM 均有测试，无 double-free/block 泄漏。
- 报告内部碎片、可容纳请求数及其计算方法。

### v0.5 Paged Attention CUDA

- 首版支持 query length 1、GQA/MQA、block table 间接寻址和 online softmax。
- 多种长度、非连续物理块、边界 block 下与 FP32/SDPA reference 对拍。
- 每次优化只改变一个主要变量，保留 correctness、benchmark 和 profiler 证据。

### v0.6 Continuous Batching

- waiting/running queue 可在每个 step 加入和移除不同长度请求。
- token/block budget、请求完成和 cache 回收行为均有确定性测试。
- 与 Static Batching 在同一 workload 下比较 TTFT、TPOT、吞吐和显存。

当前已完成第一小步的同步 CPU 状态机与 token budget baseline，详见
[Scheduler 状态机学习记录](scheduler_state_machine.md)。block-aware admission、batched
ModelRunner 和性能对比仍未实现。

### v1.0 收尾

- README、架构图、测试矩阵、机器可读与 Markdown benchmark 全部可复现。
- Nsight Systems/Compute 分析能解释主要瓶颈，并记录有价值的失败实验。
- 简历数字只引用仓库内正式结果；创建 release 前稳定入口通过全部测试。
