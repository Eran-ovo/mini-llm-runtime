# Packed Prefill 的分层正确性门禁

## 为什么要分层

`serial` 逐请求 Prefill 与 `packed` 批量 Prefill 的矩阵形状不同。FP16/cuBLAS
可能为不同形状选择不同的计算路径；即使逻辑模型相同，中间 hidden/KV 也未必
逐元素一致。特别是在 `(128,128,128,1)` 中，1-token 请求的跨层 V Cache
相对 L2 达到 `0.0137077`。逐层、同输入的 FP32 对照和首个分叉位置见
`docs/ragged_kv_numerics.md`。

因此 `scripts/benchmark_packed_prefill.py` 的 schema v2 分开问三个问题：

1. **模型数值**：跨执行形状比较最终 logits。要求两侧有限、greedy token 完全
   相同、全体 logits 相对 L2 不超过 `0.01`。`allclose` 只作诊断，不是额外门禁。
2. **KV 写入等价**：保持同一个 packed Attention/模型计算，只切换 scalar 与
   vectorized 写入。masked 和 segmented 两条路径分别对拍；要求最终 logits
   以及每条请求 gather 后的逻辑 K/V 逐元素完全相同。这里检验的是已提交的
   逻辑 KV，不涵盖未分配物理槽；独立 block 映射测试仍然必要。
3. **跨形状 KV 诊断**：保留最大绝对误差、最大相对 L2、最坏请求/张量和旧
   1% 结果，便于发现回归；非有限 KV 仍是硬错误。跨形状相对 L2 本身不再
   作为拒绝 benchmark 的单独理由。

这个判据不是说 1% logits 阈值对任意模型都成立；它只是当前 Qwen2.5-0.5B
FP16 工作负载的显式、可测试约定。它也不能单独证明 Cache 续写后的 Decode
正确性；Prefill→Decode 的 HF 续写对拍已在 commit `a75b400` 补上，见下文。

## 计时生命周期

Correctness 对拍会构造多份 Paged KV Cache。它们必须在交错计时前释放，
否则 peak memory 被额外的对拍 Cache 污染。现在所有对拍对象只在
`verify_correctness()` 的局部作用域内存活；返回后用 weak reference 检查
Manager 都已销毁，再调用 `measure_cuda_interleaved()`。结果 JSON 中的
`correctness_caches_released_before_timing=true` 记录该检查已通过。

早期 commit `982bc03` 的 `v2_7_layered_gate_*_clean_982bc03` JSON 虽然来自
clean tree，但对拍 Cache 仍被局部变量持有：**正确性字段可用于追溯，时间和
显存字段不得作为正式性能证据**。保留这些文件作为失败实验，不删改原始结果。

## 可复现结果

修正后的 source commit 是 `1df7406fafd28a83156c03a8c5beaa83f5d9e1c4`。
在干净 worktree、Qwen2.5-0.5B FP16、RTX 3060 Laptop 上，用 CUDA Event
交错计时；每组 warmup 3，保存每轮原始样本和 median。完整 GPU、CUDA、
PyTorch、Git 状态及 peak memory 见本地（默认不进 Git）JSON：

| Prompt 长度 | 测量轮数 | correctness | packed + vectorized median | segmented + vectorized median | 原始结果 |
| --- | ---: | --- | ---: | ---: | --- |
| `(128,128,128,1)` | 10 | 通过；最大 logits rel L2 `0.006740`，同形状写入完全一致 | `35.726 ms` | `30.259 ms` | `benchmarks/results/v2_7_layered_gate_ragged_clean_1df7406/result.json` |
| `(64,64,64,64)` | 8 | 通过；最大 logits rel L2 `0.002520`，同形状写入完全一致 | `26.442 ms` | `26.481 ms` | `benchmarks/results/v2_7_layered_gate_4x64_clean_1df7406/result.json` |

这里的耗时范围只包含 Prefill ModelRunner 和 KV 写入，不含 tokenizer、请求
调度或端到端 TTFT。单次运行不足以宣称 segmented 在所有机器/功耗状态下
稳定更快；当前步骤的结论是**正确性门禁和显存计时生命周期已修复**，不是
新的 dispatch 策略。测试为 `229 passed, 1 warning`（CUDA 架构编译提示）。

复现示例：

```bash
python scripts/benchmark_packed_prefill.py --local-files-only \
  --prompt-lengths 128 128 128 1 --compare-segmented-sdpa \
  --compare-vectorized-kv-write --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/<new-run-name>
```

## 变长 Prefill→Decode 续写

`scripts/check_ragged_prefill_decode.py` 用相同的四条 tokenized prompt 构造
`(128,128,128,1)`，先做 packed Prefill，再执行两轮 batched Paged Decode，
每条请求共生成 3 个 token。HF reference 显式调用 Prefill 和带
`past_key_values` 的 Decode，不调用 `generate()`。Gate 检查逐请求 token 序列、
每轮输入位置与提交后 Cache 长度、活动物理 block 不重叠、跨 block 增长和
最终释放全部 block。

在 clean-tree commit `a75b400bb6a80d770766bae8e6d3745f580874bc` 上，
`masked` 与 `segmented_sdpa` Prefill 均通过全部 8 项检查；四条请求的
Cache 长度从 `(128,128,128,1)` 经第一次 Decode 变为 `(129,129,129,2)`，
第二次变为 `(130,130,130,3)`。三条长请求第一次 Decode 从各 8 个 block
增长到 9 个。完整 token IDs、block IDs、GPU/CUDA/PyTorch 和 Git 状态位于：

- `benchmarks/results/ragged_prefill_decode_masked_clean_a75b400/result.json`
- `benchmarks/results/ragged_prefill_decode_segmented_clean_a75b400/result.json`

这是固定 workload 的 correctness 证据，不是性能 benchmark；它未覆盖
Scheduler 的动态加入/离开、长序列数值漂移或其他模型/GPU。相应单元和完整测试
为 `236 passed, 1 warning`（CUDA 架构编译提示）。

## Engine/Scheduler 路径的续写

`scripts/check_engine_ragged_continuation.py` 把**同一** `(128,128,128,1)`
workload 送入 `ContinuousBatchEngine`。第 0 步四请求 packed Prefill（`auto`
选择 `masked`），第 1/2 步分别对四请求 batched Paged Decode。HF 仍独立显式
Prefill/Decode；四条请求各 3 个 greedy token 全部一致。

这里须区分“已提交长度”和“物理预留容量”：Admission 在第 0 步就为三条
128-token 请求各预留 9 个 block，即容量 144，但已提交长度仍为 128；
第 1 步长度增至 129，跨越**逻辑** block 边界，物理 block 数保持 9，
不是此时新分配。第 2 步发出最后一个 token 后，Engine 释放 28/28 个 block，
waiting/running 队列与 reservation 均为空。Gate 还逐步检查发出的 token、
block table 的独占和稳定性、最终释放的 block IDs。

clean-tree commit `80d9e40821c283e6ce299f04a70d930b2aa34dae` 的完整记录位于
`benchmarks/results/engine_ragged_continuation_clean_80d9e40/result.json`；
`244 passed, 1 warning`。这仍是固定 workload 的 correctness 检查，不测时间，
也没有覆盖新请求在 Decode 过程中加入的 mixed step；该场景需要单独验证。

## 晚到短请求的 mixed step

在 `scripts/check_engine_ragged_continuation.py --late-short-request` 模式中，
第 0 步仍对 `(128,128,128,1)` 做四请求 packed Prefill；该步结束后才提交
一条 7-token 短请求。第 1 步 Scheduler 先安排四条旧请求 Decode，再接纳新请求
Prefill，合计 token budget `4+7=11`。Engine 实际先执行 Prefill、再执行
Decode，但写回仍遵循 Scheduler 的请求顺序；第 2 步五条请求一起 Decode。

独立 HF reference 对拍全部五条 greedy token 序列；Gate 同时检查每步发出的
token 与请求行对应、已提交长度、预留容量、活动 block 独占与稳定、逻辑 block
边界及最终释放。初始四请求预留 28 个 block，晚到请求另预留 1 个；结束后
`29/29` 个 block 空闲。clean-tree commit
`bbf46351f0fb6945c98e0f1d90158d101b9eb6c4` 的完整记录见
`benchmarks/results/engine_ragged_late_mixed_clean_bbf4635/result.json`。
旧四请求模式也在同一 commit 复测通过，见
`benchmarks/results/engine_ragged_base_clean_bbf4635/result.json`；
完整测试 `249 passed, 1 warning`。

这只是一个人为控制到达时刻的 correctness case，不代表真实线上到达分布，
没有测量 TTFT、TPOT 或吞吐量；该模式本身不测试内存不足时的排队。
下节单独补充资源压力排队，但仍不包括抢占。

## Block pool 压力下的等待与复用

`--block-pressure-reuse` 把物理池固定为恰好容纳首批四请求的 28 个 block。
第 0 步 Prefill 后提交 7-token 晚到请求，池内 free=0；第 1 步旧请求继续
Decode，晚到请求仍在 waiting，且没有 Cache 或 reservation。首条长请求
只生成 2 token，于第 1 步结束并释放 block `0…8`；第 2 步晚到请求才
与其余旧请求 Decode 同轮 Prefill，取得刚释放的 block `0`。其余请求
结束后，第 3 步晚到请求 Decode 并释放 block，最终 free=28/28。

Gate 检查等待期无副作用、每步 token 行与 HF 独立 reference 一致、活动
block 无重叠、复用 ID 属于已释放集合、分阶段释放顺序和最终资源清空。
干净提交 `d0d1c33c45ef9fbfcc01ae21b35883960e81686d` 的完整 token、
block 与队列轨迹在
`benchmarks/results/engine_block_pressure_reuse_clean_d0d1c33/result.json`。
同一提交也复测了原四请求与无压力晚到模式；完整测试为
`254 passed, 1 warning`。这里的“复用”只指物理 block ID，
**不**允许新请求读到旧请求的 KV；HF token 对拍覆盖了当前固定输入的这一风险。

此实验仍不测延迟，也不证明任意到达分布下不会饥饿；没有实现 preemption，
更不能把“此处等待一步”写成通用调度保证。
