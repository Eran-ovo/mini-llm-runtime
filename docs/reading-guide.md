# 源码导航与阅读路线

本页以 v1.1.0 稳定实现为准。先看一个请求如何走完，再下钻到模型、显存与 CUDA。
分阶段教学实验见 [开发指南](development-guide.md)，对外性能口径见
[公开 benchmark 摘录](benchmarks/README.md)。

## 十分钟看懂整体能力

| 要回答的问题 | 建议阅读 |
|---|---|
| 请求如何动态加入、完成和回收？ | [engine.py](../src/mini_llm_runtime/engine.py)、[scheduler.py](../src/mini_llm_runtime/scheduler.py) |
| 谁决定可以接纳多少请求？ | [block_admission.py](../src/mini_llm_runtime/block_admission.py) |
| 如何摆脱完整 HF model forward？ | [qwen_loader.py](../src/mini_llm_runtime/qwen_loader.py)、[qwen_model_runner.py](../src/mini_llm_runtime/qwen_model_runner.py) |
| KV 的地址与生命周期怎么管理？ | [paged_kv_cache.py](../src/mini_llm_runtime/paged_kv_cache.py)、[paged_kv_manager.py](../src/mini_llm_runtime/paged_kv_manager.py) |
| CUDA kernel 如何消费 block table？ | [paged_attention_cuda.cu](../csrc/paged_attention_cuda.cu)、[绑定](../csrc/paged_attention_binding.cpp) |
| 哪些性能数字能复核？ | [benchmark 摘录](benchmarks/README.md)、[证据索引](evidence_index.md) |

## 一次完整请求

从 [ContinuousBatchEngine](../src/mini_llm_runtime/engine.py) 开始：

1. `submit()`：保存 request id、prompt、生成长度上限，并记录 arrival。
2. Scheduler 先为 running Decode 分配预算，再从 waiting 队列接纳 Prefill。
3. Admission 根据最大 cache token 数申请生命周期 blocks。
4. 新请求组成 packed Prefill；每个 segment 独立 RoPE position、attention 范围和 KV table。
5. 旧请求组成 batched Decode；新 token 写 KV，每层 CUDA Attention 读取各自历史 pages。
6. GPU argmax 结果按请求顺序合并，一次 D2H；Scheduler 原子提交 token/状态。
7. 完成事件触发释放；blocks 回到 free list，下一个 step 可复用。

验证入口：

```bash
python -m experiments.continuous_batch_engine_runner \
  --output-dir benchmarks/results/engine_walkthrough

python scripts/check_engine_ragged_continuation.py \
  --block-pressure-reuse \
  --output-dir benchmarks/results/pressure_walkthrough
```

输出不仅含 token，还包含 step 轨迹、block table、释放记录与机器可读 gate。
首次下载模型；缓存后可加 `--local-files-only`。

## 三条深入路线

### 模型执行

`qwen_config → qwen_weights → qwen_loader → qwen_model_runner`。
关注 Q/K/V shape、GQA head 映射、RoPE position、SwiGLU、residual 与 tied LM Head。

测试：[test_manual_qwen_model_runner.py](../tests/test_manual_qwen_model_runner.py)；
教学逐层对拍见 [开发指南](development-guide.md)。

### KV 生命周期

`FixedBlockAllocator → RequestBlockTable → PagedKVStorage → PagedKVCacheManager → PagedBatch adapters`。
关注 reserve/write/commit 的先后、committed length 与 reserved capacity 的区别、
cross-block 增长、OOM 原子失败和释放复用。

文档：[Batched Decode Adapter](paged_batch_decode.md)、
[KV Reservation 分析](kv_reservation_analysis.md)。

### CUDA 与端到端性能

`paged_attention_binding.cpp → paged_attention_cuda.cu → benchmark_paged_attention.py`。
当前 v1 是 correctness-first：一个 CTA 负责 request/query head，沿 token 串行，
FP32 online softmax。split-KV 是独立实验路径。

再回到 `benchmark_batching_policy.py` 与 `request_metrics.py`，
区分 kernel latency、GPU timeline 和 CPU 请求延迟。

文档：[v1 NCU 分析](paged_attention_v1_profile.md)、
[split-KV 实验](split_kv_experiment.md)、
[Systems 采集边界](nsys_batching_profile.md)。

## 测试与性能证据怎么读

- 普通 `pytest` 不下载真实权重；CUDA/NVCC 缺失时相关测试会跳过。
- 真实 HF gate 独立执行，覆盖 token、cache、动态路径和完整回收。
- 不同 FP16 GEMM shape 会有舍入差异，token 一致与 tensor bitwise equal 是不同标准。
- 相同计算 shape 下的 scalar/vectorized KV write 要求完全等价。
- 所有公开性能数字必须带 source commit、硬件、计时范围和 raw samples。
- [v1.0.0 Release](https://github.com/Eran-ovo/mini-llm-runtime/releases/tag/v1.0.0)
  提供完整证据归档；v1.1.0 更新源码与教程，历史性能仍绑定原采样提交。

## 文档的版本边界

学习记录保留了某一阶段“尚未实现”的描述。当前源码支持变长 packed Prefill、
batched Paged Decode、请求指标及容量压力验证；查当前能力优先看
[README](../README.md)、本页和 [Engine 文档](continuous_batch_engine.md)。

历史实验的测试数量和性能数字属于当时 commit，不自动更新为最新提交。
