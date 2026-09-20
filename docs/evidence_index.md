# 项目证据索引

本页区分“正式可引用结果、正确性证据、profiler 证据、smoke 和失败实验”。README/简历中的
数字只能来自 formal benchmark 或 clean-tree release bundle；其他结果只能解释开发过程。

## Clean-tree Release Candidate

- source commit：`e01c3c737a5ee55d9e14544c5108c0ae0a5b6a66`；
- full tests：`180 passed`；
- bundle：`benchmarks/results/release_candidate_e01c3c7/`；
- manifest：`bundle_manifest.json`，`passed=true`；
- manifest SHA-256：
  `09603b59aa6a2a8b09c5624634edd80eee8c24b11ed05f4ddb9ef3a11ca9b055`；
- 三个正式 JSON 均记录相同 commit 且 `git_dirty=false`；
- bundle 内 14 个非 manifest 文件均有相对路径、大小和 SHA-256。

Release bundle 不进入 Git；通过
`scripts/run_release_evaluation.py --output-dir <new-directory>` 复现，或在未来 GitHub Release
中作为独立 artifact 发布。

## Formal Benchmark

### Static vs Continuous Batching

- release 路径：`release_candidate_e01c3c7/static_vs_continuous/`；
- workload：8-request burst，generation lengths `2,4,8,12`；
- warmup / measured：3 / 10；
- case 顺序逐轮交错；
- correctness：所有 policy token sequence 相同；
- 指标：throughput、TTFT、TPOT、E2E、CUDA timeline、peak memory 和 raw samples。

此前两次独立 run 位于 `batching_policy_v06` 和 `batching_policy_v06_repeat2`，方向一致，但
记录为 `git_dirty=true`，只用于重复性佐证；headline 数字使用 clean-tree bundle。

### Mixed Prefill Budget 与 Tail/Fairness

- release 路径：`release_candidate_e01c3c7/mixed_prefill_tail/`；
- cases：unbounded、8、4；
- warmup / measured：3 / 12；
- percentile：每个 trial 内 linear interpolation，再对 trial percentile 取 median；
- 正式报告同时保存 arrival-position TTFT/E2E，避免 pooled median 掩盖 waiting tail。

该实验的决策是保留默认关闭的教学旋钮，不推荐 budget 4/8，也不继续针对单一 workload 扫参。

## Formal Correctness

### Continuous Batching HF Gate

- release 路径：`release_candidate_e01c3c7/continuous_batch_correctness/`；
- oracle：HF 显式 Prefill/Decode，不调用 `generate()`；
- exact match：逐请求完整 greedy token sequence；
- 路径覆盖：late admission、mixed step、batched Decode、跨 block sequence、释放后物理 block
  复用、最终全部资源释放。

### 参数化/分层测试

- ModelRunner 与 HF 逐层/最终 logits：`manual_qwen_*`、`independent_qwen_model_runner`；
- Paged KV 地址/生命周期：`test_paged_kv_*`、`test_paged_batch.py`；
- Paged Attention 与 SDPA/Python reference：`test_paged_attention*.py`；
- Engine rollback 与请求指标：`test_engine.py`、`test_request_metrics.py`。

## Profiler Evidence

### Paged Attention Nsight Compute

- v1 kernel 与 split-KV 的 `.ncu-rep`、profile context 和分析位于
  `benchmarks/results/paged_profile_*`、`split_profile_*`；
- 用于解释 occupancy、memory access、warp stall 和短序列 split-KV 开销；
- 不与端到端 TTFT/TPOT 混为同一种计时。

### Continuous Batching Nsight Systems

- 路径：`benchmarks/results/nsys_batching_v06/`；
- NVTX 证据：mixed step 的 host 编排为 Prefill 完成后才进入 batched Decode，每 step 末存在
  token D2H barrier；
- 限制：WSL2 + Nsight Systems 2023.4 没有 CUDA GPU kernel timeline，因此不能判断 kernel
  overlap、SM 利用率或真实 device idle。

## Smoke、失败实验与停止条件

- 名称含 `smoke` 的目录只验证脚本、依赖和输出 schema，不能引用性能数字；
- split-KV、mixed Prefill budget 等负结果保留在 `experiments/` 和对应文档，不进入稳定 dispatch；
- v0.6 manifest 明确拒绝把 smoke 路径列为正式证据；
- 失败 capture、工具限制和被数据否决的优化必须保留原因，不能只展示成功实验。

## 已知不能外推的范围

- 当前正式 workload 是短 prompt、8-request burst，不代表真实在线 arrival distribution；
- Qwen2.5-0.5B FP16/RTX 3060 Laptop 的结果不能外推到其他模型和 GPU；
- release bundle 没有重新采集 Nsight，使用的是先前机制分析；
- 同步 Engine 不支持 chunked prefill、preemption、async overlap、量化或分布式推理；
- 所有简历表述都必须保留 workload 和硬件上下文。
