# v0.6 Continuous Batching 验收与冻结

## 为什么要做 Milestone Freeze

功能代码、测试、benchmark 和 profiler 分散存在时，很容易出现“某个报告已经过期”“smoke
被误当正式结果”“文档引用的 commit 与数据不一致”等问题。冻结不是停止开发，而是固定一组
可审计证据，明确这个版本已经证明了什么、尚未证明什么。

本项目用 `milestones/v0.6.json` 建立白名单。只有 manifest 中列出的 artifact 属于 v0.6
验收证据；目录中其他 `*_smoke`、预提交结果和失败尝试仍保留学习价值，但不参与性能结论。

## Requirement 验收

| 原始要求 | 状态 | 主要证据 |
|---|---|---|
| waiting/running request queue | PASS | Scheduler 状态机与单元测试 |
| 每个 step 动态加入和移除请求 | PASS | Engine 测试与 HF 端到端 gate |
| 不同 prompt/generation length | PASS | 8-request burst benchmark 原始规格 |
| 简单策略与 token budget | PASS | Static/Continuous、总 budget、mixed Prefill budget |
| Static vs Continuous 指标对比 | PASS | 两次 3-warmup/10-repeat 正式 benchmark |

因此 v0.6 的原始功能范围已经完成。mixed Prefill budget 是从 profiler 证据派生的受控实验，
默认保持关闭，不改变 v0.6 的基础策略结论。

## 白名单 Artifact

1. `batching_policy_v06`：Static vs Continuous 正式 run 1；
2. `batching_policy_v06_repeat2`：独立重复 run；
3. `mixed_prefill_tail_v06`：schema v2 tail/fairness 正式实验；
4. `continuous_batch_correctness_v06`：HF 外部 oracle 与 Paged KV 生命周期 gate；
5. `nsys_batching_v06`：Prefill/Decode interference 的 host-side profiler 证据。

正式 benchmark JSON、CSV 和 profiler 二进制按 `.gitignore` 不进入 Git；manifest 记录相对路径、
关键字段、数据 commit 和主文件 SHA-256。这样本机 artifact 被覆盖后 checker 会失败；新 clone
缺少原始数据时也会明确提示需要复现或从 release artifact 获取。

## Checker 能与不能验证什么

`scripts/check_v06_acceptance.py` 验证：

- 每条 requirement 的 implementation/test/doc 路径存在；
- 正式 artifact 和配套 report/CSV/profile 文件存在；
- 主 artifact SHA-256 未变化；
- warmup、repeats、schema、correctness flag 和 Git commit 符合冻结记录。

它不会重新运行 GPU benchmark，也不会判断性能数字“是否够快”。SHA-256 只能证明文件与冻结
时一致，不能替代实验设计审查。代码 correctness 仍由 `pytest` 和 HF gate 单独负责。

本机验收：

```bash
source /home/eran/venvs/torch/bin/activate
python -m pytest -q
python scripts/check_v06_acceptance.py
```

## 已知限制

- 正式 benchmark 记录 `git_dirty=true`，绑定到明确 commit，但不是 clean-tree release run；
- 性能 workload 是 8-request burst，不能外推到真实在线 arrival distribution；
- WSL2 下 Nsight Systems 没有 GPU kernel timeline，只能验证 NVTX host 编排和 CUDA API；
- correctness gate 使用固定短请求，不能替代所有长度/batch/block size 的参数化覆盖；
- mixed budget 4/8 牺牲 waiting tail 与吞吐，不作为默认策略。

## Freeze 决策

v0.6 状态为 `frozen`：不再继续搜索 mixed budget，不在这个里程碑加入 Chunked Prefill、
异步 stream、Web Server 或新 kernel。后续若修复 correctness bug，必须重新运行受影响 gate；
若改变性能路径，必须生成新 artifact，不能静默覆盖 v0.6 白名单结果。

进入 v1.0 前仍缺少的关键证据不是新功能，而是：**在最终 release commit 上完成一次 clean-tree
的一键 correctness + benchmark 复现，并生成可发布的 artifact bundle。** 这将是下一阶段的
第一条主线。
