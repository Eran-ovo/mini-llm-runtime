# Clean-Tree Release Evaluation

## 目标

v0.6 的单项 artifact 来自明确 commit，但当时工作区存在文档改动，因此环境记录为
`git_dirty=true`。v1.0 release 需要更强的可复现性条件：correctness、测试和正式 benchmark
必须在同一个最终 commit、同一个 clean tree 中运行，并打包成不可静默覆盖的结果目录。

## 为什么使用 Detached Worktree

主工作区可能有用户正在编辑的文件。`git stash` 会移动用户状态，`git reset` 可能破坏数据，
直接要求提交又会混入未审查内容。release launcher 使用：

```text
git worktree add --detach <temporary-path> <HEAD>
```

临时 worktree 只包含目标 commit。运行结束后删除的只是该临时 checkout；主工作区的 branch、
index 和未提交文件都不移动。

## Editable Install 陷阱

虚拟环境中的 editable install 可能仍指向主工作区。仅仅把 `cwd` 切到临时 worktree 不足以
保证导入 clean 源码。launcher 会显式设置：

```text
PYTHONPATH=<temporary-worktree>/src:<temporary-worktree>:<old-PYTHONPATH>
```

内层 runner 随后检查 `mini_llm_runtime.__file__` 必须位于临时 worktree 的 `src/` 下，并将
实际 import provenance 写入 bundle manifest。如果仍导入主工作区，release 立即失败。

## 正式执行序列

同一个 clean worktree、同一个 Python/CUDA 环境依次运行：

1. 全量 `pytest -q`；
2. Qwen Continuous Batching HF correctness gate；
3. Static vs Continuous：warmup 3、measured repeats 10；
4. Mixed Prefill tail/fairness：warmup 3、measured repeats 12。

任一步非零退出都会停止后续步骤。已经生成的日志和 failure manifest 保留，方便诊断；失败
bundle 不能作为 release evidence。

## 双重 Clean Check

内层在执行前后都验证：

```text
git rev-parse HEAD == expected_commit
git status --porcelain == empty
```

此外，每个正式 JSON 自己记录的 `environment_after.git_commit` 必须等于 release commit，
`git_dirty` 必须为 false。这样同时检查了 orchestrator 视角与 benchmark 自身视角。

## Bundle 内容

输出目录原先必须不存在，避免覆盖历史 release。bundle 包含：

- `environment_before.json` / `environment_after.json`；
- 每一步完整 stdout/stderr log；
- correctness JSON/report；
- 两套 benchmark JSON/CSV/report；
- `bundle_manifest.json`。

manifest 记录命令、return code、耗时、commit、import provenance、验证结果，以及除 manifest
自身外每个文件的相对路径、大小和 SHA-256。耗时只用于定位异常慢步骤，不属于模型性能指标。

## 正式命令

脚本默认禁止联网下载模型，确保使用已经固定在本机 cache 中的 checkpoint：

```bash
cd /home/eran/mini-llm-runtime
source /home/eran/venvs/torch/bin/activate

python scripts/run_release_evaluation.py \
  --output-dir benchmarks/results/release_candidate_v1
```

RTX 3060 默认使用 `TORCH_CUDA_ARCH_LIST=8.6`。如果确实需要下载缺失模型，必须显式提供
`--allow-download`，并在 release 说明中记录网络来源和解析到的 checkpoint revision。

## 不能解决的问题

- clean tree 不能保证实验设计正确；
- SHA-256 不能证明结果来自可信机器，只能检测 bundle 内文件变化；
- GPU 频率和功耗仍可能随时间变化，交错 case 与多轮样本仍然必要；
- 当前 bundle 没有重新采集 Nsight，因为 profiler 工具版本/WSL2 限制尚未改变；
- release candidate 通过后仍需人工审查报告，不能自动生成夸大的 README 或简历数字。
