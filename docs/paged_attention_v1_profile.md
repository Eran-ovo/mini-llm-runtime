# Paged Attention v1 profiler 诊断

2026-09-19，kernel 源码基于 `c9d9be2`。采集时工作区增加了 profiling 脚本；
没有修改 CUDA kernel。完整环境和 dirty 状态见各输出目录的 profile_context.json。

## 复现

从仓库根目录运行；必须 activate 虚拟环境，让 JIT loader 能在 PATH 中找到 Ninja。

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 ncu \
  --profile-from-start off --clock-control none --cache-control all \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --export benchmarks/results/paged_profile_v1_probe/kernel \
  python -m scripts.profile_paged_attention \
  --output-dir benchmarks/results/paged_profile_v1_probe
```

输出目录由脚本创建。已有同名报告时使用新的 export 名称，保留历史证据。
Batch 对照添加 `--batch-size 8` 并更换输出位置。

读取原始报告：

```bash
ncu --import benchmarks/results/paged_profile_v1_probe/kernel.ncu-rep --page details
ncu --import benchmarks/results/paged_profile_v1_probe/kernel.ncu-rep --page raw --csv
```

脚本先运行 checked API 与 SDPA 对拍，然后执行 50 次 warmup。profiler start/stop
范围内只有一次 unchecked Paged 调用。两个 case 的 max_abs_vs_sdpa 均为
0.00006103515625；采集输出与 checked 输出逐 bit 相同。

## 实测证据

Nsight Compute 2024.1；RTX 3060 Laptop，30 SM；FP16，Hq=14，Hkv=2，D=64，
KV length=2048，block size=16，64 threads/CTA。每次报告使用 14 passes。

| 指标 | B=1 | B=8 |
|---|---:|---:|
| CTA 数 | 14 | 112 |
| Achieved occupancy | 4.17% | 15.49% |
| Eligible warps / scheduler | 0.10 | 0.187 |
| Issue active | 10.20% | 17.71% |
| Long scoreboard，cycles/issued instruction | 4.312 | 4.596 |
| Short scoreboard，cycles/issued instruction | 1.242 | 1.381 |
| Barrier，cycles/issued instruction | 1.184 | 1.332 |

B=1：31 registers/thread，256 bytes static shared memory/CTA；理论 occupancy
66.67%，实际 active warps/SM 为 2；DRAM throughput 仅峰值的 0.21%。
Profiler duration 为约 1.77 ms（B=1）与 1.664 ms（B=8）。

原始报告：`benchmarks/results/paged_profile_v1_probe/kernel.ncu-rep` 与
`benchmarks/results/paged_profile_v1_probe/batch8.ncu-rep`；B=8 环境记录在
`benchmarks/results/paged_profile_v1_batch8/profile_context.json`。

## 解释和限制

- B=1 只有 14 CTA，无法同时覆盖 30 SM；每 CTA 只有两个 warp。增加寄存器容量
  或减少 shared memory 都不会凭空增加可调度工作。
- long scoreboard 最高，eligible warps 很少，支持“内存依赖延迟难以隐藏”的判断。
  DRAM throughput 很低，不支持“已经打满显存带宽”的判断。
- barrier 和 short scoreboard 都有成本，但不能仅凭源码 barrier 数量断言它们占主导。
- 没有采集到源指令级 PC sampling，无法把 long scoreboard 唯一归因到 K、V、
  block table 中哪条 load；报告有 optional sampling metric 缺失提示。
- NCU 中的估计 speedup 不是实测加速比，也不能把多个提示相加。
- `clock-control none` 没有锁频；`cache-control all` 在 replay 时清理缓存。
  profiler duration 不与既有 warm-cache benchmark 直接相除。
- 原 benchmark 的 Event 包围 Python launch 循环，区间可能包含 host 提交不及时造成
  的 GPU 空隙。短 SDPA 数字不应表述为纯 kernel duration；内存 delta 也包含循环中
  短暂同时存活的输出，不能据此断言 SDPA 物化完整 attention matrix。

## 下一项单变量实验

优先尝试沿 KV 序列分区，增加每个 Query Head 的 CTA 数（split-KV），保持 QKV dtype、
block layout 和基础点积计算不变。每个分区输出 FP32 `(m, l, a)`；合并时：

```text
m = max(m_j)
l = sum(exp(m_j - m) * l_j)
a = sum(exp(m_j - m) * a_j)
O = a / l
```

分区必须覆盖有效 token 且不重叠。额外 scratch、merge kernel 和 launch 成本必须
计入 v2 benchmark；这仍是待验证假设，尤其短上下文可能因合并开销变慢。

指标语义参考 NVIDIA 官方指南：
https://docs.nvidia.com/nsight-compute/ProfilingGuide/
