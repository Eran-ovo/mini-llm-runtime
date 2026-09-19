# split-KV：partial / merge 的 Nsight Compute 诊断

2026-09-19；固定 B=1，N=2048，Hq/Hkv=14/2，D=64，FP16，block size=16。
本次只比较 S=32 与 S=64，CUDA kernel 未改变。

## 这次要回答什么

split-KV 把一个 query head 对全部 KV token 的 attention 拆成两步：

1. `partial_kernel`：每个 `(request, query_head, split)` CTA 扫描一段 KV，输出
   online-softmax 状态 `(m, l, a[64])`；
2. `merge_kernel`：每个 `(request, query_head)` CTA 读取 S 份状态，用共同的全局最大值
   重新缩放并合并，最后输出 `a/l`。

S 从 32 增至 64 时，总 K/V token 数不变，但 partial grid 从 448 增至 896，单 CTA
扫描的 token 数从 64 降至 32。理论收益是更短的串行依赖链和更多 CTA；代价是 Q 被更多
CTA 重复读取、partial state/scratch 翻倍，merge 循环也翻倍。因此要分别观察两个 kernel，
不能只凭完整调用时间猜测原因。

## 正确性与采集方法

`scripts/profile_split_kv.py` 在采集前先让候选同时与 PyTorch SDPA 和 v1 CUDA kernel
对拍。S=32、64 的 `max_abs_vs_sdpa` 均为 0.00006103515625，并且被采集调用与 checked
调用逐元素一致。

```bash
source /home/eran/venvs/torch/bin/activate
mkdir -p benchmarks/results/split_profile_s32 benchmarks/results/split_profile_s64

TORCH_CUDA_ARCH_LIST=8.6 ncu --profile-from-start off \
  --clock-control none --cache-control all \
  --section SpeedOfLight --section LaunchStats --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  --section MemoryWorkloadAnalysis \
  --export benchmarks/results/split_profile_s32/report \
  python -m scripts.profile_split_kv --num-splits 32 \
  --output-dir benchmarks/results/split_profile_s32

# 第二次将路径和 --num-splits 都改为 64。
```

每个 kernel 经 14 次 replay 收集硬件计数器。GPU 未锁频，报告也给出了相应警告。
报告中的 duration 是单次 profiler 观测，不具备正式 benchmark 的多轮统计意义；它只用于
结合 grid、occupancy 和 stall 解释执行结构。

## 采集结果

### partial kernel

| 指标 | S=32 | S=64 |
|---|---:|---:|
| Grid / waves per SM | 448 / 0.93 | 896 / 1.87 |
| 每 CTA token 数 | 64 | 32 |
| Duration（单次 NCU） | 93.09 us | 94.46 us |
| Registers / thread | 42 | 42 |
| Achieved occupancy | 60.34% | 60.77% |
| Eligible warps / scheduler | 0.76 | 0.80 |
| Issued warps / scheduler | 0.49 | 0.50 |
| Max throughput | 67.96% | 67.34% |
| L1/TEX / L2 hit rate | 49.14% / 85.02% | 44.43% / 80.75% |
| L1TEX scoreboard stall / issued instruction | 5.5 cycles | 5.4 cycles |

S=64 确实增加了可调度 CTA 数，但 S=32 已经达到约 60% achieved occupancy；翻倍 grid
没有改善 occupancy、eligible warp、issue rate 或吞吐率。两者都约有一半周期没有 eligible
warp，主要提示仍是等待 L1TEX load dependency。这里不能把它简化为“DRAM 带宽瓶颈”：
两者 DRAM throughput 只有约 4.5%–4.9%，NCU 的 67% `Max throughput` 主要来自片上
memory/compute pipe 指标。

### merge kernel

| 指标 | S=32 | S=64 |
|---|---:|---:|
| Grid / waves per SM | 14 / 0.03 | 14 / 0.03 |
| Duration（单次 NCU） | 8.83 us | 15.55 us |
| Achieved occupancy | 4.16% | 4.16% |
| Eligible / issued warps per scheduler | 0.09 / 0.09 | 0.09 / 0.09 |
| L1TEX scoreboard stall / issued instruction | 7.9 cycles | 8.4 cycles |
| Scratch | 118272 bytes | 236544 bytes |

merge 的 grid 始终只有 14 个 CTA，小于 30 个 SM，先天无法铺满 GPU。S=64 没有增加
merge 并行度，只让每个线程读取和合并两倍的 partial states；因此单次采集 duration 从
8.83 us 增至 15.55 us。4.16% 是整个 GPU 在这次小 grid 上的 achieved occupancy，
不能与单个 SM 能驻留多少 block 的 theoretical occupancy 混为一谈。

## 怎样解释与 Event benchmark 的差异

边界复测中，完整 Python 调用的多轮 CUDA Event median 显示 S=64 比 S=32 快约
1.55%–2.00%；这次 NCU 单次 replay 则没有看到 partial 变快，并看到 merge 明显变慢。
两者并不构成“任选一个相信”的关系：

- Event benchmark 回答固定 harness 下完整路径的延迟分布，但包含两个 launch、可能的
  host supply gap 和调用内 scratch 分配影响；
- NCU 回答 kernel 的硬件执行结构，但 replay、cache control 和未锁频会改变时序，单次
  duration 不能替代多轮 benchmark；
- 约 2 us 的差值相对 Laptop GPU 的 17%–21% 样本 CV 很小，虽然配对胜率有信号，仍不足
  以支持更高 scratch 和更多 merge 工作的默认策略。

## 决策与主线

当前证据支持“split-KV 解决了长序列 v1 并行度不足”，但不支持“分区越多越好”。
S=32 已让 partial kernel 获得充足并行度；S=64 的收益没有体现在更好的核心效率指标上，
却确定地翻倍 scratch 和 merge 工作。因此保留 S=32 作为长序列实验候选，S=64 不设为
默认值，也暂不设计 shape dispatch。

至此停止继续扫描 split 数。下一步回到推理引擎主线：把已经验证的稳定 Paged Attention
v1 接入 Qwen ModelRunner 的单 token Decode，以真实请求的 block table 和 sequence
lengths 替换 correctness 阶段的 gather 路径。split-KV 继续留在 `experiments/`，等端到端
TPOT 显示 attention 确实是瓶颈后，再决定是否进入稳定 dispatch。
