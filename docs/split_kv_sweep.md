# 固定 shape 的 split-KV 分区扫描

2026-09-19；B=1，N=2048，Hq/Hkv=14/2，D=64，FP16，block size=16。
本轮只增加实验 harness，CUDA kernel 不变。稳定入口仍用 v1。

## 原理

128 个逻辑 KV block 按 S=1/2/4/8/16/32 分区。每个 partial CTA 扫描 2048/S
个 token，grid 有 14*S 个 partial CTA；merge 仍有 14 个 CTA。
增加 S 能缩短单个 CTA 的依赖链，并提供更多并行工作，但 scratch 和 merge 工作量
随 S 增长。scratch 为 `14*S*66*4` bytes。S=1 仍有两次 launch，与原 v1 不等价。

## 复现及测量边界

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.sweep_split_kv \
  --output-dir benchmarks/results/split_kv_sweep_run1
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.sweep_split_kv \
  --output-dir benchmarks/results/split_kv_sweep_run2
```

7 个候选共享相同 Q/K/V 和随机打散的 block table。每个候选先经 checked API 与
SDPA/v1 对拍；14 轮 warmup，28 轮正式测量，每个 sample 为 20 次完整调用。
每 7 轮循环轮换位置，下一组反向轮换；正式测量中每个候选在每个位置出现 4 次。
这种安排减少位置偏差，但无法完全消除候选运行时间不同造成的温度/功耗历史影响。

Event 区间包含 partial+merge 的 GPU 执行及可能的 host 提交间隙；调用内部包含
scratch 分配，Event 不能独立测出 CPU 分配耗时。这不是纯 kernel duration，不能
外推模型 TPOT。固定重复输入属于 warm-cache workload。

每轮生成 result.json、samples.csv、report.md；JSON 包含 196 个原始 Event/显存
样本、实际顺序、median、CV、correctness error、环境与源码 SHA256。
基于 c9d9be2 dirty 工作区，git_dirty=true 被如实保存；尚非 release 性能数据。

## 实测结果

| Case | 第一轮 median us | 第二轮 median us | Scratch bytes |
|---|---:|---:|---:|
| v1 | 1200.529 | 1214.566 | 0 |
| S=1 | 1418.522 | 1403.418 | 3696 |
| S=2 | 717.338 | 704.691 | 7392 |
| S=4 | 314.112 | 329.190 | 14784 |
| S=8 | 169.779 | 167.270 | 29568 |
| S=16 | 95.974 | 96.102 | 59136 |
| S=32 | 77.081 | 77.184 | 118272 |

所有候选 max_abs_vs_sdpa 均为 0.00006103515625，满足 atol=rtol=2e-3。
分区正确性与顺序均衡测试共 22 项通过。

两轮 warmup 后/结束的 GPU 快照均为 P0、SM 1965 MHz，温度从 54°C 到 57°C。
这只是快照，不是全程锁频。S=32 样本 CV 分别为 12.70% 与 11.59%，应继续保留
原始样本而不只看两轮 median 接近。

## 判断

S=32 为本次已测集合中最快者；相对同轮 v1 的 Event 路径时间比为 15.57/15.74，
相对 S=16 为 1.245。此处不能表述为超过 SDPA 或整个模型加速。

S=16 到 32 时，scratch 翻倍，而延迟只下降约 19.7%，收益已经递减，但并未观察到
性能转差。最快点位于扫描边界，因此没有证据称 S=32 为最优值。

S=1 比 v1 慢约 189–218 us；额外 launch/merge、partial 边界与状态写入、两个 CUDA
函数不同的编译结果都可能贡献差值。没有单 kernel 分解证据，不能把差值全算给 merge。

本轮不修改默认 num_splits，也不加入 shape dispatch。
下一步只比较 S=16/32/64，在同一批输入上交错复测，以检查扫描边界外是否仍有收益。

## 边界复测：S=16/32/64

2026-09-19，沿用相同 shape、输入 seed 和 CUDA kernel。扫描入口增加 --splits 和
--exclude-v1，默认仍保留原七项扫描。v1 虽不参与本次计时，仍参与正确性验证。

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.sweep_split_kv \
  --splits 16,32,64 --exclude-v1 --warmup 12 --repeats 30 --iterations 20 \
  --output-dir benchmarks/results/split_kv_boundary_run1
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.sweep_split_kv \
  --splits 16,32,64 --exclude-v1 --warmup 12 --repeats 30 --iterations 20 \
  --output-dir benchmarks/results/split_kv_boundary_run2
```

每个候选在每个位置出现 10 次；每轮报告 90 个正式 raw samples。
参数解析和测量位置均衡测试 9 项通过，所有候选执行前与 v1/SDPA 对拍通过，
max_abs_vs_sdpa 均为 0.00006103515625。

| S | Run1 median us | Run2 median us | Scratch bytes |
|---:|---:|---:|---:|
| 16 | 99.608 | 100.962 | 59136 |
| 32 | 79.793 | 80.559 | 118272 |
| 64 | 78.559 | 78.950 | 236544 |

64 相比 32 的 median 延迟下降分别为 1.55%、2.00%；同轮配对的 `T32-T64` 中位数
为 2.537/2.582 us，两轮中 64 均赢得 24/30 对。配对差值的中位数不等于两个
median 的差值；它们是不同统计量。配对也不能完全消除时间相关的 GPU 状态变化。

本轮 CV 约 17%–21%。环境快照出现 SM 1425–1897 MHz，故不能跨报告直接比值；
更不能把微小 median 差异视为已经确认的纯 kernel 加速。

决策：保留 32 为低 scratch 的折中候选，64 在已测范围内略快，暂不修改默认参数，
不加入 shape dispatch。没有测到明确回升点，不能宣称找到了全局最优值。
从 32 到 64，partial CTA 数 448→896，每 CTA token 数 64→32；merge 每个输出
维度的分区遍历量和 scratch 均翻倍。收益递减符合这种成本竞争，但要定位实际瓶颈
仍需单 kernel 证据。

后续已完成 S=32 与 64 的 partial/merge 诊断；硬件指标、测量边界与停止继续扫参的
决策见 [split-KV profiler 学习记录](split_kv_profile.md)。
