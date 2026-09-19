# split-KV 最小实现与实验记录

## 本轮唯一主要变量

把 v1 每个 Query Head 的序列扫描分为多个 CTA，新增 FP32 partial state 与 merge。
点积仍为 64-thread shared reduction，K/V 仍为 scalar load；没有引入 shuffle、
Tensor Core、cp.async 或改变 Cache 布局。稳定 Python API 仍调用 v1，实验入口位于
`experiments/split_kv_attention.py`。

每个请求有效 block 数为 C，分区 j 负责 `[floor(C*j/S),floor(C*(j+1)/S))`。
末尾 token 边界截断到 sequence length。分区保存 `(m,l,a)`，其中 a 有 64 个 FP32
元素。空分区完整写入 `(-inf,0,0)`，merge 按 l=0 跳过。合并公式：

```text
m = max_j(m_j)
w_j = exp(m_j-m)
O = sum_j(w_j*a_j) / sum_j(w_j*l_j)
```

分区输出不能提前转为 FP16，也不能直接平均各分区归一化输出。两个 kernel 使用同一个
PyTorch current stream；scratch 在调用内分配，并由 PyTorch allocator 管理生命周期。
unchecked 路径要求调用方保证 metadata 已验证且不变，不作为公开稳定接口。

## 验证和运行

```bash
source /home/eran/venvs/torch/bin/activate
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.6 python -m pytest -q
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.benchmark_split_kv \
  --output-dir benchmarks/results/split_kv_v2_run1
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.benchmark_split_kv \
  --output-dir benchmarks/results/split_kv_v2_run2
```

111 项测试通过。新增测试覆盖 MHA/GQA/MQA、分区数 1/2/3/8/64、长度
1/15/16/17/65/2048/2049、非连续物理块、NaN 尾部、非默认 stream、无效分区数/metadata，
以及 score 差异巨大时全局权重不能等同局部平均的构造案例。

补充尝试 Compute Sanitizer memcheck（长上下文和强烈不均匀权重两例）：工具报告
`Failed to initialize WDDM debugger interface` 和 `Device not supported`，退出码 1。
子进程两个数值测试通过，但 sanitizer 没有正常采集，不能声称内存检查通过。
本轮未修改 Windows debugger 配置。

## 2026-09-19 结果

RTX 3060 Laptop，FP16，B=1，Hq/Hkv=14/2，D=64，block size=16，S=8。
每条路径 10 次 warmup sample、30 个正式 sample，每个 sample 20 次完整调用。
v1/split 交错并奇偶轮反转。计时前完成 JIT 与 checked metadata 验证。

| Run | N | v1 median us | split median us | v1/split |
|---|---:|---:|---:|---:|
| 1 | 16 | 15.530 | 22.861 | 0.679 |
| 2 | 16 | 15.691 | 23.706 | 0.662 |
| 1 | 2048 | 1210.931 | 172.058 | 7.038 |
| 2 | 2048 | 1214.282 | 172.800 | 7.027 |

split 对 SDPA 最大绝对误差：N=16 为 0.0009765625；N=2048 为 0.00006103515625。
FP32 scratch 为 29,568 bytes；实际 peak allocated delta 为 v1 4,096 bytes、split
33,792 bytes，后者还包含输出及 allocator 粒度影响。

结果保存于上述两个目录的 result.json。实验基于 c9d9be2 的 dirty 工作区，JSON
明确记录 git_dirty=true，并保存实际 CUDA、Python 计算入口、计时代码的 SHA256。
这些是内部实验数据，尚未形成 clean release 的性能报告。

## 决策及边界

保留 split-KV 作为长上下文候选；不替换 v1 稳定入口。N=16 出现退化，保留这项结果。
对于仅一个有效 block 的输入，8 分区中只有一个分区真正工作，其余 CTA 和 merge
产生额外开销。N=2048 时，128 个逻辑块分给 8 个 CTA，每 CTA 处理 256 token，
总 partial CTA 数从 14 增加到 112。

计时覆盖 scratch 分配调用、partial 与 merge；CUDA Event 本身不会计量所有 CPU
分配工作，但可能包含 host 提交间隙。结果是完整路径 Event 均摊时间，并非纯 kernel
duration；还不能推导整模型 TPOT。短配置 CV 约 20%–28%，长配置约 6%–8%，没有锁频。
split 数只测试了固定 8 的性能，不能把它当成所有 shape 的最优值。

下一步：固定 B=1、N=2048，仅扫描分区数 S=1/2/4/8/16/32，计入 merge 与 scratch，
找出并行度收益和合并开销之间的平衡，再考虑是否形成 shape dispatch。

相关算法背景：PyTorch [Flash-Decoding](https://pytorch.org/blog/flash-decoding/)。
