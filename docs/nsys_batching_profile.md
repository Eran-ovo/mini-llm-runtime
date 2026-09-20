# 用 Nsight Systems 验证 Prefill/Decode Interference

当前瓶颈假设是：Continuous mixed step 中逐请求 Prefill 排在 batched Decode 前面，使已有
请求的下一 token 延迟增加。Nsight Systems 用于验证时间顺序和 stream gap；此阶段不修改
Scheduler 策略，也不优化 kernel。

## NVTX 层级

仅在 `ContinuousBatchEngine(enable_nvtx=True)` 时标注：

```text
scheduler.schedule_step
engine.step:<index>:prefill=<P>:decode=<D>
  model.prefill:<request_id>:tokens=<N>
  model.decode_batch:batch=<D>
  token.d2h_sync
  scheduler.apply_and_release
```

默认关闭 NVTX，普通 correctness/benchmark 路径不变。所有 range 使用 context manager，
异常时仍会 pop，避免后续 timeline 嵌套错乱。

## Capture 边界

profile workload 先完整 warmup，再创建全新 Engine。模型加载、CUDA Extension 初始化和
warmup 不进入 capture；脚本用 `cudaProfilerStart/Stop` 包住唯一一次正式 workload，命令
用 `--capture-range=cudaProfilerApi` 只采集该区间。

```bash
source /home/eran/venvs/torch/bin/activate
mkdir -p benchmarks/results/nsys_batching_v06

TORCH_CUDA_ARCH_LIST=8.6 nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output=benchmarks/results/nsys_batching_v06/continuous \
  python -m experiments.profile_batching_policy \
  --policy continuous --local-files-only
```

Static 使用完全相同命令，只替换 policy 和 output。查看重点：

1. Continuous 的 mixed step 是否明确先出现一个或多个 Prefill range；
2. Decode range 是否只能在 Prefill 全部完成后开始；
3. `token.d2h_sync` 是否是每 step 的 CPU/GPU barrier；
4. Static decode-only step 是否没有 Prefill range；
5. kernel 之间是否存在明显 CPU launch gap。

NVTX range 只能解释时间归属，不能给出 SM occupancy、memory throughput 或 warp stall；这些
属于 Nsight Compute。只有 Systems timeline 支持当前假设后，才决定是否研究 chunked
prefill 或 Prefill/Decode overlap。

当前 WSL2 环境的 Nsight Systems 2023.4 能采集 NVTX 与 CUDA API，但生成的 SQLite 中没有
CUDA kernel data，`cuda_gpu_kern_sum` 会明确标记为 skipped。因此本轮只能验证 host-side
编排顺序、range duration、API 调用与同步边界，不能声称观察到了 GPU kernel overlap、SM
利用率或真实 device idle。完整限制和实测证据记录在结果目录的 `analysis.md`。
