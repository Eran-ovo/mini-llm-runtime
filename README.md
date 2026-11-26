# Mini LLM Runtime

**从模型前向到请求调度，独立实现一个可验证的单 GPU LLM 推理引擎。**

A single-GPU LLM inference runtime built from scratch: Qwen ModelRunner, paged KV cache,
CUDA decode attention, and continuous batching.

[![Version](https://img.shields.io/badge/version-v1.1.0-2563eb)](https://github.com/Eran-ovo/mini-llm-runtime/releases/tag/v1.1.0)
[![Model](https://img.shields.io/badge/model-Qwen2.5--0.5B-7c3aed)](#项目能力)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76b900)](#快速开始)
[![GPU](https://img.shields.io/badge/GPU-RTX%203060%20Laptop%20%2F%206%20GB-334155)](#实测结果)

[项目能力](#项目能力) · [运行架构](#运行架构) · [实测结果](#实测结果) ·
[快速开始](#快速开始) · [源码导航](docs/reading-guide.md) ·
[正确性与证据](docs/evidence_index.md)

## 项目能力

在 **RTX 3060 Laptop 6 GB** 上运行 `Qwen/Qwen2.5-0.5B` FP16，
覆盖一次请求从入队、Prefill、逐 token Decode 到 KV 回收的完整生命周期。
模型计算由自有 ModelRunner 编排；Hugging Face 用作 tokenizer 与外部 correctness oracle。
同卡交错测试中，三组固定负载的输出吞吐中位数比 vLLM 0.8.5 V1 高 **2.2%–14.1%**。
见 [测试配置与结果](#与-vllm-同机对照)。

| 层次 | 独立实现的能力 |
|---|---|
| **模型执行** | 直接加载 config/safetensors；自有 24 层 Qwen forward；RMSNorm、RoPE、QKV、GQA、SwiGLU、LM Head；显式 Prefill/Decode |
| **状态与显存** | 连续/Paged KV Cache；GPU block pool、CPU free list、每请求 block table；事务式追加、跨块增长、释放复用与 OOM 防护 |
| **CUDA Attention** | 手写 Decode Paged Attention：FP16、q_len=1、head_dim=64、GQA/MQA、非连续物理块、FP32 online softmax |
| **动态批处理** | 变长 packed Prefill、batched Decode、waiting/running/finished 状态机；Decode-priority 与 request/token/block budget |
| **验证与测量** | HF token/logits/KV 对拍；TTFT、TPOT、吞吐、尾延迟与显存；CUDA Event、raw samples、clean-tree release bundle |

线性层使用 PyTorch/cuBLAS，Prefill Attention 使用 PyTorch matmul 或 SDPA。
自定义 CUDA 算子覆盖 Paged Attention、RMSNorm、SwiGLU 和 RoPE/KV 写入，
保留 PyTorch 路径用于逐层对拍和 KV 写入验证。

## 运行架构

```mermaid
flowchart TD
    R["Request / Tokenizer"] --> S["Scheduler<br/>waiting · running · finished"]
    S -->|"step plan · token budget"| E["ContinuousBatchEngine"]
    S -->|"block-aware admission"| K["Paged KV Manager<br/>block pool · free list · block table"]
    E --> M["Qwen ModelRunner<br/>24 decoder layers → logits"]
    M --> P["Prefill<br/>packed prompt · per-request position"]
    M --> D["Decode<br/>one new token / running request"]
    P -->|"write prompt K/V"| K
    D -->|"append current K/V"| K
    K -->|"physical K/V · block tables · lengths"| A["CUDA Paged Attention<br/>GQA · online softmax"]
    D -->|"query"| A
    A -->|"attention output"| M
    M -->|"next tokens · one batched D2H"| E
    E -->|"apply results"| S
    E -->|"release finished blocks"| K
```

- **Prefill**：处理有效 prompt token，创建各层 KV，产出首 token。
  `auto` 对达到阈值的等长 prompt 使用 batched causal SDPA，对变长 prompt
  按形状选择 dense block-diagonal mask 或 segmented SDPA。变长组可选择
  packed Flash Attention 或自研 CUDA GQA。FP16 批量路径可用
  CUDA 融合 RMSNorm、SwiGLU 以及 vectorized KV 写入时的 RoPE。
- **Decode**：逐请求 position 延续，只计算新 token；每层一次 batched Paged Attention，
  从 block table 读取历史 KV。FP16 batched Decode 合并 Q/K RoPE 与 KV 写入，
  position、物理 slot 和 block table 共用一次 H2D 传输。
  Warp 并行 Attention 在单分区时直接输出，多分区时额外合并 FP32 softmax 状态。
- **Continuous Batching**：每 step 移除完成请求，并用剩余预算接纳 waiting 请求。
  Engine 同步执行 `Prefill → Decode → D2H → commit/release`。可选分块 Prefill
  为混合 step 保留 Decode 机会，并在最后一个 prompt chunk 完成后返回首 token。
- **Paged KV**：逻辑 token 与物理页解耦，完成时归还 blocks。
  Admission 保守预留请求整个生成生命周期所需 blocks，避免无抢占机制时中途 OOM。

KV 地址映射：

```text
logical_block = token_position // block_size
block_offset  = token_position % block_size
physical_block = block_table[request_id, logical_block]

K/V layout: [layer, physical_block, kv_head, block_offset, head_dim]
```

设计与验收见 [架构文档](docs/architecture.md)；
按真实源码跟读见 [源码导航](docs/reading-guide.md)。

## 实测结果

**RTX 3060 Laptop 6 GB（Ampere sm_86）、WSL2 Ubuntu 22.04、CUDA 12.4、
PyTorch 2.6.0+cu124、Qwen2.5-0.5B FP16。**

### 与 vLLM 同机对照

两边使用相同的预分词 token ID、greedy 解码和每请求固定输出 32 token。
每组预热 3 轮，正式交错测量 12 轮，轮换执行顺序。mini 启用 Prefill/Decode CUDA Graph，
vLLM 使用 0.8.5 V1、CUDA Graph，关闭前缀缓存。调度上限为 8 个运行请求和每步 4096 token。

| 并发 × 输入 token | mini 吞吐中位数 | vLLM 吞吐中位数 | mini 相对增幅 | 配对胜场 |
|---|---:|---:|---:|---:|
| 4 × 32 | **549.7 tok/s** | 481.8 tok/s | **+14.1%** | 12/12 |
| 8 × 256 | **668.0 tok/s** | 632.4 tok/s | **+5.6%** | 11/12 |
| 1 × 2048 | **80.6 tok/s** | 78.8 tok/s | **+2.2%** | 10/12 |

计时窗口从提交请求到引擎返回最后一个 token，跳过模型加载、预热、tokenizer 和 HTTP。
长上下文组的 TTFT 中位数为 146.5 ms（mini）和 144.0 ms（vLLM）。
结果适用于上述单 GPU、固定 burst 设置。采样运行于包含优化的本地工作区。

[12 轮原始样本与顺序](docs/benchmarks/vllm-interleaved.json) ·
[测量细节](docs/benchmarks/README.md#与-vllm-同机交错) · [复现命令](#复现-vllm-交错对照)

### 错峰到达与变长 Prefill

在同卡、相同 KV 池大小的错峰 trace 中，4 条短请求进入 Decode 后，再提交 3 条
128/512/1024-token 请求。自研 packed GQA 将后到请求 P95 TTFT 从 **125.2 ms**
降至 **123.4 ms**，输出吞吐从 **465.8** 升至 **467.7 tok/s**。
同轮 vLLM 0.8.5 V1 为 **111.5 ms** 和 **549.2 tok/s**。
[四后端结果与对拍](docs/benchmarks/README.md#错峰变长-prefill-对照) ·
[原始样本](docs/benchmarks/vllm-varlen-mixed.json)

同一错峰 trace 启用 512-token 分块后，首批请求最大 token 间隔的 12 轮中位数由
**107.7 ms** 降到 **46.2 ms**，达标吞吐为 **11.82 req/s**，同轮 vLLM 为
**5.95 req/s**。达标条件为 TTFT ≤ 200 ms 且请求的每个 token 间隔 ≤ 50 ms。
[分块预算、吞吐与延迟取舍](docs/benchmarks/README.md#分块-prefill-对照) ·
[分块原始样本](docs/benchmarks/vllm-chunked-prefill.json)

### Continuous vs Static Batching

8-request burst；generation lengths 循环 `2,4,8,12`；
max running `4`，token budget `64`，block size `16`；
warmup `3`，measured `10`，逐轮交错并反转 case 顺序。
采样源码：[`d31ded2`](https://github.com/Eran-ovo/mini-llm-runtime/tree/d31ded234702d9f51d419d8c2d200e51c5994950)。

| 指标 | Static | Continuous | 相对变化 |
|---|---:|---:|---:|
| Throughput | 65.43 tok/s | **71.70 tok/s** | **+9.58%** |
| TTFT median | 334.78 ms | **178.08 ms** | **−46.81%** |
| E2E median | 478.58 ms | **414.06 ms** | **−13.48%** |
| TPOT median | **24.06 ms** | 25.56 ms | +6.23%（变慢） |

动态 refill 改善排队和总吞吐；mixed Prefill 同时会干扰已有请求 Decode，
因此 TPOT 有代价。TTFT/TPOT/E2E 使用 CPU 请求事件时间线，
CUDA Event 单独保存 GPU timeline，二者没有混用。
该固定 workload 的结果不外推为生产流量或相对其他推理框架的结论。

[数据与统计口径](docs/benchmarks/README.md) ·
[原始样本摘录](docs/benchmarks/static-continuous.json) ·
[v1.0.0 完整证据归档](https://github.com/Eran-ovo/mini-llm-runtime/releases/download/v1.0.0/mini-llm-runtime-v1.0.0-release-evidence.tar.gz)

### Prefill Attention Backend

4 个 512-token prompt；两条路径均使用 vectorized KV write，
**只改变 Attention backend**。warmup `2`，measured `6`。
采样源码：[`7f4c844`](https://github.com/Eran-ovo/mini-llm-runtime/tree/7f4c8444c34b78732775f8df86582e5c7f4075ce)。

| 指标 | Dense masked | Segmented SDPA | 降幅 |
|---|---:|---:|---:|
| Prefill ModelRunner + KV write median | 372.129 ms | **136.475 ms** | **63.3%** |
| Peak allocated memory | 1453.53 MiB | **1055.99 MiB** | **27.3%** |

测量不含 tokenizer、输入构造和 Scheduler，不能当作端到端 TTFT。
该历史实验的 Segmented SDPA 按请求调用 attention，共享 packed projection/MLP。
变长 fused 路径的对照见 [错峰到达与变长 Prefill](#错峰到达与变长-prefill)。

[原始样本摘录](docs/benchmarks/prefill-4x512.json) ·
[Backend 原理与形状扫描](docs/prefill_attention_backends.md)

## 快速开始

已验证：Python 3.10、CUDA toolkit 12.4（含 `nvcc`）、GCC/G++ 11、
Ninja、PyTorch 2.6.0+cu124、RTX 3060 Laptop。
CUDA extension 首次使用时 JIT 编译，编译时间不进入 benchmark。

```bash
git clone https://github.com/Eran-ovo/mini-llm-runtime.git
cd mini-llm-runtime

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e '.[dev]' ninja
python scripts/check_environment.py
```

### 验证 Runtime 主链路

```bash
# 单元与 CUDA 测试：不下载模型；CUDA/NVCC 不可用时相关测试会 skip。
python -m pytest -q

# 真实 Qwen + HF oracle：首次下载权重，覆盖动态进出队、跨块与回收。
python -m experiments.continuous_batch_engine_runner \
  --output-dir benchmarks/results/engine_correctness

# 变长 Prefill、晚到请求、显存压力等待、物理块复用与全部回收。
python scripts/check_engine_ragged_continuation.py \
  --block-pressure-reuse \
  --output-dir benchmarks/results/block_pressure_correctness
```

权重已缓存时可追加 `--local-files-only`。参考模型先释放，再加载自有 Runner，
降低 6 GB GPU 上的同时驻留开销。`Qwen2.5-0.5B` 是 base model，输出用于
推理路径验证，并非 chat/instruct 效果演示。

### 复现正式实验

```bash
# 从当前 HEAD 创建 detached clean worktree：
# tests → HF correctness → Static/Continuous → tail benchmark。
python scripts/run_release_evaluation.py \
  --output-dir benchmarks/results/release_local

# 独立复现 Prefill backend 对照。
python scripts/benchmark_packed_prefill.py \
  --fixed-prompt-length 512 \
  --compare-segmented-sdpa --compare-vectorized-kv-write \
  --warmup 2 --repeats 6 \
  --output-dir benchmarks/results/prefill_local
```

在当前 HEAD 复现得到的是当前版本结果；复核历史数字须使用对应 source commit。
完整参数与归档语义见 [Release Evaluation](docs/release_evaluation.md)。

### 复现 vLLM 交错对照

在 mini 环境运行下列命令。脚本启动独立的 vLLM Python 进程，两个引擎同时驻留，
依次占用同一张 GPU。使用同一份本地 Qwen checkpoint。已验证的 vLLM 环境为
`vllm==0.8.5`、`transformers==4.51.3` 和 PyTorch 2.6。脚本记录逐轮样本、顺序、
吞吐、TTFT、TPOT、ITL 和生成 token 的 SHA-256。
[基准脚本](scripts/benchmark_vllm.py) · [vLLM GPU 安装文档](https://docs.vllm.ai/en/v0.8.5/getting_started/installation/gpu.html)

```bash
MODEL_DIR=/path/to/local/Qwen2.5-0.5B
VLLM_PYTHON=/path/to/vllm-venv/bin/python

OMP_NUM_THREADS=2 python scripts/benchmark_vllm.py --backend mini \
  --interleave-with "$VLLM_PYTHON" --model "$MODEL_DIR" \
  --mini-cuda-graph --mini-prefill-cuda-graph --gpu-memory-utilization 0.40 \
  --warmup 3 --repeats 12 --output benchmarks/results/vllm_interleaved.json
```

ModelRunner 通过 `enable_decode_cuda_graph=True` 和 `enable_prefill_cuda_graph=True`
分别启用 Decode 和 Prefill 图重放。Engine 的 Decode 图也包含 greedy argmax，
随后按批次回传 token ID。Prefill 图目前适用于总计不超过 512 token 的
等长、vectorized KV 写入批次。图按 KV 池和输入形状缓存，基准默认用两轮预热完成捕获。

### 变长 Prefill 与错峰请求

变长 FP16 prompt 的 Q/K/V 按 token 打包。一次因果 Attention 调用处理本层所有请求，
直接使用模型的 GQA head 映射。Engine 可通过 `prefill_attention_backend` 选择
`auto_varlen_sdpa`（PyTorch 2.6 packed Flash Attention）或 `auto_varlen_cuda`（自研 CUDA GQA）。
两种配置在符合 segmented SDPA 分发条件的变长批次上启用 packed 路径。
CUDA 路径需在当前 Python/PyTorch 环境构建带 `flashattention_varlen` 的 `ai_infra_ops`。
[CUDA 扩展源码](https://github.com/Eran-ovo/CUDA-Operators/tree/main/torch_ext)

```bash
OPS_DIR=/path/to/CUDA-Operators/torch_ext
(cd "$OPS_DIR" && MAX_JOBS=1 python setup.py build_ext --inplace)

PYTHONPATH="$OPS_DIR${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2 \
python scripts/benchmark_vllm.py --backend mini \
  --interleave-with "$VLLM_PYTHON" --model "$MODEL_DIR" \
  --mini-cuda-graph --mini-prefill-cuda-graph --gpu-memory-utilization 0.40 \
  --compare-mini-prefill-backend auto_varlen_sdpa \
  --compare-mini-prefill-backend auto_varlen_cuda \
  --mixed --kv-blocks 160 --ttft-slo-ms 200 --itl-slo-ms 50 \
  --warmup 3 --repeats 12 --output benchmarks/results/vllm_varlen.json
```

`--mixed` 增加一组错峰 trace。4 条短请求先进入，60 ms 后加入 3 条变长请求。
TTFT 从计划到达时刻计入排队和提交延迟。基准同时记录首批请求的 ITL、
后到请求的 TTFT，以及达到首 token 和逐 token 延迟阈值的请求吞吐。

### 分块 Prefill

通过 `RequestScheduler(prefill_chunk_tokens=512, ...)` 为已有 Decode 的 step 限制
Prefill token 数。已接纳的 prompt 在后续 step 沿用其物理 KV 页和绝对 position，
按 Q 与 KV 的各自边界计算因果 Attention。每步成功后提交 KV 和 prompt 进度。
分块中途失败可回滚本轮写入，从已提交的 prefix 重试。

```bash
PYTHONPATH="$OPS_DIR${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=2 \
python scripts/benchmark_vllm.py --backend mini \
  --interleave-with "$VLLM_PYTHON" --model "$MODEL_DIR" \
  --mini-cuda-graph --mini-prefill-cuda-graph --gpu-memory-utilization 0.55 \
  --mini-prefill-backend auto_varlen_cuda --compare-mini-prefill-chunk-tokens 384 \
  --compare-mini-prefill-chunk-tokens 512 --case 4x32x32 --case 1x2048x32 \
  --mixed --kv-blocks 160 --ttft-slo-ms 200 --itl-slo-ms 50 \
  --warmup 3 --repeats 12 --output benchmarks/results/vllm_chunked.json
```

单独运行 mini 时，用 `--mini-prefill-chunk-tokens 512` 启用相同预算。

## 正确性与工程质量

当前主分支的全量回归为 **302 passed**，两条变长 Prefill 路径均通过真实 Qwen 的 Hugging Face 对拍。
单元测试与真实模型 oracle 分开运行，普通回归可在本地独立执行。

| 验证层次 | 覆盖重点 |
|---|---|
| 模型数学 | 权重映射、RoPE、GQA、单层/整模型 logits、greedy token |
| KV 生命周期 | reserve/write/commit、跨块追加、OOM 原子失败、double-free 防护、释放复用 |
| CUDA Attention | SDPA/Python reference、非连续物理块、变长 batch、GQA/MQA |
| 调度与 Engine | late admission、mixed step、token 行归属、outstanding batch、失败回滚 |
| 证据管理 | warmup、median、raw samples、峰值显存、环境、commit、clean-tree、SHA-256 |

**有价值的负结果也保留。** Mixed-prefill budget 限制改善部分 TPOT，
却损害吞吐和尾延迟，默认关闭；split-KV 的分区/合并开销与 profiler 边界
保存在实验文档。稳定入口不自动采用未经端到端验证的实验策略。

[证据索引](docs/evidence_index.md) ·
[正确性 Gate](docs/continuous_batch_correctness_gate.md) ·
[split-KV 实验](docs/split_kv_experiment.md) ·
[Mixed Prefill 取舍](docs/mixed_prefill_budget_benchmark.md)

## 源码与文档入口

```text
src/mini_llm_runtime/
  qwen_loader.py / qwen_model_runner.py    # 权重加载、24 层前向、Prefill/Decode
  paged_kv_cache.py / paged_kv_manager.py   # block pool、表、事务与生命周期
  scheduler.py / block_admission.py        # 状态机、预算与准入
  engine.py / request_metrics.py           # 编排、回收与请求时间线
csrc/                                     # PyTorch 绑定、Paged Attention CUDA
tests/                                    # 分层/边界/集成回归
scripts/                                  # 环境、正确性与正式 benchmark
experiments/                              # 教学、对拍与优化消融
docs/                                     # 架构、证据、教程和失败分析
```

- **快速读懂项目**：[源码导航与阅读路线](docs/reading-guide.md)。
- **学习实现过程**：[分阶段开发指南](docs/development-guide.md)。
- **理解系统数据流**：[交互式可视化源码](docs/vllm_runtime_flow.html)。
- **从基础到代码**：[13 页 HTML 教程](docs/tutorial/index.html)。

HTML 文档可在本地直接打开，或从仓库根目录运行 `python -m http.server 8000`，
访问 `http://localhost:8000/docs/tutorial/`。GitHub 文件页提供源码浏览。

## 当前边界与后续方向

单 GPU、FP16、greedy。Decode CUDA kernel 当前支持 `head_dim=64`，
按 batch 和上下文形状选择基础 Paged Attention 或 split-KV。
同步 Engine 采用 whole-prefill、strict FIFO 和保守全生命周期 block reservation。
尚不支持 chunked prefill、preemption、异步 CPU/GPU overlap、prefix caching、
量化、分布式推理或 Speculative Decoding。

后续优先建立 Paged Decode 的分层 profile 与优化闭环，
再研究 chunked prefill 和增量 KV admission。所有收益都须重新通过
correctness 和正式 benchmark。

**相关项目**：[CUDA Operators](https://github.com/Eran-ovo/CUDA-Operators) —
Tensor Core GEMM、FlashAttention 与 Norm 算子的手写实现及 Nsight Compute 优化证据。
