# Mini LLM Runtime

面向单 GPU 的轻量级 LLM 推理引擎学习项目。目标模型为
`Qwen/Qwen2.5-0.5B`，主线是从可信的 Hugging Face reference 出发，逐步实现
ModelRunner、KV Cache、Paged Attention 和 Continuous Batching。

当前阶段：**v0.6 Continuous Batching correctness baseline（进行中）**。已具备独立权重
加载、Qwen ModelRunner、连续与 Paged KV Cache、Decode CUDA Paged Attention，以及
单请求多 token 生成闭环；当前正在固定请求状态机和调度语义，尚未接入 batched Runner。

## 架构主线

```text
Request / Tokenizer
        |
        v
Scheduler ---- token/block budget ----> KV Cache Manager
        |                                      |
        v                                      v
ModelRunner: embedding -> decoder layers -> logits
                            |
                            +-> Prefill: prompt Q/K/V -> 写入 KV Cache -> first token
                            +-> Decode: 1-token Q/K/V -> 追加 KV -> Paged Attention
```

- **Prefill** 一次处理 prompt，计算密度较高，产出首 token，并初始化各层 KV。
- **Decode** 每步只处理新 token，读取历史 KV，通常更受显存带宽和 launch 开销影响。
- **Paged KV Cache** 用固定大小 block 承载 KV；每个请求的 block table 将逻辑 token
  映射到物理 block，避免为最大长度预留连续空间。
- **Paged Attention** 按 block table 间接读取 K/V，并用 online softmax 避免物化完整
  attention matrix。
- **Continuous Batching** 在每个 step 接纳新请求、移除完成请求，并通过 token/block
  budget 控制工作集；释放请求时把物理 block 归还 free list。

详细阶段设计见 [docs/architecture.md](docs/architecture.md)。

## 环境

本项目不绑定私人虚拟环境路径。当前机器可使用已有环境：

```bash
source /home/eran/venvs/torch/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_environment.py
```

环境脚本输出 JSON，包括 GPU/驱动、可用显存、CUDA、PyTorch、Python、编译器和
Git commit。显存空闲量是瞬时值，不应写成固定性能结论。

## Hugging Face baseline

baseline 不调用 `transformers.generate()`。它显式执行：

1. `prefill(input_ids)`：完整 prompt forward，返回 logits 与 `past_key_values`；
2. `decode_one(token, past_key_values)`：只输入一个新 token；
3. 手写 greedy loop：首 token 来自 Prefill logits，后续 token 来自逐 token Decode。

运行真实模型（首次会从 Hugging Face 下载权重）：

```bash
python scripts/run_hf_baseline.py \
  --prompt '请用一句话解释 KV Cache。' \
  --max-new-tokens 8 \
  --warmup 2 \
  --repeats 10 \
  --output-dir benchmarks/results/hf_smoke
```

输出包括：

- `result.json`：生成 token、文本、TTFT、TPOT、tokens/s、raw samples、peak memory
  与完整环境元数据；
- `logits.pt`：Prefill 最后位置以及每个 Decode step 的完整 logits，用于后续对拍。

这里的 TTFT 是 tokenizer 之后的 GPU Prefill + argmax 时间；TPOT 是固定 context
length 的单步 Decode + argmax 时间。每轮 Decode 都在计时区间外重建 cache，避免
不同轮次因 cache 被原地扩展而测到不同 shape。

## 测试

```bash
pytest
```

快速测试不下载模型，使用小型 fake causal LM 验证 Prefill/Decode 的边界和手写
greedy 数据流。真实模型 smoke test 由上面的 CLI 单独执行。

## 交互式学习实验

下面的脚本使用真实 Qwen 第 0 层展示 Prefill/Decode 的 Q/K/V、GQA Head 映射、
Attention probability 和 Dynamic KV Cache 增长：

```bash
python experiments/attention_walkthrough.py --local-files-only
```

它会强制 eager attention 并注册 forward hook，因此只用于学习和正确性观察，不能
用于性能 benchmark。

进一步使用真实第 0 层权重，从基本 PyTorch 算子手写完整 Prefill Attention：

```bash
python experiments/manual_qwen_attention.py --local-files-only
```

该实验手动实现 Q/K/V Projection、RoPE、Causal Mask、GQA、FP32 Softmax、Head
合并和 Output Projection，并逐检查点与 Hugging Face eager reference 对拍。

继续观察“有状态”的推理：为第 0 层预分配连续 K/V buffer，Prefill 批量写入，
Decode 只追加一个位置，并与 Hugging Face DynamicCache 对拍：

```bash
python -m experiments.manual_contiguous_kv_cache --local-files-only
```

该版本有意只支持单请求、单层、无 padding 和固定容量，用来隔离 Cache 的写入顺序、
有效长度、容量边界及历史前缀不变性；它不是稳定 runtime API。

把 Attention 与 Qwen 的两次 RMSNorm、两条 Residual 和 SwiGLU MLP 组合成完整第 0
个 Decoder Layer，并逐检查点对拍：

```bash
python -m experiments.manual_qwen_decoder_layer --local-files-only
```

最后将同一套手写层逻辑堆叠 24 次，加上 Embedding、Final RMSNorm 和 tied LM Head，
形成只支持无 padding Prefill 的完整教学版 ModelRunner：

```bash
python -m experiments.manual_qwen_model_runner --local-files-only
```

将配置和权重映射提升到稳定 `src/` 后，可运行不持有 Hugging Face 模块对象的
`QwenPrefillRunner` 集成对拍：

```bash
python -m experiments.independent_qwen_model_runner --local-files-only
```

最后绕过 `AutoModelForCausalLM`，直接从 `config.json` 和单文件/分片 safetensors
构造 Candidate；实验会先释放 HF Reference，再加载自有 Runner，适合 6 GB GPU：

```bash
python -m experiments.direct_safetensors_runner --local-files-only
```

## 连续 KV Cache

v0.3 从稳定的多层连续 Cache 数据结构开始。它预分配
`[layer, batch, kv_head, capacity, head_dim]` 的 K/V buffer，并用
`begin_append → write_layer → commit_append` 保证 24 层全部写完后才推进全局长度。
当前实现固定 batch、等长请求，并已接入 ModelRunner 的 Prefill 与单 token Decode。

Prefill 集成实验会在 24 层中逐层写入旋转后的 K 和原始 V，并与 Hugging Face
`past_key_values` 对拍：

```bash
python -m experiments.prefill_kv_cache_runner --local-files-only
```

单 token Decode 会使用追加前的 Cache 长度作为 RoPE position，只计算当前 token 的
Q/K/V，再让当前 Q 读取完整历史 K/V。下面的实验与 Hugging Face 对拍增长后的 24 层
Cache 和 Decode logits：

```bash
python -m experiments.decode_kv_cache_runner --local-files-only
```

在此基础上，`greedy_generate` 用一次 Prefill 和最多 `max_new_tokens - 1` 次
Decode 组成完整的单请求生成循环，并处理 EOS、最大位置与 Cache 容量。逐 step
logits 和最终 token 序列可用下面的真实模型实验对拍：

```bash
python -m experiments.greedy_generation_runner \
  --max-new-tokens 8 \
  --local-files-only
```

正式比较“每步完整重算”和“连续 KV Cache”的固定长度生成路径：

```bash
python scripts/benchmark_kv_cache.py \
  --max-new-tokens 16 \
  --warmup 5 \
  --repeats 20 \
  --local-files-only \
  --output-dir benchmarks/results/kv_cache_v03
```

两条路径按轮交错，并在奇偶轮反转先后次序，以降低 Laptop GPU 温度、频率和功耗
漂移造成的顺序偏差。结果目录包含保存全部 latency/memory 原始样本与环境信息的
`result.json`，以及便于阅读的 `report.md`。

## Paged KV Cache 元数据

v0.4 先实现地址管理层：`FixedBlockAllocator` 管理固定数量的物理 block ID 和
free list，`RequestBlockTable` 保存单个请求从逻辑 block 到物理 block 的映射。
逻辑 token `t` 通过 `t // block_size` 选择 block table 项，再通过
`t % block_size` 得到块内 offset。`PagedKVStorage` 进一步预分配布局为
`[layer, physical_block, kv_head, block_offset, head_dim]` 的 K/V tensor，并提供
事务式写入与仅供 correctness 使用的逻辑连续 gather。`PagedKVCacheManager` 统一
管理 request registry，并按 Scheduler 指定顺序生成带 `-1` padding 的 GPU int32
block table、sequence lengths 和碎片统计。

当前版本覆盖跨块增长、OOM 原子失败、请求释放、物理块复用、double-free 防护，
以及 GPU 物理 block 的写入/gather 对拍；ModelRunner Prefill 与单 token Decode
已可通过逐层 adapter 写入非连续物理块。单请求 Decode 可显式选择 `paged_cuda`
backend，直接读取物理 Cache，不再经过连续 K/V gather。

```bash
python -m experiments.paged_block_table_walkthrough
python -m experiments.paged_kv_storage_walkthrough
python -m experiments.paged_cache_manager_walkthrough
python -m experiments.paged_qwen_prefill_runner --local-files-only
python -m experiments.paged_qwen_decode_runner --local-files-only
python -m experiments.paged_greedy_generation_runner \
  --max-new-tokens 8 --block-size 3 --local-files-only
```

单 token 入口会强制第一次 Decode 跨 block，并逐层比较 CUDA Attention 与独立
Python reference。事务顺序、metadata 复用、三层正确性标准和真实 Qwen 结果见
[ModelRunner Paged Decode 集成记录](docs/model_runner_paged_decode.md)。
多 token 入口进一步覆盖块内复用、反复跨块、EOS 和容量预检，结果与状态不变量见
[多 token Paged CUDA Decode 记录](docs/paged_greedy_decode.md)。

在编写 CUDA kernel 前，先运行 Decode-only PyTorch Paged Attention reference。该实现
直接按 block table 读取非连续物理 K/V，支持变长 batch 和 GQA，并与 gather 后的连续
Attention 数学结果对拍：

```bash
python -m experiments.paged_attention_reference_walkthrough
```

Reference 会显式物化 score/probability，并包含 Python loop 与 CUDA 同步，只用于定义
正确语义，不能用于 benchmark。未来 CUDA kernel 必须保持相同的地址映射和跨 block
Softmax 结果，但会用 online softmax 避免保存完整 attention vector。

第一版 CUDA correctness kernel 采用一个 CTA 处理一个 `(request, query_head)`，支持
Qwen2.5-0.5B 所需的 FP16、`head_dim=64`、GQA/MQA 和变长 batch。首次调用会通过
PyTorch JIT extension 编译，产物进入用户级 cache：

```bash
python -m pytest -q tests/test_paged_attention_cuda.py
```

该版本逐 token 串行扫描，并使用 shared-memory reduction 与 FP32 online softmax；
它用于验证 CUDA 地址映射与数值语义，尚未进行 warp reduction、向量化加载或 token
并行，不能作为最终性能数据。

优化前先建立 v1 baseline。benchmark 在计时前验证一次固定 metadata，计时内使用
unchecked hot path；Paged 路径直接读取打散的物理 block，SDPA 路径使用预先准备的
连续 K/V，且不把 gather 算入 SDPA 时间：

```bash
python scripts/benchmark_paged_attention.py \
  --batch-sizes 1,8 \
  --sequence-lengths 16,128,512,2048 \
  --block-size 16 \
  --warmup 5 \
  --repeats 20 \
  --iterations-per-sample 20 \
  --output-dir benchmarks/results/paged_attention_v1
```

每个 sample 用 CUDA Event 包围多次 launch，再除以迭代次数，以降低微秒级 kernel
的测量噪声。两条路径按轮交错，并在奇数轮反转先后顺序。JIT 编译、输入构造和一次性
metadata 验证都在 warmup/计时区间之外。

CUDA Event 包围 Python launch 循环时可能包含 GPU 等待 host 提交的空隙，尤其需要
谨慎解释短 SDPA 时间。单 kernel 的 Nsight Compute 采集入口、原始报告位置与
瓶颈分析见 [v1 profiler 学习记录](docs/paged_attention_v1_profile.md)。

沿 KV 序列拆分 CTA 的实验实现、online-softmax 状态合并和两轮对照数据见
[split-KV 实验记录](docs/split_kv_experiment.md)。该入口留在 `experiments/`，
稳定 runtime 暂不自动切换到 split-KV。

固定 B=1/N=2048 的分区数扫描与原始样本说明见
[split-KV 分区扫描](docs/split_kv_sweep.md)。
S=32/64 的 partial/merge 硬件指标、profiler 与 Event 测量边界以及停止继续扫参的决策见
[split-KV profiler 学习记录](docs/split_kv_profile.md)。

## Scheduler 状态机

v0.6 的第一步是纯 CPU、同步的请求 Scheduler。它使用 Decode-priority、whole-prefill、
strict-FIFO baseline，在每个 step 按 `max_batch_tokens` 和
`max_running_requests` 动态组成 Prefill/Decode batch：

```bash
python -m pytest -q tests/test_scheduler.py
python -m experiments.scheduler_walkthrough
```

Scheduler 只维护 waiting/running/finished 状态并产生完成事件；Engine 消费事件后才让
KV Cache Manager 释放物理块。策略取舍、outstanding batch 约束和动态进出队示例见
[Scheduler 状态机学习记录](docs/scheduler_state_machine.md)。

保守的 block-aware baseline 会在接纳时按
`prompt_length + max_new_tokens - 1` 预留完整生命周期 blocks，防止尚无 preemption
机制时 Decode 中途 OOM。它会牺牲可接纳请求数，因此不是最终策略：

```bash
python -m pytest -q tests/test_block_admission.py
python -m experiments.block_aware_scheduler_walkthrough
```

原子预留、资源阻塞、完成释放及其利用率代价见
[Block-aware Admission 学习记录](docs/block_aware_admission.md)。当前尚未实现 chunked
prefill、按需增长/preemption 或真实 GPU Continuous Batch。

多请求 Decode adapter 已能按 Scheduler 指定顺序，为不同历史长度的请求原子追加一枚
K/V，构造 padded GPU block table，并让一次 CUDA Paged Attention 与逐请求 CUDA/Python
reference 对拍：

```bash
python -m pytest -q tests/test_paged_batch.py
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.paged_batch_decode_walkthrough
```

Batch row、RoPE position 与可读 length 的区别、跨请求事务回滚和 padding 规则见
[多请求 Paged Decode Batch Adapter](docs/paged_batch_decode.md)。Adapter 现已接入完整
Qwen ModelRunner：变长请求共享 batched QKV/MLP，并在每层只调用一次 Paged Attention。

```bash
python -m pytest -q tests/test_batched_qwen_decode.py
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.batched_qwen_decode_runner \
  --local-files-only
```

数据流、逐请求 position、跨层事务边界与对拍口径见
[Qwen 多请求 Batched Decode](docs/batched_qwen_decode.md)。当前 Scheduler 尚未自动驱动
ModelRunner，也尚未形成真实 Continuous Batching benchmark。

在固定的纯 KV Cache 显存预算下，下面的确定性模拟会让连续预留和不同 block size
处理同一批 FIFO 请求，并输出接纳请求数、block/预留区利用率、slot 利用率和内部碎片：

```bash
python scripts/analyze_kv_cache_capacity.py \
  --cache-budget-mib 64 \
  --max-sequence-length 2048 \
  --block-sizes 1,4,8,16,32,64 \
  --num-requests 128 \
  --seed 2027 \
  --output-dir benchmarks/results/paged_capacity_v04
```

这里的 request length 表示请求需要驻留在 Cache 中的总 token 数。脚本只做 K/V
tensor storage 的整数容量分析，不计 block table/Python allocator metadata，也不运行
GPU kernel，因此其结果不能用于声称 Paged Attention 更快，不需要 CUDA Event 或
warmup。原始请求长度、首个被拒请求、完整配置、环境与 Git commit 会保存在
`result.json` 中。

## 目录

```text
mini-llm-runtime/
├── src/mini_llm_runtime/   # 可复用 baseline、计时与环境采集
├── scripts/                # 环境检查和真实模型入口
├── tests/                  # 无网络快速正确性测试
├── benchmarks/results/     # 正式结果（默认不提交）
├── docs/                   # 架构与阶段验收标准
├── pyproject.toml
└── README.md
```

## Benchmark 纪律

任何可对外引用的数据都必须来自固定配置的多轮测试：包含 warmup、CUDA Event、
median、原始样本、peak memory、GPU/驱动/CUDA/PyTorch 和 Git commit。Laptop GPU
还需记录功耗/频率状态，并至少重复测试，不能依据单次结果下结论。
