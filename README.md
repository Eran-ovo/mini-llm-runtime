# Mini LLM Runtime

面向单 GPU 的轻量级 LLM 推理引擎学习项目。目标模型为
`Qwen/Qwen2.5-0.5B`，主线是从可信的 Hugging Face reference 出发，逐步实现
ModelRunner、KV Cache、Paged Attention 和 Continuous Batching。

当前里程碑：**v0.3 连续 KV Cache**。已具备独立权重加载、Qwen ModelRunner、
Prefill、单 token Decode、单请求 greedy generation，以及有/无 Cache 的正式 benchmark。

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
