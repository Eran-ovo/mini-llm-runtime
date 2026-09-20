# Continuous Batching 端到端 Correctness Gate

## 为什么已有单元测试仍然不够

Paged Attention、KV Manager、Scheduler 和 Engine 都有独立单元测试，但模块分别正确不自动
推出集成后正确。典型系统性错误包括：

- batched Decode 的第 `i` 行绑定到错误 request ID；
- block table 行顺序和 token 输出顺序不一致；
- 一个请求释放后，复用同一物理 block 的新请求读到旧 KV；
- mixed step 中 Prefill/Decode 的结果写回发生串位；
- sequence length 或 RoPE position 在 block 边界上偏移一位；
- 所有 batching policy 共用同一个错误，因此彼此 token 一致却仍然错误。

所以 gate 必须使用 Runtime 外部的 oracle，而不能只比较不同 Scheduler 配置。

## Oracle 选择

参考路径使用 Hugging Face Qwen，并显式执行：

```text
HF Prefill(prompt, use_cache=True)
  -> greedy argmax
  -> HF Decode(token, past_key_values)
  -> greedy argmax
  -> ...
```

没有调用 `transformers.generate()`，因此 Prefill、逐 token Decode 和 KV Cache 的数据流仍然
可见。HF 模型使用 FP16、eager attention；Runtime 使用相同 checkpoint 和 tokenizer，底层
Decode 则走自研 Paged Attention CUDA kernel。

Greedy decoding 对每一步 logits 执行 argmax。如果任意一步选出不同 token，后续输入和 KV
都会分叉，因此 gate 比较完整 token sequence，并在首个差异出现时失败。这里要求 token ID
exact match，而不是给 token ID 设置浮点 tolerance。

## 固定 workload 覆盖的执行路径

```text
step 0: Prefill A, B
        B 只生成一个 token，完成并释放 block 2

step 1: Decode A + Prefill C
        mixed step；C 得到 blocks [2,3]，复用 B 的 block 2

step 2: Decode A, C
        batched Decode batch size=2；A/C 完成并释放所有 blocks
```

block size 为 4。C 的 prompt 有 7 个 token，因此实际 cache snapshot 中出现
`token_count=7 > block_size=4`，不是仅根据输入长度推断跨 block。A、C 的 block table 行与
batched Decode 行顺序也会进入真实 CUDA 路径。

## Gate 条件

所有条件必须同时成立：

1. 每个请求的 Engine token sequence 与 HF exact match；
2. step 0 之后发生新 Prefill admission；
3. 至少一个 step 同时包含 Prefill 和 Decode；
4. 至少一次 Decode batch size >= 2；
5. 实际 cache snapshot 中观察到 sequence 跨越 block boundary；
6. 后续请求实际复用先前释放的物理 block ID；
7. 结束后 request registry、reservation 和 allocator 全部归零/归还。

只要 workload 因代码变化不再命中某条路径，即使 token 恰好匹配，gate 也会失败。这避免了
测试悄悄退化成只跑 Prefill 或 batch size=1。

## Artifact

`result.json` 保存：

- checkpoint、dtype、block size；
- prompt、token IDs、generation length；
- 每个 step 的 Prefill/Decode/finished request IDs；
- step 后每个活跃请求的物理 block IDs、token count 和 capacity；
- released blocks、allocator/cache stats；
- HF/Engine 完整 token sequence；
- 每条 gate check；
- Python、PyTorch、CUDA、GPU 和 Git commit 信息。

请求时间线也会保存，但标记为 debug only。Correctness run 没有 warmup、多轮样本或公平的
计时控制，因此其中的 TTFT/TPOT 绝不能作为性能数据。

## 能证明与不能证明的事

它能证明当前固定短请求 workload 下，HF 与 Runtime 的 greedy token 一致，并且动态加入、
mixed step、batched Paged Decode、block boundary、物理复用和最终释放都实际发生。

它不能证明：

- 所有 prompt length、block size、batch size 都正确；
- logits 在数值上逐元素一致；
- EOS、多种采样策略或异常回滚路径正确；
- 长上下文下没有累计误差；
- 性能优于 Hugging Face。

这些分别属于参数化 kernel tests、逐层/logits 对拍、异常路径测试、长上下文测试和正式
benchmark，不能从一个端到端 gate 过度外推。

正式运行：

```bash
source /home/eran/venvs/torch/bin/activate
TORCH_CUDA_ARCH_LIST=8.6 python -m experiments.continuous_batch_engine_runner \
  --local-files-only \
  --output-dir benchmarks/results/continuous_batch_correctness_v06
```
