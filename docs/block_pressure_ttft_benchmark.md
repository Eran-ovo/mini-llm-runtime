# Block pool 压力对晚到请求 TTFT 的影响

## 假设与单变量

Admission 在请求接纳时预留完整生成周期需要的 KV blocks。首批四请求
`(128,128,128,1)` 共占 28 个 block；晚到的 7-token 请求另需 1 个。
只把 block pool 从 29 改为 28，晚到请求就会在旧请求释放前留在 waiting。
假设是：此等待会增加它的 queue wait 和 TTFT；不改 Scheduler、模型、
prompt、生成长度、block size 或请求到达的 **step 边界**。

首条 128-token 请求生成 2 token，其余首批请求生成 3 token，晚到请求生成
2 token。晚到请求总在第 0 个 Engine step 返回后立即提交。28-block case
在第 1 步仍等待，第 2 步复用已释放 block；29-block case 在第 1 步即可接纳。
这两个 case 的计算路径/step 数因 admission 结果而不同，正是要测的机制，
不能把两者的每步 CUDA 时间直接解释为同一算子性能差异。

## 测量口径

- 晚到请求 TTFT：`engine.submit()` 的 CPU 单调时钟 arrival，到首 token
  经 D2H 同步后可见的时间；包含排队、调度与模型执行。
- Queue wait：arrival 到 Prefill 真正开始；可解释 TTFT 的排队部分，
  但不能把两组中位数之差当作严格的时延分解。
- 每个 Engine step 用 CUDA Event 记录 device timeline；它不是 TTFT，
  也不是 kernel-only 时间。记录模型/Cache 在场时的 PyTorch peak allocated。
- 两个 case 每轮交错，奇偶轮反转先后顺序。每组 warmup 3、measured 10；
  保存每轮原始 timing、CUDA Event、调度轨迹和请求事件。先要求两组
  greedy token 全部等于独立 HF 显式 Prefill/Decode reference，且符合
  预期等待/复用路径，否则停止报告。
- 两组每轮的 TTFT 差值先配对计算，再对 10 个差值取 median；它与
  “两组 TTFT median 相减”不是同一个统计量。

## 正式结果

Qwen2.5-0.5B FP16、RTX 3060 Laptop、PyTorch 2.6.0+cu124，干净提交
`2c6a9bac774f4169393efc465cd1cba59e8d0618`。两次独立运行的全部
correctness gate 通过，20/20 个配对差值为正：

| Clean-tree run | 28-block TTFT median | 29-block TTFT median | 28-block queue wait median | 29-block queue wait median | 配对 TTFT 差值 median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 第一次 | 76.73 ms | 52.46 ms | 26.23 ms | 0.11 ms | 23.60 ms |
| 独立重复 | 76.88 ms | 52.05 ms | 26.29 ms | 0.11 ms | 24.82 ms |

原始结果（默认不进 Git）保留每个 trial 的全部样本、step CUDA Event、峰值
显存、GPU/CUDA/PyTorch 及 Git 状态：

- `benchmarks/results/block_pressure_ttft_clean_2c6a9ba/result.json`
- `benchmarks/results/block_pressure_ttft_repeat2_clean_2c6a9ba/result.json`

本实验支持的有限结论是：在这个固定到达边界与模型形状下，保守预留使
晚到请求多等待一个旧请求的 Decode step，TTFT 随之增加。它不说明
所有在线到达分布的吞吐/公平性，也不能据此决定改用增量分配；那会引入
Decode 中途 OOM、抢占或回滚问题，需另设正确性与性能实验。

已对这两份原始结果进一步分解“未来预留 block”与“当前块尾部空位”，
并检查不改变下一 step 执行语义时的容量下界，见
`docs/kv_reservation_analysis.md`。该分析没有新增任何时延测量。

复现：

```bash
python scripts/benchmark_block_pressure_ttft.py --local-files-only \
  --warmup 3 --repeats 10 \
  --output-dir benchmarks/results/<new-run-name>
```
