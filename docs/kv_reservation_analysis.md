# KV 预留容量与已提交长度的分解

## 先区分三个量

当前 `PagedKVCacheManager` 的 block pool 在创建时已经整体分配为 GPU tensor。
Admission 把其中的物理 block ID 预留给请求；`token_count` 则只统计已提交、
可被 Attention 读取的 token。这两种“分配”不能混为一谈：释放请求的 block ID
会增加可接纳容量，但**不会**使固定大小的 GPU pool tensor 立刻缩小。

对每条活动请求，令 `B` 为 block size，`N` 为已提交 token，`A` 为已预留
block 数。当前长度至少需要 `ceil(N/B)` 块，于是：

- 未来预留 block：`A - ceil(N/B)`；
- 当前最后一个必要 block 的尾部空位：`ceil(N/B) * B - N`；
- 未提交 slot：`A * B - N = 未来预留 block * B + 尾部空位`。

尾部空位虽未写入 KV，但属于当前请求的物理 block，不能作为独立 block
接纳另一请求。未来预留 block 是 allocator 层面的 headroom，也不能直接
称为“浪费”：当前调度可能在下一步就需要它们。

## 固定场景的阶段快照

`scripts/analyze_kv_reservation.py` 对此前两份 clean-tree TTFT benchmark
原始 JSON 做确定性派生，不重新运行模型，也不产生新的时延样本。两份来源
各 10 轮的容量轨迹一致。28-block 压力 case：

| Step 结束后 | 活动预留 block | 已提交 token | 当前长度最少 block | 未来预留 block | 当前块尾部空位 slot | free block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0：首批 Prefill | 28 | 385 | 25 | 3 | 15 | 0 |
| 1：第一条长请求完成 | 19 | 260 | 19 | 0 | 44 | 9 |
| 2：晚到请求 Prefill | 1 | 7 | 1 | 0 | 9 | 27 |
| 3：全部完成 | 0 | 0 | 0 | 0 | 0 | 28 |

Step 0 的 448 个已预留 slot 中有 63 个未提交：48 个来自三块未来预留，
15 个是 1-token 请求的当前 block 尾部空位。后者并不等于可接纳容量。

## 为什么简单改成增量分配仍不够

反事实只改变物理 block 的预留时机，**保持同一个下一 step 的请求集合与
当前 Engine 执行顺序**。Step 0 若只分配已提交长度所需的 25 块，
28-block pool 会留下 3 块。但 Step 1 的旧请求 Decode 同时需要增长 3 块
（三条 128-token 请求均需写入第 129 个 token）；晚到请求 Prefill 另需 1 块，
合计 4 块，仍短缺 1 块。29-block pool 下可用 4 块，恰好满足。

当前 Engine 的 mixed step 先运行新请求 Prefill，再运行旧请求 Decode；
完成事件与释放发生在本 step 末。因此不能提前花掉最后一块、寄望于
“即将完成”的旧请求在同一步中释放。若想改变结果，需要明确的调度重排、
抢占或其他容量保护语义，并重新证明正确性；单纯去掉 admission 预留会
引入 Decode 中途 OOM 风险。

## 证据与边界

来源是 source commit `2c6a9ba` 的两份正式 benchmark JSON；派生脚本 commit
`e370dd59785007f27308c35d2685a8afa549a683`。每份分析保存来源
SHA-256、分析 Git 状态与 20 条逐 case/轮次的推导：

- `benchmarks/results/kv_reservation_derived_run1_clean_e370dd5/result.json`
- `benchmarks/results/kv_reservation_derived_run2_clean_e370dd5/result.json`

完整测试 `267 passed, 1 warning`。这是固定 workload 和当前 Engine
语义的容量推算，不是增量分配的实测性能，也不代表其他请求长度分布。

复现示例：

```bash
python scripts/analyze_kv_reservation.py \
  --source benchmarks/results/block_pressure_ttft_clean_2c6a9ba/result.json \
  --output-dir benchmarks/results/<new-analysis-name>
```
