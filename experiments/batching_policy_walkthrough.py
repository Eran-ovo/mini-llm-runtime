#!/usr/bin/env python3
"""用相同请求和输出对比 Static 与 Continuous Batching 的组批差异。"""

from mini_llm_runtime.scheduler import BatchingPolicy, RequestScheduler


def run(policy: BatchingPolicy) -> list[tuple[int, tuple[str, ...], tuple[str, ...]]]:
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=6,
        batching_policy=policy,
    )
    scheduler.submit("A", (1, 2, 3), max_new_tokens=3)
    scheduler.submit("B", (4, 5), max_new_tokens=1)
    next_tokens = {
        "A": iter((10, 11, 12)),
        "B": iter((20,)),
        "C": iter((30, 31)),
    }
    records: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
    submitted_c = False

    while scheduler.has_unfinished_requests:
        batch = scheduler.schedule_step()
        if batch is None:
            raise RuntimeError("确定性演示不应出现无法取得进展的 step")
        records.append(
            (
                batch.step_index,
                batch.prefill_request_ids,
                batch.decode_request_ids,
            )
        )
        results = {item.request_id: next(next_tokens[item.request_id]) for item in batch.items}
        scheduler.apply_step_results(results)
        if not submitted_c:
            # 两种策略都在相同逻辑边界看到 C 到达。
            scheduler.submit("C", (6,), max_new_tokens=2)
            submitted_c = True
    return records


def main() -> None:
    for policy in (BatchingPolicy.CONTINUOUS, BatchingPolicy.STATIC):
        print(f"===== {policy.value} =====")
        for step, prefill, decode in run(policy):
            print(f"step={step}: prefill={prefill}, decode={decode}")
        print()


if __name__ == "__main__":
    main()
