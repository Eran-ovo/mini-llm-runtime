#!/usr/bin/env python3
"""确定性演示 mixed Prefill budget 如何把 refill 分散到多个 Decode step。"""

from dataclasses import dataclass

from mini_llm_runtime.scheduler import RequestScheduler


@dataclass(frozen=True)
class StepRecord:
    step: int
    prefill: tuple[str, ...]
    decode: tuple[str, ...]
    total_tokens: int


def run(max_mixed_prefill_tokens: int | None) -> list[StepRecord]:
    scheduler = RequestScheduler(
        max_running_requests=4,
        max_batch_tokens=20,
        max_mixed_prefill_tokens=max_mixed_prefill_tokens,
    )
    scheduler.submit("A", (1, 2, 3, 4), max_new_tokens=4)
    scheduler.submit("B", (5, 6, 7, 8), max_new_tokens=1)

    # token 值本身不重要；长度决定每个请求何时完成。
    next_tokens = {
        "A": iter((10, 11, 12, 13)),
        "B": iter((20,)),
        "C": iter((30,)),
        "D": iter((40,)),
    }
    records: list[StepRecord] = []
    submitted_refill = False

    while scheduler.has_unfinished_requests:
        batch = scheduler.schedule_step()
        if batch is None:
            raise RuntimeError("确定性演示不应出现无法取得进展的 step")
        records.append(
            StepRecord(
                step=batch.step_index,
                prefill=batch.prefill_request_ids,
                decode=batch.decode_request_ids,
                total_tokens=batch.token_count,
            )
        )
        scheduler.apply_step_results(
            {
                item.request_id: next(next_tokens[item.request_id])
                for item in batch.items
            }
        )

        if not submitted_refill:
            # 两次实验都在完全相同的 step 边界看到 C、D 到达。
            scheduler.submit("C", (9, 10, 11), max_new_tokens=1)
            scheduler.submit("D", (12, 13), max_new_tokens=1)
            submitted_refill = True

    return records


def main() -> None:
    for budget in (None, 3):
        label = "unbounded" if budget is None else str(budget)
        print(f"===== max_mixed_prefill_tokens={label} =====")
        for record in run(budget):
            print(
                f"step={record.step}: decode={record.decode}, "
                f"prefill={record.prefill}, total_tokens={record.total_tokens}"
            )
        print()


if __name__ == "__main__":
    main()
