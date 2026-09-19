#!/usr/bin/env python3
"""演示 token budget 与 Paged KV block reservation 共同控制请求准入。"""

import torch

from mini_llm_runtime.block_admission import PagedBlockAdmissionController
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.scheduler import RequestScheduler


def main() -> None:
    manager = PagedKVCacheManager(
        total_blocks=5,
        block_size=2,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        dtype=torch.float32,
        device="cpu",
    )
    admission = PagedBlockAdmissionController(manager)
    scheduler = RequestScheduler(
        max_running_requests=3,
        max_batch_tokens=5,
        admission_callback=admission.try_admit,
    )
    scheduler.submit("A", (1, 2, 3), max_new_tokens=2)  # 4 slots / 2 blocks
    scheduler.submit("B", (4, 5), max_new_tokens=3)     # 4 slots / 2 blocks
    scheduler.submit("C", (6, 7, 8), max_new_tokens=2)  # 4 slots / 2 blocks
    model_outputs = {"A": (10, 11), "B": (20, 21, 22), "C": (30, 31)}

    while scheduler.has_unfinished_requests:
        batch = scheduler.schedule_step()
        if batch is None:
            raise RuntimeError("仍有请求但没有可执行 batch，且本演示没有外部资源变化")
        print(f"\n===== step {batch.step_index} =====")
        print(
            f"prefill={list(batch.prefill_request_ids)} "
            f"decode={list(batch.decode_request_ids)} tokens={batch.token_count}/5"
        )
        print(
            f"waiting={list(scheduler.waiting_request_ids)} "
            f"running={list(scheduler.running_request_ids)} "
            f"free_blocks={manager.allocator.free_count}/5"
        )
        print(
            "reservations="
            + str(
                {
                    item.request_id: item.block_count
                    for item in admission.reservations
                }
            )
        )

        results = {}
        for item in batch.items:
            request = scheduler.get_request(item.request_id)
            # fake ModelRunner：只推进逻辑 Cache 长度；block 已在 admission 预留。
            allocated = manager.get_request(item.request_id).append_tokens(
                len(item.input_token_ids)
            )
            if allocated:
                raise RuntimeError("执行期不应再申请 block")
            results[item.request_id] = model_outputs[item.request_id][
                len(request.generated_token_ids)
            ]

        update = scheduler.apply_step_results(results)
        released = admission.release_finished(update.finished_request_ids)
        print(f"emitted={list(update.emitted_tokens)}")
        print(f"finished={list(update.finished_request_ids)} released={released}")

    print("\n===== final =====")
    print(f"finished order = {list(scheduler.finished_request_ids)}")
    print(f"manager requests = {list(manager.request_ids)}")
    print(
        f"free blocks = {manager.allocator.free_count}/"
        f"{manager.allocator.total_blocks}"
    )


if __name__ == "__main__":
    main()
