#!/usr/bin/env python3
"""用确定性 token 演示 Continuous Batching Scheduler 的动态进出队。"""

from mini_llm_runtime.scheduler import RequestScheduler


def main() -> None:
    scheduler = RequestScheduler(max_running_requests=3, max_batch_tokens=6)
    scheduler.submit("A", (1, 2, 3, 4), max_new_tokens=3)
    scheduler.submit("B", (5, 6), max_new_tokens=4, eos_token_ids={99})
    scheduler.submit("C", (7,), max_new_tokens=1)

    # 用固定 token 模拟 ModelRunner 输出；Scheduler 本身不计算 logits。
    model_outputs = {
        "A": (10, 11, 12),
        "B": (20, 99),
        "C": (30,),
        "D": (40,),
    }
    submitted_d = False
    released_cache_ids: list[str] = []

    while scheduler.has_unfinished_requests:
        batch = scheduler.schedule_step()
        if batch is None:
            break
        print(f"\n===== step {batch.step_index} schedule =====")
        print(f"token budget = {batch.token_count}/6")
        for item in batch.items:
            print(
                f"{item.kind.value:7s} request={item.request_id} "
                f"input={list(item.input_token_ids)} cost={item.token_cost}"
            )
        print(f"waiting = {list(scheduler.waiting_request_ids)}")
        print(f"running = {list(scheduler.running_request_ids)}")

        # 演示 batch 0 尚在“GPU 执行”时 D 到达；它不能混入 outstanding batch。
        if not submitted_d:
            scheduler.submit("D", (8,), max_new_tokens=1)
            submitted_d = True
            print("new arrival: D -> waiting")

        results = {}
        for item in batch.items:
            request = scheduler.get_request(item.request_id)
            result_index = len(request.generated_token_ids)
            results[item.request_id] = model_outputs[item.request_id][result_index]
        update = scheduler.apply_step_results(results)

        print(f"emitted  = {list(update.emitted_tokens)}")
        print(f"finished = {list(update.finished_request_ids)}")
        # 真实 Engine 会在这里调用 KVCacheManager.release_request(request_id)。
        released_cache_ids.extend(update.finished_request_ids)

    print("\n===== final =====")
    print(f"finished order     = {list(scheduler.finished_request_ids)}")
    print(f"released cache IDs = {released_cache_ids}")
    for request_id in scheduler.finished_request_ids:
        request = scheduler.get_request(request_id)
        print(
            f"{request_id}: tokens={list(request.generated_token_ids)} "
            f"reason={request.finish_reason.value}"
        )


if __name__ == "__main__":
    main()
