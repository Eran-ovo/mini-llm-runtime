#!/usr/bin/env python3
"""为一个 Static/Continuous burst workload 生成带 NVTX 的 Nsight Systems trace。"""

from __future__ import annotations

import argparse
import gc

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.scheduler import BatchingPolicy
from scripts.benchmark_batching_policy import (
    PROMPTS,
    RequestSpec,
    build_engine,
    parse_positive_int_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument(
        "--policy", choices=tuple(item.value for item in BatchingPolicy), required=True
    )
    parser.add_argument("--request-count", type=int, default=8)
    parser.add_argument(
        "--generation-lengths",
        type=parse_positive_int_list,
        default=(2, 4, 8, 12),
    )
    parser.add_argument("--max-running-requests", type=int, default=4)
    parser.add_argument("--max-batch-tokens", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def submit_all(engine, specs: tuple[RequestSpec, ...]) -> None:
    for spec in specs:
        engine.submit(
            spec.request_id,
            spec.prompt_token_ids,
            max_new_tokens=spec.max_new_tokens,
        )


def run_to_completion(engine) -> tuple[dict[str, tuple[int, ...]], list[dict]]:
    steps = []
    while engine.scheduler.has_unfinished_requests:
        result = engine.step()
        if result is None:
            raise RuntimeError("存在未完成请求，但 Scheduler 无法取得进展")
        steps.append(
            {
                "step": result.batch.step_index,
                "prefill": result.batch.prefill_request_ids,
                "decode": result.batch.decode_request_ids,
            }
        )
    tokens = {
        request_id: engine.scheduler.get_request(request_id).generated_token_ids
        for request_id in engine.scheduler.finished_request_ids
    }
    return tokens, steps


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 profiler workload 要求 CUDA")
    if min(
        args.request_count,
        args.max_running_requests,
        args.max_batch_tokens,
        args.block_size,
    ) <= 0:
        raise SystemExit("request/budget/block 参数必须 > 0")

    from transformers import AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    specs = tuple(
        RequestSpec(
            request_id=f"request_{index:03d}",
            prompt=PROMPTS[index % len(PROMPTS)],
            prompt_token_ids=tuple(
                tokenizer(PROMPTS[index % len(PROMPTS)])["input_ids"]
            ),
            max_new_tokens=args.generation_lengths[
                index % len(args.generation_lengths)
            ],
        )
        for index in range(args.request_count)
    )
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )
    policy = BatchingPolicy(args.policy)

    # 完整 warmup 不开启 NVTX，也不在 cudaProfilerApi capture range 中。
    warmup_engine = build_engine(
        policy=policy,
        runner=runner,
        specs=specs,
        block_size=args.block_size,
        max_running_requests=args.max_running_requests,
        max_batch_tokens=args.max_batch_tokens,
    )
    submit_all(warmup_engine, specs)
    expected_tokens, _ = run_to_completion(warmup_engine)
    del warmup_engine
    gc.collect()

    profile_engine = build_engine(
        policy=policy,
        runner=runner,
        specs=specs,
        block_size=args.block_size,
        max_running_requests=args.max_running_requests,
        max_batch_tokens=args.max_batch_tokens,
        enable_nvtx=True,
    )
    submit_all(profile_engine, specs)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    actual_tokens, steps = run_to_completion(profile_engine)
    torch.cuda.cudart().cudaProfilerStop()
    torch.cuda.synchronize()

    if actual_tokens != expected_tokens:
        raise SystemExit("profiled run 与 warmup run 生成 token 不一致")
    print(f"policy={policy.value}, steps={len(steps)}, token_match=True")
    for item in steps:
        print(
            f"step={item['step']} prefill={item['prefill']} "
            f"decode={item['decode']}"
        )


if __name__ == "__main__":
    main()
