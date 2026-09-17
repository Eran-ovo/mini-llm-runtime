#!/usr/bin/env python3
"""运行 Qwen2.5-0.5B correctness artifact 与分离的 Prefill/Decode benchmark。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.hf_baseline import HuggingFaceBaseline
from mini_llm_runtime.timing import measure_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="请用一句话解释 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 benchmark 要求 CUDA；请先运行 scripts/check_environment.py")

    repo_root = Path(__file__).resolve().parents[1]
    runner = HuggingFaceBaseline.from_pretrained(args.model, device="cuda")
    input_ids, attention_mask = runner.encode(args.prompt)

    # Correctness artifact 与 benchmark 分开：计时轮次不会改变要保存的 reference。
    generation = runner.greedy_generate(args.prompt, args.max_new_tokens)

    def prefill_operation(_: Any) -> None:
        output = runner.prefill(input_ids, attention_mask)
        torch.argmax(output.logits[:, -1, :], dim=-1)

    prefill_timing = measure_cuda(
        prefill_operation, warmup=args.warmup, repeats=args.repeats, device="cuda"
    )

    # 每轮在 Event 外重建相同 prompt 的 cache，计时内始终只有一个 token Decode。
    def prepare_decode() -> tuple[torch.Tensor, torch.Tensor, Any]:
        state = runner.prefill(input_ids, attention_mask)
        token = torch.argmax(state.logits[:, -1, :], dim=-1, keepdim=True)
        decode_mask = torch.cat([attention_mask, torch.ones_like(token)], dim=1)
        return token, decode_mask, state.past_key_values

    def decode_operation(context: tuple[torch.Tensor, torch.Tensor, Any]) -> None:
        token, decode_mask, cache = context
        output = runner.decode_one(token, decode_mask, cache)
        torch.argmax(output.logits[:, -1, :], dim=-1)

    decode_timing = measure_cuda(
        decode_operation,
        prepare=prepare_decode,
        warmup=args.warmup,
        repeats=args.repeats,
        device="cuda",
    )

    result = {
        "schema_version": 1,
        "model": args.model,
        "prompt": args.prompt,
        "prompt_token_ids": generation.prompt_token_ids,
        "generated_token_ids": generation.generated_token_ids,
        "generated_text": generation.text,
        "metrics": {
            "ttft_ms": prefill_timing.median_ms,
            "tpot_ms": decode_timing.median_ms,
            "single_stream_decode_tokens_per_second": 1000.0 / decode_timing.median_ms,
            "prefill": prefill_timing.to_dict(),
            "decode": decode_timing.to_dict(),
        },
        "measurement_definition": {
            "ttft": "GPU Prefill forward + argmax; tokenizer/model load excluded",
            "tpot": "one-token GPU Decode forward + argmax at fixed prompt context",
            "peak_memory": "torch.cuda.max_memory_allocated; includes live model/cache allocations",
        },
        "environment": collect_environment(repo_root),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "prompt_token_ids": generation.prompt_token_ids,
            "generated_token_ids": generation.generated_token_ids,
            "next_token_logits": generation.next_token_logits,
        },
        args.output_dir / "logits.pt",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

