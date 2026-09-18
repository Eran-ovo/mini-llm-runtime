#!/usr/bin/env python3
"""正式比较完整重算与连续 KV Cache 的单请求生成延迟和显存。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.environment import collect_environment
from mini_llm_runtime.generation import greedy_generate
from mini_llm_runtime.kv_cache import ContiguousKVCache
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.timing import CudaBenchmarkCase, measure_cuda_interleaved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="请用一句话解释 KV Cache。")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def make_cache(
    runner: QwenPrefillRunner, *, prompt_length: int, max_new_tokens: int
) -> ContiguousKVCache:
    config = runner.config
    return ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        num_kv_heads=config.num_key_value_heads,
        capacity=prompt_length + max_new_tokens - 1,
        head_dim=config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )


@torch.inference_mode()
def generate_without_cache(
    runner: QwenPrefillRunner,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """正确性 oracle：每一步都重新计算 prompt 与全部已生成 token。"""
    sequence = input_ids
    generated: list[torch.Tensor] = []
    for index in range(max_new_tokens):
        output = runner.prefill(sequence)
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        if index != max_new_tokens - 1:
            sequence = torch.cat((sequence, token), dim=1)
    return torch.cat(generated, dim=1)


def timing_with_derived_metrics(
    timing: Any, *, measured_tokens: int
) -> dict[str, Any]:
    data = timing.to_dict()
    data["average_tpot_ms"] = timing.median_ms / measured_tokens
    data["tokens_per_second"] = measured_tokens * 1000.0 / timing.median_ms
    return data


def render_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    no_cache = metrics["post_prefill_generation"]["full_recompute"]
    cached = metrics["post_prefill_generation"]["contiguous_kv_cache"]
    prefill_no_cache = metrics["prefill"]["without_cache"]
    prefill_cached = metrics["prefill"]["with_cache"]
    parameters = result["parameters"]
    environment = result["environment_after"]
    dirty = environment.get("git_dirty")
    commit = environment.get("git_commit")

    return f"""# Continuous KV Cache Benchmark

- model: `{result['model']}`
- prompt tokens: {parameters['prompt_tokens']}
- generated tokens: {parameters['max_new_tokens']}
- warmup / repeats: {parameters['warmup']} / {parameters['repeats']}
- dtype: `{parameters['dtype']}`
- git commit: `{commit}` (dirty={dirty})

## Results

| Path | Median total (ms) | Average TPOT (ms) | tokens/s | Peak allocated (MiB) |
|---|---:|---:|---:|---:|
| Full recompute | {no_cache['median_ms']:.4f} | {no_cache['average_tpot_ms']:.4f} | {no_cache['tokens_per_second']:.2f} | {no_cache['peak_memory_bytes'] / 2**20:.2f} |
| Contiguous KV Cache | {cached['median_ms']:.4f} | {cached['average_tpot_ms']:.4f} | {cached['tokens_per_second']:.2f} | {cached['peak_memory_bytes'] / 2**20:.2f} |

- post-Prefill speedup: **{metrics['post_prefill_speedup']:.3f}x**
- Cache physical storage: **{metrics['cache_storage_bytes'] / 2**20:.4f} MiB**

## Prefill

| Path | Median TTFT (ms) | Peak allocated (MiB) |
|---|---:|---:|
| Without Cache write | {prefill_no_cache['median_ms']:.4f} | {prefill_no_cache['peak_memory_bytes'] / 2**20:.2f} |
| With Cache write | {prefill_cached['median_ms']:.4f} | {prefill_cached['peak_memory_bytes'] / 2**20:.2f} |

Raw timing and memory samples, measurement definitions, GPU telemetry and complete
software versions are stored in `result.json`.
"""


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该 benchmark 要求 CUDA")
    if args.max_new_tokens < 2:
        raise SystemExit("比较 Decode 路径要求 --max-new-tokens >= 2")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("--warmup 必须 >= 0，--repeats 必须 > 0")

    repo_root = Path(__file__).resolve().parents[1]
    environment_before = collect_environment(repo_root)
    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to("cuda")
    prompt_length = input_ids.shape[1]

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(config, weights)
    required_capacity = prompt_length + args.max_new_tokens - 1
    if required_capacity > config.max_position_embeddings:
        raise SystemExit("prompt 与生成长度超过模型 max_position_embeddings")

    # correctness gate：benchmark 之前先确认唯一变量确实只是 Cache 策略。
    expected_tokens = generate_without_cache(
        runner, input_ids, args.max_new_tokens
    )
    candidate = greedy_generate(
        runner,
        input_ids,
        max_new_tokens=args.max_new_tokens,
    )
    if not torch.equal(candidate.generated_token_ids, expected_tokens):
        raise SystemExit("正确性检查失败：cached 与 full recompute token 序列不同")
    generated_ids = candidate.generated_token_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    del candidate, expected_tokens
    gc.collect()
    torch.cuda.empty_cache()

    # Prefill：Cache allocation 在 Event 外；计时内比较是否写入 Cache。
    def prefill_without_cache(_: Any) -> None:
        output = runner.prefill(input_ids)
        torch.argmax(output.logits[:, -1], dim=-1)

    def prepare_empty_cache() -> ContiguousKVCache:
        return make_cache(
            runner,
            prompt_length=prompt_length,
            max_new_tokens=args.max_new_tokens,
        )

    def prefill_with_cache(cache: ContiguousKVCache) -> None:
        output = runner.prefill(input_ids, cache=cache)
        torch.argmax(output.logits[:, -1], dim=-1)

    prefill_timings = measure_cuda_interleaved(
        {
            "without_cache": CudaBenchmarkCase(prefill_without_cache),
            "with_cache": CudaBenchmarkCase(
                prefill_with_cache, prepare=prepare_empty_cache
            ),
        },
        warmup=args.warmup,
        repeats=args.repeats,
        device="cuda",
    )

    measured_tokens = args.max_new_tokens - 1

    # 两条 post-Prefill 路径都从同一个首 token 开始，计时内生成剩余 token。
    def prepare_full_recompute() -> torch.Tensor:
        output = runner.prefill(input_ids)
        first_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        return torch.cat((input_ids, first_token), dim=1)

    def run_full_recompute(initial_sequence: torch.Tensor) -> None:
        sequence = initial_sequence
        for index in range(measured_tokens):
            output = runner.prefill(sequence)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            if index != measured_tokens - 1:
                sequence = torch.cat((sequence, token), dim=1)

    def prepare_cached_decode() -> tuple[ContiguousKVCache, torch.Tensor]:
        cache = prepare_empty_cache()
        output = runner.prefill(input_ids, cache=cache)
        first_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        return cache, first_token

    def run_cached_decode(
        context: tuple[ContiguousKVCache, torch.Tensor]
    ) -> None:
        cache, token = context
        for _ in range(measured_tokens):
            output = runner.decode_one(token, cache=cache)
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)

    generation_timings = measure_cuda_interleaved(
        {
            "full_recompute": CudaBenchmarkCase(
                run_full_recompute, prepare=prepare_full_recompute
            ),
            "contiguous_kv_cache": CudaBenchmarkCase(
                run_cached_decode, prepare=prepare_cached_decode
            ),
        },
        warmup=args.warmup,
        repeats=args.repeats,
        device="cuda",
    )

    example_cache = prepare_empty_cache()
    no_cache_metrics = timing_with_derived_metrics(
        generation_timings["full_recompute"], measured_tokens=measured_tokens
    )
    cached_metrics = timing_with_derived_metrics(
        generation_timings["contiguous_kv_cache"], measured_tokens=measured_tokens
    )
    result = {
        "schema_version": 1,
        "benchmark": "contiguous_kv_cache_vs_full_recompute",
        "model": args.model,
        "prompt": args.prompt,
        "prompt_token_ids": input_ids[0].cpu().tolist(),
        "generated_token_ids": generated_ids,
        "generated_text": generated_text,
        "correctness": {
            "token_sequences_equal": True,
            "comparison": "fixed-length greedy token IDs before timing",
        },
        "parameters": {
            "prompt_tokens": prompt_length,
            "max_new_tokens": args.max_new_tokens,
            "measured_post_prefill_tokens": measured_tokens,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "dtype": str(weights.embedding.dtype),
            "batch_size": 1,
            "cache_capacity": required_capacity,
            "case_order": "interleaved; reversed on odd rounds",
        },
        "metrics": {
            "prefill": {
                name: timing.to_dict()
                for name, timing in prefill_timings.items()
            },
            "post_prefill_generation": {
                "full_recompute": no_cache_metrics,
                "contiguous_kv_cache": cached_metrics,
            },
            "post_prefill_speedup": (
                no_cache_metrics["median_ms"] / cached_metrics["median_ms"]
            ),
            "cache_storage_bytes": example_cache.storage_nbytes,
            "peak_memory_delta_bytes": (
                cached_metrics["peak_memory_bytes"]
                - no_cache_metrics["peak_memory_bytes"]
            ),
        },
        "measurement_definition": {
            "timer": "CUDA Event on the current default stream",
            "prefill": "GPU Prefill forward + argmax; Cache allocation excluded",
            "post_prefill_generation": (
                "N-1 GPU forwards + argmax after a shared logical first token; "
                "prompt Prefill and setup excluded"
            ),
            "average_tpot": "median total post-Prefill time divided by N-1",
            "peak_memory": (
                "torch.cuda.max_memory_allocated; model and prepared live Cache "
                "included; allocator reserved memory excluded"
            ),
        },
        "environment_before": environment_before,
        "environment_after": collect_environment(repo_root),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nMarkdown report: {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
