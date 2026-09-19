#!/usr/bin/env python3
"""真实 Qwen 多 token Paged CUDA greedy Decode 与 HF 逐 step 对拍。"""

from __future__ import annotations

import argparse
import gc
import math

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from experiments.paged_qwen_decode_runner import (
    END_TO_END_ATOL,
    END_TO_END_RTOL,
    install_layer_attention_verifier,
)
from mini_llm_runtime.generation import greedy_generate
from mini_llm_runtime.hf_baseline import HuggingFaceBaseline
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.max_new_tokens <= 0 or args.block_size <= 0:
        raise SystemExit("--max-new-tokens 和 --block-size 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # Reference 也显式执行 Prefill/Decode，不调用 transformers.generate()。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    reference_runner = HuggingFaceBaseline(hf_model, tokenizer, "cuda")
    reference = reference_runner.greedy_generate(args.prompt, args.max_new_tokens)
    del reference_runner, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    prompt_ids = torch.tensor(
        [reference.prompt_token_ids], dtype=torch.long, device="cuda"
    )
    prompt_length = prompt_ids.shape[1]
    required_tokens = prompt_length + args.max_new_tokens - 1
    if required_tokens > config.max_position_embeddings:
        raise SystemExit("prompt + generation 超过 max_position_embeddings")
    target_blocks = math.ceil(required_tokens / args.block_size)

    # 多准备一个只属于 blocker 的块。target 会复用 block 0，后续得到 2,3,...，
    # 从而真实验证逻辑 block 与物理 block 不相等的情况。
    manager = PagedKVCacheManager(
        total_blocks=target_blocks + 1,
        block_size=args.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    cache = PagedRequestKVCache(manager, "target")
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )

    # correctness 实验才安装逐层 reference；这里的同步和 Python loop 禁止用于计时。
    attention_records = install_layer_attention_verifier()
    candidate = greedy_generate(
        runner,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=tokenizer.eos_token_id,
        cache=cache,
        return_step_logits=True,
    )
    candidate_ids = candidate.generated_token_ids[0].cpu().tolist()
    candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)

    print("===== 多 token Paged CUDA greedy generation =====")
    print(f"prompt token ids       = {reference.prompt_token_ids}")
    print(f"reference tokens       = {reference.generated_token_ids}")
    print(f"candidate tokens       = {candidate_ids}")
    print(f"reference text         = {reference.text!r}")
    print(f"candidate text         = {candidate_text!r}")
    print(f"block size             = {args.block_size}")
    print(f"final block table      = {list(cache.table.block_ids)}")
    print(f"decode steps           = {candidate.decode_steps}")
    print(f"cache length/capacity  = {candidate.cache_length}/{candidate.cache_capacity}")
    print(f"stopped by EOS         = {candidate.stopped_by_eos}")

    first_mismatch: str | None = None
    if candidate_ids != reference.generated_token_ids:
        first_mismatch = "generated_token_ids"
    if candidate.step_logits is None:
        raise AssertionError("实验要求 return_step_logits=True")
    if len(candidate.step_logits) != len(reference.next_token_logits):
        first_mismatch = first_mismatch or "step_count"

    print("\n===== 逐 step logits 对拍 =====")
    for index, (actual, expected) in enumerate(
        zip(candidate.step_logits, reference.next_token_logits)
    ):
        actual_cpu = actual.float().cpu()
        expected_fp32 = expected.float()
        report = compare(actual_cpu, expected_fp32)
        within_budget = torch.allclose(
            actual_cpu,
            expected_fp32,
            atol=END_TO_END_ATOL,
            rtol=END_TO_END_RTOL,
        )
        actual_token = int(actual.argmax(dim=-1).item())
        expected_token = int(expected.argmax(dim=-1).item())
        print(
            f"step_{index:02d} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} strict={report.allclose} "
            f"fp16_budget={within_budget} token={actual_token}/{expected_token}"
        )
        if not within_budget and first_mismatch is None:
            first_mismatch = f"step_{index:02d}_logits_budget"
        if actual_token != expected_token and first_mismatch is None:
            first_mismatch = f"step_{index:02d}_token"

    expected_attention_calls = candidate.decode_steps * config.num_hidden_layers
    attention_passed = (
        len(attention_records) == expected_attention_calls
        and all(bool(record["allclose"]) for record in attention_records)
    )
    max_attention_error = max(
        (float(record["max_abs"]) for record in attention_records), default=0.0
    )
    print("\n===== CUDA Attention 与 block 生命周期 =====")
    print(f"attention calls        = {len(attention_records)}/{expected_attention_calls}")
    print(f"max attention max_abs  = {max_attention_error:.8f}")
    print(f"all attention close    = {attention_passed}")

    checked_records = [
        record for record in attention_records if record["entry"] == "checked"
    ]
    previous_table: tuple[int, ...] | None = None
    for step_index, record in enumerate(checked_records):
        length = int(record["sequence_length"])
        table = tuple(record["block_table"])
        expected_length = prompt_length + step_index + 1
        expected_blocks = math.ceil(length / args.block_size)
        print(
            f"decode_{step_index:02d} length={length} "
            f"block_table={list(table)} required_blocks={expected_blocks}"
        )
        if len(table) != expected_blocks and first_mismatch is None:
            first_mismatch = f"decode_{step_index:02d}_block_count"
        if length != expected_length and first_mismatch is None:
            first_mismatch = f"decode_{step_index:02d}_sequence_length"
        if (
            previous_table is not None
            and table[: len(previous_table)] != previous_table
            and first_mismatch is None
        ):
            first_mismatch = f"decode_{step_index:02d}_block_prefix"
        previous_table = table

    if len(checked_records) != candidate.decode_steps and first_mismatch is None:
        first_mismatch = "checked_call_count"

    expected_cache_length = prompt_length + candidate.decode_steps
    if candidate.cache_length != expected_cache_length and first_mismatch is None:
        first_mismatch = "cache_length"
    if cache.pending is not None and first_mismatch is None:
        first_mismatch = "pending_transaction"
    if not attention_passed and first_mismatch is None:
        first_mismatch = "attention_reference"

    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：多步 Paged Decode 的 token、Attention 与 block 生命周期一致。")


if __name__ == "__main__":
    main()
