#!/usr/bin/env python3
"""逐 step 对拍 HF reference 与自有 cached greedy generation。"""

from __future__ import annotations

import argparse
import gc

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.generation import greedy_generate
from mini_llm_runtime.hf_baseline import HuggingFaceBaseline
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # 第一阶段：HF 也使用显式 Prefill/Decode 循环，禁止 transformers.generate()。
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

    # 第二阶段：直接从 safetensors 加载自有 ModelRunner，执行同一 prompt。
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    candidate_runner = QwenPrefillRunner(config, weights)
    prompt_ids = torch.tensor(
        [reference.prompt_token_ids], dtype=torch.long, device="cuda"
    )
    candidate = greedy_generate(
        candidate_runner,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=tokenizer.eos_token_id,
        return_step_logits=True,
    )
    candidate_ids = candidate.generated_token_ids[0].cpu().tolist()
    candidate_text = tokenizer.decode(candidate_ids, skip_special_tokens=True)

    print("===== 自有 cached greedy generation =====")
    print(f"prompt token ids   = {reference.prompt_token_ids}")
    print(f"reference tokens   = {reference.generated_token_ids}")
    print(f"candidate tokens   = {candidate_ids}")
    print(f"reference text     = {reference.text!r}")
    print(f"candidate text     = {candidate_text!r}")
    print(f"prefill tokens     = {candidate.prefill_tokens}")
    print(f"decode steps       = {candidate.decode_steps}")
    print(
        f"cache length/capacity = "
        f"{candidate.cache_length}/{candidate.cache_capacity}"
    )
    print(f"stopped by EOS     = {candidate.stopped_by_eos}")

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
        report = compare(actual.float().cpu(), expected)
        actual_token = int(actual.argmax(dim=-1).item())
        expected_token = int(expected.argmax(dim=-1).item())
        print(
            f"step_{index:02d} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose} "
            f"token={actual_token}/{expected_token}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = f"step_{index:02d}_logits"

    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：一次 Prefill 与连续 cached Decode 生成了相同 token 序列。")


if __name__ == "__main__":
    main()
