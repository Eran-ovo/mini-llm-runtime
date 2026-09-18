#!/usr/bin/env python3
"""把稳定 24 层连续 KV Cache 接入 Prefill，并与 HF past_key_values 对拍。"""

from __future__ import annotations

import argparse
import gc
from typing import Any

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.kv_cache import ContiguousKVCache
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--extra-capacity", type=int, default=8)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.extra_capacity < 0:
        raise SystemExit("--extra-capacity 必须 >= 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # 先生成 HF Reference，并把 24 层 Cache/Logits 搬到 CPU 后释放模型。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    with torch.inference_mode():
        reference = hf_model(
            **encoded, use_cache=True, output_attentions=False, return_dict=True
        )
    reference_logits = reference.logits.detach().float().cpu()
    reference_cache = [
        (
            layer.keys.detach().float().cpu(),
            layer.values.detach().float().cpu(),
        )
        for layer in reference.past_key_values.layers
    ]
    del reference, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    prompt_length = encoded["input_ids"].shape[1]
    cache = ContiguousKVCache(
        num_layers=config.num_hidden_layers,
        batch_size=encoded["input_ids"].shape[0],
        num_kv_heads=config.num_key_value_heads,
        capacity=prompt_length + args.extra_capacity,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )
    runner = QwenPrefillRunner(config, weights)
    candidate = runner.prefill(encoded["input_ids"], cache=cache)

    print("===== 完整 Prefill + 24 层连续 KV Cache =====")
    print(f"prompt token ids     = {encoded['input_ids'][0].tolist()}")
    print(f"cache buffer shape   = {list(cache.key.shape)}")
    print(f"cache length         = {cache.length}")
    print(f"cache capacity       = {cache.capacity}")
    print(f"cache storage        = {cache.storage_nbytes / 2**20:.4f} MiB")

    first_mismatch: str | None = None
    print("\n===== 逐层 Cache 对拍 =====")
    for index, (reference_key, reference_value) in enumerate(reference_cache):
        candidate_key, candidate_value = cache.view_layer(index)
        key_report = compare(candidate_key.float().cpu(), reference_key)
        value_report = compare(candidate_value.float().cpu(), reference_value)
        print(
            f"layer_{index:02d} K max_abs={key_report.max_abs:.8f} "
            f"V max_abs={value_report.max_abs:.8f} "
            f"allclose={key_report.allclose and value_report.allclose}"
        )
        if (
            not key_report.allclose or not value_report.allclose
        ) and first_mismatch is None:
            first_mismatch = f"layer_{index:02d}_cache"

    logits_report = compare(candidate.logits.float().cpu(), reference_logits)
    candidate_token = candidate.logits[:, -1].argmax(-1).cpu()
    reference_token = reference_logits[:, -1].argmax(-1)
    print("\n===== 最终输出 =====")
    print(
        f"logits max_abs={logits_report.max_abs:.8f} "
        f"mean_abs={logits_report.mean_abs:.8f} allclose={logits_report.allclose}"
    )
    print(f"candidate token = {candidate_token.item()} {tokenizer.decode(candidate_token)!r}")
    print(f"reference token = {reference_token.item()} {tokenizer.decode(reference_token)!r}")

    if not logits_report.allclose and first_mismatch is None:
        first_mismatch = "logits"
    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：Prefill 正确初始化了 24 层连续 K/V Cache。")


if __name__ == "__main__":
    main()

