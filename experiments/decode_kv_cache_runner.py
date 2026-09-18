#!/usr/bin/env python3
"""用真实 Qwen 验证一次 cached Decode，并逐层对拍 K/V Cache 与 logits。"""

from __future__ import annotations

import argparse
import gc

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
    if args.extra_capacity < 1:
        raise SystemExit("单 token Decode 要求 --extra-capacity >= 1")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    prompt_length = encoded["input_ids"].shape[1]

    # 第一阶段：生成 HF Reference。DynamicCache 会原地增长，因此 Decode 后再复制最终 Cache。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    with torch.inference_mode():
        reference_prefill = hf_model(
            **encoded, use_cache=True, output_attentions=False, return_dict=True
        )
        reference_prefill_last_logits = (
            reference_prefill.logits[:, -1:].detach().float().cpu()
        )
        decode_input = reference_prefill.logits[:, -1].argmax(
            dim=-1, keepdim=True
        )
        decode_attention_mask = torch.cat(
            (
                encoded["attention_mask"],
                torch.ones_like(decode_input, dtype=encoded["attention_mask"].dtype),
            ),
            dim=1,
        )
        reference_decode = hf_model(
            input_ids=decode_input,
            attention_mask=decode_attention_mask,
            past_key_values=reference_prefill.past_key_values,
            use_cache=True,
            output_attentions=False,
            return_dict=True,
        )

    reference_decode_logits = reference_decode.logits.detach().float().cpu()
    reference_cache = [
        (
            layer.keys.detach().float().cpu(),
            layer.values.detach().float().cpu(),
        )
        for layer in reference_decode.past_key_values.layers
    ]
    reference_decode_input = decode_input.detach().cpu()
    del reference_prefill, reference_decode, hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # 第二阶段：直接加载 safetensors，用自有 Runner 执行同一条 token 流。
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
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
    candidate_prefill = runner.prefill(encoded["input_ids"], cache=cache)
    old_length = cache.length
    old_prefixes = [
        tuple(tensor.clone() for tensor in cache.view_layer(layer_index))
        for layer_index in range(config.num_hidden_layers)
    ]
    candidate_decode = runner.decode_one(
        reference_decode_input.to(weights.embedding.device), cache=cache
    )

    print("===== 完整 Prefill + 一次 cached Decode =====")
    print(f"prompt token ids       = {encoded['input_ids'][0].tolist()}")
    print(f"decode input token     = {reference_decode_input.item()} "
          f"{tokenizer.decode(reference_decode_input[0])!r}")
    print(f"cache length           = {old_length} -> {cache.length}")
    print(f"physical KV shape      = {list(cache.key.shape)}")
    print(
        "decode attention shape = "
        f"[B={cache.batch_size}, Hq={config.num_attention_heads}, "
        f"Q=1, K={cache.length}]"
    )

    first_mismatch: str | None = None
    prefill_report = compare(
        candidate_prefill.logits[:, -1:].float().cpu(),
        reference_prefill_last_logits,
    )
    print("\n===== Logits 对拍 =====")
    print(
        f"Prefill(last) max_abs={prefill_report.max_abs:.8f} "
        f"mean_abs={prefill_report.mean_abs:.8f} allclose={prefill_report.allclose}"
    )
    if not prefill_report.allclose:
        first_mismatch = "prefill_logits"

    decode_report = compare(
        candidate_decode.logits.float().cpu(), reference_decode_logits
    )
    print(
        f"Decode        max_abs={decode_report.max_abs:.8f} "
        f"mean_abs={decode_report.mean_abs:.8f} allclose={decode_report.allclose}"
    )
    if not decode_report.allclose and first_mismatch is None:
        first_mismatch = "decode_logits"

    print("\n===== 逐层增长后 Cache 对拍 =====")
    history_unchanged = True
    for index, (reference_key, reference_value) in enumerate(reference_cache):
        candidate_key, candidate_value = cache.view_layer(index)
        old_key, old_value = old_prefixes[index]
        layer_history_unchanged = torch.equal(
            candidate_key[:, :, :old_length], old_key
        ) and torch.equal(candidate_value[:, :, :old_length], old_value)
        history_unchanged = history_unchanged and layer_history_unchanged

        key_report = compare(candidate_key.float().cpu(), reference_key)
        value_report = compare(candidate_value.float().cpu(), reference_value)
        layer_matches = key_report.allclose and value_report.allclose
        print(
            f"layer_{index:02d} K max_abs={key_report.max_abs:.8f} "
            f"V max_abs={value_report.max_abs:.8f} "
            f"history_unchanged={layer_history_unchanged} allclose={layer_matches}"
        )
        if not layer_matches and first_mismatch is None:
            first_mismatch = f"layer_{index:02d}_cache"

    candidate_token = candidate_decode.logits[:, -1].argmax(-1).cpu()
    reference_token = reference_decode_logits[:, -1].argmax(-1)
    print("\n===== 下一 token =====")
    print(
        f"candidate = {candidate_token.item()} "
        f"{tokenizer.decode(candidate_token)!r}"
    )
    print(
        f"reference = {reference_token.item()} "
        f"{tokenizer.decode(reference_token)!r}"
    )
    print(f"all history unchanged = {history_unchanged}")

    if not history_unchanged and first_mismatch is None:
        first_mismatch = "historical_cache_prefix"
    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：Decode 只计算一个新 token，并正确追加了 24 层 K/V。")


if __name__ == "__main__":
    main()
