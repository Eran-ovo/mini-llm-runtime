#!/usr/bin/env python3
"""验证真实 Qwen 第一次 Paged Decode 跨块增长，并与 HF 逐层对拍。"""

from __future__ import annotations

import argparse
import gc

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help="默认等于 prompt token 数，使第一次 Decode 必然跨块",
    )
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.block_size is not None and args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    prompt_length = encoded["input_ids"].shape[1]
    block_size = args.block_size or prompt_length
    if prompt_length % block_size != 0:
        raise SystemExit(
            "该边界实验要求 prompt_length 能被 block_size 整除，"
            "以便第一次 Decode 申请新块"
        )

    # HF Reference：保存增长后的 24 层 Cache 与第一次 Decode logits。
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

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    prompt_blocks = prompt_length // block_size
    manager = PagedKVCacheManager(
        total_blocks=prompt_blocks + 3,
        block_size=block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )

    # block 1 被 blocker 占用；target 从复用的 block 0 开始增长。
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    paged_cache = PagedRequestKVCache(manager, "target")
    runner = QwenPrefillRunner(config, weights)

    candidate_prefill = runner.prefill(encoded["input_ids"], cache=paged_cache)
    prefill_block_table = paged_cache.table.block_ids
    old_length = paged_cache.length
    old_prefixes = [
        tuple(tensor.clone() for tensor in paged_cache.view_layer(layer_index))
        for layer_index in range(config.num_hidden_layers)
    ]
    candidate_decode = runner.decode_one(
        reference_decode_input.to(weights.embedding.device), cache=paged_cache
    )
    metadata = manager.build_batch_metadata(("target",))

    print("===== Qwen Paged Decode 跨 block 边界 =====")
    print(f"prompt token ids       = {encoded['input_ids'][0].tolist()}")
    print(f"decode input token     = {reference_decode_input.item()} "
          f"{tokenizer.decode(reference_decode_input[0])!r}")
    print(f"block size             = {block_size}")
    print(f"block table            = {list(prefill_block_table)} -> "
          f"{list(paged_cache.table.block_ids)}")
    print(f"sequence length        = {old_length} -> {paged_cache.length}")
    print(f"GPU block table        = {metadata.block_table.cpu().tolist()}")

    first_mismatch: str | None = None
    prefill_report = compare(
        candidate_prefill.logits[:, -1:].float().cpu(),
        reference_prefill_last_logits,
    )
    decode_report = compare(
        candidate_decode.logits.float().cpu(), reference_decode_logits
    )
    print("\n===== Logits 对拍 =====")
    print(
        f"Prefill(last) max_abs={prefill_report.max_abs:.8f} "
        f"allclose={prefill_report.allclose}"
    )
    print(
        f"Decode        max_abs={decode_report.max_abs:.8f} "
        f"allclose={decode_report.allclose}"
    )
    if not prefill_report.allclose:
        first_mismatch = "prefill_logits"
    if not decode_report.allclose and first_mismatch is None:
        first_mismatch = "decode_logits"

    print("\n===== 逐层增长后 Paged Cache 对拍 =====")
    history_unchanged = True
    for layer_index, (reference_key, reference_value) in enumerate(reference_cache):
        candidate_key, candidate_value = paged_cache.view_layer(layer_index)
        old_key, old_value = old_prefixes[layer_index]
        layer_history_unchanged = torch.equal(
            candidate_key[:, :, :old_length], old_key
        ) and torch.equal(candidate_value[:, :, :old_length], old_value)
        history_unchanged = history_unchanged and layer_history_unchanged
        key_report = compare(candidate_key.float().cpu(), reference_key)
        value_report = compare(candidate_value.float().cpu(), reference_value)
        matches = key_report.allclose and value_report.allclose
        print(
            f"layer_{layer_index:02d} K max_abs={key_report.max_abs:.8f} "
            f"V max_abs={value_report.max_abs:.8f} "
            f"history_unchanged={layer_history_unchanged} allclose={matches}"
        )
        if not matches and first_mismatch is None:
            first_mismatch = f"layer_{layer_index:02d}_cache"

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
    print("\n全部通过：第一次 Paged Decode 正确跨块追加并保持历史不变。")


if __name__ == "__main__":
    main()
