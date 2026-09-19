#!/usr/bin/env python3
"""把真实 Qwen Prefill 接入非连续 Paged K/V，并与 HF 逐层对拍。"""

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
    parser.add_argument("--block-size", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")

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
    required_blocks = (prompt_length + args.block_size - 1) // args.block_size
    manager = PagedKVCacheManager(
        total_blocks=required_blocks + 2,
        block_size=args.block_size,
        num_layers=config.num_hidden_layers,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=weights.embedding.dtype,
        device=weights.embedding.device,
    )

    # 制造物理碎片：target 的第一块复用 0，后续块从 2 开始。
    temporary = manager.create_request("temporary")
    blocker = manager.create_request("blocker")
    temporary.append_tokens(1)  # block 0
    blocker.append_tokens(1)    # block 1
    manager.release_request("temporary")
    manager.create_request("target")
    paged_cache = PagedRequestKVCache(manager, "target")

    runner = QwenPrefillRunner(config, weights)
    candidate = runner.prefill(encoded["input_ids"], cache=paged_cache)
    metadata = manager.build_batch_metadata(("target",))

    print("===== Qwen Prefill + 非连续 Paged KV Cache =====")
    print(f"prompt token ids     = {encoded['input_ids'][0].tolist()}")
    print(f"block size           = {args.block_size}")
    print(f"target block table   = {list(paged_cache.table.block_ids)}")
    print(f"GPU block table      = {metadata.block_table.cpu().tolist()}")
    print(f"sequence lengths     = {metadata.sequence_lengths.cpu().tolist()}")
    print(f"physical K shape     = {list(manager.storage.key.shape)}")
    print(f"storage              = {manager.storage.storage_nbytes / 2**20:.4f} MiB")

    first_mismatch: str | None = None
    print("\n===== 逐层 Paged Cache 对拍 =====")
    for layer_index, (reference_key, reference_value) in enumerate(reference_cache):
        candidate_key, candidate_value = paged_cache.view_layer(layer_index)
        key_report = compare(candidate_key.float().cpu(), reference_key)
        value_report = compare(candidate_value.float().cpu(), reference_value)
        matches = key_report.allclose and value_report.allclose
        print(
            f"layer_{layer_index:02d} K max_abs={key_report.max_abs:.8f} "
            f"V max_abs={value_report.max_abs:.8f} allclose={matches}"
        )
        if not matches and first_mismatch is None:
            first_mismatch = f"layer_{layer_index:02d}_cache"

    logits_report = compare(candidate.logits.float().cpu(), reference_logits)
    candidate_token = candidate.logits[:, -1].argmax(-1).cpu()
    reference_token = reference_logits[:, -1].argmax(-1)
    print("\n===== 最终输出 =====")
    print(
        f"logits max_abs={logits_report.max_abs:.8f} "
        f"mean_abs={logits_report.mean_abs:.8f} allclose={logits_report.allclose}"
    )
    print(
        f"candidate token = {candidate_token.item()} "
        f"{tokenizer.decode(candidate_token)!r}"
    )
    print(
        f"reference token = {reference_token.item()} "
        f"{tokenizer.decode(reference_token)!r}"
    )

    if not logits_report.allclose and first_mismatch is None:
        first_mismatch = "logits"
    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：ModelRunner Prefill 正确读写了非连续 Paged K/V。")


if __name__ == "__main__":
    main()
