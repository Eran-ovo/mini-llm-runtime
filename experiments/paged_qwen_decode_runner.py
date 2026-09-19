#!/usr/bin/env python3
"""验证真实 Qwen 第一次 Paged Decode 跨块增长，并与 HF 逐层对拍。"""

from __future__ import annotations

import argparse
import gc

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
import mini_llm_runtime.qwen_model_runner as qwen_model_runner_module
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


# 单 kernel 同输入仍使用严格阈值；这里单独定义 24 层不同归约顺序的累计预算。
END_TO_END_ATOL = 3e-2
END_TO_END_RTOL = 3e-3


def install_layer_attention_verifier() -> list[dict[str, float | bool | str]]:
    """仅在 correctness 实验中包装 CUDA 调用，逐层与独立 reference 对拍。"""
    records: list[dict[str, float | bool | str]] = []
    original_checked = qwen_model_runner_module.paged_decode_attention_cuda
    original_unchecked = (
        qwen_model_runner_module._paged_decode_attention_cuda_unchecked
    )

    def wrap(operation, entry: str):
        def verified(query, key, value, table, lengths, **kwargs):
            actual = operation(query, key, value, table, lengths, **kwargs)
            expected = paged_decode_attention_reference(
                query,
                key,
                value,
                table,
                lengths,
                scale=kwargs.get("scale"),
            ).output
            error = (actual.float() - expected.float()).abs()
            records.append(
                {
                    "entry": entry,
                    "max_abs": float(error.max().item()),
                    "mean_abs": float(error.mean().item()),
                    "allclose": bool(
                        torch.allclose(
                            actual.float(), expected.float(), atol=2e-3, rtol=2e-3
                        )
                    ),
                }
            )
            return actual

        return verified

    qwen_model_runner_module.paged_decode_attention_cuda = wrap(
        original_checked, "checked"
    )
    qwen_model_runner_module._paged_decode_attention_cuda_unchecked = wrap(
        original_unchecked, "unchecked"
    )
    return records


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
    # Prefill 仍走 PyTorch correctness 路径；单 token Decode 直接读取物理 block pool。
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )

    candidate_prefill = runner.prefill(encoded["input_ids"], cache=paged_cache)
    prefill_block_table = paged_cache.table.block_ids
    old_length = paged_cache.length
    old_prefixes = [
        tuple(tensor.clone() for tensor in paged_cache.view_layer(layer_index))
        for layer_index in range(config.num_hidden_layers)
    ]
    layer_attention_records = install_layer_attention_verifier()
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
    decode_within_accumulation_budget = torch.allclose(
        candidate_decode.logits.float().cpu(),
        reference_decode_logits,
        atol=END_TO_END_ATOL,
        rtol=END_TO_END_RTOL,
    )
    print("\n===== Logits 对拍 =====")
    print(
        f"Prefill(last) max_abs={prefill_report.max_abs:.8f} "
        f"allclose={prefill_report.allclose}"
    )
    print(
        f"Decode        max_abs={decode_report.max_abs:.8f} "
        f"mean_abs={decode_report.mean_abs:.8f} "
        f"strict_allclose={decode_report.allclose} "
        f"fp16_budget_allclose={decode_within_accumulation_budget}"
    )
    if not prefill_report.allclose:
        first_mismatch = "prefill_logits"

    print("\n===== 每层 CUDA Attention 对同输入 reference =====")
    layer_attention_passed = (
        len(layer_attention_records) == config.num_hidden_layers
        and all(bool(record["allclose"]) for record in layer_attention_records)
    )
    print(f"checked/unchecked calls = 1/{len(layer_attention_records) - 1}")
    print(
        "max(max_abs)            = "
        f"{max(float(record['max_abs']) for record in layer_attention_records):.8f}"
    )
    print(f"all {config.num_hidden_layers} layers close = {layer_attention_passed}")
    if not layer_attention_passed and first_mismatch is None:
        first_mismatch = "layer_attention_reference"

    print("\n===== 逐层增长后 Paged Cache 对拍 =====")
    history_unchanged = True
    all_cache_finite = True
    layer_zero_matches = False
    for layer_index, (reference_key, reference_value) in enumerate(reference_cache):
        candidate_key, candidate_value = paged_cache.view_layer(layer_index)
        old_key, old_value = old_prefixes[layer_index]
        layer_history_unchanged = torch.equal(
            candidate_key[:, :, :old_length], old_key
        ) and torch.equal(candidate_value[:, :, :old_length], old_value)
        history_unchanged = history_unchanged and layer_history_unchanged
        layer_finite = bool(
            torch.isfinite(candidate_key).all()
            and torch.isfinite(candidate_value).all()
        )
        all_cache_finite = all_cache_finite and layer_finite
        key_report = compare(candidate_key.float().cpu(), reference_key)
        value_report = compare(candidate_value.float().cpu(), reference_value)
        matches = key_report.allclose and value_report.allclose
        if layer_index == 0:
            layer_zero_matches = matches
        print(
            f"layer_{layer_index:02d} K max_abs={key_report.max_abs:.8f} "
            f"V max_abs={value_report.max_abs:.8f} "
            f"history_unchanged={layer_history_unchanged} finite={layer_finite} "
            f"vs_HF_allclose={matches}"
        )

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
    print(f"all cache finite       = {all_cache_finite}")
    print(f"layer 0 K/V close      = {layer_zero_matches}")

    if not history_unchanged and first_mismatch is None:
        first_mismatch = "historical_cache_prefix"
    if not all_cache_finite and first_mismatch is None:
        first_mismatch = "non_finite_cache"
    if not layer_zero_matches and first_mismatch is None:
        first_mismatch = "layer_00_cache"
    if not decode_within_accumulation_budget and first_mismatch is None:
        first_mismatch = "decode_logits_fp16_budget"
    if not torch.equal(candidate_token, reference_token) and first_mismatch is None:
        first_mismatch = "decode_argmax"
    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：第一次 Paged Decode 正确跨块追加并保持历史不变。")


if __name__ == "__main__":
    main()
