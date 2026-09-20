#!/usr/bin/env python3
"""真实 Qwen 多请求 Paged Decode：对拍逐请求 ModelRunner 与 Hugging Face。"""

from __future__ import annotations

import argparse
import gc
import math

import torch
from huggingface_hub import snapshot_download

from mini_llm_runtime.paged_attention import paged_decode_attention_reference
from mini_llm_runtime.paged_batch import PagedBatchDecodeAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


REQUEST_ORDER = ("C", "A", "B")
PROMPTS = {
    "A": "你好，GPU",
    "B": "请简要解释 KV Cache。",
    "C": "CUDA 是什么？",
}
END_TO_END_ATOL = 3e-2
END_TO_END_RTOL = 3e-3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def build_candidate_state(
    runner: QwenPrefillRunner,
    encoded: dict[str, torch.Tensor],
    block_size: int,
) -> tuple[PagedKVCacheManager, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """逐请求 Prefill，刻意不把 padding 写入任何请求的 Cache。"""
    required_blocks = sum(
        math.ceil((tokens.shape[1] + 1) / block_size)
        for tokens in encoded.values()
    )
    manager = PagedKVCacheManager(
        total_blocks=required_blocks + 2,
        block_size=block_size,
        num_layers=runner.config.num_hidden_layers,
        num_kv_heads=runner.config.num_key_value_heads,
        head_dim=runner.config.head_dim,
        dtype=runner.weights.embedding.dtype,
        device=runner.weights.embedding.device,
    )
    decode_inputs: dict[str, torch.Tensor] = {}
    prefill_logits: dict[str, torch.Tensor] = {}
    for request_id in ("A", "B", "C"):
        manager.create_request(request_id)
        output = runner.prefill(
            encoded[request_id],
            cache=PagedRequestKVCache(manager, request_id),
        )
        last_logits = output.logits[:, -1:]
        prefill_logits[request_id] = last_logits.detach().float().cpu()
        decode_inputs[request_id] = last_logits[:, -1].argmax(
            dim=-1, keepdim=True
        )
    return manager, decode_inputs, prefill_logits


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")
    if args.block_size <= 0:
        raise SystemExit("--block-size 必须 > 0")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    encoded = {
        request_id: tokenizer(prompt, return_tensors="pt")["input_ids"].cuda()
        for request_id, prompt in PROMPTS.items()
    }

    # HF reference 逐请求运行，避免 padding token 混入 Cache，也使比较语义清楚。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    hf_prefill: dict[str, torch.Tensor] = {}
    hf_decode: dict[str, torch.Tensor] = {}
    hf_inputs: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for request_id in ("A", "B", "C"):
            input_ids = encoded[request_id]
            prefill = hf_model(input_ids=input_ids, use_cache=True, return_dict=True)
            decode_input = prefill.logits[:, -1].argmax(dim=-1, keepdim=True)
            decoded = hf_model(
                input_ids=decode_input,
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
            hf_prefill[request_id] = prefill.logits[:, -1:].float().cpu()
            hf_decode[request_id] = decoded.logits.float().cpu()
            hf_inputs[request_id] = decode_input.cpu()
            del prefill, decoded
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    runner = QwenPrefillRunner(
        config, weights, decode_attention_backend="paged_cuda"
    )

    # 基准 1：相同权重、相同 prompt，每个请求单独调用 Decode。
    single_manager, single_inputs, single_prefill = build_candidate_state(
        runner, encoded, args.block_size
    )
    single_decode: dict[str, torch.Tensor] = {}
    for request_id in REQUEST_ORDER:
        output = runner.decode_one(
            single_inputs[request_id],
            cache=PagedRequestKVCache(single_manager, request_id),
        )
        single_decode[request_id] = output.logits.detach().float().cpu()

    # 被测路径：另一份相同初始状态，一次推进三个不同长度请求。
    batch_manager, batch_inputs, batch_prefill = build_candidate_state(
        runner, encoded, args.block_size
    )
    adapter = PagedBatchDecodeAdapter(batch_manager, REQUEST_ORDER)
    token_ids = torch.cat([batch_inputs[rid] for rid in REQUEST_ORDER], dim=0)

    records: list[dict[str, object]] = []
    original_checked = model_runner_module.paged_decode_attention_cuda
    original_unchecked = model_runner_module._paged_decode_attention_cuda_unchecked

    def wrap(operation, entry: str):
        def verified(query, key, value, table, lengths, **kwargs):
            actual = operation(query, key, value, table, lengths, **kwargs)
            expected = paged_decode_attention_reference(
                query, key, value, table, lengths, scale=kwargs.get("scale")
            ).output
            error = (actual.float() - expected.float()).abs()
            records.append(
                {
                    "entry": entry,
                    "lengths": tuple(int(x) for x in lengths.tolist()),
                    "max_abs": float(error.max().item()),
                    "allclose": bool(
                        torch.allclose(
                            actual.float(), expected.float(), atol=2e-3, rtol=2e-3
                        )
                    ),
                }
            )
            return actual

        return verified

    model_runner_module.paged_decode_attention_cuda = wrap(
        original_checked, "checked"
    )
    model_runner_module._paged_decode_attention_cuda_unchecked = wrap(
        original_unchecked, "unchecked"
    )
    try:
        batch_output = runner.decode_batch(token_ids, cache=adapter)
    finally:
        model_runner_module.paged_decode_attention_cuda = original_checked
        model_runner_module._paged_decode_attention_cuda_unchecked = original_unchecked

    print("===== Batched Qwen Paged Decode =====")
    print(f"request order = {REQUEST_ORDER}")
    print(
        "prompt lengths = "
        f"{ {rid: encoded[rid].shape[1] for rid in REQUEST_ORDER} }"
    )
    metadata = batch_manager.build_batch_metadata(REQUEST_ORDER)
    print(f"decode lengths = {metadata.sequence_lengths.cpu().tolist()}")
    print(f"block table = {metadata.block_table.cpu().tolist()}")
    print(f"attention calls = {len(records)} (expected {config.num_hidden_layers})")

    failed: list[str] = []
    attention_ok = (
        len(records) == config.num_hidden_layers
        and records[0]["entry"] == "checked"
        and all(record["allclose"] for record in records)
    )
    print(
        f"attention reference allclose = {attention_ok}, "
        f"max_abs={max(float(r['max_abs']) for r in records):.8f}"
    )
    if not attention_ok:
        failed.append("layer_attention")

    print("\nrequest | prefill_vs_HF | batch_vs_single | batch_vs_HF | argmax")
    for row, request_id in enumerate(REQUEST_ORDER):
        batch_logits = batch_output.logits[row : row + 1].float().cpu()
        prefill_ok = torch.allclose(
            batch_prefill[request_id], hf_prefill[request_id], atol=2e-3, rtol=2e-3
        ) and torch.allclose(
            single_prefill[request_id], hf_prefill[request_id], atol=2e-3, rtol=2e-3
        )
        single_ok = torch.allclose(
            batch_logits,
            single_decode[request_id],
            atol=END_TO_END_ATOL,
            rtol=END_TO_END_RTOL,
        )
        hf_ok = torch.allclose(
            batch_logits,
            hf_decode[request_id],
            atol=END_TO_END_ATOL,
            rtol=END_TO_END_RTOL,
        )
        argmax_ok = int(batch_logits.argmax(-1).item()) == int(
            hf_decode[request_id].argmax(-1).item()
        )
        print(
            f"{request_id:7s} | {str(prefill_ok):13s} | {str(single_ok):15s} | "
            f"{str(hf_ok):11s} | {argmax_ok}"
        )
        if not all((prefill_ok, single_ok, hf_ok, argmax_ok)):
            failed.append(request_id)
        if not torch.equal(batch_inputs[request_id].cpu(), hf_inputs[request_id]):
            failed.append(f"{request_id}_decode_input")

    if failed:
        raise SystemExit(f"对拍失败：{failed}")
    print("\n全部通过：三个变长请求由每层一次 Paged Attention 正确推进。")


if __name__ == "__main__":
    main()
