#!/usr/bin/env python3
"""诊断变长 Packed Prefill 与逐请求 Prefill 的 KV 数值差异；不做性能计时。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

from mini_llm_runtime.environment import collect_environment
import mini_llm_runtime.qwen_model_runner as model_runner_module
from mini_llm_runtime.paged_batch import PagedBatchPrefillAdapter
from mini_llm_runtime.paged_kv_adapter import PagedRequestKVCache
from mini_llm_runtime.paged_kv_manager import PagedKVCacheManager
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner, _rms_norm
from scripts.benchmark_packed_prefill import PROMPTS, construct_prompts


def error_stats(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float | bool]:
    """同时保留绝对差、分子与分母，避免只看相对 L2 误判小范数张量。"""
    if reference.shape != actual.shape:
        raise ValueError("对拍 tensor 形状不同")
    difference = reference.float() - actual.float()
    numerator = float(torch.linalg.vector_norm(difference))
    denominator = float(torch.linalg.vector_norm(reference.float()))
    return {
        "exact": bool(torch.equal(reference, actual)),
        "max_abs": float(difference.abs().max()),
        "difference_l2": numerator,
        "reference_l2": denominator,
        "relative_l2": numerator / max(denominator, 1e-12),
    }


def capture_forward(operation):
    """只在诊断脚本中 hook 私有函数，不给稳定 ModelRunner 添加调试 API。"""
    attention_outputs = []
    layer_outputs = []
    original_attention = model_runner_module._attention
    original_decoder_layer = model_runner_module._decoder_layer

    def capture_attention(*args, **kwargs):
        output = original_attention(*args, **kwargs)
        attention_outputs.append(output.detach().clone())
        return output

    def capture_layer(*args, **kwargs):
        output = original_decoder_layer(*args, **kwargs)
        layer_outputs.append(output.detach().clone())
        return output

    with (
        patch.object(model_runner_module, "_attention", capture_attention),
        patch.object(model_runner_module, "_decoder_layer", capture_layer),
    ):
        result = operation()
    return result, attention_outputs, layer_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=(128, 128, 128, 1))
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("诊断需要 CUDA")
    if args.block_size <= 0 or any(length <= 0 for length in args.prompt_lengths):
        raise SystemExit("block size 与所有 prompt length 必须 > 0")
    repo_root = Path(__file__).resolve().parents[1]
    before = collect_environment(repo_root)

    from transformers import AutoTokenizer

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    templates = tuple(
        tuple(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        for prompt in PROMPTS
    )
    lengths = tuple(args.prompt_lengths)
    prompts = construct_prompts(templates, prompt_lengths=lengths)
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=getattr(torch, args.dtype)
    )
    if max(lengths) > config.max_position_embeddings:
        raise SystemExit("prompt 超出模型位置范围")
    runner = QwenPrefillRunner(config, weights, decode_attention_backend="paged_cuda")
    request_ids = tuple(f"request-{index}" for index in range(len(prompts)))
    single_inputs = tuple(
        torch.tensor((prompt,), dtype=torch.long, device="cuda") for prompt in prompts
    )
    packed_input = torch.tensor(
        (tuple(token for prompt in prompts for token in prompt),),
        dtype=torch.long,
        device="cuda",
    )

    def new_manager() -> PagedKVCacheManager:
        manager = PagedKVCacheManager(
            total_blocks=sum(math.ceil(length / args.block_size) for length in lengths),
            block_size=args.block_size,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            dtype=weights.embedding.dtype,
            device=weights.embedding.device,
        )
        for request_id, length in zip(request_ids, lengths, strict=True):
            manager.create_request(request_id)
            manager.reserve_request_capacity(request_id, length)
        return manager

    serial_cache = new_manager()
    def run_serial():
        return torch.cat(
            [
                runner.prefill(input_ids, cache=PagedRequestKVCache(serial_cache, request_id))
                .logits[:, -1]
                for request_id, input_ids in zip(request_ids, single_inputs, strict=True)
            ],
            dim=0,
        )

    serial_logits, serial_attention, serial_hidden = capture_forward(run_serial)

    def run_packed(write_backend: str, attention_backend: str, *, capture: bool = False):
        manager = new_manager()
        adapter = PagedBatchPrefillAdapter(
            manager, request_ids, lengths, write_backend=write_backend
        )
        def operation():
            return runner.prefill_batch(
                packed_input, cache=adapter, attention_backend=attention_backend
            ).logits[:, -1]

        if capture:
            logits, attention_outputs, hidden_outputs = capture_forward(operation)
        else:
            logits = operation()
            attention_outputs, hidden_outputs = [], []
        return manager, logits, attention_outputs, hidden_outputs

    scalar_cache, scalar_logits, _, _ = run_packed("scalar", "masked")
    vector_cache, vector_logits, vector_attention, vector_hidden = run_packed(
        "vectorized", "masked", capture=True
    )
    segmented_cache, segmented_logits, segmented_attention, segmented_hidden = run_packed(
        "vectorized", "segmented_sdpa", capture=True
    )

    # 首层输入尚未经过 Attention：这里可区分地址错误与投影数值差异。
    first_layer = weights.layers[0]
    packed_embedding = F.embedding(packed_input, weights.embedding)
    packed_normalized = _rms_norm(
        packed_embedding,
        first_layer.input_norm,
        config.rms_norm_eps,
    )
    packed_after_attention = packed_embedding + vector_attention[0]
    packed_post_norm = _rms_norm(
        packed_after_attention, first_layer.post_attention_norm, config.rms_norm_eps
    )
    packed_gate_linear = F.linear(packed_post_norm, first_layer.mlp.gate_proj.weight)
    packed_up_linear = F.linear(packed_post_norm, first_layer.mlp.up_proj.weight)
    packed_gate_fp32 = F.linear(
        packed_post_norm.float(), first_layer.mlp.gate_proj.weight.float()
    )
    packed_up_fp32 = F.linear(
        packed_post_norm.float(), first_layer.mlp.up_proj.weight.float()
    )
    packed_mlp_product = F.silu(packed_gate_linear) * packed_up_linear
    packed_down = F.linear(packed_mlp_product, first_layer.mlp.down_proj.weight)
    projection_weights = {
        "q": first_layer.attention.q_proj,
        "k": first_layer.attention.k_proj,
        "v": first_layer.attention.v_proj,
    }
    layer_zero_projections = []
    layer_zero_mlp = []
    cursor = 0
    for request_index, (request_id, input_ids, length) in enumerate(
        zip(request_ids, single_inputs, lengths, strict=True)
    ):
        single_embedding = F.embedding(input_ids, weights.embedding)
        single_normalized = _rms_norm(
            single_embedding,
            first_layer.input_norm,
            config.rms_norm_eps,
        )
        packed_slice = packed_normalized[:, cursor : cursor + length]
        single_after_attention = (
            single_embedding + serial_attention[request_index * config.num_hidden_layers]
        )
        single_post_norm = _rms_norm(
            single_after_attention, first_layer.post_attention_norm, config.rms_norm_eps
        )
        single_gate_linear = F.linear(single_post_norm, first_layer.mlp.gate_proj.weight)
        single_up_linear = F.linear(single_post_norm, first_layer.mlp.up_proj.weight)
        single_gate_fp32 = F.linear(
            single_post_norm.float(), first_layer.mlp.gate_proj.weight.float()
        )
        single_up_fp32 = F.linear(
            single_post_norm.float(), first_layer.mlp.up_proj.weight.float()
        )
        single_mlp_product = F.silu(single_gate_linear) * single_up_linear
        single_down = F.linear(single_mlp_product, first_layer.mlp.down_proj.weight)
        layer_zero_mlp.append(
            {
                "request_id": request_id,
                "length": length,
                "after_attention_residual": error_stats(
                    single_after_attention,
                    packed_after_attention[:, cursor : cursor + length],
                ),
                "post_attention_norm": error_stats(
                    single_post_norm, packed_post_norm[:, cursor : cursor + length]
                ),
                "gate_projection": error_stats(
                    single_gate_linear, packed_gate_linear[:, cursor : cursor + length]
                ),
                "gate_projection_fp32_control": error_stats(
                    single_gate_fp32, packed_gate_fp32[:, cursor : cursor + length]
                ),
                "up_projection": error_stats(
                    single_up_linear, packed_up_linear[:, cursor : cursor + length]
                ),
                "up_projection_fp32_control": error_stats(
                    single_up_fp32, packed_up_fp32[:, cursor : cursor + length]
                ),
                "mlp_product": error_stats(
                    single_mlp_product, packed_mlp_product[:, cursor : cursor + length]
                ),
                "down_projection": error_stats(
                    single_down, packed_down[:, cursor : cursor + length]
                ),
            }
        )
        projections = {}
        for name, projection in projection_weights.items():
            projections[name] = {
                "fp16": error_stats(
                    F.linear(single_normalized, projection.weight, projection.bias),
                    F.linear(packed_slice, projection.weight, projection.bias),
                ),
                "fp32_control": error_stats(
                    F.linear(
                        single_normalized.float(),
                        projection.weight.float(),
                        projection.bias.float(),
                    ),
                    F.linear(
                        packed_slice.float(),
                        projection.weight.float(),
                        projection.bias.float(),
                    ),
                ),
            }
        layer_zero_projections.append(
            {
                "request_id": request_id,
                "length": length,
                "normalized_input": error_stats(single_normalized, packed_slice),
                "projections": projections,
            }
        )
        cursor += length

    layer_errors = []
    forward_errors = []
    scalar_vector_exact = True
    cursor = 0
    for request_index, (request_id, length) in enumerate(
        zip(request_ids, lengths, strict=True)
    ):
        serial_key, serial_value = serial_cache.gather(request_id)
        scalar_key, scalar_value = scalar_cache.gather(request_id)
        vector_key, vector_value = vector_cache.gather(request_id)
        segmented_key, segmented_value = segmented_cache.gather(request_id)
        for layer_index in range(config.num_hidden_layers):
            serial_index = request_index * config.num_hidden_layers + layer_index
            for name, serial_outputs, vector_outputs, segmented_outputs in (
                ("attention_output", serial_attention, vector_attention, segmented_attention),
                ("layer_output", serial_hidden, vector_hidden, segmented_hidden),
            ):
                reference = serial_outputs[serial_index]
                forward_errors.append(
                    {
                        "request_id": request_id,
                        "layer": layer_index,
                        "stage": name,
                        "serial_vs_vectorized": error_stats(
                            reference,
                            vector_outputs[layer_index][:, cursor : cursor + length],
                        ),
                        "serial_vs_segmented": error_stats(
                            reference,
                            segmented_outputs[layer_index][:, cursor : cursor + length],
                        ),
                    }
                )
            for kind, reference, scalar, vector, segmented in (
                ("key", serial_key, scalar_key, vector_key, segmented_key),
                ("value", serial_value, scalar_value, vector_value, segmented_value),
            ):
                exact = bool(torch.equal(scalar[layer_index], vector[layer_index]))
                scalar_vector_exact &= exact
                layer_errors.append(
                    {
                        "request_id": request_id,
                        "layer": layer_index,
                        "kind": kind,
                        "serial_vs_scalar": error_stats(
                            reference[layer_index], scalar[layer_index]
                        ),
                        "serial_vs_vectorized": error_stats(
                            reference[layer_index], vector[layer_index]
                        ),
                        "serial_vs_segmented": error_stats(
                            reference[layer_index], segmented[layer_index]
                        ),
                        "scalar_vs_vectorized_exact": exact,
                    }
                )
        cursor += length

    worst = max(
        layer_errors,
        key=lambda row: row["serial_vs_vectorized"]["relative_l2"],
    )
    result = {
        "classification": "correctness_diagnostic_not_benchmark",
        "model": args.model,
        "dtype": args.dtype,
        "prompt_lengths": list(lengths),
        "prompt_token_ids": [list(prompt) for prompt in prompts],
        "block_size": args.block_size,
        "torch_cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "logits": {
            "serial_vs_scalar": error_stats(serial_logits, scalar_logits),
            "serial_vs_vectorized": error_stats(serial_logits, vector_logits),
            "serial_vs_segmented": error_stats(serial_logits, segmented_logits),
            "scalar_vs_vectorized_exact": bool(torch.equal(scalar_logits, vector_logits)),
            "serial_vs_vectorized_greedy_equal": bool(
                torch.equal(serial_logits.argmax(-1), vector_logits.argmax(-1))
            ),
        },
        "packed_scalar_vs_vectorized_kv_exact": scalar_vector_exact,
        "layer_zero_projections": layer_zero_projections,
        "layer_zero_mlp": layer_zero_mlp,
        "forward_errors": forward_errors,
        "layer_errors": layer_errors,
        "worst_serial_vs_vectorized": worst,
        "environment_before": before,
        "environment_after": collect_environment(repo_root),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "result.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(target),
                "scalar_vector_kv_exact": scalar_vector_exact,
                "scalar_vector_logits_exact": result["logits"]["scalar_vs_vectorized_exact"],
                "greedy_equal": result["logits"]["serial_vs_vectorized_greedy_equal"],
                "worst_request": worst["request_id"],
                "worst_layer": worst["layer"],
                "worst_kind": worst["kind"],
                "worst_relative_l2": worst["serial_vs_vectorized"]["relative_l2"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
