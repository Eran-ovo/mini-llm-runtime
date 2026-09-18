#!/usr/bin/env python3
"""验证自有 QwenConfig/QwenWeights/QwenPrefillRunner 与 HF Reference 一致。"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.qwen_config import QwenConfig
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.qwen_weights import QwenWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt", default="你好，GPU")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("该实验需要 CUDA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).cuda().eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    reference_layers: list[torch.Tensor] = []

    def capture_layer(_module: Any, _inputs: Any, _kwargs: Any, output: torch.Tensor) -> None:
        reference_layers.append(output.detach().clone())

    handles = [
        layer.register_forward_hook(capture_layer, with_kwargs=True)
        for layer in hf_model.model.layers
    ]
    try:
        with torch.inference_mode():
            reference = hf_model(
                **encoded, use_cache=False, output_attentions=False, return_dict=True
            )
        reference_logits = reference.logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()

    config = QwenConfig.from_hf_config(hf_model.config)
    state_dict = hf_model.state_dict()
    weights = QwenWeights.from_state_dict(config, state_dict)
    del state_dict
    runner = QwenPrefillRunner(config, weights)
    candidate = runner.prefill(encoded["input_ids"], return_layer_outputs=True)
    assert candidate.layer_outputs is not None

    print("===== 独立权重表示的 Qwen Prefill Runner =====")
    print(f"layers                  = {len(weights.layers)}")
    print(f"head_dim                = {config.head_dim}")
    print(f"GQA group_size          = {config.gqa_group_size}")
    print(f"weight dtype/device     = {weights.embedding.dtype}/{weights.embedding.device}")
    print(f"runner 持有 HF model    = {hasattr(runner, 'model')}")

    first_mismatch: str | None = None
    for index, (actual, expected) in enumerate(
        zip(candidate.layer_outputs, reference_layers, strict=True)
    ):
        report = compare(actual, expected)
        print(
            f"layer_{index:02d} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = f"layer_{index:02d}"

    logits_report = compare(candidate.logits, reference_logits)
    candidate_token = candidate.logits[:, -1].argmax(-1)
    reference_token = reference_logits[:, -1].argmax(-1)
    print(
        f"logits   max_abs={logits_report.max_abs:.8f} "
        f"mean_abs={logits_report.mean_abs:.8f} allclose={logits_report.allclose}"
    )
    print(f"candidate token = {candidate_token.item()} {tokenizer.decode(candidate_token)!r}")
    print(f"reference token = {reference_token.item()} {tokenizer.decode(reference_token)!r}")

    if not logits_report.allclose and first_mismatch is None:
        first_mismatch = "logits"
    if first_mismatch is not None:
        raise SystemExit(f"对拍失败，first_mismatch={first_mismatch}")
    print("\n全部通过：Candidate Runner 只依赖自有 config/weights，不访问 HF 模块树。")


if __name__ == "__main__":
    main()

