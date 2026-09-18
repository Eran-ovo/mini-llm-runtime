#!/usr/bin/env python3
"""HF 仅生成 Reference；Candidate 直接从 config.json/safetensors 加载并执行。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from experiments.manual_qwen_attention import compare
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner


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

    model_dir = snapshot_download(args.model, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True
    )
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    # Reference 阶段：中间结果立刻转 FP32 CPU，之后释放整个 HF 模型。
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=True,
    ).cuda().eval()
    reference_layers: list[torch.Tensor] = []

    def capture_layer(_module: Any, _inputs: Any, _kwargs: Any, output: torch.Tensor) -> None:
        reference_layers.append(output.detach().float().cpu())

    handles = [
        layer.register_forward_hook(capture_layer, with_kwargs=True)
        for layer in hf_model.model.layers
    ]
    try:
        with torch.inference_mode():
            reference = hf_model(
                **encoded, use_cache=False, output_attentions=False, return_dict=True
            )
        reference_logits = reference.logits.detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()

    del reference, hf_model
    gc.collect()
    torch.cuda.empty_cache()
    memory_before_load = torch.cuda.memory_allocated()

    # Candidate 阶段：没有 AutoModelForCausalLM，只读取 JSON 与 safetensors。
    config, weights = load_qwen_checkpoint(
        model_dir, device="cuda", dtype=torch.float16
    )
    disk_config = json.loads((Path(model_dir) / "config.json").read_text(encoding="utf-8"))
    memory_after_load = torch.cuda.memory_allocated()
    runner = QwenPrefillRunner(config, weights)
    candidate = runner.prefill(encoded["input_ids"], return_layer_outputs=True)
    assert candidate.layer_outputs is not None

    print("===== Direct safetensors Qwen Runner =====")
    print(f"model directory          = {model_dir}")
    print(f"disk dtype               = {disk_config.get('torch_dtype')} (来自 config.json)")
    print(f"runtime dtype/device     = {weights.embedding.dtype}/{weights.embedding.device}")
    print(f"GPU allocated before     = {memory_before_load / 2**20:.2f} MiB")
    print(f"GPU allocated after      = {memory_after_load / 2**20:.2f} MiB")
    print(f"runner 持有 HF model     = {hasattr(runner, 'model')}")

    first_mismatch: str | None = None
    for index, (actual, expected) in enumerate(
        zip(candidate.layer_outputs, reference_layers, strict=True)
    ):
        report = compare(actual.float().cpu(), expected)
        print(
            f"layer_{index:02d} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = f"layer_{index:02d}"

    logits_report = compare(candidate.logits.float().cpu(), reference_logits)
    candidate_token = candidate.logits[:, -1].argmax(-1).cpu()
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
    print("\n全部通过：Candidate 加载和执行均不依赖 Hugging Face 模型类。")


if __name__ == "__main__":
    main()
