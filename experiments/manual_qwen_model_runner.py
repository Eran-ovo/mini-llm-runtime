#!/usr/bin/env python3
"""把手写 Decoder Layer 堆叠 24 次，形成只支持 Prefill 的最小 Qwen ModelRunner。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from experiments.manual_qwen_attention import compare
from experiments.manual_qwen_decoder_layer import (
    manual_decoder_layer_prefill,
    manual_rms_norm,
)


def embedding_lookup(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """输入端：把整数 token id 查表为 hidden vector。"""
    if input_ids.ndim != 2:
        raise ValueError("input_ids 必须是 [batch, sequence]")
    return F.embedding(input_ids, weight)


def tied_lm_head(hidden_states: torch.Tensor, embedding_weight: torch.Tensor) -> torch.Tensor:
    """输出端：复用 Embedding 权重，把 hidden state 投影到词表 logits。"""
    return F.linear(hidden_states, embedding_weight)


@dataclass(frozen=True)
class PrefillResult:
    logits: torch.Tensor
    embedding_output: torch.Tensor
    layer_outputs: list[torch.Tensor]
    final_norm: torch.Tensor


class ManualQwenPrefillRunner:
    """教学版完整 Qwen Prefill；只借用 HF 对象持有的只读权重。"""

    def __init__(self, hf_model: Any) -> None:
        self.model = hf_model
        self.config = hf_model.config
        if not self.config.tie_word_embeddings:
            raise ValueError("当前教学 runner 只支持 tied word embeddings")
        if len(hf_model.model.layers) != self.config.num_hidden_layers:
            raise ValueError("Decoder Layer 数量与 config 不一致")

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> PrefillResult:
        """无 padding Prefill；不调用 hf_model.forward。"""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids 必须是非空 [batch, sequence]")
        batch_size, sequence_length = input_ids.shape
        embedding_weight = self.model.model.embed_tokens.weight
        hidden_states = embedding_lookup(input_ids, embedding_weight)
        embedding_output = hidden_states

        # 当前只支持无 padding batch，所以每个请求的位置都是 0..S-1。
        position_ids = torch.arange(
            sequence_length, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)

        layer_outputs: list[torch.Tensor] = []
        for layer in self.model.model.layers:
            result = manual_decoder_layer_prefill(
                hidden_states, position_ids, layer, self.config
            )
            hidden_states = result.output
            layer_outputs.append(hidden_states)

        final_norm = manual_rms_norm(
            hidden_states,
            self.model.model.norm.weight,
            self.model.model.norm.variance_epsilon,
        )
        logits = tied_lm_head(final_norm, embedding_weight)
        return PrefillResult(logits, embedding_output, layer_outputs, final_norm)


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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        local_files_only=args.local_files_only,
    ).cuda().eval()
    encoded = tokenizer(args.prompt, return_tensors="pt").to("cuda")

    reference: dict[str, Any] = {"layer_outputs": []}

    def capture_embedding(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        reference["embedding"] = output.detach().clone()

    def capture_layer(_module: Any, _inputs: Any, _kwargs: Any, output: torch.Tensor) -> None:
        reference["layer_outputs"].append(output.detach().clone())

    def capture_final_norm(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
        reference["final_norm"] = output.detach().clone()

    handles = [
        model.model.embed_tokens.register_forward_hook(capture_embedding),
        model.model.norm.register_forward_hook(capture_final_norm),
    ]
    handles.extend(
        layer.register_forward_hook(capture_layer, with_kwargs=True)
        for layer in model.model.layers
    )
    try:
        with torch.inference_mode():
            reference_output = model(
                **encoded,
                use_cache=False,
                output_attentions=False,
                return_dict=True,
            )
        reference["logits"] = reference_output.logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()

    runner = ManualQwenPrefillRunner(model)
    candidate = runner.prefill(encoded["input_ids"])

    print("===== 24 层手写 Qwen Prefill ModelRunner =====")
    print(f"prompt token ids       = {encoded['input_ids'][0].tolist()}")
    print(f"embedding              = {list(candidate.embedding_output.shape)}")
    print(f"decoder layers         = {len(candidate.layer_outputs)}")
    print(f"final norm             = {list(candidate.final_norm.shape)}")
    print(f"logits                 = {list(candidate.logits.shape)}")
    tied_storage = (
        model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    )
    print(f"Embedding/LM Head 共享存储 = {tied_storage}")

    if len(reference["layer_outputs"]) != self_config_layers(model):
        raise RuntimeError("Reference 捕获的 Decoder Layer 数量不正确")

    first_mismatch: str | None = None
    embedding_report = compare(candidate.embedding_output, reference["embedding"])
    print("\n===== 逐层输出对拍 =====")
    print(
        f"embedding  max_abs={embedding_report.max_abs:.8f} "
        f"mean_abs={embedding_report.mean_abs:.8f} allclose={embedding_report.allclose}"
    )
    if not embedding_report.allclose:
        first_mismatch = "embedding"

    for index, (actual, expected) in enumerate(
        zip(candidate.layer_outputs, reference["layer_outputs"], strict=True)
    ):
        report = compare(actual, expected)
        print(
            f"layer_{index:02d}   max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = f"layer_{index:02d}"

    tail_reports = {
        "final_norm": compare(candidate.final_norm, reference["final_norm"]),
        "all_logits": compare(candidate.logits, reference["logits"]),
        "last_logits": compare(candidate.logits[:, -1], reference["logits"][:, -1]),
    }
    for name, report in tail_reports.items():
        print(
            f"{name:10s} max_abs={report.max_abs:.8f} "
            f"mean_abs={report.mean_abs:.8f} allclose={report.allclose}"
        )
        if not report.allclose and first_mismatch is None:
            first_mismatch = name

    candidate_token = torch.argmax(candidate.logits[:, -1], dim=-1)
    reference_token = torch.argmax(reference["logits"][:, -1], dim=-1)
    print("\n===== 下一个 token =====")
    print(f"candidate = {candidate_token.item()} {tokenizer.decode(candidate_token)!r}")
    print(f"reference = {reference_token.item()} {tokenizer.decode(reference_token)!r}")
    print(f"token match = {torch.equal(candidate_token, reference_token)}")

    if first_mismatch is not None:
        raise SystemExit(f"完整模型对拍失败，first_mismatch={first_mismatch}")
    if not tied_storage or not torch.equal(candidate_token, reference_token):
        raise SystemExit("权重共享或最终 token 检查失败")
    print("\n全部检查点通过：候选路径没有调用 HF model forward，完整 Prefill 数据流正确。")


def self_config_layers(model: Any) -> int:
    """独立小函数使错误消息保持清晰。"""
    return model.config.num_hidden_layers


if __name__ == "__main__":
    main()

