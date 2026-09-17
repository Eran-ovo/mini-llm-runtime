"""Qwen2.5 Hugging Face reference；禁止调用 transformers.generate()。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class StepOutput:
    logits: torch.Tensor
    past_key_values: Any


@dataclass
class GenerationResult:
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    text: str
    # 只保留每一步用于选下一个 token 的 logits，避免保存 [prompt_len, vocab] 巨量数据。
    next_token_logits: list[torch.Tensor]


class HuggingFaceBaseline:
    """把 Prefill 与单 token Decode 明确拆开的 reference runner。"""

    def __init__(self, model: Any, tokenizer: Any, device: torch.device | str) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
    ) -> "HuggingFaceBaseline":
        # 延迟 import，使不联网的 fake-model 单测无需先加载 transformers。
        from transformers import AutoModelForCausalLM, AutoTokenizer

        target = torch.device(device)
        if target.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA，但 torch.cuda.is_available() 为 False")
        dtype = torch.float16 if target.type == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(target)
        return cls(model=model, tokenizer=tokenizer, device=target)

    def encode(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not prompt:
            raise ValueError("prompt 不能为空")
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
        return input_ids, attention_mask

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> StepOutput:
        """完整 prompt forward；返回首 token logits 和初始化后的 KV Cache。"""
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("Prefill input_ids 必须是 [batch, prompt_len] 且 prompt_len >= 1")
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        return StepOutput(output.logits, output.past_key_values)

    @torch.inference_mode()
    def decode_one(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> StepOutput:
        """只处理一个新 token；历史 token 仅通过 past_key_values 参与 attention。"""
        if token_ids.ndim != 2 or token_ids.shape[1] != 1:
            raise ValueError("Decode token_ids 必须是 [batch, 1]")
        if past_key_values is None:
            raise ValueError("Decode 必须提供 Prefill/前一步产生的 past_key_values")
        output = self.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        return StepOutput(output.logits, output.past_key_values)

    @torch.inference_mode()
    def greedy_generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        """手写 greedy generation，用它固定后续自有 runtime 的 correctness oracle。"""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须 > 0")
        input_ids, attention_mask = self.encode(prompt)
        prompt_ids = input_ids[0].tolist()
        step = self.prefill(input_ids, attention_mask)

        generated: list[int] = []
        logits_trace: list[torch.Tensor] = []
        eos_ids = self._eos_token_ids()

        for index in range(max_new_tokens):
            last_logits = step.logits[:, -1, :]
            logits_trace.append(last_logits.detach().cpu())
            next_token = torch.argmax(last_logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id in eos_ids or index == max_new_tokens - 1:
                break

            # Decode 的 mask 覆盖“历史 cache + 当前新 token”的完整可见长度。
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(next_token, device=self.device)], dim=1
            )
            step = self.decode_one(next_token, attention_mask, step.past_key_values)

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return GenerationResult(prompt_ids, generated, text, logits_trace)

    def _eos_token_ids(self) -> set[int]:
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, int):
            return {eos}
        return {int(token_id) for token_id in eos}
