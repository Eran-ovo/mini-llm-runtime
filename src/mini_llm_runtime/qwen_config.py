"""Mini LLM Runtime 使用的最小 Qwen2 配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class QwenConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        integer_fields = (
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        )
        if any(value <= 0 for value in integer_fields):
            raise ValueError("QwenConfig 的整数维度必须全部 > 0")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("rms_norm_eps 和 rope_theta 必须 > 0")
        if self.hidden_act != "silu":
            raise ValueError("当前 Runner 只支持 Qwen 的 SiLU/SwiGLU")
        if not self.tie_word_embeddings:
            raise ValueError("当前 Runner 只支持 tied word embeddings")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def gqa_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_config(cls, config: Any) -> "QwenConfig":
        return cls.from_dict(config.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QwenConfig":
        """从 config.json 字典构造最小配置，并拒绝未实现的结构变体。"""
        if data.get("model_type") != "qwen2":
            raise ValueError(f"只支持 model_type=qwen2，实际为 {data.get('model_type')!r}")
        if data.get("rope_scaling") is not None:
            raise ValueError("当前 Runner 尚不支持 rope_scaling")
        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "hidden_act",
            "tie_word_embeddings",
        )
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"config.json 缺少字段：{', '.join(missing)}")
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data["num_key_value_heads"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            hidden_act=str(data["hidden_act"]),
            tie_word_embeddings=bool(data["tie_word_embeddings"]),
        )
