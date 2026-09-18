import pytest
import torch

from mini_llm_runtime.qwen_config import QwenConfig
from mini_llm_runtime.qwen_model_runner import QwenPrefillRunner
from mini_llm_runtime.qwen_weights import QwenWeights


def tiny_config() -> QwenConfig:
    return QwenConfig(
        vocab_size=5,
        hidden_size=4,
        intermediate_size=6,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        hidden_act="silu",
        tie_word_embeddings=True,
    )


def tiny_state_dict(config: QwenConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator)

    state = {
        "model.embed_tokens.weight": rand(config.vocab_size, config.hidden_size),
        "model.norm.weight": rand(config.hidden_size),
    }
    state["lm_head.weight"] = state["model.embed_tokens.weight"]
    kv_width = config.num_key_value_heads * config.head_dim
    for index in range(config.num_hidden_layers):
        prefix = f"model.layers.{index}"
        state.update(
            {
                f"{prefix}.input_layernorm.weight": rand(config.hidden_size),
                f"{prefix}.post_attention_layernorm.weight": rand(config.hidden_size),
                f"{prefix}.self_attn.q_proj.weight": rand(
                    config.hidden_size, config.hidden_size
                ),
                f"{prefix}.self_attn.q_proj.bias": rand(config.hidden_size),
                f"{prefix}.self_attn.k_proj.weight": rand(kv_width, config.hidden_size),
                f"{prefix}.self_attn.k_proj.bias": rand(kv_width),
                f"{prefix}.self_attn.v_proj.weight": rand(kv_width, config.hidden_size),
                f"{prefix}.self_attn.v_proj.bias": rand(kv_width),
                f"{prefix}.self_attn.o_proj.weight": rand(
                    config.hidden_size, config.hidden_size
                ),
                f"{prefix}.mlp.gate_proj.weight": rand(
                    config.intermediate_size, config.hidden_size
                ),
                f"{prefix}.mlp.up_proj.weight": rand(
                    config.intermediate_size, config.hidden_size
                ),
                f"{prefix}.mlp.down_proj.weight": rand(
                    config.hidden_size, config.intermediate_size
                ),
            }
        )
    return state


def test_config_validates_gqa_divisibility() -> None:
    with pytest.raises(ValueError, match="num_key_value_heads"):
        QwenConfig(
            vocab_size=5,
            hidden_size=12,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=3,
            num_key_value_heads=2,
            max_position_embeddings=8,
            rms_norm_eps=1e-6,
            rope_theta=10_000.0,
            hidden_act="silu",
            tie_word_embeddings=True,
        )


def test_weight_mapping_and_tiny_prefill() -> None:
    config = tiny_config()
    state = tiny_state_dict(config)
    weights = QwenWeights.from_state_dict(config, state)
    assert len(weights.layers) == 2
    assert weights.embedding.data_ptr() == state["model.embed_tokens.weight"].data_ptr()

    output = QwenPrefillRunner(config, weights).prefill(torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, 5)
    assert torch.isfinite(output.logits).all()


def test_weight_mapping_rejects_wrong_shape_and_unknown_key() -> None:
    config = tiny_config()
    wrong = tiny_state_dict(config)
    wrong["model.layers.0.self_attn.k_proj.weight"] = torch.ones((3, 4))
    with pytest.raises(ValueError, match="k_proj.weight shape"):
        QwenWeights.from_state_dict(config, wrong)

    unknown = tiny_state_dict(config)
    unknown["surprise.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="未识别权重"):
        QwenWeights.from_state_dict(config, unknown)


def test_hf_config_rejects_non_default_rope_scaling() -> None:
    data = {"model_type": "qwen2", "rope_scaling": {"rope_type": "dynamic"}}
    with pytest.raises(ValueError, match="rope_scaling"):
        QwenConfig.from_dict(data)
