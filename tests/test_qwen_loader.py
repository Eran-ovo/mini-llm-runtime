import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from mini_llm_runtime.qwen_config import QwenConfig
from mini_llm_runtime.qwen_loader import load_qwen_checkpoint

from test_qwen_weights import tiny_config, tiny_state_dict


def write_config(path: Path, config: QwenConfig) -> None:
    data = {
        "model_type": "qwen2",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "rope_scaling": None,
        "hidden_act": config.hidden_act,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    (path / "config.json").write_text(json.dumps(data), encoding="utf-8")


def checkpoint_tensors(config: QwenConfig) -> dict[str, torch.Tensor]:
    # tied lm_head 不应重复保存在 safetensors 中。
    state = tiny_state_dict(config)
    state.pop("lm_head.weight")
    return state


def test_load_single_safetensors_without_hf_model(tmp_path: Path) -> None:
    config = tiny_config()
    write_config(tmp_path, config)
    save_file(checkpoint_tensors(config), tmp_path / "model.safetensors")

    loaded_config, weights = load_qwen_checkpoint(
        tmp_path, device="cpu", dtype=torch.float64
    )
    assert loaded_config == config
    assert weights.embedding.dtype == torch.float64
    assert len(weights.layers) == config.num_hidden_layers


def test_load_sharded_checkpoint_uses_index_locations(tmp_path: Path) -> None:
    config = tiny_config()
    write_config(tmp_path, config)
    state = checkpoint_tensors(config)
    keys = sorted(state)
    midpoint = len(keys) // 2
    first_keys, second_keys = keys[:midpoint], keys[midpoint:]
    first_name = "model-00001-of-00002.safetensors"
    second_name = "model-00002-of-00002.safetensors"
    save_file({key: state[key] for key in first_keys}, tmp_path / first_name)
    save_file({key: state[key] for key in second_keys}, tmp_path / second_name)
    weight_map = {
        **{key: first_name for key in first_keys},
        **{key: second_name for key in second_keys},
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    _, weights = load_qwen_checkpoint(tmp_path, device="cpu")
    assert len(weights.layers) == 2


def test_shards_without_index_are_rejected(tmp_path: Path) -> None:
    config = tiny_config()
    write_config(tmp_path, config)
    save_file(
        checkpoint_tensors(config), tmp_path / "model-00001-of-00001.safetensors"
    )
    with pytest.raises(FileNotFoundError, match="缺少 model.safetensors.index.json"):
        load_qwen_checkpoint(tmp_path, device="cpu")

