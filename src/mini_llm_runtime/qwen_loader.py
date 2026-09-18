"""直接从 config.json 和 safetensors 加载 Qwen2，不构造 HF 模型对象。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from .qwen_config import QwenConfig
from .qwen_weights import QwenWeights


def load_qwen_config(model_dir: str | Path) -> QwenConfig:
    model_dir = Path(model_dir)
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    try:
        data: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"config.json 不是合法 JSON：{error}") from error
    if not isinstance(data, dict):
        raise ValueError("config.json 顶层必须是 JSON object")
    return QwenConfig.from_dict(data)


def _checkpoint_plan(model_dir: Path) -> tuple[list[Path], dict[str, str] | None]:
    """返回需要读取的 shard，以及可选的 key→shard 索引。"""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"safetensors index 不是合法 JSON：{error}") from error
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index 缺少非空 weight_map")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
            raise ValueError("safetensors weight_map 必须是 string→string")
        names = sorted(set(weight_map.values()))
        files = [model_dir / name for name in names]
        missing = [str(path) for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"index 引用的 shard 不存在：{', '.join(missing)}")
        return files, weight_map

    single = model_dir / "model.safetensors"
    if single.is_file():
        return [single], None
    shards = sorted(model_dir.glob("model-*.safetensors"))
    if shards:
        raise FileNotFoundError("发现分片 safetensors，但缺少 model.safetensors.index.json")
    raise FileNotFoundError(f"{model_dir} 中找不到 model.safetensors")


def load_qwen_weights(
    model_dir: str | Path,
    config: QwenConfig,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> QwenWeights:
    """逐 shard 读取 Tensor，转换到目标 dtype/device，再执行严格 schema 映射。"""
    model_dir = Path(model_dir)
    files, weight_map = _checkpoint_plan(model_dir)
    target_device = torch.device(device)
    state_dict: dict[str, torch.Tensor] = {}
    actual_locations: dict[str, str] = {}

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in state_dict:
                    raise ValueError(f"多个 shard 重复定义权重：{key}")
                tensor = handle.get_tensor(key)
                target_dtype = dtype if dtype is not None else tensor.dtype
                state_dict[key] = tensor.to(device=target_device, dtype=target_dtype)
                actual_locations[key] = path.name

    if weight_map is not None:
        indexed_keys = set(weight_map)
        loaded_keys = set(state_dict)
        missing = sorted(indexed_keys - loaded_keys)
        extra = sorted(loaded_keys - indexed_keys)
        if missing or extra:
            raise ValueError(
                f"index 与 shard key 不一致：missing={missing[:5]}, extra={extra[:5]}"
            )
        wrong_location = [
            key for key, filename in weight_map.items() if actual_locations[key] != filename
        ]
        if wrong_location:
            raise ValueError(f"权重位于错误 shard（前 5 项）：{wrong_location[:5]}")

    # QwenWeights 保存 Tensor 引用；局部 state_dict 释放后不会释放其底层 storage。
    return QwenWeights.from_state_dict(config, state_dict)


def load_qwen_checkpoint(
    model_dir: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> tuple[QwenConfig, QwenWeights]:
    config = load_qwen_config(model_dir)
    weights = load_qwen_weights(model_dir, config, device=device, dtype=dtype)
    return config, weights

