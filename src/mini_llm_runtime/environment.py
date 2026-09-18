"""采集可复现 benchmark 所需的软硬件环境元数据。"""

from __future__ import annotations

import json
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    """运行只读命令；缺少工具时返回 None，而不是让环境检查整体失败。"""
    executable = shutil.which(command[0])
    if executable is None:
        # 允许 `venv/bin/python script.py` 这种未 activate 的调用方式找到同目录工具。
        sibling = Path(sys.executable).parent / command[0]
        executable = str(sibling) if sibling.is_file() else None
    if executable is None:
        return None
    resolved_command = [executable, *command[1:]]
    try:
        return subprocess.check_output(
            resolved_command,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def collect_environment(repo_root: Path | None = None) -> dict[str, Any]:
    """返回 JSON 可序列化环境信息；显存字段是采集时刻的瞬时快照。"""
    import torch

    data: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "tools": {
            "nvcc": _run(["nvcc", "--version"]),
            "gcc": _run(["gcc", "--version"]),
            "g++": _run(["g++", "--version"]),
            "cmake": _run(["cmake", "--version"]),
            "ninja": _run(["ninja", "--version"]),
        },
        "packages": {
            name: _package_version(name)
            for name in ("transformers", "safetensors", "tokenizers")
        },
    }

    smi_query = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,"
        "pstate,temperature.gpu,clocks.sm,clocks.mem,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]
    data["nvidia_smi"] = _run(smi_query)

    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_bytes": total_bytes,
                    "free_memory_bytes": free_bytes,
                }
            )
        data["cuda_devices"] = devices

    if repo_root is not None:
        data["git_commit"] = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
        data["git_dirty"] = bool(_run(["git", "status", "--porcelain"], cwd=repo_root))
    return data


def _package_version(name: str) -> str | None:
    """读取已安装版本，不 import 重型依赖。"""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def dump_environment(repo_root: Path | None = None) -> str:
    return json.dumps(collect_environment(repo_root), ensure_ascii=False, indent=2)
