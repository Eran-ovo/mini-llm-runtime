"""按需编译并加载 Mini LLM Runtime 的本地 CUDA extension。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def load_cuda_extension() -> ModuleType:
    """首次调用时 JIT 编译；产物由 PyTorch 放在用户级 cache，不写入仓库。"""
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("找不到 CUDA toolkit，无法编译 Paged Attention extension")
    repo_root = Path(__file__).resolve().parents[2]
    sources = [
        repo_root / "csrc" / "paged_attention_binding.cpp",
        repo_root / "csrc" / "paged_attention_cuda.cu",
        repo_root / "csrc" / "paged_attention_split_cuda.cu",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"CUDA extension 源文件缺失：{missing}")

    return load(
        name="mini_llm_runtime_paged_attention_cuda_v1",
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=os.environ.get("MINI_LLM_RUNTIME_CUDA_VERBOSE") == "1",
    )
