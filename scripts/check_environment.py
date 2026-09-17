#!/usr/bin/env python3
"""打印可直接附在 benchmark 结果中的环境 JSON。"""

from pathlib import Path

from mini_llm_runtime.environment import dump_environment


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(dump_environment(root))

