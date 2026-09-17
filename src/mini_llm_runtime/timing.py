"""CUDA benchmark 基础设施：warmup、CUDA Event、原始样本与中位数。"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class TimingResult:
    warmup: int
    repeats: int
    samples_ms: list[float]
    median_ms: float
    peak_memory_bytes: int
    peak_memory_samples_bytes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_samples(samples_ms: list[float], warmup: int) -> TimingResult:
    """纯函数便于 CPU 单测；正式 GPU 路径由 measure_cuda 产生样本。"""
    if warmup < 0:
        raise ValueError("warmup 必须 >= 0")
    if not samples_ms:
        raise ValueError("至少需要一个 timing sample")
    return TimingResult(
        warmup=warmup,
        repeats=len(samples_ms),
        samples_ms=samples_ms,
        median_ms=float(statistics.median(samples_ms)),
        peak_memory_bytes=0,
        peak_memory_samples_bytes=[],
    )


def measure_cuda(
    operation: Callable[[Any], Any],
    *,
    prepare: Callable[[], Any] | None = None,
    warmup: int = 2,
    repeats: int = 10,
    device: torch.device | str = "cuda",
) -> TimingResult:
    """测量 operation；prepare 在 Event 计时区间外运行，适合重建 Decode cache。"""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup 必须 >= 0 且 repeats 必须 > 0")
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("measure_cuda 只接受当前可用的 CUDA device")

    def one(measure: bool) -> tuple[float, int]:
        context = prepare() if prepare is not None else None
        torch.cuda.synchronize(device)
        if measure:
            torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(context)
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        peak = int(torch.cuda.max_memory_allocated(device)) if measure else 0
        return elapsed, peak

    for _ in range(warmup):
        one(measure=False)

    samples: list[float] = []
    memory_samples: list[int] = []
    for _ in range(repeats):
        elapsed, peak = one(measure=True)
        samples.append(elapsed)
        memory_samples.append(peak)

    return TimingResult(
        warmup=warmup,
        repeats=repeats,
        samples_ms=samples,
        median_ms=float(statistics.median(samples)),
        peak_memory_bytes=max(memory_samples),
        peak_memory_samples_bytes=memory_samples,
    )

