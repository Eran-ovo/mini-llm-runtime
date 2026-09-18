"""CUDA benchmark 基础设施：warmup、CUDA Event、原始样本与中位数。"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

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


@dataclass(frozen=True)
class CudaBenchmarkCase:
    """一条待计时路径，以及在 Event 区间外重建输入状态的函数。"""

    operation: Callable[[Any], Any]
    prepare: Callable[[], Any] | None = None


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
    device = _validate_cuda_device(device)
    case = CudaBenchmarkCase(operation=operation, prepare=prepare)

    for _ in range(warmup):
        _measure_cuda_once(case, device=device, measure_memory=False)

    samples: list[float] = []
    memory_samples: list[int] = []
    for _ in range(repeats):
        elapsed, peak = _measure_cuda_once(
            case, device=device, measure_memory=True
        )
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


def measure_cuda_interleaved(
    cases: Mapping[str, CudaBenchmarkCase],
    *,
    warmup: int = 2,
    repeats: int = 10,
    device: torch.device | str = "cuda",
) -> dict[str, TimingResult]:
    """交错测量多条路径，奇偶轮反转顺序以减轻温度/频率漂移偏差。"""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup 必须 >= 0 且 repeats 必须 > 0")
    if len(cases) < 2:
        raise ValueError("interleaved benchmark 至少需要两个 case")
    if any(not name for name in cases):
        raise ValueError("benchmark case name 不能为空")
    device = _validate_cuda_device(device)
    names = list(cases)

    def order_for(round_index: int) -> list[str]:
        return names if round_index % 2 == 0 else list(reversed(names))

    for round_index in range(warmup):
        for name in order_for(round_index):
            _measure_cuda_once(
                cases[name], device=device, measure_memory=False
            )

    samples = {name: [] for name in names}
    memory_samples = {name: [] for name in names}
    for round_index in range(repeats):
        for name in order_for(round_index):
            elapsed, peak = _measure_cuda_once(
                cases[name], device=device, measure_memory=True
            )
            samples[name].append(elapsed)
            memory_samples[name].append(peak)

    return {
        name: TimingResult(
            warmup=warmup,
            repeats=repeats,
            samples_ms=samples[name],
            median_ms=float(statistics.median(samples[name])),
            peak_memory_bytes=max(memory_samples[name]),
            peak_memory_samples_bytes=memory_samples[name],
        )
        for name in names
    }


def _validate_cuda_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark 只接受当前可用的 CUDA device")
    return resolved


def _measure_cuda_once(
    case: CudaBenchmarkCase,
    *,
    device: torch.device,
    measure_memory: bool,
) -> tuple[float, int]:
    context = case.prepare() if case.prepare is not None else None
    torch.cuda.synchronize(device)
    if measure_memory:
        # reset 后的 peak 会从当前 live allocation 起算，因此包含模型与准备好的 Cache。
        torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    case.operation(context)
    end.record()
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    peak = int(torch.cuda.max_memory_allocated(device)) if measure_memory else 0
    return elapsed, peak
