"""KV Cache 容量与内部碎片的确定性模拟。

这个模块只回答“固定显存预算能容纳多少请求”，不执行模型，也不推断性能。
因此这里不依赖 CUDA，所有结果都由整数运算得到，适合用于设计 block size 前的
容量分析和单元测试。
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


@dataclass(frozen=True)
class KVCacheGeometry:
    """描述每个 token 的 K/V Cache 几何形状与元素大小。"""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    bytes_per_element: int

    def __post_init__(self) -> None:
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("num_kv_heads", self.num_kv_heads)
        _require_positive_int("head_dim", self.head_dim)
        _require_positive_int("bytes_per_element", self.bytes_per_element)

    @property
    def bytes_per_token(self) -> int:
        # 乘 2 是因为每层、每个 KV head 都同时保存 Key 和 Value。
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * self.bytes_per_element
        )

    def bytes_per_block(self, block_size: int) -> int:
        _require_positive_int("block_size", block_size)
        return block_size * self.bytes_per_token


@dataclass(frozen=True)
class CapacitySimulationResult:
    """一种分配策略在给定 FIFO 请求序列上的容量结果。"""

    policy: str
    allocation_unit_tokens: int
    budget_bytes: int
    bytes_per_token: int
    allocation_unit_bytes: int
    total_allocation_units: int
    allocated_units: int
    free_allocation_units: int
    total_token_slots: int
    allocated_token_slots: int
    used_tokens: int
    internal_fragmentation_tokens: int
    admitted_request_lengths: tuple[int, ...]
    first_rejected_request_index: int | None
    first_rejected_request_length: int | None
    allocation_unit_utilization: float
    slot_utilization: float
    effective_budget_utilization: float
    unusable_tail_bytes: int

    @property
    def admitted_request_count(self) -> int:
        return len(self.admitted_request_lengths)

    def to_dict(self) -> dict[str, object]:
        """转换为 JSON 友好的字典，并显式保留派生的请求数。"""
        result = asdict(self)
        result["admitted_request_lengths"] = list(self.admitted_request_lengths)
        result["admitted_request_count"] = self.admitted_request_count
        return result


def _validate_budget_and_workload(
    workload_lengths: Iterable[int], budget_bytes: int
) -> tuple[int, ...]:
    if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
        raise ValueError("budget_bytes 必须是整数")
    if budget_bytes < 0:
        raise ValueError("budget_bytes 不能为负数")

    lengths = tuple(workload_lengths)
    for length in lengths:
        _require_positive_int("request length", length)
    return lengths


def _build_result(
    *,
    policy: str,
    allocation_unit_tokens: int,
    budget_bytes: int,
    bytes_per_token: int,
    total_units: int,
    allocated_units: int,
    admitted_lengths: list[int],
    first_rejected_index: int | None,
    first_rejected_length: int | None,
) -> CapacitySimulationResult:
    unit_bytes = allocation_unit_tokens * bytes_per_token
    total_slots = total_units * allocation_unit_tokens
    allocated_slots = allocated_units * allocation_unit_tokens
    used_tokens = sum(admitted_lengths)
    fragmentation = allocated_slots - used_tokens

    return CapacitySimulationResult(
        policy=policy,
        allocation_unit_tokens=allocation_unit_tokens,
        budget_bytes=budget_bytes,
        bytes_per_token=bytes_per_token,
        allocation_unit_bytes=unit_bytes,
        total_allocation_units=total_units,
        allocated_units=allocated_units,
        free_allocation_units=total_units - allocated_units,
        total_token_slots=total_slots,
        allocated_token_slots=allocated_slots,
        used_tokens=used_tokens,
        internal_fragmentation_tokens=fragmentation,
        admitted_request_lengths=tuple(admitted_lengths),
        first_rejected_request_index=first_rejected_index,
        first_rejected_request_length=first_rejected_length,
        allocation_unit_utilization=(allocated_units / total_units if total_units else 0.0),
        slot_utilization=(used_tokens / allocated_slots if allocated_slots else 0.0),
        effective_budget_utilization=(
            used_tokens * bytes_per_token / budget_bytes if budget_bytes else 0.0
        ),
        # 不足一个完整分配单元的末尾字节不能被当前策略使用。
        unusable_tail_bytes=budget_bytes - total_units * unit_bytes,
    )


def simulate_contiguous_reservation(
    workload_lengths: Iterable[int],
    *,
    budget_bytes: int,
    reservation_tokens: int,
    geometry: KVCacheGeometry,
) -> CapacitySimulationResult:
    """模拟“每个请求都连续预留最大长度”的基线策略。"""
    _require_positive_int("reservation_tokens", reservation_tokens)
    lengths = _validate_budget_and_workload(workload_lengths, budget_bytes)
    oversized = [length for length in lengths if length > reservation_tokens]
    if oversized:
        raise ValueError(
            "请求长度不能超过 contiguous reservation_tokens；"
            f"发现长度 {oversized[0]} > {reservation_tokens}"
        )

    unit_bytes = reservation_tokens * geometry.bytes_per_token
    total_units = budget_bytes // unit_bytes
    admitted_count = min(len(lengths), total_units)
    admitted = list(lengths[:admitted_count])
    rejected_index = admitted_count if admitted_count < len(lengths) else None

    return _build_result(
        policy="contiguous_reservation",
        allocation_unit_tokens=reservation_tokens,
        budget_bytes=budget_bytes,
        bytes_per_token=geometry.bytes_per_token,
        total_units=total_units,
        allocated_units=admitted_count,
        admitted_lengths=admitted,
        first_rejected_index=rejected_index,
        first_rejected_length=(lengths[rejected_index] if rejected_index is not None else None),
    )


def simulate_paged_allocation(
    workload_lengths: Iterable[int],
    *,
    budget_bytes: int,
    block_size: int,
    geometry: KVCacheGeometry,
) -> CapacitySimulationResult:
    """模拟按需申请固定大小 block 的 Paged KV Cache。"""
    _require_positive_int("block_size", block_size)
    lengths = _validate_budget_and_workload(workload_lengths, budget_bytes)
    unit_bytes = geometry.bytes_per_block(block_size)
    total_blocks = budget_bytes // unit_bytes

    admitted: list[int] = []
    allocated_blocks = 0
    rejected_index: int | None = None
    rejected_length: int | None = None
    for index, length in enumerate(lengths):
        required_blocks = (length + block_size - 1) // block_size
        if allocated_blocks + required_blocks > total_blocks:
            # FIFO stop-on-first-OOM：不跳过队首长请求去接纳后续短请求。
            rejected_index = index
            rejected_length = length
            break
        admitted.append(length)
        allocated_blocks += required_blocks

    return _build_result(
        policy="paged",
        allocation_unit_tokens=block_size,
        budget_bytes=budget_bytes,
        bytes_per_token=geometry.bytes_per_token,
        total_units=total_blocks,
        allocated_units=allocated_blocks,
        admitted_lengths=admitted,
        first_rejected_index=rejected_index,
        first_rejected_length=rejected_length,
    )


def generate_log_uniform_lengths(
    *, num_requests: int, max_sequence_length: int, seed: int
) -> tuple[int, ...]:
    """生成可复现的、偏向短请求但覆盖多个长度尺度的合成 workload。"""
    _require_positive_int("num_requests", num_requests)
    _require_positive_int("max_sequence_length", max_sequence_length)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed 必须是整数")

    if max_sequence_length == 1:
        return (1,) * num_requests

    rng = random.Random(seed)
    log_max = math.log(max_sequence_length)
    return tuple(
        max(1, min(max_sequence_length, round(math.exp(rng.uniform(0.0, log_max)))))
        for _ in range(num_requests)
    )
