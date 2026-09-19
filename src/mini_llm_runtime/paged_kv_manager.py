"""集中管理 Paged KV Cache 请求生命周期、batch metadata 与碎片统计。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .paged_kv_cache import (
    FixedBlockAllocator,
    PagedKVStorage,
    RequestBlockTable,
)


@dataclass(frozen=True)
class PagedBatchMetadata:
    """按显式 request 顺序打包、可直接交给未来 CUDA kernel 的元数据。"""

    request_ids: tuple[str, ...]
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def max_blocks_per_request(self) -> int:
        return self.block_table.shape[1]


@dataclass(frozen=True)
class PagedCacheStats:
    """一个时刻的 block pool、token slot 与物理 storage 快照。"""

    active_requests: int
    total_blocks: int
    allocated_blocks: int
    free_blocks: int
    block_size: int
    total_token_slots: int
    allocated_token_slots: int
    committed_tokens: int
    pending_tokens: int
    internal_fragmentation_tokens: int
    block_utilization: float
    slot_utilization: float
    storage_nbytes: int


class PagedKVCacheManager:
    """Paged KV Cache 的单一所有者和 Scheduler-facing API。"""

    def __init__(
        self,
        *,
        total_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        self.allocator = FixedBlockAllocator(total_blocks=total_blocks)
        self.storage = PagedKVStorage(
            allocator=self.allocator,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
        self._requests: dict[str, RequestBlockTable] = {}

    @property
    def block_size(self) -> int:
        return self.storage.block_size

    @property
    def request_ids(self) -> tuple[str, ...]:
        """按请求创建顺序返回只读 registry 快照。"""
        return tuple(self._requests)

    def create_request(self, request_id: str) -> RequestBlockTable:
        if request_id in self._requests:
            raise ValueError(f"request_id={request_id!r} 已经存在")
        table = RequestBlockTable(
            request_id=request_id,
            block_size=self.block_size,
            allocator=self.allocator,
        )
        self._requests[request_id] = table
        return table

    def get_request(self, request_id: str) -> RequestBlockTable:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"未知 request_id={request_id!r}") from error

    def append_all(
        self,
        request_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """通过 request ID 事务式追加全层 K/V。"""
        self.storage.append_all(self.get_request(request_id), key, value)

    def gather(
        self, request_id: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑顺序 gather 单个请求，仅供 correctness 验证。"""
        return self.storage.gather(self.get_request(request_id))

    def release_request(self, request_id: str) -> tuple[int, ...]:
        """先成功释放 block，再从 registry 移除请求。"""
        table = self.get_request(request_id)
        released = table.release()
        del self._requests[request_id]
        return released

    def build_batch_metadata(
        self,
        request_ids: Sequence[str],
        *,
        device: torch.device | str | None = None,
    ) -> PagedBatchMetadata:
        """按调用者给定的 batch 顺序打包 block table 与 sequence length。"""
        ordered_ids = tuple(request_ids)
        if not ordered_ids:
            raise ValueError("batch request_ids 不能为空")
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("同一个 request_id 不能在 batch 中重复出现")

        tables = [self.get_request(request_id) for request_id in ordered_ids]
        pending_ids = [
            table.request_id for table in tables if table.pending is not None
        ]
        if pending_ids:
            raise RuntimeError(f"请求存在未提交 append：{pending_ids}")

        max_blocks = max(table.block_count for table in tables)
        padded_rows = [
            list(table.block_ids) + [-1] * (max_blocks - table.block_count)
            for table in tables
        ]
        target_device = self.storage.device if device is None else torch.device(device)
        if max_blocks == 0:
            block_table = torch.empty(
                (len(tables), 0), dtype=torch.int32, device=target_device
            )
        else:
            block_table = torch.tensor(
                padded_rows, dtype=torch.int32, device=target_device
            )
        sequence_lengths = torch.tensor(
            [table.token_count for table in tables],
            dtype=torch.int32,
            device=target_device,
        )
        return PagedBatchMetadata(
            request_ids=ordered_ids,
            block_table=block_table,
            sequence_lengths=sequence_lengths,
        )

    def stats(self) -> PagedCacheStats:
        committed_tokens = sum(
            table.token_count for table in self._requests.values()
        )
        pending_tokens = sum(
            table.pending.token_count
            for table in self._requests.values()
            if table.pending is not None
        )
        allocated_blocks = self.allocator.allocated_count
        allocated_token_slots = allocated_blocks * self.block_size
        reserved_tokens = committed_tokens + pending_tokens
        internal_fragmentation = allocated_token_slots - reserved_tokens
        return PagedCacheStats(
            active_requests=len(self._requests),
            total_blocks=self.allocator.total_blocks,
            allocated_blocks=allocated_blocks,
            free_blocks=self.allocator.free_count,
            block_size=self.block_size,
            total_token_slots=self.allocator.total_blocks * self.block_size,
            allocated_token_slots=allocated_token_slots,
            committed_tokens=committed_tokens,
            pending_tokens=pending_tokens,
            internal_fragmentation_tokens=internal_fragmentation,
            block_utilization=(
                allocated_blocks / self.allocator.total_blocks
            ),
            slot_utilization=(
                reserved_tokens / allocated_token_slots
                if allocated_token_slots > 0
                else 0.0
            ),
            storage_nbytes=self.storage.storage_nbytes,
        )
