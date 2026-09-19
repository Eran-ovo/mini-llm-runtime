"""Scheduler 与 Paged KV Cache Manager 之间的保守 block 预留策略。"""

from __future__ import annotations

from dataclasses import dataclass

from .paged_kv_manager import PagedKVCacheManager
from .scheduler import RequestState


@dataclass(frozen=True)
class BlockReservation:
    request_id: str
    max_cache_tokens: int
    block_count: int


class PagedBlockAdmissionController:
    """接纳时预留请求最大生命周期容量，避免 Decode 中途 OOM。"""

    def __init__(self, manager: PagedKVCacheManager) -> None:
        self.manager = manager
        self._reservations: dict[str, BlockReservation] = {}

    @property
    def reservations(self) -> tuple[BlockReservation, ...]:
        return tuple(self._reservations.values())

    def reservation_for(self, request_id: str) -> BlockReservation:
        try:
            return self._reservations[request_id]
        except KeyError as error:
            raise KeyError(f"请求 {request_id!r} 尚未预留 block") from error

    def try_admit(self, request: RequestState) -> bool:
        """资源足够则原子创建 Cache request 并预留，否则无副作用返回 False。"""
        if request.request_id in self._reservations:
            raise RuntimeError(f"请求 {request.request_id!r} 已经完成 block 预留")
        block_size = self.manager.block_size
        required_blocks = (
            request.max_cache_tokens + block_size - 1
        ) // block_size
        if required_blocks > self.manager.allocator.total_blocks:
            raise ValueError(
                f"请求 {request.request_id!r} 永久无法接纳：required_blocks="
                f"{required_blocks}, total_blocks={self.manager.allocator.total_blocks}"
            )
        if required_blocks > self.manager.allocator.free_count:
            return False

        self.manager.create_request(request.request_id)
        try:
            new_blocks = self.manager.reserve_request_capacity(
                request.request_id, request.max_cache_tokens
            )
            if len(new_blocks) != required_blocks:
                raise RuntimeError("新请求预留的 block 数与计算结果不一致")
        except Exception:
            # create_request 已进入 registry；任何后续失败都必须恢复 manager。
            self.manager.release_request(request.request_id)
            raise
        self._reservations[request.request_id] = BlockReservation(
            request_id=request.request_id,
            max_cache_tokens=request.max_cache_tokens,
            block_count=required_blocks,
        )
        return True

    def release_finished(
        self, request_ids: tuple[str, ...]
    ) -> dict[str, tuple[int, ...]]:
        """由 Engine 消费完成事件；预校验全部 ID 后再逐请求释放。"""
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("finished request IDs 不能重复")
        missing = [
            request_id
            for request_id in request_ids
            if request_id not in self._reservations
        ]
        if missing:
            raise KeyError(f"完成请求没有 block reservation：{missing}")

        released: dict[str, tuple[int, ...]] = {}
        for request_id in request_ids:
            released[request_id] = self.manager.release_request(request_id)
            del self._reservations[request_id]
        return released
