"""Scheduler 与 Paged KV Cache Manager 之间的物理页准入和逐步预算。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from .paged_kv_manager import PagedKVCacheManager
from .scheduler import RequestState


KVAllocationPolicy = Literal["lifetime", "on_demand"]


@dataclass(frozen=True)
class BlockReservation:
    """请求的生命周期上限；按需模式下实际持页数见 request table。"""

    request_id: str
    max_cache_tokens: int
    block_count: int


class PagedBlockAdmissionController:
    """管理请求的容量上限，以及整段预留或按步分配的策略。"""

    def __init__(
        self, manager: PagedKVCacheManager, *, allocation_policy: KVAllocationPolicy = "lifetime"
    ) -> None:
        if allocation_policy not in ("lifetime", "on_demand"):
            raise ValueError("allocation_policy 必须是 lifetime 或 on_demand")
        self.manager = manager
        self.allocation_policy = allocation_policy
        self._reservations: dict[str, BlockReservation] = {}

    @property
    def reservations(self) -> tuple[BlockReservation, ...]:
        return tuple(self._reservations.values())

    def reservation_for(self, request_id: str) -> BlockReservation:
        try:
            return self._reservations[request_id]
        except KeyError as error:
            raise KeyError(f"请求 {request_id!r} 尚未预留 block") from error

    def validate_request_capacity(self, request: RequestState) -> int:
        """校验单条请求能否放入整个 KV 池，返回所需 block 数。"""
        required_blocks = self._blocks_for(request.max_cache_tokens)
        if required_blocks > self.manager.allocator.total_blocks:
            raise ValueError(
                f"请求 {request.request_id!r} 永久无法接纳：required_blocks="
                f"{required_blocks}, total_blocks={self.manager.allocator.total_blocks}"
            )
        return required_blocks

    def try_admit(self, request: RequestState) -> bool:
        """注册请求。整段模式同时预留物理页，按需模式建立空表。"""
        if request.request_id in self._reservations:
            raise RuntimeError(f"请求 {request.request_id!r} 已经完成 block 预留")
        required_blocks = self.validate_request_capacity(request)
        reserved_blocks = required_blocks if self.allocation_policy == "lifetime" else 0
        if reserved_blocks > self.manager.allocator.free_count:
            return False

        self.manager.create_request(request.request_id)
        try:
            if reserved_blocks:
                new_blocks = self.manager.reserve_request_capacity(
                    request.request_id, request.max_cache_tokens
                )
                if len(new_blocks) != reserved_blocks:
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

    def _blocks_for(self, tokens: int) -> int:
        block_size = self.manager.block_size
        return (tokens + block_size - 1) // block_size

    def plan_step(self) -> Callable[[RequestState, int], bool] | None:
        """只记录本轮拟用页数；真正的分配由各 Adapter 的 begin_append 完成。"""
        if self.allocation_policy == "lifetime":
            return None

        held = {}
        committed = {}
        limits = {}
        for reservation in self._reservations.values():
            request_id = reservation.request_id
            table = self.manager.get_request(request_id)
            held[request_id] = table.block_count
            committed[request_id] = table.token_count
            limits[request_id] = reservation.block_count
        free = self.manager.allocator.free_count

        def allow(request: RequestState, token_count: int) -> bool:
            nonlocal free
            request_id = request.request_id
            old = held.get(request_id, 0)
            target = max(old, self._blocks_for(committed.get(request_id, 0) + token_count))
            growth = target - old
            if growth == 0:
                return True
            if growth > free:
                return False

            limit = limits.get(request_id)
            if limit is None:
                limit = self.validate_request_capacity(request)
            rest = [(maximum - held[rid], held[rid]) for rid, maximum in limits.items() if rid != request_id]
            rest.append((limit - target, target))

            # 先模拟剩余需求最小的请求跑完。完成后的净增空闲量是它当前持有的页。
            available = free - growth
            for remaining, owned in sorted(rest):
                if remaining > available:
                    return False
                available += owned

            held[request_id] = target
            limits[request_id] = limit
            free -= growth
            return True

        return allow

    def release_finished(
        self, request_ids: tuple[str, ...]
    ) -> dict[str, tuple[int, ...]]:
        """由 Engine 消费完成事件并释放对应 reservation。"""
        return self.release_admitted(request_ids)

    def release_admitted(
        self, request_ids: tuple[str, ...]
    ) -> dict[str, tuple[int, ...]]:
        """释放一组已接纳请求；也用于执行失败时撤销本轮 Prefill。"""
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("request IDs 不能重复")
        missing = [
            request_id
            for request_id in request_ids
            if request_id not in self._reservations
        ]
        if missing:
            raise KeyError(f"请求没有 block reservation：{missing}")

        released: dict[str, tuple[int, ...]] = {}
        for request_id in request_ids:
            released[request_id] = self.manager.release_request(request_id)
            del self._reservations[request_id]
        return released
