"""只读实验：估算一个 mixed step 需要新增多少 KV block。

这不是 Scheduler 的准入策略：它只证明给定 step 的容量是否足够，
不保证请求剩余生命周期内的所有 Decode 都不会耗尽 block pool。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ActiveRequest:
    request_id: str
    committed_tokens: int
    allocated_blocks: int


@dataclass(frozen=True)
class PlannedPrefill:
    request_id: str
    prompt_tokens: int


@dataclass(frozen=True)
class OneStepBlockDemand:
    free_blocks: int
    decode_growth_by_request: tuple[tuple[str, int], ...]
    prefill_blocks_by_request: tuple[tuple[str, int], ...]
    required_new_blocks: int
    shortfall_blocks: int
    fits_without_release: bool


def _blocks_for_tokens(token_count: int, block_size: int) -> int:
    return (token_count + block_size - 1) // block_size


def estimate_one_step_block_demand(
    *,
    block_size: int,
    total_blocks: int,
    free_blocks: int,
    active_requests: Sequence[ActiveRequest],
    decode_request_ids: Sequence[str],
    prefills: Sequence[PlannedPrefill],
) -> OneStepBlockDemand:
    """在本 step 内任何完成释放之前，计算同时运行的新增 block 需求。

    active_requests 必须包含当前占用 pool 的全部请求，而不只是本步 Decode 的请求。
    committed_tokens 是已写入 KV 的长度；allocated_blocks 是已经归属该请求的
    物理 block 数，可能包含尚未写入 KV 的未来预留。此函数不会修改输入。
    """
    if block_size <= 0 or total_blocks <= 0:
        raise ValueError("block_size 与 total_blocks 必须为正")
    if not 0 <= free_blocks <= total_blocks:
        raise ValueError("free_blocks 必须位于 pool 容量范围内")

    active_by_id: dict[str, ActiveRequest] = {}
    allocated_total = 0
    for request in active_requests:
        if not request.request_id or request.request_id in active_by_id:
            raise ValueError("活动请求 ID 必须非空且唯一")
        if request.committed_tokens <= 0:
            raise ValueError("活动请求的 committed_tokens 必须为正")
        minimal_blocks = _blocks_for_tokens(request.committed_tokens, block_size)
        if request.allocated_blocks < minimal_blocks:
            raise ValueError("活动请求的 block 数装不下已提交 token")
        active_by_id[request.request_id] = request
        allocated_total += request.allocated_blocks
    # 快照不完整时不能继续估算，否则遗漏的占用可能被误当成 free block。
    if allocated_total + free_blocks != total_blocks:
        raise ValueError("活动请求已分配 block + free_blocks 必须等于 total_blocks")

    decode_growth: list[tuple[str, int]] = []
    seen_decode: set[str] = set()
    for request_id in decode_request_ids:
        if request_id in seen_decode:
            raise ValueError("同一请求不能在一个 step 内 Decode 两次")
        seen_decode.add(request_id)
        if request_id not in active_by_id:
            raise ValueError(f"Decode 请求 {request_id} 不在活动快照中")
        request = active_by_id[request_id]
        next_blocks = _blocks_for_tokens(request.committed_tokens + 1, block_size)
        # 已预留 block 或当前 block 尾部 slot 都能承接追加，无须新 block。
        decode_growth.append((request_id, max(0, next_blocks - request.allocated_blocks)))

    prefill_blocks: list[tuple[str, int]] = []
    seen_prefill: set[str] = set()
    for request in prefills:
        if not request.request_id or request.request_id in seen_prefill:
            raise ValueError("Prefill 请求 ID 必须非空且唯一")
        seen_prefill.add(request.request_id)
        if request.request_id in active_by_id:
            raise ValueError("新 Prefill 请求不能已占用活动 Cache")
        if request.prompt_tokens <= 0:
            raise ValueError("Prefill 的 prompt_tokens 必须为正")
        prefill_blocks.append(
            (request.request_id, _blocks_for_tokens(request.prompt_tokens, block_size))
        )

    required = sum(blocks for _, blocks in decode_growth + prefill_blocks)
    return OneStepBlockDemand(
        free_blocks=free_blocks,
        decode_growth_by_request=tuple(decode_growth),
        prefill_blocks_by_request=tuple(prefill_blocks),
        required_new_blocks=required,
        shortfall_blocks=max(0, required - free_blocks),
        fits_without_release=required <= free_blocks,
    )
