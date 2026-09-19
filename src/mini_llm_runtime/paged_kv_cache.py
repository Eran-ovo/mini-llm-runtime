"""Paged KV Cache 的地址元数据与 PyTorch 物理 K/V block pool。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


class BlockPoolExhaustedError(RuntimeError):
    """物理 block pool 无法原子满足一次分配请求。"""


@dataclass(frozen=True)
class BlockLocation:
    """一个逻辑 token 在物理 block pool 中的位置。"""

    block_id: int
    block_offset: int


@dataclass(frozen=True)
class PendingBlockAppend:
    """已预留物理地址、但尚未对读取者提交的逻辑 token 区间。"""

    start: int
    end: int
    new_block_ids: tuple[int, ...]

    @property
    def token_count(self) -> int:
        return self.end - self.start


class FixedBlockAllocator:
    """只管理物理 block ID 的固定容量 free list。

    本对象不持有 K/V tensor。`allocated` 集合用于发现 double-free；free list
    使用栈结构，使 allocate/free 都只操作少量 CPU 元数据。
    """

    def __init__(self, total_blocks: int) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks 必须 > 0")
        self._total_blocks = total_blocks
        # 反向初始化后 pop() 会按 0, 1, 2... 分配，便于测试与日志复现。
        self._free_block_ids = list(reversed(range(total_blocks)))
        self._allocated_block_ids: set[int] = set()

    @property
    def total_blocks(self) -> int:
        return self._total_blocks

    @property
    def free_count(self) -> int:
        return len(self._free_block_ids)

    @property
    def allocated_count(self) -> int:
        return len(self._allocated_block_ids)

    @property
    def free_block_ids(self) -> tuple[int, ...]:
        """返回排序快照，防止调用者修改内部 free list。"""
        return tuple(sorted(self._free_block_ids))

    @property
    def allocated_block_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._allocated_block_ids))

    def allocate(self, block_count: int) -> tuple[int, ...]:
        """原子分配 `block_count` 个 ID；容量不足时不改变任何状态。"""
        if block_count <= 0:
            raise ValueError("block_count 必须 > 0")
        if block_count > self.free_count:
            raise BlockPoolExhaustedError(
                f"block pool 容量不足：requested={block_count}, "
                f"free={self.free_count}, total={self.total_blocks}"
            )

        block_ids = tuple(self._free_block_ids.pop() for _ in range(block_count))
        self._allocated_block_ids.update(block_ids)
        return block_ids

    def free(self, block_ids: Iterable[int]) -> None:
        """原子归还一组 ID；任何 ID 非法时整组都不释放。"""
        requested = tuple(block_ids)
        if not requested:
            raise ValueError("至少需要释放一个 block")
        if len(set(requested)) != len(requested):
            raise ValueError("同一次 free 中包含重复 block ID")

        out_of_range = [
            block_id
            for block_id in requested
            if not 0 <= block_id < self.total_blocks
        ]
        if out_of_range:
            raise IndexError(f"block ID 越界：{out_of_range}")
        not_allocated = [
            block_id
            for block_id in requested
            if block_id not in self._allocated_block_ids
        ]
        if not_allocated:
            raise RuntimeError(f"试图释放未分配或已释放的 block：{not_allocated}")

        # 所有检查通过后才修改状态，避免一组 ID 被部分释放。
        for block_id in requested:
            self._allocated_block_ids.remove(block_id)
        # 反向压栈，使按升序传入的 table 在复用时仍优先得到较小 ID。
        self._free_block_ids.extend(reversed(requested))


class RequestBlockTable:
    """一个请求的逻辑 token 到物理 block 的映射与生命周期。"""

    def __init__(
        self,
        *,
        request_id: str,
        block_size: int,
        allocator: FixedBlockAllocator,
    ) -> None:
        if not request_id:
            raise ValueError("request_id 不能为空")
        if block_size <= 0:
            raise ValueError("block_size 必须 > 0")
        self.request_id = request_id
        self.block_size = block_size
        self._allocator = allocator
        self._block_ids: list[int] = []
        self._token_count = 0
        self._released = False
        self._pending: PendingBlockAppend | None = None

    @property
    def allocator(self) -> FixedBlockAllocator:
        return self._allocator

    @property
    def block_ids(self) -> tuple[int, ...]:
        """只读 block table；第 i 项对应逻辑 block i。"""
        return tuple(self._block_ids)

    @property
    def token_count(self) -> int:
        return self._token_count

    @property
    def block_count(self) -> int:
        return len(self._block_ids)

    @property
    def token_capacity(self) -> int:
        return self.block_count * self.block_size

    @property
    def internal_fragmentation_tokens(self) -> int:
        return self.token_capacity - self.token_count

    @property
    def released(self) -> bool:
        return self._released

    @property
    def pending(self) -> PendingBlockAppend | None:
        return self._pending

    def append_tokens(self, token_count: int) -> tuple[int, ...]:
        """只操作元数据的便利入口：预留后立即提交。"""
        pending = self.begin_append(token_count)
        self.commit_append()
        return pending.new_block_ids

    def begin_append(self, token_count: int) -> PendingBlockAppend:
        """预留 token 地址与必要 block，但暂不推进可见 token_count。"""
        self._require_active()
        if self._pending is not None:
            raise RuntimeError("已有未提交 append，不能嵌套 begin_append")
        if token_count <= 0:
            raise ValueError("token_count 必须 > 0")

        new_token_count = self.token_count + token_count
        required_blocks = (
            new_token_count + self.block_size - 1
        ) // self.block_size
        additional_blocks = required_blocks - self.block_count

        # allocator.allocate 本身具有失败原子性；成功后才更新 table/token_count。
        new_block_ids: tuple[int, ...] = ()
        if additional_blocks > 0:
            new_block_ids = self._allocator.allocate(additional_blocks)
            self._block_ids.extend(new_block_ids)
        self._pending = PendingBlockAppend(
            start=self.token_count,
            end=new_token_count,
            new_block_ids=new_block_ids,
        )
        return self._pending

    def commit_append(self) -> None:
        """使 pending token 对普通 locate/gather 可见。"""
        pending = self._require_pending()
        self._token_count = pending.end
        self._pending = None

    def abort_append(self) -> None:
        """放弃 pending token，并归还仅为本次增长新分配的 block。"""
        pending = self._require_pending()
        if pending.new_block_ids:
            suffix_length = len(pending.new_block_ids)
            if tuple(self._block_ids[-suffix_length:]) != pending.new_block_ids:
                raise RuntimeError("block table 尾部与 pending 新 block 不一致")
            self._allocator.free(pending.new_block_ids)
            del self._block_ids[-suffix_length:]
        self._pending = None

    def locate(
        self, token_index: int, *, include_pending: bool = False
    ) -> BlockLocation:
        """把有效逻辑 token index 映射为物理 block ID 与块内 offset。"""
        self._require_active()
        visible_token_count = self.token_count
        if include_pending:
            visible_token_count = self._require_pending().end
        if not 0 <= token_index < visible_token_count:
            raise IndexError(
                f"token_index={token_index} 越界，有效范围 [0,{visible_token_count})"
            )
        logical_block = token_index // self.block_size
        block_offset = token_index % self.block_size
        return BlockLocation(
            block_id=self._block_ids[logical_block],
            block_offset=block_offset,
        )

    def release(self) -> tuple[int, ...]:
        """归还该请求的全部物理 block，并永久关闭这张 table。"""
        self._require_active()
        if self._pending is not None:
            raise RuntimeError("存在未提交 append，必须先 commit 或 abort 再 release")
        released_ids = tuple(self._block_ids)
        if released_ids:
            self._allocator.free(released_ids)
        self._block_ids.clear()
        self._token_count = 0
        self._released = True
        return released_ids

    def _require_active(self) -> None:
        if self._released:
            raise RuntimeError(f"request {self.request_id!r} 的 block table 已释放")

    def _require_pending(self) -> PendingBlockAppend:
        if self._pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        return self._pending


class PagedKVStorage:
    """绑定一个 allocator 的多层 K/V 物理 block pool。

    物理布局为 [layer, physical_block, kv_head, block_offset, head_dim]。
    当前 append/gather 是用于验证地址映射的 PyTorch 正确性实现。
    """

    def __init__(
        self,
        *,
        allocator: FixedBlockAllocator,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = (
            num_layers,
            allocator.total_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Paged K/V storage 所有维度必须 > 0")
        if not dtype.is_floating_point:
            raise ValueError("Paged K/V storage dtype 必须是浮点类型")
        self.allocator = allocator
        self.key = torch.empty(dimensions, dtype=dtype, device=device)
        self.value = torch.empty(dimensions, dtype=dtype, device=device)

    @property
    def num_layers(self) -> int:
        return self.key.shape[0]

    @property
    def total_blocks(self) -> int:
        return self.key.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.key.shape[2]

    @property
    def block_size(self) -> int:
        return self.key.shape[3]

    @property
    def head_dim(self) -> int:
        return self.key.shape[4]

    @property
    def dtype(self) -> torch.dtype:
        return self.key.dtype

    @property
    def device(self) -> torch.device:
        return self.key.device

    @property
    def storage_nbytes(self) -> int:
        return (
            self.key.numel() * self.key.element_size()
            + self.value.numel() * self.value.element_size()
        )

    @torch.inference_mode()
    def write_layer(
        self,
        table: RequestBlockTable,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """写一个 layer 的 pending K/V，输入为 [kv_head,new_token,head_dim]。"""
        self._validate_table(table)
        pending = table._require_pending()
        self._validate_layer_index(layer_index)
        if key.shape != value.shape:
            raise ValueError("新 K/V shape 必须相同")
        expected = (self.num_kv_heads, pending.token_count, self.head_dim)
        if tuple(key.shape) != expected:
            raise ValueError(f"新 K/V shape={tuple(key.shape)}，预期={expected}")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("新 K/V dtype 必须与 Paged storage 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("新 K/V device 必须与 Paged storage 相同")

        for relative_index in range(pending.token_count):
            logical_index = pending.start + relative_index
            location = table.locate(logical_index, include_pending=True)
            self.key[layer_index, location.block_id, :, location.block_offset, :].copy_(
                key[:, relative_index, :]
            )
            self.value[layer_index, location.block_id, :, location.block_offset, :].copy_(
                value[:, relative_index, :]
            )

    @torch.inference_mode()
    def gather_layer(
        self,
        table: RequestBlockTable,
        layer_index: int,
        *,
        include_pending: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑顺序重建一层 [kv_head,token,head_dim] K/V。"""
        self._validate_table(table)
        self._validate_layer_index(layer_index)
        visible_length = table.token_count
        if include_pending:
            visible_length = table._require_pending().end
        if visible_length == 0:
            shape = (self.num_kv_heads, 0, self.head_dim)
            return self.key.new_empty(shape), self.value.new_empty(shape)

        gathered_key = []
        gathered_value = []
        for token_index in range(visible_length):
            location = table.locate(
                token_index, include_pending=include_pending
            )
            gathered_key.append(
                self.key[layer_index, location.block_id, :, location.block_offset, :]
            )
            gathered_value.append(
                self.value[layer_index, location.block_id, :, location.block_offset, :]
            )
        return (
            torch.stack(gathered_key, dim=1),
            torch.stack(gathered_value, dim=1),
        )

    @torch.inference_mode()
    def append_all(
        self,
        table: RequestBlockTable,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """事务式写入全层新 K/V，输入为 [layer,kv_head,new_token,head_dim]。"""
        self._validate_table(table)
        if key.shape != value.shape:
            raise ValueError("新 K/V shape 必须相同")
        if key.ndim != 4:
            raise ValueError("新 K/V 必须是 [layer,kv_head,new_token,head_dim]")
        expected_prefix = (self.num_layers, self.num_kv_heads)
        if tuple(key.shape[:2]) != expected_prefix or key.shape[3] != self.head_dim:
            raise ValueError(
                f"新 K/V shape={tuple(key.shape)} 与 Paged storage 布局不匹配"
            )
        if key.shape[2] == 0:
            raise ValueError("new_token 维度必须 > 0")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("新 K/V dtype 必须与 Paged storage 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("新 K/V device 必须与 Paged storage 相同")

        pending = table.begin_append(key.shape[2])
        try:
            for relative_index in range(pending.token_count):
                logical_index = pending.start + relative_index
                location = table.locate(logical_index, include_pending=True)
                self.key[:, location.block_id, :, location.block_offset, :].copy_(
                    key[:, :, relative_index, :]
                )
                self.value[:, location.block_id, :, location.block_offset, :].copy_(
                    value[:, :, relative_index, :]
                )
            table.commit_append()
        except Exception:
            # 已写入字节保持为不可见垃圾；新分配 block 会归还 free list。
            if table.pending is not None:
                table.abort_append()
            raise

    @torch.inference_mode()
    def gather(
        self, table: RequestBlockTable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按逻辑 token 顺序重建连续 K/V，仅用作 correctness reference。"""
        self._validate_table(table)
        if table.pending is not None:
            raise RuntimeError("存在未提交 append，不能 gather")
        if table.token_count == 0:
            shape = (self.num_layers, self.num_kv_heads, 0, self.head_dim)
            return self.key.new_empty(shape), self.value.new_empty(shape)

        gathered_key = []
        gathered_value = []
        for token_index in range(table.token_count):
            location = table.locate(token_index)
            gathered_key.append(
                self.key[:, location.block_id, :, location.block_offset, :]
            )
            gathered_value.append(
                self.value[:, location.block_id, :, location.block_offset, :]
            )
        return (
            torch.stack(gathered_key, dim=2),
            torch.stack(gathered_value, dim=2),
        )

    def _validate_table(self, table: RequestBlockTable) -> None:
        if table.released:
            raise RuntimeError("不能访问已经释放的 request block table")
        if table.allocator is not self.allocator:
            raise ValueError("request block table 与 Paged storage 不属于同一 allocator")
        if table.block_size != self.block_size:
            raise ValueError("request block_size 与 Paged storage 不一致")

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )
