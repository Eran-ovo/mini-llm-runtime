"""把单请求 Paged KV Cache 适配为 ModelRunner 的逐层 Cache 接口。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .paged_kv_cache import PendingBlockAppend, RequestBlockTable
from .paged_kv_manager import PagedKVCacheManager


@dataclass(frozen=True)
class PagedDecodeAttentionInputs:
    """单层 CUDA Paged Attention 所需的物理 K/V 与本次请求元数据。"""

    key_cache: torch.Tensor
    value_cache: torch.Tensor
    block_table: torch.Tensor
    sequence_lengths: torch.Tensor


class PagedRequestKVCache:
    """绑定 manager 中一个请求，提供 begin/write/view/commit 逐层事务。"""

    def __init__(self, manager: PagedKVCacheManager, request_id: str) -> None:
        self.manager = manager
        self.request_id = request_id
        self._table = manager.get_request(request_id)
        self._written_layers = [False] * self.num_layers
        # 同一次 append 中所有层共享地址元数据；只在第一次 Attention 时创建。
        self._pending_attention_metadata: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def table(self) -> RequestBlockTable:
        return self._table

    @property
    def num_layers(self) -> int:
        return self.manager.storage.num_layers

    @property
    def batch_size(self) -> int:
        return 1

    @property
    def num_kv_heads(self) -> int:
        return self.manager.storage.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.manager.storage.head_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.manager.storage.dtype

    @property
    def device(self) -> torch.device:
        return self.manager.storage.device

    @property
    def length(self) -> int:
        return self.table.token_count

    @property
    def pending(self) -> PendingBlockAppend | None:
        return self.table.pending

    @property
    def available_token_capacity(self) -> int:
        """当前请求最后一块余量，加上 pool 尚未分配的全部 token slots。"""
        visible_end = self.pending.end if self.pending is not None else self.length
        slack = self.table.token_capacity - visible_end
        return slack + self.manager.allocator.free_count * self.table.block_size

    def begin_append(self, token_count: int) -> PendingBlockAppend:
        pending = self.table.begin_append(token_count)
        self._written_layers = [False] * self.num_layers
        self._pending_attention_metadata = None
        return pending

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        if self.pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )
        if self._written_layers[layer_index]:
            raise RuntimeError(f"layer {layer_index} 在当前 append 中已经写入")
        expected = (
            self.batch_size,
            self.num_kv_heads,
            self.pending.token_count,
            self.head_dim,
        )
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"新 K/V 必须是 {expected}")

        # 一张 request table 对应单请求，因此移除 batch=1 维后写入物理池。
        self.manager.storage.write_layer(
            self.table, layer_index, key[0], value[0]
        )
        self._written_layers[layer_index] = True

    def paged_attention_inputs(
        self, layer_index: int
    ) -> PagedDecodeAttentionInputs:
        """返回包含当前 pending token 的单请求 CUDA Attention 输入。

        该入口只能在对应层已经写入后调用。这样 CUDA kernel 看到的 sequence
        length 与物理 K/V 内容始终一致，不会读取尚未初始化的 pending slot。
        """
        if self.pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )
        if not self._written_layers[layer_index]:
            raise RuntimeError(
                f"layer {layer_index} 尚未写入 pending K/V，不能执行 Paged Attention"
            )

        if self._pending_attention_metadata is None:
            # begin_append 已把新 block 追加到 table；pending.end 才是包含当前
            # Decode token 的可读长度，table.token_count 仍是提交前的历史长度。
            block_table = torch.tensor(
                (self.table.block_ids,), dtype=torch.int32, device=self.device
            )
            sequence_lengths = torch.tensor(
                (self.pending.end,), dtype=torch.int32, device=self.device
            )
            self._pending_attention_metadata = (block_table, sequence_lengths)
        block_table, sequence_lengths = self._pending_attention_metadata
        storage = self.manager.storage
        return PagedDecodeAttentionInputs(
            # 去掉 layer 维后布局正好是
            # [physical_block,kv_head,block_offset,head_dim]。
            key_cache=storage.key[layer_index],
            value_cache=storage.value[layer_index],
            block_table=block_table,
            sequence_lengths=sequence_lengths,
        )

    def view_layer(
        self, layer_index: int, *, include_pending: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if include_pending:
            if self.pending is None:
                raise RuntimeError("当前没有 begin_append 创建的事务")
            if not 0 <= layer_index < self.num_layers:
                raise IndexError(
                    f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
                )
            if not self._written_layers[layer_index]:
                raise RuntimeError(
                    f"layer {layer_index} 尚未写入 pending K/V，不能读取未提交区间"
                )
        key, value = self.manager.storage.gather_layer(
            self.table, layer_index, include_pending=include_pending
        )
        return key.unsqueeze(0), value.unsqueeze(0)

    def commit_append(self) -> None:
        if self.pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        missing = [
            index for index, written in enumerate(self._written_layers) if not written
        ]
        if missing:
            raise RuntimeError(f"不能 commit，尚未写入的层：{missing}")
        self.table.commit_append()
        self._written_layers = [False] * self.num_layers
        self._pending_attention_metadata = None

    def abort_append(self) -> None:
        if self.pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        self.table.abort_append()
        self._written_layers = [False] * self.num_layers
        self._pending_attention_metadata = None
