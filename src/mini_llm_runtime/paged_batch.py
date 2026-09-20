"""按 Scheduler 顺序组织多请求、单 token Paged Decode 事务。"""

from __future__ import annotations

import torch

from .paged_kv_adapter import PagedDecodeAttentionInputs
from .paged_kv_cache import RequestBlockTable
from .paged_kv_manager import PagedKVCacheManager


class PagedBatchDecodeAdapter:
    """把多个 request table 组合成逐层 Decode batch。

    Adapter 只负责 Cache 写入与 metadata，不计算 Q/K/V projection。调用者必须保证
    query/key/value 的 batch row 与 `request_ids` 顺序完全一致。
    """

    def __init__(
        self, manager: PagedKVCacheManager, request_ids: tuple[str, ...]
    ) -> None:
        if not request_ids:
            raise ValueError("Decode batch request_ids 不能为空")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Decode batch 不能包含重复 request_id")
        self.manager = manager
        self.request_ids = tuple(request_ids)
        self._tables: tuple[RequestBlockTable, ...] = tuple(
            manager.get_request(request_id) for request_id in self.request_ids
        )
        self._active = False
        self._written_layers = [False] * self.num_layers
        self._metadata: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def num_layers(self) -> int:
        return self.manager.storage.num_layers

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
    def active(self) -> bool:
        return self._active

    def build_position_ids(
        self, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """返回 `[B,1]` Decode absolute positions，即 append 前的各请求长度。"""
        positions = []
        for table in self._tables:
            if table.pending is None:
                position = table.token_count
            else:
                # begin_decode 后 token_count 尚未 commit，pending.start 与旧长度相同。
                position = table.pending.start
            if position <= 0:
                raise ValueError(
                    f"请求 {table.request_id!r} Cache 为空，不能执行 Decode"
                )
            positions.append(position)
        target = self.device if device is None else torch.device(device)
        return torch.tensor(positions, dtype=torch.long, device=target).unsqueeze(1)

    def begin_decode(self) -> None:
        """为 batch 中每个请求原子开启一个单 token append。"""
        if self._active:
            raise RuntimeError("当前 Decode batch 已经存在 active transaction")
        # mutation 前完成可检查的条件，避免明显错误造成部分 begin。
        for table in self._tables:
            if table.pending is not None:
                raise RuntimeError(f"请求 {table.request_id!r} 已存在 pending append")
            if table.token_count <= 0:
                raise ValueError(
                    f"请求 {table.request_id!r} Cache 为空，必须先完成 Prefill"
                )

        begun: list[RequestBlockTable] = []
        try:
            for table in self._tables:
                table.begin_append(1)
                begun.append(table)
        except Exception:
            # 后面的请求分配失败时，撤销前面请求可能新申请的物理 block。
            for table in reversed(begun):
                table.abort_append()
            raise
        self._active = True
        self._written_layers = [False] * self.num_layers
        self._metadata = None

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """写入 `[B,Hkv,1,D]` 当前 token K/V，batch row 遵循 request_ids。"""
        self._require_active()
        self._validate_layer_index(layer_index)
        if self._written_layers[layer_index]:
            raise RuntimeError(f"layer {layer_index} 在当前 Decode 中已经写入")
        expected = (self.batch_size, self.num_kv_heads, 1, self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"Decode 新 K/V 必须是 {expected}")

        for batch_index, table in enumerate(self._tables):
            self.manager.storage.write_layer(
                table,
                layer_index,
                key[batch_index],
                value[batch_index],
            )
        self._written_layers[layer_index] = True

    def paged_attention_inputs(
        self, layer_index: int
    ) -> PagedDecodeAttentionInputs:
        """返回当前层共享 storage view 和包含 pending token 的变长 batch metadata。"""
        self._require_active()
        self._validate_layer_index(layer_index)
        if not self._written_layers[layer_index]:
            raise RuntimeError(
                f"layer {layer_index} 尚未写入全部请求 K/V，不能执行 Attention"
            )
        if self._metadata is None:
            max_blocks = max(table.block_count for table in self._tables)
            rows = [
                list(table.block_ids) + [-1] * (max_blocks - table.block_count)
                for table in self._tables
            ]
            block_table = torch.tensor(
                rows, dtype=torch.int32, device=self.device
            )
            sequence_lengths = torch.tensor(
                [table.pending.end for table in self._tables],
                dtype=torch.int32,
                device=self.device,
            )
            self._metadata = (block_table, sequence_lengths)
        block_table, sequence_lengths = self._metadata
        storage = self.manager.storage
        return PagedDecodeAttentionInputs(
            key_cache=storage.key[layer_index],
            value_cache=storage.value[layer_index],
            block_table=block_table,
            sequence_lengths=sequence_lengths,
        )

    def commit_decode(self) -> None:
        """仅当所有层都写完后，一次性推进所有请求的可见长度。"""
        self._require_active()
        missing = [
            index for index, written in enumerate(self._written_layers) if not written
        ]
        if missing:
            raise RuntimeError(f"不能 commit，尚未写入的层：{missing}")
        # 上面的完整预校验后，RequestBlockTable.commit_append 不再包含分配操作。
        for table in self._tables:
            table.commit_append()
        self._reset()

    def abort_decode(self) -> None:
        """撤销所有请求的 pending append；已写 K/V 字节保持为不可见垃圾。"""
        self._require_active()
        for table in reversed(self._tables):
            table.abort_append()
        self._reset()

    def _reset(self) -> None:
        self._active = False
        self._written_layers = [False] * self.num_layers
        self._metadata = None

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Decode batch transaction")

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )
