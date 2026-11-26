"""按 Scheduler 顺序组织多请求、单 token Paged Decode 事务。"""

from __future__ import annotations

from array import array
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import accumulate, chain

import torch

from .cuda_extension import load_cuda_extension
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
        self._decode_buffer: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self._metadata: tuple[torch.Tensor, torch.Tensor] | None = None
        self._physical_indices: tuple[torch.Tensor, torch.Tensor] | None = None

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

    @property
    def current_positions(self) -> tuple[int, ...]:
        """返回 CPU metadata 中的 Decode positions，不触发 GPU 同步。"""
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
        return tuple(positions)

    def build_position_ids(
        self, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """返回 `[B,1]` Decode absolute positions，即 append 前的各请求长度。"""
        target = self.device if device is None else torch.device(device)
        if self._position_ids is not None and target == self.device:
            return self._position_ids
        return torch.tensor(
            self.current_positions, dtype=torch.long, device=target
        ).unsqueeze(1)

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
            locations = [
                table.locate(table.pending.start, include_pending=True)
                for table in self._tables
            ]
            wide = array("q", chain(
                (location.block_id for location in locations),
                (location.block_offset for location in locations),
                (table.pending.start for table in self._tables),
            ))
            max_blocks = max(table.block_count for table in self._tables)
            padded_rows = chain.from_iterable(
                table.block_ids + (-1,) * (max_blocks - table.block_count)
                for table in self._tables
            )
            narrow = array("i", chain(padded_rows, (table.pending.end for table in self._tables)))
            # 索引与 position 使用 int64，Attention 表与长度使用 int32。
            # 按元素宽度排列以便共享一次 H2D，并为 CUDA Graph 保留单个输入缓冲区。
            payload = bytearray(wide)
            payload.extend(narrow.tobytes())
            buffer = torch.frombuffer(payload, dtype=torch.uint8).to(self.device)
            self._set_decode_buffer(buffer)
        except Exception:
            # 后面的请求分配失败时，撤销前面请求可能新申请的物理 block。
            for table in reversed(begun):
                table.abort_append()
            raise
        self._active = True
        self._written_layers = [False] * self.num_layers

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """写入 `[B,Hkv,1,D]` 当前 token K/V，batch row 遵循 request_ids。"""
        blocks, offsets = self._write_slots(layer_index, key, value)
        storage = self.manager.storage
        storage.key[layer_index, blocks, :, offsets, :] = key.squeeze(2)
        storage.value[layer_index, blocks, :, offsets, :] = value.squeeze(2)
        self._written_layers[layer_index] = True

    def rotate_and_write_layer(
        self, layer_index: int, query: torch.Tensor, key: torch.Tensor,
        value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
    ) -> torch.Tensor:
        """返回旋转后的 Q，并将旋转后的 K 和原始 V 写入当前 pending slot。"""
        blocks, offsets = self._write_slots(layer_index, key, value)
        storage = self.manager.storage
        rotated_query = load_cuda_extension().decode_rope_and_write(
            query, key, value, cosine, sine,
            storage.key[layer_index], storage.value[layer_index], blocks, offsets,
        )
        self._written_layers[layer_index] = True
        return rotated_query

    def _write_slots(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_active()
        self._validate_layer_index(layer_index)
        if self._written_layers[layer_index]:
            raise RuntimeError(f"layer {layer_index} 在当前 Decode 中已经写入")
        expected = (self.batch_size, self.num_kv_heads, 1, self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"Decode 新 K/V 必须是 {expected}")

        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("新 K/V dtype 必须与 Paged storage 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("新 K/V device 必须与 Paged storage 相同")
        assert self._physical_indices is not None
        return self._physical_indices

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
        block_table, sequence_lengths = self._attention_metadata()
        storage = self.manager.storage
        return PagedDecodeAttentionInputs(
            key_cache=storage.key[layer_index],
            value_cache=storage.value[layer_index],
            block_table=block_table,
            sequence_lengths=sequence_lengths,
        )

    def _attention_metadata(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._metadata is not None
        return self._metadata

    def _graph_inputs(self) -> torch.Tensor:
        """返回本步 Decode 的 position、slot 与 Attention 元数据缓冲区。"""
        self._require_active()
        assert self._decode_buffer is not None
        return self._decode_buffer

    def _set_decode_buffer(self, buffer: torch.Tensor) -> None:
        batch = self.batch_size
        wide_bytes = 3 * batch * 8
        wide = buffer[:wide_bytes].view(torch.int64).view(3, batch)
        narrow = buffer[wide_bytes:].view(torch.int32)
        block_table = narrow[:-batch].view(batch, -1)
        blocks, offsets, positions = wide.unbind()
        self._decode_buffer = buffer
        self._physical_indices = (blocks, offsets)
        self._position_ids = positions.unsqueeze(1)
        self._metadata = (block_table, narrow[-batch:])

    @contextmanager
    def _use_graph_inputs(self, buffer: torch.Tensor) -> Iterator[None]:
        original = self._graph_inputs()
        self._set_decode_buffer(buffer)
        try:
            yield
        finally:
            self._set_decode_buffer(original)

    def _set_all_layers_written(self, written: bool) -> None:
        self._require_active()
        self._written_layers = [written] * self.num_layers

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
        self._decode_buffer = None
        self._position_ids = None
        self._metadata = None
        self._physical_indices = None

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Decode batch transaction")

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )


class PagedBatchPrefillAdapter:
    """把 packed Prefill 写入各自的物理 KV block，支持追加到已提交的 prefix。

    `lengths[i]` 对应 `request_ids[i]`；所有层成功后才一起提交长度。
    默认使用批量物理 slot 索引写入；scalar 路径保留为独立正确性基线。
    """

    def __init__(
        self,
        manager: PagedKVCacheManager,
        request_ids: tuple[str, ...],
        lengths: tuple[int, ...],
        *,
        write_backend: str = "vectorized",
        prefix_lengths: tuple[int, ...] | None = None,
    ) -> None:
        if not request_ids or len(request_ids) != len(lengths):
            raise ValueError("Prefill request_ids 与 lengths 必须非空且等长")
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Prefill batch 不能包含重复 request_id")
        if any(length <= 0 for length in lengths):
            raise ValueError("Prefill lengths 必须全部 > 0")
        if write_backend not in {"scalar", "vectorized"}:
            raise ValueError("write_backend 必须是 scalar 或 vectorized")
        if prefix_lengths is None:
            prefix_lengths = (0,) * len(lengths)
        if len(prefix_lengths) != len(lengths) or any(prefix < 0 for prefix in prefix_lengths):
            raise ValueError("Prefill prefix_lengths 必须与 lengths 对齐且非负")
        self.manager = manager
        self.request_ids = tuple(request_ids)
        self.lengths = tuple(lengths)
        self._offsets = (0, *accumulate(self.lengths))
        self.prefix_lengths = prefix_lengths
        self.has_prefix = any(prefix_lengths)
        self.kv_lengths = tuple(p + n for p, n in zip(prefix_lengths, lengths, strict=True))
        self.kv_offsets = (0, *accumulate(self.kv_lengths))
        self.write_backend = write_backend
        self._tables = tuple(manager.get_request(rid) for rid in request_ids)
        self._active = False
        self._written_layers = [False] * self.num_layers
        self._physical_indices: tuple[torch.Tensor, torch.Tensor] | None = None
        self._full_indices: tuple[torch.Tensor, torch.Tensor] | None = None

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

    @property
    def offsets(self) -> tuple[int, ...]:
        """packed token 的边界，例如 lengths=(3,2) 对应 (0,3,5)。"""
        return self._offsets

    def begin_prefill(self) -> None:
        if self._active:
            raise RuntimeError("Prefill batch 已经有 active transaction")
        for table, prefix in zip(self._tables, self.prefix_lengths, strict=True):
            if table.pending is not None or table.token_count != prefix:
                raise ValueError(f"请求 {table.request_id!r} 的 Cache 必须为空或与已提交 prefix 一致")
        begun: list[RequestBlockTable] = []
        try:
            for table, length in zip(self._tables, self.lengths, strict=True):
                table.begin_append(length)
                begun.append(table)
            if self.write_backend == "vectorized":
                self._physical_indices = self._build_slots(self.prefix_lengths)
            if self.has_prefix:
                self._full_indices = self._build_slots((0,) * self.batch_size)
        except Exception:
            for table in reversed(begun):
                table.abort_append()
            self._physical_indices = None
            self._full_indices = None
            raise
        self._active = True
        self._written_layers = [False] * self.num_layers

    def _build_slots(self, starts: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        block_size = self.manager.block_size
        used_blocks: set[int] = set()
        blocks: list[int] = []
        offsets: list[int] = []
        for table, start, end in zip(self._tables, starts, self.kv_lengths, strict=True):
            block_ids = table.block_ids
            for page in range(start // block_size, (end + block_size - 1) // block_size):
                block_id = block_ids[page]
                if block_id in used_blocks:
                    raise RuntimeError("packed Prefill 存在重复物理 slot")
                used_blocks.add(block_id)
                page_start = page * block_size
                tile = range(max(start - page_start, 0), min(end - page_start, block_size))
                blocks.extend([block_id] * len(tile))
                offsets.extend(tile)
        return torch.tensor((blocks, offsets), dtype=torch.long, device=self.device).unbind()

    def gather_prefill(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """在本层写入后打包该层的 prefix 与当前 chunk。"""
        if self._full_indices is None or not self._written_layers[layer_index]:
            raise RuntimeError("Prefill gather 需要已写入的 chunk")
        blocks, offsets = self._full_indices
        storage = self.manager.storage
        return (
            storage.key[layer_index, blocks, :, offsets, :],
            storage.value[layer_index, blocks, :, offsets, :],
        )

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        self._validate_write_layer(layer_index, key, value)
        if self.write_backend == "vectorized":
            blocks, offsets = self._vectorized_slots(key, value)
            storage = self.manager.storage
            # 配对索引 [block,offset] 保留 storage 中独立的 kv_head 维。
            storage.key[layer_index, blocks, :, offsets, :] = key[0].transpose(0, 1)
            storage.value[layer_index, blocks, :, offsets, :] = value[0].transpose(0, 1)
        else:
            for table, start, end in zip(
                self._tables, self.offsets[:-1], self.offsets[1:], strict=True
            ):
                self.manager.storage.write_layer(
                    table, layer_index, key[0, :, start:end], value[0, :, start:end]
                )
        self._written_layers[layer_index] = True

    def rotate_and_write_layer(
        self, layer_index: int, query: torch.Tensor, key: torch.Tensor,
        value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """保留 Attention 所需的旋转 Q/K，同时写入当前 packed K/V。"""
        self._validate_write_layer(layer_index, key, value)
        blocks, offsets = self._vectorized_slots(key, value)
        storage = self.manager.storage
        # [1,heads,tokens,dim] → [tokens,heads,1,dim]，供每个 token 独立寻址。
        packed = tuple(tensor.transpose(1, 2).squeeze(0).unsqueeze(2)
                       for tensor in (query, key, value))
        rotated = load_cuda_extension().prefill_rope_and_write(
            *packed, cosine.reshape(-1, 1, 1, self.head_dim),
            sine.reshape(-1, 1, 1, self.head_dim),
            storage.key[layer_index], storage.value[layer_index], blocks, offsets,
        )
        self._written_layers[layer_index] = True
        query_out, key_out = rotated
        return (
            query_out.squeeze(2).transpose(0, 1).unsqueeze(0),
            key_out.squeeze(2).transpose(0, 1).unsqueeze(0),
        )

    def _validate_write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Prefill batch transaction")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index={layer_index} 越界")
        if self._written_layers[layer_index]:
            raise RuntimeError(f"layer {layer_index} 已写入")
        expected = (1, self.num_kv_heads, self.offsets[-1], self.head_dim)
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError(f"packed K/V 必须是 {expected}")

    def _vectorized_slots(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("packed K/V dtype 必须与 Paged storage 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("packed K/V device 必须与 Paged storage 相同")
        if self._physical_indices is None:
            raise RuntimeError("缺少 Prefill 物理 slot 索引")
        return self._physical_indices

    def _graph_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        if not self._active or self._physical_indices is None:
            raise RuntimeError("Prefill CUDA Graph 需要活跃的 vectorized transaction")
        return self._physical_indices

    @contextmanager
    def _use_graph_inputs(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> Iterator[None]:
        physical = self._physical_indices
        self._physical_indices = inputs
        try:
            yield
        finally:
            self._physical_indices = physical

    def _set_all_layers_written(self, written: bool) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Prefill batch transaction")
        self._written_layers = [written] * self.num_layers

    def commit_prefill(self) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Prefill batch transaction")
        missing = [i for i, written in enumerate(self._written_layers) if not written]
        if missing:
            raise RuntimeError(f"不能 commit，尚未写入的层：{missing}")
        for table in self._tables:
            table.commit_append()
        self._active = False
        self._physical_indices = None
        self._full_indices = None

    def abort_prefill(self) -> None:
        if not self._active:
            raise RuntimeError("当前没有 active Prefill batch transaction")
        for table in reversed(self._tables):
            table.abort_append()
        self._active = False
        self._written_layers = [False] * self.num_layers
        self._physical_indices = None
        self._full_indices = None
