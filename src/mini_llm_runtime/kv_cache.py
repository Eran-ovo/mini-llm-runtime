"""固定容量、多层连续 KV Cache，以及事务式追加生命周期。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class LayerKVCache(Protocol):
    """ModelRunner 按层追加 K/V 所需的最小结构化接口。"""

    num_layers: int
    batch_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    length: int

    @property
    def pending(self) -> object | None: ...

    @property
    def available_token_capacity(self) -> int: ...

    def begin_append(self, token_count: int) -> object: ...

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None: ...

    def view_layer(
        self, layer_index: int, *, include_pending: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def commit_append(self) -> None: ...

    def abort_append(self) -> None: ...


@dataclass(frozen=True)
class PendingAppend:
    """尚未提交的逻辑 token 区间。"""

    start: int
    end: int

    @property
    def token_count(self) -> int:
        return self.end - self.start


class ContiguousKVCache:
    """固定 batch、固定容量、所有请求等长的多层 K/V buffer。

    布局为 [layer, batch, kv_head, token, head_dim]。一次 append 必须让所有层
    写完同一个 token 区间后才能 commit，避免全局 length 提前可见。
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        num_kv_heads: int,
        capacity: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        dimensions = (num_layers, batch_size, num_kv_heads, capacity, head_dim)
        if any(value <= 0 for value in dimensions):
            raise ValueError("KV Cache 所有维度必须 > 0")
        if not dtype.is_floating_point:
            raise ValueError("KV Cache dtype 必须是浮点类型")

        self.key = torch.empty(dimensions, dtype=dtype, device=device)
        self.value = torch.empty(dimensions, dtype=dtype, device=device)
        self.length = 0
        self._pending: PendingAppend | None = None
        self._written_layers: list[bool] = [False] * num_layers

    @property
    def num_layers(self) -> int:
        return self.key.shape[0]

    @property
    def batch_size(self) -> int:
        return self.key.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.key.shape[2]

    @property
    def capacity(self) -> int:
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
    def pending(self) -> PendingAppend | None:
        return self._pending

    @property
    def storage_nbytes(self) -> int:
        """K 与 V 两块物理 storage 的总字节数。"""
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()

    @property
    def available_token_capacity(self) -> int:
        return self.capacity - self.length

    def begin_append(self, token_count: int) -> PendingAppend:
        """预留一个尚不可全局读取的逻辑区间，不立即修改 length。"""
        if self._pending is not None:
            raise RuntimeError("已有未提交 append，不能嵌套 begin_append")
        if token_count <= 0:
            raise ValueError("token_count 必须 > 0")
        end = self.length + token_count
        if end > self.capacity:
            raise RuntimeError(
                f"KV Cache 容量不足：length={self.length}, append={token_count}, "
                f"capacity={self.capacity}"
            )
        self._pending = PendingAppend(self.length, end)
        self._written_layers = [False] * self.num_layers
        return self._pending

    def write_layer(
        self, layer_index: int, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """写入当前事务中某一层的新 K/V，但尚不推进全局 length。"""
        pending = self._require_pending()
        self._validate_layer_index(layer_index)
        if self._written_layers[layer_index]:
            raise RuntimeError(f"layer {layer_index} 在当前 append 中已经写入")
        if key.shape != value.shape:
            raise ValueError("新 K/V shape 必须相同")
        expected = (
            self.batch_size,
            self.num_kv_heads,
            pending.token_count,
            self.head_dim,
        )
        if tuple(key.shape) != expected:
            raise ValueError(f"新 K/V shape={tuple(key.shape)}，预期={expected}")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("新 K/V dtype 必须与 Cache 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("新 K/V device 必须与 Cache 相同")

        token_slice = slice(pending.start, pending.end)
        self.key[layer_index, :, :, token_slice, :].copy_(key)
        self.value[layer_index, :, :, token_slice, :].copy_(value)
        self._written_layers[layer_index] = True

    def view_layer(
        self, layer_index: int, *, include_pending: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回某层有效前缀；已写层可选择看到当前尚未 commit 的 token。"""
        self._validate_layer_index(layer_index)
        visible_length = self.length
        if include_pending:
            pending = self._require_pending()
            if not self._written_layers[layer_index]:
                raise RuntimeError(
                    f"layer {layer_index} 尚未写入 pending K/V，不能读取未提交区间"
                )
            visible_length = pending.end
        return (
            self.key[layer_index, :, :, :visible_length, :],
            self.value[layer_index, :, :, :visible_length, :],
        )

    def commit_append(self) -> None:
        """仅当所有层都写完时，使 pending 区间对整个 Cache 可见。"""
        pending = self._require_pending()
        missing = [index for index, written in enumerate(self._written_layers) if not written]
        if missing:
            raise RuntimeError(f"不能 commit，尚未写入的层：{missing}")
        self.length = pending.end
        self._pending = None
        self._written_layers = [False] * self.num_layers

    def abort_append(self) -> None:
        """放弃 pending 可见性；已写字节留作垃圾，下一次 append 会覆盖。"""
        self._require_pending()
        self._pending = None
        self._written_layers = [False] * self.num_layers

    def append_all(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """便利入口：一次追加所有层，输入为 [L,B,Hkv,Snew,D]。"""
        if key.shape != value.shape:
            raise ValueError("新 K/V shape 必须相同")
        if key.ndim != 5:
            raise ValueError("全层 K/V 必须是 [layer,batch,kv_head,token,head_dim]")
        expected_prefix = (self.num_layers, self.batch_size, self.num_kv_heads)
        if tuple(key.shape[:3]) != expected_prefix or key.shape[4] != self.head_dim:
            raise ValueError(
                f"全层 K/V shape={tuple(key.shape)} 与 Cache 布局不匹配"
            )
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("新 K/V dtype 必须与 Cache 相同")
        if key.device != self.device or value.device != self.device:
            raise ValueError("新 K/V device 必须与 Cache 相同")

        self.begin_append(key.shape[3])
        try:
            for layer_index in range(self.num_layers):
                self.write_layer(layer_index, key[layer_index], value[layer_index])
            self.commit_append()
        except Exception:
            # append_all 对调用者提供原子可见性；失败后 length 保持原值。
            if self._pending is not None:
                self.abort_append()
            raise

    def reset(self) -> None:
        """清空逻辑内容但不清零物理 buffer，后续写入会覆盖旧字节。"""
        self.length = 0
        self._pending = None
        self._written_layers = [False] * self.num_layers

    def _require_pending(self) -> PendingAppend:
        if self._pending is None:
            raise RuntimeError("当前没有 begin_append 创建的事务")
        return self._pending

    def _validate_layer_index(self, layer_index: int) -> None:
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(
                f"layer_index={layer_index} 越界，有效范围 [0,{self.num_layers})"
            )
