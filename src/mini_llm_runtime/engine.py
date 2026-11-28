"""连接 Scheduler、Qwen ModelRunner 与 Paged KV Cache 的同步执行循环。"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .block_admission import PagedBlockAdmissionController
from .paged_batch import PagedBatchDecodeAdapter, PagedBatchPrefillAdapter
from .qwen_model_runner import PACKED_PREFILL_ATTENTION_BACKENDS, QwenPrefillRunner
from .request_metrics import RequestMetricsCollector
from .scheduler import (
    RequestScheduler,
    RequestState,
    SchedulerBatch,
    SchedulerStepUpdate,
    WorkKind,
)


@contextmanager
def _nvtx_range(enabled: bool, message: str) -> Iterator[None]:
    """可选 NVTX range；异常路径同样保证 pop，避免破坏后续 timeline。"""
    if enabled:
        torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@dataclass(frozen=True)
class EngineStepResult:
    """一个成功 step 的调度输入、生成事件与资源释放记录。"""

    batch: SchedulerBatch
    update: SchedulerStepUpdate
    released_blocks: tuple[tuple[str, tuple[int, ...]], ...]
    started_ns: int
    tokens_ready_ns: int
    completed_ns: int
    prefill_attention_backend: str | None


AUTO_PREFILL_ATTENTION_BACKENDS = {
    "auto": "segmented_sdpa",
    "auto_varlen_sdpa": "varlen_sdpa",
    "auto_varlen_cuda": "varlen_cuda",
}
PREFILL_ATTENTION_BACKEND_CHOICES = tuple(sorted(
    PACKED_PREFILL_ATTENTION_BACKENDS | AUTO_PREFILL_ATTENTION_BACKENDS.keys()
))


def select_prefill_attention_backend(
    configured: str, lengths: Sequence[int]
) -> str:
    """根据本轮 prompt 形状选择 Attention；显式 backend 永远优先。"""
    if configured not in PREFILL_ATTENTION_BACKEND_CHOICES:
        raise ValueError(f"未知的 prefill_attention_backend: {configured}")
    if configured not in AUTO_PREFILL_ATTENTION_BACKENDS:
        return configured
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("auto Prefill 需要非空且为正数的 prompt lengths")
    total_tokens = sum(lengths)
    if total_tokens >= 128 and lengths[0] >= 32 and all(length == lengths[0] for length in lengths):
        return "batched_sdpa"
    if total_tokens >= 512 and (
        max(lengths) >= 128 or total_tokens >= 32 * len(lengths)
    ):
        return AUTO_PREFILL_ATTENTION_BACKENDS[configured]
    return "masked"


class ContinuousBatchEngine:
    """同步、greedy 的最小 Continuous Batching Engine。

    同轮 Prefill 使用 packed batch，已有请求的 Decode 使用 Paged batch。Scheduler
    必须使用本 Engine 所持 Admission Controller 的 `try_admit` 作为 callback。
    """

    def __init__(
        self,
        *,
        scheduler: RequestScheduler,
        runner: QwenPrefillRunner,
        admission: PagedBlockAdmissionController,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        metrics: RequestMetricsCollector | None = None,
        enable_nvtx: bool = False,
        prefill_attention_backend: str = "auto",
        prefill_kv_write_backend: str = "vectorized",
    ) -> None:
        if runner.decode_attention_backend != "paged_cuda":
            raise ValueError("ContinuousBatchEngine 需要 paged_cuda ModelRunner")
        if prefill_attention_backend not in PREFILL_ATTENTION_BACKEND_CHOICES:
            raise ValueError(f"未知的 prefill_attention_backend: {prefill_attention_backend}")
        if prefill_kv_write_backend not in {"scalar", "vectorized"}:
            raise ValueError("prefill_kv_write_backend 必须是 scalar 或 vectorized")
        storage = admission.manager.storage
        expected = (
            runner.config.num_hidden_layers,
            runner.config.num_key_value_heads,
            runner.config.head_dim,
            runner.weights.embedding.dtype,
            runner.weights.embedding.device,
        )
        actual = (
            storage.num_layers,
            storage.num_kv_heads,
            storage.head_dim,
            storage.dtype,
            storage.device,
        )
        if actual != expected:
            raise ValueError(
                f"ModelRunner 与 Paged KV storage 布局不匹配："
                f"actual={actual}, expected={expected}"
            )
        if enable_nvtx and runner.weights.embedding.device.type != "cuda":
            raise ValueError("NVTX profiling 只支持 CUDA Engine")
        self.scheduler = scheduler
        self.runner = runner
        self.admission = admission
        self._clock_ns = clock_ns
        self.metrics = metrics or RequestMetricsCollector()
        self.enable_nvtx = enable_nvtx
        self.prefill_attention_backend = prefill_attention_backend
        self.prefill_kv_write_backend = prefill_kv_write_backend

    @property
    def manager(self):
        return self.admission.manager

    def submit(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: int | Collection[int] | None = None,
    ) -> RequestState:
        """提交请求并以同一单调时钟记录 arrival。"""
        arrival_ns = self._clock_ns()
        request = self.scheduler.submit(
            request_id,
            prompt_token_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            validate_request=self._validate_request,
        )
        self.metrics.register_request(request_id, arrival_ns)
        return request

    def _validate_request(self, request: RequestState) -> None:
        if (
            self.scheduler.prefill_chunk_tokens is not None
            and request.max_cache_tokens > self.runner.config.max_position_embeddings
        ):
            raise ValueError("请求超过模型 max_position_embeddings")
        self.admission.validate_request_capacity(request)

    @torch.inference_mode()
    def step(self) -> EngineStepResult | None:
        """调度并执行一步；无可运行工作时返回 None。

        新 Prefill 先执行，旧请求 Decode 后执行。取回 token 后才提交 Decode。
        执行或取回结果失败时，Engine 撤销 Decode append、Prefill 预留与 Scheduler batch。
        """
        started_ns = self._clock_ns()
        with _nvtx_range(self.enable_nvtx, "scheduler.schedule_step"):
            batch = self.scheduler.schedule_step(
                rollback_admissions=self.admission.release_admitted,
                allow_tokens=self.admission.plan_step(),
            )
        if batch is None:
            return None
        message = (
            f"engine.step:{batch.step_index}:"
            f"prefill={len(batch.prefill_request_ids)}:"
            f"decode={len(batch.decode_request_ids)}"
        )
        with _nvtx_range(self.enable_nvtx, message):
            return self._execute_batch(batch, started_ns)

    def _execute_batch(
        self, batch: SchedulerBatch, started_ns: int
    ) -> EngineStepResult:
        """执行已经选好的 Scheduler batch；由 `step()` 负责创建外层 NVTX range。"""

        selected_ids: list[str] = []
        selected_tokens: list[torch.Tensor] = []
        selected_prefill_backend: str | None = None
        decode_cache = None
        prefill_cache = None
        device = self.runner.weights.embedding.device
        try:
            self.metrics.require_registered(
                tuple(item.request_id for item in batch.items)
            )
            prefill_items = tuple(
                item for item in batch.items if item.kind is WorkKind.PREFILL
            )
            if prefill_items:
                prefill_lengths = tuple(len(item.input_token_ids) for item in prefill_items)
                selected_prefill_backend = select_prefill_attention_backend(
                    self.prefill_attention_backend, prefill_lengths
                )
                with _nvtx_range(
                    self.enable_nvtx,
                    f"model.prefill_batch:requests={len(prefill_items)}:"
                    f"tokens={sum(prefill_lengths)}:backend={selected_prefill_backend}",
                ):
                    for request_id in batch.admitted_request_ids:
                        self.metrics.record_prefill_started(request_id, self._clock_ns())
                    # Scheduler 顺序同时决定 packed token 区间和结果行归属。
                    input_ids = torch.tensor(
                        (tuple(
                            token
                            for item in prefill_items
                            for token in item.input_token_ids
                        ),),
                        dtype=torch.long,
                        device=device,
                    )
                    prefill_cache = PagedBatchPrefillAdapter(
                        self.manager,
                        batch.prefill_request_ids,
                        prefill_lengths,
                        write_backend=self.prefill_kv_write_backend,
                        prefix_lengths=tuple(item.prompt_start for item in prefill_items),
                    )
                    emit_rows = tuple(i for i, item in enumerate(prefill_items) if item.emits_token)
                    options = {}
                    if (
                        self.scheduler.prefill_chunk_tokens is not None
                        or self.admission.allocation_policy == "on_demand"
                    ):
                        options["defer_commit"] = True
                        if len(emit_rows) != len(prefill_items):
                            options["emit_rows"] = emit_rows
                    output = self.runner.prefill_batch(
                        input_ids,
                        cache=prefill_cache,
                        attention_backend=selected_prefill_backend,
                        **options,
                    )
                    if prefill_cache.has_prefix:
                        selected_prefill_backend = "chunked_sdpa"
                    if emit_rows:
                        selected_ids.extend(prefill_items[i].request_id for i in emit_rows)
                        selected_tokens.append(output.logits[:, -1].argmax(dim=-1))

            decode_items = tuple(
                item
                for item in batch.items
                if item.kind is WorkKind.DECODE
            )
            if decode_items:
                decode_ids = tuple(item.request_id for item in decode_items)
                with _nvtx_range(
                    self.enable_nvtx,
                    f"model.decode_batch:batch={len(decode_ids)}",
                ):
                    token_ids = torch.tensor(
                        tuple(item.input_token_ids for item in decode_items),
                        dtype=torch.long,
                        device=device,
                    )
                    decode_cache = PagedBatchDecodeAdapter(self.manager, decode_ids)
                    decode_tokens = self.runner.decode_batch_greedy(
                        token_ids,
                        cache=decode_cache,
                        defer_commit=True,
                    )
                    selected_ids.extend(decode_ids)
                    selected_tokens.append(decode_tokens)

            # 两阶段共享一次 D2H，Scheduler 写回时按原始 batch 的顺序发出事件。
            with _nvtx_range(self.enable_nvtx, "token.d2h_sync"):
                if selected_tokens:
                    tokens = (
                        selected_tokens[0] if len(selected_tokens) == 1
                        else torch.cat(selected_tokens)
                    )
                    cpu_tokens = tokens.cpu().tolist()
                else:
                    if device.type == "cuda":
                        torch.cuda.current_stream(device).synchronize()
                    cpu_tokens = []
            tokens_ready_ns = self._clock_ns()
            generated = dict(zip(selected_ids, cpu_tokens, strict=True))
            if decode_cache is not None:
                decode_cache.commit_decode()
            if prefill_cache is not None and prefill_cache.active:
                prefill_cache.commit_prefill()
        except Exception:
            with _nvtx_range(self.enable_nvtx, "engine.rollback"):
                if decode_cache is not None and decode_cache.active:
                    decode_cache.abort_decode()
                if prefill_cache is not None and prefill_cache.active:
                    prefill_cache.abort_prefill()
                self.admission.release_admitted(batch.admitted_request_ids)
                self.scheduler.abort_step()
            raise

        with _nvtx_range(self.enable_nvtx, "scheduler.apply_and_release"):
            update = self.scheduler.apply_step_results(generated)
            released = self.admission.release_finished(update.finished_request_ids)
        completed_ns = self._clock_ns()
        self.metrics.record_tokens(update.emitted_tokens, tokens_ready_ns)
        self.metrics.record_completed(update.finished_request_ids, completed_ns)
        return EngineStepResult(
            batch=batch,
            update=update,
            released_blocks=tuple(released.items()),
            started_ns=started_ns,
            tokens_ready_ns=tokens_ready_ns,
            completed_ns=completed_ns,
            prefill_attention_backend=selected_prefill_backend,
        )
