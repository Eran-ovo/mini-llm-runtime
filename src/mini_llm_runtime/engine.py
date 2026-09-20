"""连接 Scheduler、Qwen ModelRunner 与 Paged KV Cache 的同步执行循环。"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

import torch

from .block_admission import PagedBlockAdmissionController
from .paged_batch import PagedBatchDecodeAdapter
from .paged_kv_adapter import PagedRequestKVCache
from .qwen_model_runner import QwenPrefillRunner
from .request_metrics import RequestMetricsCollector
from .scheduler import (
    RequestScheduler,
    RequestState,
    SchedulerBatch,
    SchedulerStepUpdate,
)


@dataclass(frozen=True)
class EngineStepResult:
    """一个成功 step 的调度输入、生成事件与资源释放记录。"""

    batch: SchedulerBatch
    update: SchedulerStepUpdate
    released_blocks: tuple[tuple[str, tuple[int, ...]], ...]
    started_ns: int
    tokens_ready_ns: int
    completed_ns: int


class ContinuousBatchEngine:
    """同步、greedy 的最小 Continuous Batching Engine。

    当前 Prefill 逐请求执行，已有请求的 Decode 合并为一个 Paged batch。Scheduler
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
    ) -> None:
        if runner.decode_attention_backend != "paged_cuda":
            raise ValueError("ContinuousBatchEngine 需要 paged_cuda ModelRunner")
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
        self.scheduler = scheduler
        self.runner = runner
        self.admission = admission
        self._clock_ns = clock_ns
        self.metrics = metrics or RequestMetricsCollector()

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
        )
        self.metrics.register_request(request_id, arrival_ns)
        return request

    @torch.inference_mode()
    def step(self) -> EngineStepResult | None:
        """调度并执行一步；无可运行工作时返回 None。

        新 Prefill 先执行，旧请求 Decode 后执行。若任一模型调用失败，Decode 自身
        事务先回滚，随后 Engine 释放本轮 Prefill reservation 并撤销 Scheduler batch。
        """
        started_ns = self._clock_ns()
        batch = self.scheduler.schedule_step()
        if batch is None:
            return None

        # request_id -> GPU scalar token。最后按 batch.items 顺序一次性同步回 CPU。
        selected_tokens: dict[str, torch.Tensor] = {}
        prefill_ids = batch.prefill_request_ids
        try:
            self.metrics.require_registered(
                tuple(item.request_id for item in batch.items)
            )
            for item in batch.items:
                if item.request_id not in prefill_ids:
                    continue
                self.metrics.record_prefill_started(
                    item.request_id, self._clock_ns()
                )
                input_ids = torch.tensor(
                    (item.input_token_ids,),
                    dtype=torch.long,
                    device=self.runner.weights.embedding.device,
                )
                output = self.runner.prefill(
                    input_ids,
                    cache=PagedRequestKVCache(self.manager, item.request_id),
                )
                selected_tokens[item.request_id] = output.logits[
                    :, -1
                ].argmax(dim=-1)

            decode_items = tuple(
                item
                for item in batch.items
                if item.request_id in batch.decode_request_ids
            )
            if decode_items:
                decode_ids = tuple(item.request_id for item in decode_items)
                token_ids = torch.tensor(
                    tuple(item.input_token_ids for item in decode_items),
                    dtype=torch.long,
                    device=self.runner.weights.embedding.device,
                )
                output = self.runner.decode_batch(
                    token_ids,
                    cache=PagedBatchDecodeAdapter(self.manager, decode_ids),
                )
                decode_tokens = output.logits[:, -1].argmax(dim=-1)
                for index, request_id in enumerate(decode_ids):
                    selected_tokens[request_id] = decode_tokens[index : index + 1]
        except Exception:
            # Prefill 请求均为本 step 新接纳，释放后可从空 Cache 完整重试。
            self.admission.release_admitted(prefill_ids)
            self.scheduler.abort_step()
            raise

        # 按 Scheduler 原始顺序拼接，避免执行顺序（Prefill→Decode）改变结果归属。
        ordered_gpu_tokens = torch.cat(
            [selected_tokens[item.request_id] for item in batch.items]
        )
        ordered_cpu_tokens = ordered_gpu_tokens.cpu().tolist()
        tokens_ready_ns = self._clock_ns()
        generated = {
            item.request_id: int(token)
            for item, token in zip(batch.items, ordered_cpu_tokens, strict=True)
        }
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
        )
