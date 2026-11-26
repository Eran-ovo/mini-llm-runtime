"""Static/Continuous Batching 共用的同步 CPU 请求状态机。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


class RequestStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class FinishReason(str, Enum):
    EOS = "eos"
    MAX_TOKENS = "max_tokens"


class WorkKind(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class BatchingPolicy(str, Enum):
    """是否允许在已有 running cohort 中补入新请求。"""

    CONTINUOUS = "continuous"
    STATIC = "static"


@dataclass
class RequestState:
    """Scheduler 拥有的单请求逻辑状态；不直接持有 GPU K/V tensor。"""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    arrival_index: int
    status: RequestStatus = RequestStatus.WAITING
    prefilled: bool = False
    prefill_tokens: int = 0
    finish_reason: FinishReason | None = None
    _generated_token_ids: list[int] = field(default_factory=list, repr=False)

    @property
    def generated_token_ids(self) -> tuple[int, ...]:
        return tuple(self._generated_token_ids)

    @property
    def max_cache_tokens(self) -> int:
        # 最后一个生成 token 不再作为 Decode 输入，因此不写入 KV Cache。
        return len(self.prompt_token_ids) + self.max_new_tokens - 1


@dataclass(frozen=True)
class ScheduledRequest:
    """一个 engine step 中某个请求提交给 ModelRunner 的输入。"""

    request_id: str
    kind: WorkKind
    input_token_ids: tuple[int, ...]
    prompt_start: int = 0
    emits_token: bool = True

    @property
    def token_cost(self) -> int:
        # Decode 恰好消费一个 token；Prefill 可以消费 prompt 的一部分。
        return len(self.input_token_ids)


@dataclass(frozen=True)
class SchedulerBatch:
    step_index: int
    items: tuple[ScheduledRequest, ...]
    token_count: int

    @property
    def prefill_request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.items if item.kind is WorkKind.PREFILL)

    @property
    def decode_request_ids(self) -> tuple[str, ...]:
        return tuple(item.request_id for item in self.items if item.kind is WorkKind.DECODE)

    @property
    def admitted_request_ids(self) -> tuple[str, ...]:
        return tuple(
            item.request_id for item in self.items
            if item.kind is WorkKind.PREFILL and item.prompt_start == 0
        )


@dataclass(frozen=True)
class SchedulerStepUpdate:
    """执行结果写回后的事件；Engine 据此释放完成请求的 KV blocks。"""

    step_index: int
    emitted_tokens: tuple[tuple[str, int], ...]
    finished_request_ids: tuple[str, ...]


class RequestScheduler:
    """Decode-priority、strict-FIFO 的同步调度器，支持可选的分块 Prefill。

    `schedule_step()` 与 `apply_step_results()` 必须交替调用。同一时刻只允许一个
    outstanding batch，模拟 GPU batch 尚未完成时请求不能被重复调度。Static 与
    Continuous 只改变是否给未结束的 running cohort 补入新请求。
    """

    def __init__(
        self,
        *,
        max_running_requests: int,
        max_batch_tokens: int,
        max_mixed_prefill_tokens: int | None = None,
        prefill_chunk_tokens: int | None = None,
        admission_callback: Callable[[RequestState], bool] | None = None,
        batching_policy: BatchingPolicy = BatchingPolicy.CONTINUOUS,
    ) -> None:
        if max_running_requests <= 0 or max_batch_tokens <= 0:
            raise ValueError("max_running_requests 和 max_batch_tokens 必须 > 0")
        if max_running_requests > max_batch_tokens:
            raise ValueError(
                "max_running_requests 不能大于 max_batch_tokens，"
                "否则无法保证每个 running 请求每步 Decode 一次"
            )
        if not isinstance(batching_policy, BatchingPolicy):
            raise ValueError("batching_policy 必须是 BatchingPolicy")
        if max_mixed_prefill_tokens is not None and (
            isinstance(max_mixed_prefill_tokens, bool)
            or not isinstance(max_mixed_prefill_tokens, int)
            or max_mixed_prefill_tokens <= 0
        ):
            raise ValueError("max_mixed_prefill_tokens 必须是正整数或 None")
        if prefill_chunk_tokens is not None and (
            isinstance(prefill_chunk_tokens, bool)
            or not isinstance(prefill_chunk_tokens, int)
            or prefill_chunk_tokens <= 0
        ):
            raise ValueError("prefill_chunk_tokens 必须是正整数或 None")
        self.max_running_requests = max_running_requests
        self.max_batch_tokens = max_batch_tokens
        self.max_mixed_prefill_tokens = max_mixed_prefill_tokens
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self.batching_policy = batching_policy
        self._admission_callback = admission_callback
        self._requests: dict[str, RequestState] = {}
        self._waiting: deque[str] = deque()
        self._running: list[str] = []
        self._finished: list[str] = []
        self._next_arrival_index = 0
        self._next_step_index = 0
        self._outstanding: SchedulerBatch | None = None

    @property
    def waiting_request_ids(self) -> tuple[str, ...]:
        return tuple(self._waiting)

    @property
    def running_request_ids(self) -> tuple[str, ...]:
        return tuple(self._running)

    @property
    def finished_request_ids(self) -> tuple[str, ...]:
        return tuple(self._finished)

    @property
    def outstanding_batch(self) -> SchedulerBatch | None:
        return self._outstanding

    @property
    def has_unfinished_requests(self) -> bool:
        return bool(self._waiting or self._running or self._outstanding)

    def get_request(self, request_id: str) -> RequestState:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"未知 request_id={request_id!r}") from error

    def submit(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        eos_token_ids: int | Collection[int] | None = None,
        validate_request: Callable[[RequestState], object] | None = None,
    ) -> RequestState:
        """把新请求追加到 waiting 队尾；GPU batch 执行期间也允许到达。"""
        if not request_id:
            raise ValueError("request_id 不能为空")
        if request_id in self._requests:
            raise ValueError(f"request_id={request_id!r} 已经存在")
        prompt = self._validate_token_sequence(prompt_token_ids, "prompt_token_ids")
        if len(prompt) > self.max_batch_tokens and self.prefill_chunk_tokens is None:
            raise ValueError(
                f"prompt length={len(prompt)} 超过 max_batch_tokens="
                f"{self.max_batch_tokens}；当前版本不支持 chunked prefill"
            )
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens 必须是正整数")
        eos = self._normalize_eos(eos_token_ids)
        request = RequestState(
            request_id=request_id,
            prompt_token_ids=prompt,
            max_new_tokens=int(max_new_tokens),
            eos_token_ids=eos,
            arrival_index=self._next_arrival_index,
        )
        if validate_request is not None:
            validate_request(request)
        self._next_arrival_index += 1
        self._requests[request_id] = request
        self._waiting.append(request_id)
        return request

    def schedule_step(
        self, *, rollback_admissions: Callable[[tuple[str, ...]], object] | None = None
    ) -> SchedulerBatch | None:
        """先安排 Decode，再以剩余 budget 按 FIFO 准入 Prefill。

        外部准入回调预留的资源由 `rollback_admissions` 在同轮后续准入失败时回收。
        """
        if self._outstanding is not None:
            raise RuntimeError("上一个 Scheduler batch 尚未写回结果")
        if not self._running and not self._waiting:
            return None

        items: list[ScheduledRequest] = []
        # 必须在接纳新请求前记录：这个 step 是否以已有 Decode 请求开始。
        # 后面 self._running 会因 Prefill admission 增长，不能再用它判断 mixed step。
        running_at_step_start = bool(self._running)
        continuations = []
        # Decode 优先：已有 running 请求每步都获得一个 token，降低 TPOT 抖动。
        for request_id in self._running:
            request = self._requests[request_id]
            if not request.prefilled:
                continuations.append(request)
                continue
            if not request._generated_token_ids:
                raise RuntimeError(f"running 请求 {request_id!r} 缺少 Prefill 结果")
            items.append(
                ScheduledRequest(
                    request_id=request_id,
                    kind=WorkKind.DECODE,
                    input_token_ids=(request._generated_token_ids[-1],),
                )
            )

        prefill_budget = self.max_batch_tokens - len(items)
        if items:
            for limit in (self.max_mixed_prefill_tokens, self.prefill_chunk_tokens):
                if limit is not None:
                    prefill_budget = min(prefill_budget, limit)

        def append_prefill(request: RequestState) -> None:
            nonlocal prefill_budget
            start = request.prefill_tokens
            end = min(start + prefill_budget, len(request.prompt_token_ids))
            items.append(ScheduledRequest(
                request_id=request.request_id,
                kind=WorkKind.PREFILL,
                input_token_ids=request.prompt_token_ids[start:end],
                prompt_start=start,
                emits_token=end == len(request.prompt_token_ids),
            ))
            prefill_budget -= end - start

        # 已经占用 KV 的长请求继续推进，再使用剩余预算接纳新请求。
        for request in continuations:
            if prefill_budget == 0:
                break
            append_prefill(request)

        available_slots = self.max_running_requests - len(self._running)
        # 必须在 admission 前固定本 step 的决定。Static cohort 初建时可一次接纳
        # 多个请求；已有 running 请求时则完全禁止 refill。
        may_admit_prefill = (
            self.batching_policy is BatchingPolicy.CONTINUOUS
            or not running_at_step_start
        )
        admitted_ids: list[str] = []
        try:
            for request_id in self._waiting if may_admit_prefill else ():
                if available_slots == 0 or prefill_budget == 0:
                    break
                request = self._requests[request_id]
                prompt_cost = len(request.prompt_token_ids)
                # strict FIFO：队首放不下时停止，不能越过它选择更短的后续请求。
                if prompt_cost > prefill_budget and self.prefill_chunk_tokens is None:
                    break
                # False 表示暂时放不下且无副作用，True 表示已原子预留。
                if (
                    self._admission_callback is not None
                    and not self._admission_callback(request)
                ):
                    break
                admitted_ids.append(request_id)
                append_prefill(request)
                available_slots -= 1
        except Exception:
            if admitted_ids and rollback_admissions is not None:
                rollback_admissions(tuple(admitted_ids))
            raise

        if not items:
            # waiting 可能因外部 block budget 暂时无法接纳；队列保持不变。
            return None
        batch = SchedulerBatch(
            step_index=self._next_step_index,
            items=tuple(items),
            token_count=sum(item.token_cost for item in items),
        )
        for request_id in admitted_ids:
            self._waiting.popleft()
            self._requests[request_id].status = RequestStatus.RUNNING
        self._running.extend(admitted_ids)
        self._next_step_index += 1
        self._outstanding = batch
        return batch

    def apply_step_results(
        self, generated_tokens: Mapping[str, int]
    ) -> SchedulerStepUpdate:
        """原子校验并写回生成 token 与 Prefill 进度。"""
        batch = self._outstanding
        if batch is None:
            raise RuntimeError("当前没有等待写回的 Scheduler batch")
        expected_ids = tuple(item.request_id for item in batch.items if item.emits_token)
        actual_ids = tuple(generated_tokens)
        if set(actual_ids) != set(expected_ids) or len(actual_ids) != len(expected_ids):
            missing = sorted(set(expected_ids) - set(actual_ids))
            extra = sorted(set(actual_ids) - set(expected_ids))
            raise ValueError(f"step result 请求不匹配：missing={missing}, extra={extra}")
        validated = {
            request_id: self._validate_token(generated_tokens[request_id], "generated token")
            for request_id in expected_ids
        }

        finished: list[str] = []
        emitted: list[tuple[str, int]] = []
        for item in batch.items:
            request = self._requests[item.request_id]
            if item.kind is WorkKind.PREFILL:
                request.prefill_tokens += item.token_cost
                request.prefilled = item.emits_token
            if not item.emits_token:
                continue
            token = validated[item.request_id]
            request._generated_token_ids.append(token)
            emitted.append((item.request_id, token))

            if token in request.eos_token_ids:
                request.finish_reason = FinishReason.EOS
            elif len(request._generated_token_ids) >= request.max_new_tokens:
                request.finish_reason = FinishReason.MAX_TOKENS
            if request.finish_reason is not None:
                request.status = RequestStatus.FINISHED
                finished.append(item.request_id)

        if finished:
            finished_set = set(finished)
            self._running = [
                request_id
                for request_id in self._running
                if request_id not in finished_set
            ]
            self._finished.extend(finished)
        self._outstanding = None
        return SchedulerStepUpdate(
            step_index=batch.step_index,
            emitted_tokens=tuple(emitted),
            finished_request_ids=tuple(finished),
        )

    def abort_step(self) -> SchedulerBatch:
        """撤销 outstanding batch，使执行失败的 step 可以安全重试。

        Decode 请求在 schedule 时没有改变逻辑状态；新接纳的 Prefill 请求则需要
        从 running 退回 waiting 队首。GPU Cache 资源由 Engine/Admission Controller
        在调用本方法前释放。
        """
        batch = self._outstanding
        if batch is None:
            raise RuntimeError("当前没有可以撤销的 Scheduler batch")

        prefill_ids = batch.admitted_request_ids
        prefill_set = set(prefill_ids)
        for request_id in prefill_ids:
            request = self._requests[request_id]
            if request.prefilled or request._generated_token_ids:
                raise RuntimeError(
                    f"请求 {request_id!r} 已写回执行结果，不能再 abort step"
                )
            request.status = RequestStatus.WAITING
        self._running = [
            request_id
            for request_id in self._running
            if request_id not in prefill_set
        ]
        # 新请求可能在 GPU 执行期间到达；失败请求必须回到它们之前，并保持 FIFO。
        for request_id in reversed(prefill_ids):
            self._waiting.appendleft(request_id)

        self._outstanding = None
        # 失败尝试不消耗逻辑 step 编号，重试仍使用相同 index。
        self._next_step_index = batch.step_index
        return batch

    @classmethod
    def _validate_token_sequence(
        cls, values: Sequence[int], name: str
    ) -> tuple[int, ...]:
        tokens = tuple(values)
        if not tokens:
            raise ValueError(f"{name} 不能为空")
        return tuple(cls._validate_token(token, name) for token in tokens)

    @staticmethod
    def _validate_token(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} 必须是非负整数")
        return value

    @classmethod
    def _normalize_eos(
        cls, values: int | Collection[int] | None
    ) -> frozenset[int]:
        if values is None:
            return frozenset()
        raw = (values,) if isinstance(values, int) else tuple(values)
        return frozenset(cls._validate_token(value, "eos_token_ids") for value in raw)
