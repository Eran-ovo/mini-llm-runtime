"""请求级单调时钟事件与 TTFT/TPOT/E2E 派生指标。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class TokenEvent:
    """一个 token 对 CPU Scheduler 可见的时间，而不是 kernel 启动时间。"""

    token_id: int
    ready_ns: int


@dataclass(frozen=True)
class RequestMetrics:
    """请求时间线的不可变快照，保留原始事件并按需计算指标。"""

    request_id: str
    arrival_ns: int
    prefill_attempt_started_ns: tuple[int, ...]
    token_events: tuple[TokenEvent, ...]
    completed_ns: int | None

    @property
    def queue_wait_ns(self) -> int | None:
        if not self.prefill_attempt_started_ns:
            return None
        return self.prefill_attempt_started_ns[0] - self.arrival_ns

    @property
    def ttft_ns(self) -> int | None:
        if not self.token_events:
            return None
        return self.token_events[0].ready_ns - self.arrival_ns

    @property
    def inter_token_ns(self) -> tuple[int, ...]:
        return tuple(
            current.ready_ns - previous.ready_ns
            for previous, current in zip(
                self.token_events, self.token_events[1:]
            )
        )

    @property
    def median_tpot_ns(self) -> float | None:
        samples = self.inter_token_ns
        return float(statistics.median(samples)) if samples else None

    @property
    def e2e_ns(self) -> int | None:
        # 用户可见 E2E 在最后一枚 token ready 时结束，不混入资源清理开销。
        if not self.token_events:
            return None
        return self.token_events[-1].ready_ns - self.arrival_ns

    @property
    def post_token_completion_ns(self) -> int | None:
        """Scheduler 写回与 KV block 释放等最后 token 之后的收尾时间。"""
        if self.completed_ns is None or not self.token_events:
            return None
        return self.completed_ns - self.token_events[-1].ready_ns

    @staticmethod
    def ns_to_ms(value: int | float | None) -> float | None:
        if value is None:
            return None
        return float(value) / NANOSECONDS_PER_MILLISECOND

    @property
    def queue_wait_ms(self) -> float | None:
        return self.ns_to_ms(self.queue_wait_ns)

    @property
    def ttft_ms(self) -> float | None:
        return self.ns_to_ms(self.ttft_ns)

    @property
    def median_tpot_ms(self) -> float | None:
        return self.ns_to_ms(self.median_tpot_ns)

    @property
    def e2e_ms(self) -> float | None:
        return self.ns_to_ms(self.e2e_ns)

    @property
    def post_token_completion_ms(self) -> float | None:
        return self.ns_to_ms(self.post_token_completion_ns)


@dataclass
class _MutableTimeline:
    arrival_ns: int
    prefill_attempt_started_ns: list[int] = field(default_factory=list)
    token_events: list[TokenEvent] = field(default_factory=list)
    completed_ns: int | None = None

    @property
    def last_event_ns(self) -> int:
        candidates = [self.arrival_ns]
        candidates.extend(self.prefill_attempt_started_ns)
        candidates.extend(event.ready_ns for event in self.token_events)
        if self.completed_ns is not None:
            candidates.append(self.completed_ns)
        return max(candidates)


class RequestMetricsCollector:
    """由 Engine 写入的请求事件仓库；所有 timestamp 使用同一单调时钟。"""

    def __init__(self) -> None:
        self._timelines: dict[str, _MutableTimeline] = {}

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._timelines)

    def register_request(self, request_id: str, arrival_ns: int) -> None:
        if request_id in self._timelines:
            raise ValueError(f"请求 {request_id!r} 已经注册 metrics")
        self._validate_timestamp(arrival_ns)
        self._timelines[request_id] = _MutableTimeline(arrival_ns=arrival_ns)

    def require_registered(self, request_ids: tuple[str, ...]) -> None:
        missing = [rid for rid in request_ids if rid not in self._timelines]
        if missing:
            raise RuntimeError(
                f"Engine 请求必须通过 engine.submit() 注册 metrics：{missing}"
            )

    def record_prefill_started(self, request_id: str, timestamp_ns: int) -> None:
        timeline = self._get_mutable(request_id)
        self._validate_next_timestamp(timeline, timestamp_ns)
        if timeline.token_events or timeline.completed_ns is not None:
            raise RuntimeError("已经产生 token 的请求不能再次开始 Prefill")
        timeline.prefill_attempt_started_ns.append(timestamp_ns)

    def record_tokens(
        self,
        emitted_tokens: tuple[tuple[str, int], ...],
        timestamp_ns: int,
    ) -> None:
        self._validate_timestamp(timestamp_ns)
        if len({request_id for request_id, _ in emitted_tokens}) != len(
            emitted_tokens
        ):
            raise ValueError("同一个 step 不能为同一请求记录多个 token")
        invalid_token_ids = [
            (request_id, token_id)
            for request_id, token_id in emitted_tokens
            if isinstance(token_id, bool) or not isinstance(token_id, int)
        ]
        if invalid_token_ids:
            raise ValueError(f"token_id 必须是整数：{invalid_token_ids}")
        timelines = [
            self._get_mutable(request_id) for request_id, _ in emitted_tokens
        ]
        for timeline in timelines:
            self._validate_next_timestamp(timeline, timestamp_ns)
            if timeline.completed_ns is not None:
                raise RuntimeError("已完成请求不能继续记录 token")
        for (request_id, token_id), timeline in zip(
            emitted_tokens, timelines, strict=True
        ):
            timeline.token_events.append(TokenEvent(token_id, timestamp_ns))

    def record_completed(
        self, request_ids: tuple[str, ...], timestamp_ns: int
    ) -> None:
        self._validate_timestamp(timestamp_ns)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("完成请求 ID 不能重复")
        timelines = [self._get_mutable(request_id) for request_id in request_ids]
        for timeline in timelines:
            self._validate_next_timestamp(timeline, timestamp_ns)
            if timeline.completed_ns is not None:
                raise RuntimeError("请求已经记录 completion")
            if not timeline.token_events:
                raise RuntimeError("尚未产生 token 的请求不能完成")
        for timeline in timelines:
            timeline.completed_ns = timestamp_ns

    def snapshot(self, request_id: str) -> RequestMetrics:
        timeline = self._get_mutable(request_id)
        return RequestMetrics(
            request_id=request_id,
            arrival_ns=timeline.arrival_ns,
            prefill_attempt_started_ns=tuple(timeline.prefill_attempt_started_ns),
            token_events=tuple(timeline.token_events),
            completed_ns=timeline.completed_ns,
        )

    def snapshots(self) -> tuple[RequestMetrics, ...]:
        return tuple(self.snapshot(request_id) for request_id in self._timelines)

    def _get_mutable(self, request_id: str) -> _MutableTimeline:
        try:
            return self._timelines[request_id]
        except KeyError as error:
            raise KeyError(f"请求 {request_id!r} 尚未注册 metrics") from error

    @staticmethod
    def _validate_timestamp(timestamp_ns: int) -> None:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns 必须是非负整数")

    @classmethod
    def _validate_next_timestamp(
        cls, timeline: _MutableTimeline, timestamp_ns: int
    ) -> None:
        cls._validate_timestamp(timestamp_ns)
        if timestamp_ns < timeline.last_event_ns:
            raise ValueError("请求事件 timestamp 不能倒退")
