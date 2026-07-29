from __future__ import annotations

import threading
from datetime import date, datetime, timezone

import pytest

from flight_watch_agent.models import FlightOption, TrainOption
from flight_watch_agent.travel_tools import (
    CachedFlightSearchTool,
    CachedTrainSearchTool,
    FlightSearchOutput,
    FlightSearchRequest,
    RetryPolicy,
    ToolError,
    ToolErrorCode,
    ToolExecutionContext,
    ToolMetrics,
    ToolResult,
    ToolRuntime,
    ToolStatus,
    TrainSearchRequest,
)


class SequencedTrainProvider:
    def __init__(self, outcomes, *, clock=None) -> None:
        self.outcomes = list(outcomes)
        self.clock = clock
        self.calls = 0

    def query_train_options(self, _intent):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if self.clock is not None and self.calls == 1:
            self.clock["now"] = 2.0
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.parametrize(
    "code, expected",
    [
        (code, code in {
            ToolErrorCode.TIMEOUT,
            ToolErrorCode.RATE_LIMITED,
            ToolErrorCode.TOOL_UNAVAILABLE,
        })
        for code in ToolErrorCode
    ],
)
def test_retry_policy_has_an_explicit_allowlist_for_every_error_code(code, expected):
    policy = RetryPolicy(max_attempts=2, jitter_ratio=0)

    assert policy.should_retry(ToolError(code, "failure", retryable=True), 1) is expected


def test_train_runtime_retries_timeout_with_runtime_metrics():
    sleeps: list[float] = []
    runtime = ToolRuntime(
        retry_policy=RetryPolicy(
            max_attempts=3,
            base_delay_seconds=0.1,
            max_delay_seconds=1,
            jitter_ratio=0,
        ),
        sleeper=sleeps.append,
        random_value=lambda: 0.5,
    )
    provider = SequencedTrainProvider([TimeoutError("12306 timed out"), [_train_option()]])
    tool = CachedTrainSearchTool(provider, runtime=runtime)

    result = tool.search(_train_request())

    assert result.status == ToolStatus.SUCCESS
    assert provider.calls == 2
    assert sleeps == [0.1]
    assert result.metrics.attempts == 2
    assert result.metrics.termination_reason == "completed"
    assert result.metrics.retry_delays_ms == (100,)


def test_train_runtime_does_not_retry_captcha():
    provider = SequencedTrainProvider([RuntimeError("captcha required")])
    tool = CachedTrainSearchTool(provider, runtime=_runtime())

    result = tool.search(_train_request())

    assert result.status == ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code == ToolErrorCode.CAPTCHA_REQUIRED
    assert provider.calls == 1
    assert result.metrics.attempts == 1
    assert result.metrics.termination_reason == "non_retryable_error"


def test_train_batch_cancellation_stops_unstarted_requests():
    provider = SequencedTrainProvider([[_train_option()]])
    event = threading.Event()
    event.set()
    tool = CachedTrainSearchTool(provider, runtime=_runtime())

    results = tool.search_many(
        [_train_request(), TrainSearchRequest("成都东", "北京西", date(2026, 7, 31))],
        context=ToolExecutionContext(cancel_event=event),
    )

    assert provider.calls == 0
    assert [result.error.code for result in results if result.error] == [
        ToolErrorCode.CANCELLED,
        ToolErrorCode.CANCELLED,
    ]
    assert all(result.metrics.attempts == 0 for result in results)
    assert all(result.metrics.termination_reason == "cancelled" for result in results)


def test_train_batch_deadline_keeps_completed_item_and_cancels_pending_item():
    fake_clock = {"now": 0.0}
    runtime = ToolRuntime(
        retry_policy=RetryPolicy(jitter_ratio=0),
        clock=lambda: fake_clock["now"],
        sleeper=lambda _delay: None,
        random_value=lambda: 0.5,
    )
    provider = SequencedTrainProvider([[_train_option()]], clock=fake_clock)
    tool = CachedTrainSearchTool(provider, runtime=runtime)

    results = tool.search_many(
        [_train_request(), TrainSearchRequest("成都东", "北京西", date(2026, 7, 31))],
        context=ToolExecutionContext(deadline_monotonic=1.0),
    )

    assert provider.calls == 1
    assert results[0].status == ToolStatus.SUCCESS
    assert results[1].status == ToolStatus.ERROR
    assert results[1].error is not None
    assert results[1].error.code == ToolErrorCode.TIMEOUT
    assert results[1].metrics.termination_reason == "deadline_exhausted"


def test_flight_runtime_retries_only_the_failed_item_in_a_mixed_batch():
    calls: list[list[str]] = []
    first = FlightSearchRequest("CTU", "CJU", date(2026, 7, 31))
    second = FlightSearchRequest("CKG", "CJU", date(2026, 7, 31))

    def runner(requests):
        calls.append([request.request_id for request in requests])
        if len(requests) == 2:
            return [_flight_success(first), _flight_error(second, ToolErrorCode.RATE_LIMITED)]
        return [_flight_success(requests[0])]

    tool = CachedFlightSearchTool(
        runner,
        runtime=ToolRuntime(
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay_seconds=0.1,
                max_delay_seconds=1,
                jitter_ratio=0,
            ),
            sleeper=lambda _delay: None,
            random_value=lambda: 0.5,
        ),
    )

    results = tool.search_many([first, second])

    assert calls == [[first.request_id, second.request_id], [second.request_id]]
    assert [result.status for result in results] == [ToolStatus.SUCCESS, ToolStatus.SUCCESS]
    assert [result.metrics.attempts for result in results] == [1, 2]
    assert results[1].metrics.retry_delays_ms == (100,)


def _runtime() -> ToolRuntime:
    return ToolRuntime(
        retry_policy=RetryPolicy(jitter_ratio=0),
        sleeper=lambda _delay: None,
        random_value=lambda: 0.5,
    )


def _train_request() -> TrainSearchRequest:
    return TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31))


def _train_option() -> TrainOption:
    return TrainOption(
        train_code="D638",
        from_station="成都东",
        from_station_code="ICW",
        to_station="重庆北",
        to_station_code="CUW",
        travel_date=date(2026, 7, 31),
        start_time="07:05",
        arrive_time="09:16",
        duration="02:11",
        seats={"二等座": "有"},
        prices={"二等座": 93.0},
    )


def _flight_success(request: FlightSearchRequest) -> ToolResult[FlightSearchOutput]:
    return ToolResult(
        status=ToolStatus.SUCCESS,
        data=FlightSearchOutput(options=(_flight_option(request),), raw_state={}),
        metrics=_flight_metrics(request),
    )


def _flight_error(
    request: FlightSearchRequest,
    code: ToolErrorCode,
) -> ToolResult[FlightSearchOutput]:
    return ToolResult(
        status=ToolStatus.ERROR,
        data=None,
        error=ToolError(code, "temporary failure", retryable=True),
        metrics=_flight_metrics(request),
    )


def _flight_metrics(request: FlightSearchRequest) -> ToolMetrics:
    return ToolMetrics(
        request_id=request.request_id,
        started_at=datetime.now(timezone.utc),
        latency_ms=1,
        cache_hit=False,
        attempts=1,
        backend="fake",
    )


def _flight_option(request: FlightSearchRequest) -> FlightOption:
    return FlightOption(
        origin=request.origin,
        destination=request.destination,
        travel_date=request.travel_date,
        price=1000.0,
        currency="CNY",
        departure_time=None,
        arrival_time=None,
        evidence=[],
        reliability="verified",
        warnings=[],
    )
