from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from flight_watch_agent.models import FlightOption, FlightSearchIntent, TrainOption
from flight_watch_agent.travel_plan_graph import build_travel_plan_graph
from flight_watch_agent.travel_tools import (
    CachedFlightSearchTool,
    CachedTrainSearchTool,
    FlightSearchOutput,
    FlightSearchRequest,
    InMemoryToolCache,
    ToolCachePolicy,
    ToolError,
    ToolErrorCode,
    ToolMetrics,
    ToolResult,
    ToolStatus,
    TrainSearchRequest,
    as_langchain_flight_tool,
    as_langchain_train_tool,
    flight_tool_result_from_state,
)


class CountingTrainProvider:
    def __init__(self, *, error: Exception | None = None, options=None) -> None:
        self.error = error
        self.options = list(options or [])
        self.calls = 0

    def query_train_options(self, _intent):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.options)


def test_train_tool_caches_successful_result_and_reports_cache_hit():
    provider = CountingTrainProvider(options=[_train_option()])
    tool = CachedTrainSearchTool(provider)
    request = TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31), max_results=3)

    first = tool.search(request)
    second = tool.search(request)

    assert first.status == ToolStatus.SUCCESS
    assert first.data is not None
    assert first.data.options[0].train_code == "D638"
    assert first.metrics.cache_hit is False
    assert second.metrics.cache_hit is True
    assert second.metrics.attempts == 0
    assert provider.calls == 1

    tool.clear_cache()
    tool.search(request)
    assert provider.calls == 2


def test_train_tool_batch_suppresses_duplicate_requests():
    provider = CountingTrainProvider(options=[_train_option()])
    tool = CachedTrainSearchTool(provider)
    request = TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31), max_results=3)

    results = tool.search_many([request, request, request])

    assert len(results) == 3
    assert provider.calls == 1
    assert all(result.status == ToolStatus.SUCCESS for result in results)


def test_train_tool_cache_evicts_least_recently_used_entry():
    provider = CountingTrainProvider(options=[_train_option()])
    cache = InMemoryToolCache(ToolCachePolicy(ttl_seconds=300, max_entries=1))
    tool = CachedTrainSearchTool(provider, cache=cache)
    first = TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31))
    second = TrainSearchRequest("成都东", "北京西", date(2026, 7, 31))

    tool.search(first)
    tool.search(second)
    revisited = tool.search(first)

    assert provider.calls == 3
    assert revisited.metrics.cache_hit is False


def test_train_tool_can_disable_no_result_caching():
    provider = CountingTrainProvider()
    cache = InMemoryToolCache(
        ToolCachePolicy(
            ttl_seconds=300,
            max_entries=10,
            cache_no_results=False,
        )
    )
    tool = CachedTrainSearchTool(provider, cache=cache)
    request = TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31))

    first = tool.search(request)
    second = tool.search(request)

    assert first.status == ToolStatus.NO_RESULTS
    assert second.metrics.cache_hit is False
    assert provider.calls == 2


def test_train_tool_converts_backend_timeout_to_structured_error():
    tool = CachedTrainSearchTool(
        CountingTrainProvider(error=TimeoutError("12306 timed out"))
    )

    result = tool.search(
        TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31))
    )

    assert result.status == ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code == ToolErrorCode.TIMEOUT
    assert result.error.retryable is True
    assert result.data is None


def test_flight_tool_batches_unique_misses_and_reuses_cache():
    calls: list[list[str]] = []

    def runner(requests):
        calls.append([request.request_id for request in requests])
        return [
            ToolResult(
                status=ToolStatus.SUCCESS,
                data=FlightSearchOutput(
                    options=(_flight_option(request),),
                    raw_state={"termination_reason": "verified"},
                ),
                metrics=ToolMetrics(
                    request_id=request.request_id,
                    started_at=datetime.now(timezone.utc),
                    latency_ms=15,
                    cache_hit=False,
                    attempts=1,
                    backend="fake",
                ),
            )
            for request in requests
        ]

    tool = CachedFlightSearchTool(runner)
    first = FlightSearchRequest("CTU", "CJU", date(2026, 7, 31))
    second = FlightSearchRequest("CKG", "CJU", date(2026, 7, 31))

    initial = tool.search_many([first, first, second])
    cached = tool.search(first)

    assert len(initial) == 3
    assert calls == [[first.request_id, second.request_id]]
    assert cached.metrics.cache_hit is True
    assert len(calls) == 1


def test_flight_state_captcha_becomes_human_action_required():
    request = FlightSearchRequest("CTU", "CJU", date(2026, 7, 31))
    state = {
        "verified_flight_options": [],
        "observations": [
            {
                "status": "captcha_required",
                "warning": "Ctrip manual verification is required.",
            }
        ],
        "termination_reason": "human_verification_declined",
        "warnings": [],
    }

    result = flight_tool_result_from_state(
        request,
        state,
        started_at=datetime.now(timezone.utc),
        started_monotonic=0,
    )

    assert result.status == ToolStatus.HUMAN_ACTION_REQUIRED
    assert result.error is not None
    assert result.error.code == ToolErrorCode.CAPTCHA_REQUIRED
    assert result.error.retryable is False


def test_tool_requests_validate_invalid_inputs():
    with pytest.raises(ValueError, match="different"):
        FlightSearchRequest("CTU", "CTU", date(2026, 7, 31))

    with pytest.raises(ValueError, match="positive"):
        TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31), max_results=0)


def test_langchain_adapters_expose_stable_names_and_public_payloads():
    train_tool = as_langchain_train_tool(
        CachedTrainSearchTool(CountingTrainProvider(options=[_train_option()]))
    )

    def runner(requests):
        return [
            ToolResult(
                status=ToolStatus.SUCCESS,
                data=FlightSearchOutput(
                    options=(_flight_option(request),),
                    raw_state={"private_browser_state": "must_not_leak"},
                ),
                metrics=ToolMetrics(
                    request_id=request.request_id,
                    started_at=datetime.now(timezone.utc),
                    latency_ms=10,
                    cache_hit=False,
                    attempts=1,
                    backend="fake",
                ),
            )
            for request in requests
        ]

    flight_tool = as_langchain_flight_tool(CachedFlightSearchTool(runner))
    train_payload = train_tool.invoke(
        {
            "origin": "成都东",
            "destination": "重庆北",
            "travel_date": "2026-07-31",
            "max_results": 3,
        }
    )
    flight_payload = flight_tool.invoke(
        {
            "origin": "CTU",
            "destination": "CJU",
            "travel_date": "2026-07-31",
        }
    )

    assert train_tool.name == "search_trains"
    assert train_payload["status"] == "success"
    assert train_payload["options"][0]["train_code"] == "D638"
    assert flight_tool.name == "search_flights"
    assert flight_payload["status"] == "success"
    assert flight_payload["options"][0]["destination"] == "CJU"
    assert "private_browser_state" not in flight_payload


def test_travel_plan_graph_can_run_only_against_domain_tool_interfaces():
    def runner(requests):
        return [
            ToolResult(
                status=ToolStatus.SUCCESS,
                data=FlightSearchOutput(
                    options=(_flight_option(request),),
                    raw_state={
                        "verified_flight_options": [_flight_option(request)],
                        "termination_reason": "verified",
                        "warnings": [],
                    },
                ),
                metrics=ToolMetrics(
                    request_id=request.request_id,
                    started_at=datetime.now(timezone.utc),
                    latency_ms=10,
                    cache_hit=False,
                    attempts=1,
                    backend="fake",
                ),
            )
            for request in requests
        ]

    graph = build_travel_plan_graph(
        flight_tool=CachedFlightSearchTool(runner),
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="CJU",
                travel_date=date(2026, 7, 31),
            )
        }
    )

    assert state["verified_flight_options"][0].destination == "CJU"
    assert state["query_execution_stats"]["unique_flight_queries"] == 1


def test_travel_plan_graph_reuses_prefetched_human_action_result_within_run():
    calls = 0

    def runner(requests):
        nonlocal calls
        calls += 1
        return [
            ToolResult(
                status=ToolStatus.HUMAN_ACTION_REQUIRED,
                data=FlightSearchOutput(
                    options=(),
                    raw_state={
                        "verified_flight_options": [],
                        "termination_reason": "human_verification_declined",
                        "warnings": [],
                    },
                ),
                error=ToolError(
                    code=ToolErrorCode.CAPTCHA_REQUIRED,
                    message="verification required",
                    retryable=False,
                ),
                metrics=ToolMetrics(
                    request_id=request.request_id,
                    started_at=datetime.now(timezone.utc),
                    latency_ms=10,
                    cache_hit=False,
                    attempts=1,
                    backend="fake",
                ),
            )
            for request in requests
        ]

    class PlannerMarker:
        pass

    graph = build_travel_plan_graph(
        flight_tool=CachedFlightSearchTool(runner),
        hub_planner=PlannerMarker(),
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="CJU",
                travel_date=date(2026, 7, 31),
            )
        }
    )

    assert calls == 1
    assert any("captcha_required" in warning for warning in state["warnings"])


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


def _flight_option(request: FlightSearchRequest) -> FlightOption:
    return FlightOption(
        origin=request.origin,
        destination=request.destination,
        travel_date=request.travel_date,
        price=1000.0,
        currency=request.currency,
        departure_time=None,
        arrival_time=None,
        evidence=[],
        reliability="verified",
        warnings=[],
    )
