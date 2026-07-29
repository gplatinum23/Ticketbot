from __future__ import annotations

from datetime import date, datetime, timezone

from flight_watch_agent.models import FlightEvidence, FlightOption, TrainOption
from flight_watch_agent import cli
from flight_watch_agent.travel_tools import (
    CachedFlightSearchTool,
    CachedTrainSearchTool,
    FlightSearchOutput,
    FlightSearchRequest,
    SqliteToolCache,
    ToolCachePolicy,
    ToolMetrics,
    ToolResult,
    ToolStatus,
    TrainSearchRequest,
)


class CountingTrainProvider:
    def __init__(self, options) -> None:
        self.options = list(options)
        self.calls = 0

    def query_train_options(self, _intent):
        self.calls += 1
        return list(self.options)


def test_sqlite_cache_survives_tool_and_cache_recreation(tmp_path):
    path = tmp_path / "tool-cache.sqlite3"
    request = _train_request()
    first_provider = CountingTrainProvider([_train_option()])
    first = CachedTrainSearchTool(first_provider, cache=SqliteToolCache(path))

    first_result = first.search(request)
    second_provider = CountingTrainProvider([_train_option()])
    second = CachedTrainSearchTool(second_provider, cache=SqliteToolCache(path))
    cached_result = second.search(request)

    assert first_result.metrics.cache_hit is False
    assert cached_result.metrics.cache_hit is True
    assert cached_result.data is not None
    assert cached_result.data.options[0].train_code == "D638"
    assert first_provider.calls == 1
    assert second_provider.calls == 0


def test_cache_version_change_forces_a_miss(tmp_path):
    path = tmp_path / "tool-cache.sqlite3"
    request = _train_request()
    first_provider = CountingTrainProvider([_train_option()])
    CachedTrainSearchTool(
        first_provider,
        cache=SqliteToolCache(path),
        backend_version="12306-mcp-v1",
    ).search(request)
    second_provider = CountingTrainProvider([_train_option()])
    changed = CachedTrainSearchTool(
        second_provider,
        cache=SqliteToolCache(path),
        backend_version="12306-mcp-v2",
    ).search(request)

    assert changed.metrics.cache_hit is False
    assert second_provider.calls == 1


def test_no_result_uses_shorter_ttl_after_process_restart(tmp_path):
    path = tmp_path / "tool-cache.sqlite3"
    now = {"value": 1000.0}
    policy = ToolCachePolicy(ttl_seconds=300, no_result_ttl_seconds=10)
    request = _train_request()
    first_provider = CountingTrainProvider([])
    CachedTrainSearchTool(
        first_provider,
        cache=SqliteToolCache(path, policy, clock=lambda: now["value"]),
    ).search(request)

    now["value"] = 1009.0
    before_expiry_provider = CountingTrainProvider([_train_option()])
    before_expiry = CachedTrainSearchTool(
        before_expiry_provider,
        cache=SqliteToolCache(path, policy, clock=lambda: now["value"]),
    ).search(request)
    now["value"] = 1010.0
    after_expiry_provider = CountingTrainProvider([_train_option()])
    after_expiry = CachedTrainSearchTool(
        after_expiry_provider,
        cache=SqliteToolCache(path, policy, clock=lambda: now["value"]),
    ).search(request)

    assert before_expiry.status == ToolStatus.NO_RESULTS
    assert before_expiry.metrics.cache_hit is True
    assert before_expiry_provider.calls == 0
    assert after_expiry.status == ToolStatus.SUCCESS
    assert after_expiry.metrics.cache_hit is False
    assert after_expiry_provider.calls == 1


def test_persistent_flight_cache_redacts_sensitive_data_and_omits_raw_state(tmp_path):
    path = tmp_path / "tool-cache.sqlite3"
    request = FlightSearchRequest("CTU", "CJU", date(2026, 7, 31))

    def runner(requests):
        return [
            ToolResult(
                status=ToolStatus.SUCCESS,
                data=FlightSearchOutput(
                    options=(_flight_option(item),),
                    raw_state={"cookie": "secret-cookie", "driver": "private"},
                ),
                metrics=_metrics(item.request_id),
            )
            for item in requests
        ]

    CachedFlightSearchTool(runner, cache=SqliteToolCache(path)).search(request)
    persisted_text = path.read_text(encoding="utf-8", errors="ignore")
    replayed = CachedFlightSearchTool(runner, cache=SqliteToolCache(path)).search(request)

    assert "secret-cookie" not in persisted_text
    assert "secret-token" not in persisted_text
    assert "?token=" not in persisted_text
    assert replayed.metrics.cache_hit is True
    assert replayed.data is not None
    assert replayed.data.raw_state == {}


def test_error_and_human_action_results_are_never_persisted(tmp_path):
    path = tmp_path / "tool-cache.sqlite3"
    cache = SqliteToolCache(path)
    metrics = _metrics("request")
    cache.put("error", ToolResult(ToolStatus.ERROR, None, metrics))
    cache.put("human", ToolResult(ToolStatus.HUMAN_ACTION_REQUIRED, None, metrics))

    fresh_cache = SqliteToolCache(path)

    assert fresh_cache.get("error") is None
    assert fresh_cache.get("human") is None


def test_clear_cache_cli_command_does_not_start_a_backend(monkeypatch, capsys):
    calls = 0

    def clear_cache():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(cli, "clear_default_tool_cache", clear_cache)

    cli.main(["clear-cache"])

    assert calls == 1
    assert capsys.readouterr().out.strip() == "Tool cache cleared."


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


def _flight_option(request: FlightSearchRequest) -> FlightOption:
    captured_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
    return FlightOption(
        origin=request.origin,
        destination=request.destination,
        travel_date=request.travel_date,
        price=1000.0,
        currency="CNY",
        departure_time=None,
        arrival_time=None,
        evidence=[
            FlightEvidence(
                source_name="fake",
                url="https://example.test/result?token=secret-token",
                price=1000.0,
                currency="CNY",
                departure_time=None,
                arrival_time=None,
                captured_at=captured_at,
                metadata={"sessionToken": "secret-token"},
            )
        ],
        reliability="verified",
        warnings=[],
    )


def _metrics(request_id: str) -> ToolMetrics:
    return ToolMetrics(
        request_id=request_id,
        started_at=datetime.now(timezone.utc),
        latency_ms=1,
        cache_hit=False,
        attempts=1,
        backend="fake",
        termination_reason="completed",
    )
