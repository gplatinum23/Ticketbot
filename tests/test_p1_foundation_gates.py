from __future__ import annotations

import ast
import json
from datetime import date, datetime, timezone
from pathlib import Path

from flight_watch_agent.models import FlightOption, TrainOption
from flight_watch_agent.travel_tools import (
    CachedFlightSearchTool,
    CachedTrainSearchTool,
    FlightSearchOutput,
    FlightSearchRequest,
    ToolMetrics,
    ToolResult,
    ToolStatus,
    TrainSearchRequest,
    as_langchain_flight_tool,
    as_langchain_train_tool,
)


def test_orchestration_layers_do_not_import_browser_implementation_details():
    source_root = Path(__file__).parents[1] / "src" / "flight_watch_agent"
    for filename in ("travel_plan_graph.py", "flight_react.py", "travel_tools.py"):
        imports = _imported_modules(source_root / filename)
        assert not any(
            module == "selenium" or module.startswith("selenium.")
            for module in imports
        ), filename
        assert not any("cookie" in module.casefold() for module in imports), filename


def test_public_adapters_are_serializable_and_do_not_leak_raw_backend_state():
    flight_request = FlightSearchRequest("CTU", "CJU", date(2026, 7, 31))
    train_request = TrainSearchRequest("成都东", "重庆北", date(2026, 7, 31))

    def flight_runner(requests):
        return [
            ToolResult(
                status=ToolStatus.SUCCESS,
                data=FlightSearchOutput(
                    options=(_flight_option(request),),
                    raw_state={"cookie": "must-not-leak", "driver": "private"},
                ),
                metrics=_metrics(request.request_id),
            )
            for request in requests
        ]

    flight_tool = as_langchain_flight_tool(CachedFlightSearchTool(flight_runner))
    train_tool = as_langchain_train_tool(CachedTrainSearchTool(_TrainProvider()))
    flight_payload = flight_tool.invoke(flight_request.as_payload())
    train_payload = train_tool.invoke(train_request.as_payload())

    serialized = json.dumps({"flight": flight_payload, "train": train_payload})
    assert flight_payload["metrics"]["termination_reason"] == "completed"
    assert train_payload["options"][0]["train_code"] == "D638"
    assert "must-not-leak" not in serialized
    assert "private" not in serialized


class _TrainProvider:
    def query_train_options(self, _intent):
        return [_train_option()]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


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
        currency="CNY",
        departure_time=None,
        arrival_time=None,
        evidence=[],
        reliability="verified",
        warnings=[],
    )
