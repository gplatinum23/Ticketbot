from __future__ import annotations

import sys
from datetime import date, datetime, timezone

import pytest

from flight_watch_agent import cli
from flight_watch_agent.llm import TravelPlanIntent
from flight_watch_agent.models import FlightEvidence, FlightSearchIntent, SearchResult
from flight_watch_agent.progress import ConsoleProgressReporter
from flight_watch_agent.request_graph import build_travel_plan_request_graph
from flight_watch_agent.travel_plan_graph import build_travel_plan_graph


class RecordingProgressReporter:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class FakeStructuredLlm:
    def __init__(self, intent: TravelPlanIntent) -> None:
        self.intent = intent

    def invoke(self, _messages):
        return self.intent


class FakeLlm:
    def __init__(self, intent: TravelPlanIntent) -> None:
        self.intent = intent

    def with_structured_output(self, _schema):
        return FakeStructuredLlm(self.intent)


class FakeTravelPlanGraph:
    def invoke(self, state):
        intent = state["intent"]
        return {
            "response": f"planned {intent.origin}->{intent.destination}",
        }


class FakeSearchTool:
    def search(self, query: str) -> list[SearchResult]:
        return [
            SearchResult(
                title="flight",
                url="https://example.com/flight",
                snippet="flight price",
                source_name="example.com",
            )
        ]


class FakeExtractor:
    def extract(self, url: str) -> list[FlightEvidence]:
        return [
            FlightEvidence(
                source_name="example.com",
                url=url,
                price=900.0,
                currency="CNY",
                departure_time=datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc),
                arrival_time=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
                captured_at=datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc),
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
            )
        ]


def test_request_graph_reports_parse_and_planning_progress():
    reporter = RecordingProgressReporter()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
            )
        ),
        travel_plan_graph=FakeTravelPlanGraph(),
        progress_reporter=reporter,
    )

    state = graph.invoke({"user_input": "南京到新加坡"})

    assert state["response"] == "planned NKG->SIN"
    assert reporter.messages == [
        "解析自然语言出行需求...",
        "检查出行需求字段...",
        "进入综合出行规划...",
    ]
    assert "解析自然语言出行需求" not in state["response"]


def test_travel_graph_reports_node_and_query_progress():
    reporter = RecordingProgressReporter()
    graph = build_travel_plan_graph(
        web_search=FakeSearchTool(),
        page_extractor=FakeExtractor(),
        progress_reporter=reporter,
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    assert "Top travel candidates:" in state["response"]
    assert any(message.startswith("规则候选 hub(") for message in reporter.messages)
    assert any("tier:" in message and "score:" in message for message in reporter.messages)
    assert any(message.startswith("LLM 推荐 hub:") for message in reporter.messages)
    assert any(message.startswith("进入查询计划的 hub(") for message in reporter.messages)
    assert "判断出发地和目的地区域..." in reporter.messages
    assert any(message.startswith("原始候选 hub(") for message in reporter.messages)
    assert any(message.startswith("LLM 补充 hub:") for message in reporter.messages)
    assert any(message.startswith("进入查询计划的 hub(") for message in reporter.messages)
    assert "构建火车和机票查询计划..." in reporter.messages
    assert any(message.startswith("查询直达机票: NKG -> SIN") for message in reporter.messages)
    assert any(message.startswith("搜索公开机票页面:") for message in reporter.messages)
    assert "生成最终推荐结果..." in reporter.messages
    assert all(message not in state["response"] for message in reporter.messages)


def test_console_progress_reporter_writes_to_stderr(capsys):
    reporter = ConsoleProgressReporter()

    reporter.emit("测试进度...")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "total " in captured.err
    assert "[1] 测试进度..." in captured.err


def test_cli_help_includes_quiet_flag(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["ask", "--help"])

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "--quiet" in captured.out


def test_cli_plan_flight_quiet_does_not_write_progress(monkeypatch, capsys):
    class FakeGraph:
        def invoke(self, _state):
            return {"response": "ok"}

    monkeypatch.setattr(cli, "build_default_travel_plan_agent", lambda **_kwargs: FakeGraph())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "flight-watch",
            "plan-flight",
            "--origin",
            "NKG",
            "--destination",
            "SIN",
            "--travel-date",
            "2026-07-10",
            "--quiet",
        ],
    )

    cli.main()

    captured = capsys.readouterr()
    assert captured.out.strip() == "ok"
    assert captured.err == ""
