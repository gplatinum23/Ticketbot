from __future__ import annotations

from datetime import date

from flight_watch_agent.llm import MonitorIntent
from flight_watch_agent.request_graph import build_monitor_request_graph
from flight_watch_agent.storage import MonitorRepository


class FakeStructuredLlm:
    def __init__(self, intent: MonitorIntent) -> None:
        self.intent = intent

    def invoke(self, messages):
        return self.intent


class FakeLlm:
    def __init__(self, intent: MonitorIntent) -> None:
        self.intent = intent

    def with_structured_output(self, schema):
        return FakeStructuredLlm(self.intent)


def test_request_graph_creates_monitor_from_structured_llm_output(tmp_path):
    repository = MonitorRepository(tmp_path / "test.sqlite3")
    graph = build_monitor_request_graph(
        llm=FakeLlm(
            MonitorIntent(
                action="create_monitor",
                origin="sha",
                destination="nrt",
                depart_date=date(2026, 9, 20),
                threshold_price=1800,
                currency="cny",
                interval_seconds=3600,
            )
        ),
        repository=repository,
    )

    state = graph.invoke({"user_input": "watch sha to nrt"})

    assert state["monitor"].origin == "SHA"
    assert state["monitor"].destination == "NRT"
    assert repository.list_monitors()[0].threshold_price == 1800


def test_request_graph_asks_for_missing_required_fields(tmp_path):
    repository = MonitorRepository(tmp_path / "test.sqlite3")
    graph = build_monitor_request_graph(
        llm=FakeLlm(
            MonitorIntent(
                action="create_monitor",
                origin="SHA",
                destination="NRT",
                depart_date=date(2026, 9, 20),
                threshold_price=None,
                clarification="What price threshold should I use?",
            )
        ),
        repository=repository,
    )

    state = graph.invoke({"user_input": "watch sha to nrt"})

    assert state["response"] == "What price threshold should I use?"
    assert state["errors"] == ["missing_fields:threshold_price"]
    assert repository.list_monitors() == []
