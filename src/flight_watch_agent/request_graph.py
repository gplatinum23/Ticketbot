from __future__ import annotations

from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import MonitorIntent, parse_monitor_intent, required_missing_fields
from .models import Monitor
from .storage import MonitorRepository


class MonitorRequestState(TypedDict, total=False):
    user_input: str
    today: date
    intent: MonitorIntent
    monitor: Monitor
    response: str
    errors: list[str]


def build_monitor_request_graph(*, llm, repository: MonitorRepository):
    graph = StateGraph(MonitorRequestState)

    def parse_intent(state: MonitorRequestState) -> MonitorRequestState:
        intent = parse_monitor_intent(
            state["user_input"],
            llm,
            today=state.get("today"),
        )
        return {"intent": intent}

    def create_or_clarify(state: MonitorRequestState) -> MonitorRequestState:
        intent = state["intent"]
        missing_fields = required_missing_fields(intent)
        if missing_fields:
            clarification = intent.clarification or (
                "Please provide: " + ", ".join(missing_fields) + "."
            )
            return {
                "response": clarification,
                "errors": [f"missing_fields:{','.join(missing_fields)}"],
            }

        try:
            monitor = repository.add_monitor(
                origin=_required(intent.origin),
                destination=_required(intent.destination),
                depart_date=_required(intent.depart_date),
                return_date=intent.return_date,
                threshold_price=_required(intent.threshold_price),
                currency=intent.currency,
                interval_seconds=intent.interval_seconds,
            )
        except ValueError as exc:
            return {"response": str(exc), "errors": [str(exc)]}

        return {
            "monitor": monitor,
            "response": (
                f"Added monitor {monitor.id}: {monitor.origin}->{monitor.destination} "
                f"on {monitor.depart_date.isoformat()}, threshold "
                f"{monitor.threshold_price:.2f} {monitor.currency}."
            ),
        }

    graph.add_node("parse_intent", parse_intent)
    graph.add_node("create_or_clarify", create_or_clarify)
    graph.add_edge(START, "parse_intent")
    graph.add_edge("parse_intent", "create_or_clarify")
    graph.add_edge("create_or_clarify", END)

    return graph.compile()


def _required(value):
    if value is None:
        raise ValueError("Required value is missing.")
    return value
