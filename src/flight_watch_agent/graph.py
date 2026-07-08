from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .models import AlertDecision, FlightQuote, Monitor
from .notifiers import Notification, Notifier
from .providers import FlightPriceProvider
from .storage import MonitorRepository


class FlightWatchState(TypedDict, total=False):
    monitor: Monitor
    quote: FlightQuote
    decision: AlertDecision
    notification: Notification
    errors: list[str]


def build_flight_watch_graph(
    *,
    provider: FlightPriceProvider,
    notifier: Notifier,
    repository: MonitorRepository,
):
    graph = StateGraph(FlightWatchState)

    def fetch_price(state: FlightWatchState) -> FlightWatchState:
        monitor = state["monitor"]
        quote = provider.get_lowest_price(monitor.to_search_request())
        return {"quote": quote}

    def evaluate_threshold(state: FlightWatchState) -> FlightWatchState:
        monitor = state["monitor"]
        quote = state["quote"]
        should_notify = quote.price <= monitor.threshold_price
        reason = (
            "price_below_threshold"
            if should_notify
            else "price_above_threshold"
        )
        return {"decision": AlertDecision(should_notify=should_notify, reason=reason)}

    def notify(state: FlightWatchState) -> FlightWatchState:
        monitor = state["monitor"]
        quote = state["quote"]
        notification = notifier.send(monitor, quote)
        repository.record_notification(notification)
        return {"notification": notification}

    def record_result(state: FlightWatchState) -> FlightWatchState:
        repository.record_quote(state["monitor"], state["quote"])
        return {}

    def route_after_evaluation(
        state: FlightWatchState,
    ) -> Literal["notify", "record_result"]:
        if state["decision"].should_notify:
            return "notify"
        return "record_result"

    graph.add_node("fetch_price", fetch_price)
    graph.add_node("evaluate_threshold", evaluate_threshold)
    graph.add_node("notify", notify)
    graph.add_node("record_result", record_result)

    graph.add_edge(START, "fetch_price")
    graph.add_edge("fetch_price", "evaluate_threshold")
    graph.add_conditional_edges(
        "evaluate_threshold",
        route_after_evaluation,
        {"notify": "notify", "record_result": "record_result"},
    )
    graph.add_edge("notify", "record_result")
    graph.add_edge("record_result", END)

    return graph.compile()
