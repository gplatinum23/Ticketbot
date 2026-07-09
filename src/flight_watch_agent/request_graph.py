from __future__ import annotations

from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .llm import (
    TravelPlanIntent,
    parse_travel_plan_intent,
    required_missing_fields,
    to_flight_search_intent,
)
from .models import FlightSearchIntent
from .travel_plan_graph import TravelPlanState


class TravelPlanRequestState(TypedDict, total=False):
    user_input: str
    today: date
    intent: TravelPlanIntent
    flight_search_intent: FlightSearchIntent
    response: str
    errors: list[str]
    plan_state: TravelPlanState


def build_travel_plan_request_graph(*, llm, travel_plan_graph):
    graph = StateGraph(TravelPlanRequestState)

    def parse_intent(state: TravelPlanRequestState) -> TravelPlanRequestState:
        intent = parse_travel_plan_intent(
            state["user_input"],
            llm,
            today=state.get("today"),
        )
        return {"intent": intent}

    def plan_or_clarify(state: TravelPlanRequestState) -> TravelPlanRequestState:
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
            flight_search_intent = to_flight_search_intent(intent)
        except ValueError as exc:
            return {"response": str(exc), "errors": [str(exc)]}

        plan_state = travel_plan_graph.invoke({"intent": flight_search_intent})
        return {
            "flight_search_intent": flight_search_intent,
            "plan_state": plan_state,
            "response": plan_state["response"],
        }

    graph.add_node("parse_intent", parse_intent)
    graph.add_node("plan_or_clarify", plan_or_clarify)
    graph.add_edge(START, "parse_intent")
    graph.add_edge("parse_intent", "plan_or_clarify")
    graph.add_edge("plan_or_clarify", END)

    return graph.compile()
