from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .flight_react import (
    FlightEvidenceVerifier,
    PageExtractor,
    WebSearchTool,
    build_react_flight_search_graph,
)
from .models import FlightOption, FlightSearchIntent


@dataclass(frozen=True)
class CandidateRoute:
    route_type: str
    flight_option: FlightOption
    total_price: float
    summary: str


class TravelPlanState(TypedDict, total=False):
    intent: FlightSearchIntent
    verified_flight_options: list[FlightOption]
    candidate_routes: list[CandidateRoute]
    response: str
    warnings: list[str]


def build_travel_plan_graph(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    verifier: FlightEvidenceVerifier | None = None,
):
    react_search = build_react_flight_search_graph(
        web_search=web_search,
        page_extractor=page_extractor,
        verifier=verifier,
    )
    graph = StateGraph(TravelPlanState)

    def react_search_flights(state: TravelPlanState) -> TravelPlanState:
        react_state = react_search.invoke({"intent": state["intent"]})
        return {
            **state,
            "verified_flight_options": react_state.get("verified_flight_options", []),
            "warnings": state.get("warnings", []) + react_state.get("warnings", []),
        }

    def build_candidate_routes(state: TravelPlanState) -> TravelPlanState:
        routes = [
            CandidateRoute(
                route_type="flight",
                flight_option=option,
                total_price=option.price,
                summary=(
                    f"{option.origin}->{option.destination} "
                    f"{option.travel_date.isoformat()} {option.price:.2f} {option.currency}"
                ),
            )
            for option in state.get("verified_flight_options", [])
        ]
        return {**state, "candidate_routes": sorted(routes, key=lambda item: item.total_price)}

    def render_response(state: TravelPlanState) -> TravelPlanState:
        routes = state.get("candidate_routes", [])
        warnings = state.get("warnings", [])
        if not routes:
            return {
                **state,
                "response": (
                    "No verified flight options found. "
                    "At least two independent public sources are required."
                ),
            }

        lines = ["Top verified flight candidates:"]
        for index, route in enumerate(routes[:5], start=1):
            option = route.flight_option
            source_names = ", ".join(sorted({item.source_name for item in option.evidence}))
            lines.append(
                f"{index}. {route.summary}; evidence={option.evidence_count}; "
                f"sources={source_names}; reliability={option.reliability}"
            )
        if warnings:
            lines.append("Warnings: " + "; ".join(warnings))
        return {**state, "response": "\n".join(lines)}

    graph.add_node("react_search_flights", react_search_flights)
    graph.add_node("build_candidate_routes", build_candidate_routes)
    graph.add_node("render_response", render_response)

    graph.add_edge(START, "react_search_flights")
    graph.add_edge("react_search_flights", "build_candidate_routes")
    graph.add_edge("build_candidate_routes", "render_response")
    graph.add_edge("render_response", END)

    return graph.compile()
