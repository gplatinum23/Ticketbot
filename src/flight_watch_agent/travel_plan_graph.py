from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .flight_react import (
    FlightEvidenceJudge,
    FlightEvidenceVerifier,
    PageExtractor,
    WebSearchTool,
    build_react_flight_search_graph,
)
from .models import FlightOption, FlightSearchIntent, TrainOption
from .places import normalise_airport_code


class TrainProvider(Protocol):
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        """Return train options for the travel intent."""


class RoutePlanner(Protocol):
    def rank(
        self,
        *,
        intent: FlightSearchIntent,
        routes: list["CandidateRoute"],
    ) -> list["CandidateRoute"]:
        """Return routes sorted by LLM route planning preference."""


@dataclass(frozen=True)
class CandidateRoute:
    route_id: str
    route_type: str
    total_price: float | None
    summary: str
    flight_option: FlightOption | None = None
    train_option: TrainOption | None = None
    transfer_city: str | None = None
    transfer_airport: str | None = None
    transfer_wait_minutes: int | None = None
    score: float | None = None


class TravelPlanState(TypedDict, total=False):
    intent: FlightSearchIntent
    train_options: list[TrainOption]
    verified_flight_options: list[FlightOption]
    transfer_train_options: list[TrainOption]
    transfer_flight_options: list[FlightOption]
    flight_search_debug: dict[str, object]
    transfer_search_debug: dict[str, object]
    candidate_routes: list[CandidateRoute]
    response: str
    warnings: list[str]


def build_travel_plan_graph(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    train_provider: TrainProvider | None = None,
    evidence_judge: FlightEvidenceJudge | None = None,
    verifier: FlightEvidenceVerifier | None = None,
    route_planner: RoutePlanner | None = None,
    transfer_hubs: list[str] | None = None,
):
    react_search = build_react_flight_search_graph(
        web_search=web_search,
        page_extractor=page_extractor,
        evidence_judge=evidence_judge,
        verifier=verifier,
    )
    graph = StateGraph(TravelPlanState)

    def query_train_options(state: TravelPlanState) -> TravelPlanState:
        if train_provider is None:
            return {**state, "train_options": []}
        warnings = list(state.get("warnings", []))
        try:
            train_options = train_provider.query_train_options(state["intent"])
        except Exception as exc:
            warnings.append(f"train_query_failed:{exc}")
            train_options = []
        return {**state, "train_options": train_options, "warnings": warnings}

    def react_search_flights(state: TravelPlanState) -> TravelPlanState:
        react_state = react_search.invoke({"intent": state["intent"]})
        return {
            **state,
            "verified_flight_options": react_state.get("verified_flight_options", []),
            "flight_search_debug": {
                "iteration": react_state.get("iteration", 0),
                "search_queries": react_state.get("search_queries", []),
                "raw_results": react_state.get("raw_results", []),
                "extracted_evidence": react_state.get("extracted_evidence", []),
                "judged_evidence": react_state.get("judged_evidence", []),
                "verified_flight_options": react_state.get("verified_flight_options", []),
                "warnings": react_state.get("warnings", []),
            },
            "warnings": state.get("warnings", []) + react_state.get("warnings", []),
        }

    def query_transfer_options(state: TravelPlanState) -> TravelPlanState:
        if train_provider is None:
            return {
                **state,
                "transfer_train_options": [],
                "transfer_flight_options": [],
                "transfer_search_debug": {"hubs": [], "searched": []},
            }

        intent = state["intent"]
        hubs = _dedupe_hubs(
            transfer_hubs
            or _default_transfer_hubs(destination_code=intent.destination)
        )
        warnings = list(state.get("warnings", []))
        transfer_train_options: list[TrainOption] = []
        transfer_flight_options: list[FlightOption] = []
        searched: list[dict[str, object]] = []

        for hub_code in hubs:
            if hub_code in {intent.origin, intent.destination}:
                continue

            train_intent = FlightSearchIntent(
                origin=intent.origin,
                destination=hub_code,
                travel_date=intent.travel_date,
                currency=intent.currency,
            )
            try:
                hub_trains = train_provider.query_train_options(train_intent)
            except Exception as exc:
                warnings.append(f"transfer_train_query_failed:{hub_code}:{exc}")
                continue
            if not hub_trains:
                searched.append({"hub": hub_code, "trains": 0, "flights": 0})
                continue

            best_trains = sorted(hub_trains, key=_train_sort_key)[:3]
            transfer_train_options.extend(best_trains)

            flight_intent = FlightSearchIntent(
                origin=hub_code,
                destination=intent.destination,
                travel_date=intent.travel_date,
                time_preference=intent.time_preference,
                budget_threshold=intent.budget_threshold,
                currency=intent.currency,
                max_segments=intent.max_segments,
            )
            react_state = react_search.invoke({"intent": flight_intent})
            hub_flights = react_state.get("verified_flight_options", [])
            transfer_flight_options.extend(hub_flights)
            warnings.extend(
                f"transfer_flight:{hub_code}:{warning}"
                for warning in react_state.get("warnings", [])
            )
            searched.append(
                {
                    "hub": hub_code,
                    "trains": len(best_trains),
                    "flights": len(hub_flights),
                    "queries": react_state.get("search_queries", []),
                }
            )

        return {
            **state,
            "transfer_train_options": transfer_train_options,
            "transfer_flight_options": transfer_flight_options,
            "transfer_search_debug": {"hubs": hubs, "searched": searched},
            "warnings": warnings,
        }

    def build_candidate_routes(state: TravelPlanState) -> TravelPlanState:
        routes: list[CandidateRoute] = []
        for option in state.get("train_options", []):
            routes.append(
                CandidateRoute(
                    route_id=f"train:{option.train_code}:{option.start_time}",
                    route_type="train",
                    train_option=option,
                    total_price=option.lowest_price,
                    summary=_summarise_train_option(option),
                )
            )

        for option in state.get("verified_flight_options", []):
            routes.append(
                CandidateRoute(
                    route_id=f"flight:{option.origin}:{option.destination}:{option.price}",
                    route_type="flight",
                    flight_option=option,
                    total_price=option.price,
                    summary=_summarise_flight_option(option),
                )
            )

        transfer_routes = _build_transfer_routes(
            trains=state.get("transfer_train_options", []),
            flights=state.get("transfer_flight_options", []),
        )
        routes.extend(transfer_routes)

        return {**state, "candidate_routes": sorted(routes, key=_route_sort_key)}

    def rank_routes(state: TravelPlanState) -> TravelPlanState:
        routes = state.get("candidate_routes", [])
        if not routes or route_planner is None:
            return state
        warnings = list(state.get("warnings", []))
        try:
            ranked = route_planner.rank(intent=state["intent"], routes=routes)
        except Exception as exc:
            warnings.append(f"route_rank_failed:{exc}")
            return {**state, "warnings": warnings}
        return {**state, "candidate_routes": ranked, "warnings": warnings}

    def render_response(state: TravelPlanState) -> TravelPlanState:
        routes = state.get("candidate_routes", [])
        warnings = state.get("warnings", [])
        if not routes:
            warning_text = ""
            if warnings:
                warning_text = " Warnings: " + "; ".join(warnings)
            return {
                **state,
                "response": (
                    "No train or verified flight options found. "
                    "No usable public flight price evidence was found."
                    + warning_text
                ),
            }

        lines = ["Top travel candidates:"]
        for index, route in enumerate(routes[:5], start=1):
            if route.route_type == "train" and route.train_option:
                lines.append(f"{index}. {route.summary}")
                continue
            if route.route_type == "train_flight":
                lines.append(f"{index}. {route.summary}")
                continue

            option = route.flight_option
            if option is None:
                continue
            source_names = ", ".join(sorted({item.source_name for item in option.evidence}))
            lines.append(
                f"{index}. {route.summary}; evidence={option.evidence_count}; "
                f"sources={source_names}; reliability={option.reliability}"
            )
        if warnings:
            lines.append("Warnings: " + "; ".join(warnings))
        return {**state, "response": "\n".join(lines)}

    graph.add_node("query_train_options", query_train_options)
    graph.add_node("react_search_flights", react_search_flights)
    graph.add_node("query_transfer_options", query_transfer_options)
    graph.add_node("build_candidate_routes", build_candidate_routes)
    graph.add_node("rank_routes", rank_routes)
    graph.add_node("render_response", render_response)

    graph.add_edge(START, "query_train_options")
    graph.add_edge("query_train_options", "react_search_flights")
    graph.add_edge("react_search_flights", "query_transfer_options")
    graph.add_edge("query_transfer_options", "build_candidate_routes")
    graph.add_edge("build_candidate_routes", "rank_routes")
    graph.add_edge("rank_routes", "render_response")
    graph.add_edge("render_response", END)

    return graph.compile()


def _summarise_train_option(option: TrainOption) -> str:
    price_text = "price unavailable"
    if option.lowest_price is not None:
        price_text = f"from {option.lowest_price:.2f} CNY"
    seats = ", ".join(f"{name}:{value}" for name, value in option.seats.items())
    return (
        f"Train {option.train_code} {option.from_station}->{option.to_station} "
        f"{option.travel_date.isoformat()} {option.start_time}-{option.arrive_time} "
        f"duration={option.duration}; {price_text}; seats={seats}"
    )


def _summarise_flight_option(option: FlightOption) -> str:
    lowest = option.evidence[0] if option.evidence else None
    metadata = lowest.metadata or {} if lowest else {}
    flight_no = metadata.get("flight_no") or "unknown flight"
    time_text = ""
    if option.departure_time and option.arrival_time:
        time_text = f" {option.departure_time.strftime('%H:%M')}-{option.arrival_time.strftime('%H:%M')}"
    return (
        f"Flight {flight_no} {option.origin}->{option.destination} "
        f"{option.travel_date.isoformat()}{time_text} {option.price:.2f} {option.currency}"
    )


def _summarise_transfer_route(
    *,
    train: TrainOption,
    flight: FlightOption,
    total_price: float,
    wait_minutes: int | None,
) -> str:
    train_price = train.lowest_price
    train_price_text = f"{train_price:.2f}" if train_price is not None else "N/A"
    wait_text = f"; wait={wait_minutes}min" if wait_minutes is not None else ""
    flight_text = _summarise_flight_option(flight)
    return (
        f"Train+Flight via {train.to_station} total {total_price:.2f} CNY; "
        f"train {train.train_code} {train.start_time}-{train.arrive_time} {train_price_text} CNY; "
        f"{flight_text}{wait_text}"
    )


def _build_transfer_routes(
    *,
    trains: list[TrainOption],
    flights: list[FlightOption],
) -> list[CandidateRoute]:
    flights_by_origin: dict[str, list[FlightOption]] = {}
    for flight in flights:
        flights_by_origin.setdefault(flight.origin, []).append(flight)

    routes: list[CandidateRoute] = []
    for train in trains:
        hub_code = normalise_airport_code(train.to_station)
        if not hub_code:
            continue
        for flight in flights_by_origin.get(hub_code, []):
            if train.lowest_price is None:
                continue
            wait_minutes = _compute_transfer_wait_minutes(train.arrive_time, flight.departure_time)
            if wait_minutes is not None and wait_minutes < 60:
                continue
            total_price = train.lowest_price + flight.price
            routes.append(
                CandidateRoute(
                    route_id=(
                        f"train-flight:{train.train_code}:{train.to_station}:"
                        f"{flight.origin}:{flight.destination}:{flight.price}"
                    ),
                    route_type="train_flight",
                    total_price=total_price,
                    summary=_summarise_transfer_route(
                        train=train,
                        flight=flight,
                        total_price=total_price,
                        wait_minutes=wait_minutes,
                    ),
                    train_option=train,
                    flight_option=flight,
                    transfer_city=train.to_station,
                    transfer_airport=hub_code,
                    transfer_wait_minutes=wait_minutes,
                )
            )
    return sorted(routes, key=_route_sort_key)[:12]


def _route_sort_key(route: CandidateRoute) -> tuple[int, float]:
    if route.total_price is None:
        return (1, float("inf"))
    return (0, route.total_price)


def _train_sort_key(train: TrainOption) -> tuple[int, float]:
    if train.lowest_price is None:
        return (1, float("inf"))
    return (0, train.lowest_price)


def _compute_transfer_wait_minutes(train_arrive: str, flight_departure: datetime | None) -> int | None:
    if flight_departure is None:
        return None
    try:
        train_hour, train_minute = train_arrive.split(":")
        train_minutes = int(train_hour) * 60 + int(train_minute)
    except (TypeError, ValueError):
        return None
    departure_minutes = flight_departure.hour * 60 + flight_departure.minute
    return departure_minutes - train_minutes


def _dedupe_hubs(hubs: list[str]) -> list[str]:
    output: list[str] = []
    for hub in hubs:
        code = hub.strip().upper()
        if not code or code in output:
            continue
        output.append(code)
    return output


def _default_transfer_hubs(*, destination_code: str) -> list[str]:
    international_hubs = ["SHA", "HGH", "CAN", "SZX", "CTU", "TFU", "PEK", "PKX", "KMG", "CKG"]
    domestic_hubs = ["SHA", "CAN", "CTU", "PEK", "KMG"]
    if destination_code in {"SIN", "NRT", "ICN", "HKG", "BKK", "KUL"}:
        return international_hubs
    return domestic_hubs


class RouteChoice(BaseModel):
    route_id: str
    score: float = Field(ge=0, le=100)
    rationale: str


class RouteDecision(BaseModel):
    ranked: list[RouteChoice] = Field(default_factory=list)
    summary: str = ""


class LlmRoutePlanner:
    def __init__(self, llm) -> None:
        self.llm = llm

    def rank(
        self,
        *,
        intent: FlightSearchIntent,
        routes: list[CandidateRoute],
    ) -> list[CandidateRoute]:
        structured = self.llm.with_structured_output(RouteDecision)
        payload = _route_payload(routes)
        decision = structured.invoke(
            [
                ("system", _route_planning_prompt(intent)),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ]
        )
        ranked = decision if isinstance(decision, RouteDecision) else RouteDecision.model_validate(decision)
        by_id = {route.route_id: route for route in routes}
        ordered: list[CandidateRoute] = []
        for item in ranked.ranked:
            route = by_id.get(item.route_id)
            if route is None:
                continue
            ordered.append(
                CandidateRoute(
                    **{**route.__dict__, "score": item.score}
                )
            )
        remaining = [route for route in routes if route.route_id not in {r.route_id for r in ordered}]
        return ordered + sorted(remaining, key=_route_sort_key)


def _route_payload(routes: list[CandidateRoute]) -> dict[str, object]:
    serialized: list[dict[str, object]] = []
    for route in routes:
        serialized.append(
            {
                "route_id": route.route_id,
                "route_type": route.route_type,
                "total_price": route.total_price,
                "summary": route.summary,
                "transfer_wait_minutes": route.transfer_wait_minutes,
            }
        )
    return {"routes": serialized}


def _route_planning_prompt(intent: FlightSearchIntent) -> str:
    return f"""
You are ranking travel routes for value and practicality.

Trip intent:
- origin: {intent.origin}
- destination: {intent.destination}
- travel_date: {intent.travel_date.isoformat()}
- time_preference: {intent.time_preference or "none"}
- budget_threshold: {intent.budget_threshold if intent.budget_threshold is not None else "none"}

Scoring priorities:
1) Lower total price is better.
2) Route feasibility matters: avoid tight transfers.
3) Fewer transfers is better unless savings are significant.
4) Respect budget threshold when provided.

Return `ranked` with best routes first.
Every chosen route_id must exist in provided data.
Do not invent routes.
""".strip()
