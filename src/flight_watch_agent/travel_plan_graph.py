from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .agent_models import (
    CandidateHub,
    QueryBudget,
    QueryPlan,
    QueryPlanItem,
    RegionInfo,
    RouteEdge,
    StrategySelection,
)
from .agent_nodes import (
    build_query_plan as build_agent_query_plan,
    classify_region as classify_agent_region,
    generate_candidate_hubs_for_places,
    generate_candidate_hubs as generate_agent_candidate_hubs,
    select_strategies as select_agent_strategies,
)
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
    route_edges: list[RouteEdge] | None = None
    transfer_city: str | None = None
    transfer_airport: str | None = None
    transfer_wait_minutes: int | None = None
    score: float | None = None


class TravelPlanState(TypedDict, total=False):
    intent: FlightSearchIntent
    region_info: RegionInfo
    strategy_selection: StrategySelection
    candidate_hubs: list[CandidateHub]
    query_plan: QueryPlan
    route_edges: list[RouteEdge]
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

    def classify_region(state: TravelPlanState) -> TravelPlanState:
        return {**state, "region_info": classify_agent_region(state["intent"])}

    def select_strategies(state: TravelPlanState) -> TravelPlanState:
        return {
            **state,
            "strategy_selection": select_agent_strategies(
                state["intent"],
                state["region_info"],
            ),
        }

    def generate_candidate_hubs(state: TravelPlanState) -> TravelPlanState:
        hubs = generate_agent_candidate_hubs(
            state["intent"],
            state["strategy_selection"],
            budget=QueryBudget(max_hubs_per_strategy=50, max_flight_queries=50, max_train_queries=50)
            if transfer_hubs is not None
            else None,
        )
        if transfer_hubs is not None:
            allowed = set(_dedupe_hubs(transfer_hubs))
            hubs = [
                hub
                for hub in hubs
                if any(code in allowed for code in hub.airport_codes)
                or hub.hub_id.upper() in allowed
            ]
            explicit_hubs = generate_candidate_hubs_for_places(
                transfer_hubs,
                state["strategy_selection"],
            )
            existing = {hub.hub_id for hub in hubs}
            hubs.extend(hub for hub in explicit_hubs if hub.hub_id not in existing)
        return {**state, "candidate_hubs": hubs}

    def build_query_plan(state: TravelPlanState) -> TravelPlanState:
        query_plan = build_agent_query_plan(
            state["intent"],
            state["strategy_selection"],
            state["candidate_hubs"],
        )
        warnings = list(state.get("warnings", [])) + query_plan.warnings
        return {**state, "query_plan": query_plan, "warnings": warnings}

    def execute_query_plan(state: TravelPlanState) -> TravelPlanState:
        query_plan = state["query_plan"]
        warnings = list(state.get("warnings", []))
        train_options: list[TrainOption] = []
        verified_flight_options: list[FlightOption] = []
        transfer_train_options: list[TrainOption] = []
        transfer_flight_options: list[FlightOption] = []
        route_edges: list[RouteEdge] = []
        flight_search_debug: dict[str, object] = _empty_flight_debug()
        transfer_search_debug: dict[str, object] = {"hubs": [], "searched": []}

        for item in query_plan.items:
            if not item.executable:
                warnings.append(f"query_not_implemented:{item.query_id}")
                continue
            if item.mode == "train":
                if train_provider is None:
                    continue
                try:
                    options = train_provider.query_train_options(_intent_for_query_item(state["intent"], item))
                except Exception as exc:
                    warnings.append(f"train_query_failed:{item.query_id}:{exc}")
                    continue
                best_options = sorted(options, key=_train_sort_key)[:3]
                if item.strategy == "direct_train":
                    train_options.extend(best_options)
                elif item.strategy == "train_flight":
                    transfer_train_options.extend(best_options)
                route_edges.extend(_train_edges_from_options(best_options, item, state["intent"].currency))
                continue

            react_state = react_search.invoke({"intent": _intent_for_query_item(state["intent"], item)})
            flight_options = react_state.get("verified_flight_options", [])
            if item.strategy == "direct_flight":
                verified_flight_options.extend(flight_options)
                flight_search_debug = _react_debug(react_state)
                warnings.extend(react_state.get("warnings", []))
            elif item.strategy == "train_flight":
                transfer_flight_options.extend(flight_options)
                warnings.extend(
                    f"transfer_flight:{item.hub_id}:{warning}"
                    for warning in react_state.get("warnings", [])
                )
                _append_transfer_debug(transfer_search_debug, item, react_state, flight_options)
            else:
                warnings.extend(
                    f"{item.strategy}:{item.hub_id}:{warning}"
                    for warning in react_state.get("warnings", [])
                )
                _append_transfer_debug(transfer_search_debug, item, react_state, flight_options)
            route_edges.extend(_flight_edges_from_options(flight_options, item))

        return {
            **state,
            "train_options": train_options,
            "verified_flight_options": verified_flight_options,
            "transfer_train_options": transfer_train_options,
            "transfer_flight_options": transfer_flight_options,
            "route_edges": route_edges,
            "flight_search_debug": flight_search_debug,
            "transfer_search_debug": transfer_search_debug,
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
        edge_transfer_routes = _build_two_leg_routes_from_edges(state.get("route_edges", []))
        if edge_transfer_routes:
            transfer_routes = edge_transfer_routes
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
            if route.route_type in {"flight_train", "train_train", "flight_flight"}:
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

    graph.add_node("classify_region", classify_region)
    graph.add_node("select_strategies", select_strategies)
    graph.add_node("generate_candidate_hubs", generate_candidate_hubs)
    graph.add_node("build_query_plan", build_query_plan)
    graph.add_node("execute_query_plan", execute_query_plan)
    graph.add_node("build_candidate_routes", build_candidate_routes)
    graph.add_node("rank_routes", rank_routes)
    graph.add_node("render_response", render_response)

    graph.add_edge(START, "classify_region")
    graph.add_edge("classify_region", "select_strategies")
    graph.add_edge("select_strategies", "generate_candidate_hubs")
    graph.add_edge("generate_candidate_hubs", "build_query_plan")
    graph.add_edge("build_query_plan", "execute_query_plan")
    graph.add_edge("execute_query_plan", "build_candidate_routes")
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


def _empty_flight_debug() -> dict[str, object]:
    return {
        "iteration": 0,
        "search_queries": [],
        "raw_results": [],
        "extracted_evidence": [],
        "judged_evidence": [],
        "verified_flight_options": [],
        "warnings": [],
    }


def _react_debug(react_state: dict[str, object]) -> dict[str, object]:
    return {
        "iteration": react_state.get("iteration", 0),
        "search_queries": react_state.get("search_queries", []),
        "raw_results": react_state.get("raw_results", []),
        "extracted_evidence": react_state.get("extracted_evidence", []),
        "judged_evidence": react_state.get("judged_evidence", []),
        "verified_flight_options": react_state.get("verified_flight_options", []),
        "warnings": react_state.get("warnings", []),
    }


def _append_transfer_debug(
    debug: dict[str, object],
    item: QueryPlanItem,
    react_state: dict[str, object],
    flight_options: list[FlightOption],
) -> None:
    searched = debug.setdefault("searched", [])
    if isinstance(searched, list):
        searched.append(
            {
                "hub": item.hub_id,
                "query_id": item.query_id,
                "flights": len(flight_options),
                "queries": react_state.get("search_queries", []),
            }
        )
    hubs = debug.setdefault("hubs", [])
    if isinstance(hubs, list) and item.hub_id and item.hub_id not in hubs:
        hubs.append(item.hub_id)


def _intent_for_query_item(base_intent: FlightSearchIntent, item: QueryPlanItem) -> FlightSearchIntent:
    return FlightSearchIntent(
        origin=item.origin,
        destination=item.destination,
        travel_date=item.travel_date,
        time_preference=base_intent.time_preference,
        budget_threshold=base_intent.budget_threshold,
        currency=base_intent.currency,
        max_segments=base_intent.max_segments,
    )


def _train_edges_from_options(
    options: list[TrainOption],
    item: QueryPlanItem,
    currency: str,
) -> list[RouteEdge]:
    edges: list[RouteEdge] = []
    for option in options:
        edges.append(
            RouteEdge(
                edge_id=f"{item.query_id}:{option.train_code}:{option.start_time}",
                mode="train",
                strategy=item.strategy,
                origin=option.from_station,
                destination=option.to_station,
                travel_date=option.travel_date,
                price=option.lowest_price,
                currency=currency,
                departure_time=option.start_time,
                arrival_time=option.arrive_time,
                source="12306_mcp",
                confidence=0.95,
                hub_id=item.hub_id,
                leg_index=item.leg_index,
                raw_option=option,
            )
        )
    return edges


def _flight_edges_from_options(
    options: list[FlightOption],
    item: QueryPlanItem,
) -> list[RouteEdge]:
    edges: list[RouteEdge] = []
    for option in options:
        edges.append(
            RouteEdge(
                edge_id=f"{item.query_id}:{option.origin}:{option.destination}:{option.price}",
                mode="flight",
                strategy=item.strategy,
                origin=option.origin,
                destination=option.destination,
                travel_date=option.travel_date,
                price=option.price,
                currency=option.currency,
                departure_time=option.departure_time,
                arrival_time=option.arrival_time,
                source="flight_page_search",
                confidence=0.8 if option.reliability == "verified" else 0.6,
                hub_id=item.hub_id,
                leg_index=item.leg_index,
                raw_option=option,
            )
        )
    return edges


def _build_two_leg_routes_from_edges(edges: list[RouteEdge]) -> list[CandidateRoute]:
    first_edges: dict[tuple[str, str], list[RouteEdge]] = {}
    second_edges: dict[tuple[str, str], list[RouteEdge]] = {}
    for edge in edges:
        if edge.strategy in {"direct_train", "direct_flight"} or not edge.hub_id:
            continue
        key = (edge.strategy, edge.hub_id)
        if edge.leg_index == 1:
            first_edges.setdefault(key, []).append(edge)
        if edge.leg_index == 2:
            second_edges.setdefault(key, []).append(edge)

    routes: list[CandidateRoute] = []
    for key, first_group in first_edges.items():
        strategy, hub_id = key
        for first_edge in first_group:
            if first_edge.price is None:
                continue
            for second_edge in second_edges.get(key, []):
                if second_edge.price is None:
                    continue
                wait_minutes = _compute_wait_minutes(first_edge.arrival_time, second_edge.departure_time)
                if wait_minutes is not None and wait_minutes < 60:
                    continue
                total_price = first_edge.price + second_edge.price
                routes.append(
                    CandidateRoute(
                        route_id=f"{strategy}:{hub_id}:{first_edge.edge_id}:{second_edge.edge_id}",
                        route_type=strategy,
                        total_price=total_price,
                        summary=_summarise_two_leg_route(
                            strategy=strategy,
                            first_edge=first_edge,
                            second_edge=second_edge,
                            total_price=total_price,
                            wait_minutes=wait_minutes,
                        ),
                        train_option=_first_train_option(first_edge, second_edge),
                        flight_option=_first_flight_option(first_edge, second_edge),
                        route_edges=[first_edge, second_edge],
                        transfer_city=_transfer_city_from_edges(first_edge, second_edge),
                        transfer_airport=_transfer_airport_from_edges(first_edge, second_edge),
                        transfer_wait_minutes=wait_minutes,
                    )
                )
    return sorted(routes, key=_route_sort_key)[:12]


def _summarise_two_leg_route(
    *,
    strategy: str,
    first_edge: RouteEdge,
    second_edge: RouteEdge,
    total_price: float,
    wait_minutes: int | None,
) -> str:
    wait_text = f"; wait={wait_minutes}min" if wait_minutes is not None else ""
    return (
        f"{_strategy_label(strategy)} via {_transfer_city_from_edges(first_edge, second_edge) or first_edge.destination} "
        f"total {total_price:.2f} {first_edge.currency}; "
        f"{_edge_summary(first_edge)}; {_edge_summary(second_edge)}{wait_text}"
    )


def _strategy_label(strategy: str) -> str:
    return {
        "train_flight": "Train+Flight",
        "flight_train": "Flight+Train",
        "train_train": "Train+Train",
        "flight_flight": "Flight+Flight",
    }.get(strategy, strategy)


def _edge_summary(edge: RouteEdge) -> str:
    price_text = "price unavailable" if edge.price is None else f"{edge.price:.2f} {edge.currency}"
    time_text = _edge_time_text(edge)
    if edge.mode == "train" and isinstance(edge.raw_option, TrainOption):
        return f"train {edge.raw_option.train_code} {edge.origin}->{edge.destination}{time_text} {price_text}"
    if edge.mode == "flight" and isinstance(edge.raw_option, FlightOption):
        metadata = edge.raw_option.evidence[0].metadata or {} if edge.raw_option.evidence else {}
        flight_no = metadata.get("flight_no") or "unknown flight"
        return f"flight {flight_no} {edge.origin}->{edge.destination}{time_text} {price_text}"
    return f"{edge.mode} {edge.origin}->{edge.destination}{time_text} {price_text}"


def _edge_time_text(edge: RouteEdge) -> str:
    start = _format_time_value(edge.departure_time)
    end = _format_time_value(edge.arrival_time)
    if start and end:
        return f" {start}-{end}"
    return ""


def _format_time_value(value: datetime | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%H:%M")
    return str(value)


def _first_train_option(first_edge: RouteEdge, second_edge: RouteEdge) -> TrainOption | None:
    if isinstance(first_edge.raw_option, TrainOption):
        return first_edge.raw_option
    if isinstance(second_edge.raw_option, TrainOption):
        return second_edge.raw_option
    return None


def _first_flight_option(first_edge: RouteEdge, second_edge: RouteEdge) -> FlightOption | None:
    if isinstance(first_edge.raw_option, FlightOption):
        return first_edge.raw_option
    if isinstance(second_edge.raw_option, FlightOption):
        return second_edge.raw_option
    return None


def _transfer_city_from_edges(first_edge: RouteEdge, second_edge: RouteEdge) -> str | None:
    if first_edge.mode == "train":
        return first_edge.destination
    if second_edge.mode == "train":
        return second_edge.origin
    return first_edge.destination


def _transfer_airport_from_edges(first_edge: RouteEdge, second_edge: RouteEdge) -> str | None:
    if first_edge.mode == "flight":
        return first_edge.destination
    if second_edge.mode == "flight":
        return second_edge.origin
    return None


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


def _compute_wait_minutes(
    first_arrival: datetime | str | None,
    second_departure: datetime | str | None,
) -> int | None:
    first_minutes = _minutes_since_midnight(first_arrival)
    second_minutes = _minutes_since_midnight(second_departure)
    if first_minutes is None or second_minutes is None:
        return None
    return second_minutes - first_minutes


def _minutes_since_midnight(value: datetime | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    try:
        hour, minute = value.split(":")[:2]
        return int(hour) * 60 + int(minute)
    except (AttributeError, ValueError):
        return None


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
