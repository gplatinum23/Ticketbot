from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from flight_watch_agent.agent_models import RouteEdge
from flight_watch_agent.feasibility import (
    FeasibilityReasonCode,
    FeasibilityStatus,
    RouteFeasibilityEngine,
)
from flight_watch_agent.flight_react import FlightEvidenceVerifier
from flight_watch_agent.ground_transfers import StaticGroundTransferProvider
from flight_watch_agent.models import (
    FlightEvidence,
    FlightOption,
    FlightSearchIntent,
    TrainOption,
)
from flight_watch_agent.travel_plan_graph import build_travel_plan_graph
from flight_watch_agent.travel_tools import (
    CachedFlightSearchTool,
    FlightSearchOutput,
    ToolMetrics,
    ToolResult,
    ToolStatus,
)


CHINA = timezone(timedelta(hours=8), "Asia/Shanghai")
KOREA = timezone(timedelta(hours=9), "Asia/Seoul")


def test_engine_accepts_cross_day_same_terminal_connection():
    first = _flight_edge(
        "NKG",
        "CAN",
        datetime(2026, 7, 31, 20, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 23, 30, tzinfo=CHINA),
        leg_index=1,
        arrival_terminal="T2",
    )
    second = _flight_edge(
        "CAN",
        "SIN",
        datetime(2026, 8, 1, 2, 30, tzinfo=CHINA),
        datetime(2026, 8, 1, 6, 30, tzinfo=CHINA),
        leg_index=2,
        departure_terminal="T2",
    )

    result = RouteFeasibilityEngine().evaluate(
        route_type="flight_flight",
        edges=[first, second],
    )

    assert result.status == FeasibilityStatus.FEASIBLE
    assert result.connections[0].available_minutes == 180
    assert result.connections[0].required_minutes == 150


def test_engine_applies_domestic_same_terminal_and_cross_airport_policies():
    domestic_first = _flight_edge(
        "NKG",
        "CAN",
        datetime(2026, 7, 31, 8, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 10, 0, tzinfo=CHINA),
        leg_index=1,
        arrival_terminal="T2",
    )
    domestic_second = _flight_edge(
        "CAN",
        "CKG",
        datetime(2026, 7, 31, 11, 30, tzinfo=CHINA),
        datetime(2026, 7, 31, 13, 30, tzinfo=CHINA),
        leg_index=2,
        departure_terminal="T2",
    )
    domestic = RouteFeasibilityEngine().evaluate(
        route_type="flight_flight",
        edges=[domestic_first, domestic_second],
    )

    international_first = _flight_edge(
        "CTU",
        "PEK",
        datetime(2026, 7, 31, 7, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 10, 0, tzinfo=CHINA),
        leg_index=1,
    )
    transfer = StaticGroundTransferProvider().find(
        "PEK",
        "PKX",
        departure_at=international_first.arrival_time,
    )
    assert transfer is not None
    international_second = _flight_edge(
        "PKX",
        "CJU",
        datetime(2026, 7, 31, 16, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 20, 0, tzinfo=KOREA),
        leg_index=2,
    )
    cross_airport = RouteFeasibilityEngine().evaluate(
        route_type="flight_flight",
        edges=[
            international_first,
            _ground_edge(transfer),
            international_second,
        ],
    )

    assert domestic.status == FeasibilityStatus.FEASIBLE
    assert domestic.connections[0].required_minutes == 90
    assert cross_airport.status == FeasibilityStatus.FEASIBLE
    assert cross_airport.connections[0].available_minutes == 360
    assert cross_airport.connections[0].required_minutes == 360


def test_engine_rejects_connection_that_cannot_meet_ground_and_checkin_budget():
    train = _train_option()
    first = _train_edge(train)
    flight = _flight_edge(
        "CKG",
        "CJU",
        datetime(2026, 7, 31, 11, 16, tzinfo=CHINA),
        datetime(2026, 7, 31, 14, 30, tzinfo=KOREA),
        leg_index=2,
    )
    transfer = StaticGroundTransferProvider().find(
        "重庆北",
        "CKG",
        departure_at=train.arrival_at,
    )
    assert transfer is not None
    ground = _ground_edge(transfer)

    result = RouteFeasibilityEngine().evaluate(
        route_type="train_flight",
        edges=[first, ground, flight],
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    issue = next(
        issue
        for issue in result.issues
        if issue.code == FeasibilityReasonCode.INSUFFICIENT_CONNECTION_BUFFER
    )
    assert issue.available_minutes == 120
    assert issue.required_minutes == 220


def test_engine_rejects_missing_station_airport_transfer_evidence():
    train = _train_option()
    flight = _flight_edge(
        "CKG",
        "CJU",
        datetime(2026, 7, 31, 15, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 18, 0, tzinfo=KOREA),
        leg_index=2,
    )

    result = RouteFeasibilityEngine().evaluate(
        route_type="train_flight",
        edges=[_train_edge(train), flight],
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert FeasibilityReasonCode.MISSING_GROUND_TRANSFER in {
        issue.code for issue in result.issues
    }


def test_engine_rejects_wrong_city_connection():
    train = _train_option()
    beijing_flight = _flight_edge(
        "PEK",
        "CJU",
        datetime(2026, 7, 31, 20, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 23, 0, tzinfo=KOREA),
        leg_index=2,
    )

    result = RouteFeasibilityEngine().evaluate(
        route_type="train_flight",
        edges=[_train_edge(train), beijing_flight],
    )

    assert result.status == FeasibilityStatus.INFEASIBLE
    assert FeasibilityReasonCode.CROSS_CITY_TRANSFER_UNSUPPORTED in {
        issue.code for issue in result.issues
    }


def test_engine_marks_missing_time_and_default_transfer_evidence_uncertain():
    direct = RouteEdge(
        edge_id="flight:unknown-time",
        mode="flight",
        strategy="direct_flight",
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
        price=900.0,
        departure_time=None,
        arrival_time=None,
    )

    result = RouteFeasibilityEngine().evaluate(
        route_type="flight",
        edges=[direct],
    )

    assert result.status == FeasibilityStatus.UNCERTAIN
    assert {
        FeasibilityReasonCode.MISSING_DEPARTURE_TIME,
        FeasibilityReasonCode.MISSING_ARRIVAL_TIME,
    }.issubset({issue.code for issue in result.issues})


def test_engine_rejects_naive_and_reversed_edge_times():
    naive = RouteEdge(
        edge_id="train:naive",
        mode="train",
        strategy="direct_train",
        origin="成都东",
        destination="重庆北",
        travel_date=date(2026, 7, 31),
        price=93.0,
        departure_time=datetime(2026, 7, 31, 9, 0),
        arrival_time=datetime(2026, 7, 31, 10, 0),
    )
    reversed_edge = RouteEdge(
        edge_id="train:reversed",
        mode="train",
        strategy="direct_train",
        origin="成都东",
        destination="重庆北",
        travel_date=date(2026, 7, 31),
        price=93.0,
        departure_time=datetime(2026, 7, 31, 10, 0, tzinfo=CHINA),
        arrival_time=datetime(2026, 7, 31, 9, 0, tzinfo=CHINA),
    )

    naive_result = RouteFeasibilityEngine().evaluate(
        route_type="train",
        edges=[naive],
    )
    reversed_result = RouteFeasibilityEngine().evaluate(
        route_type="train",
        edges=[reversed_edge],
    )

    assert naive_result.status == FeasibilityStatus.INFEASIBLE
    assert FeasibilityReasonCode.NAIVE_DATETIME in {
        issue.code for issue in naive_result.issues
    }
    assert reversed_result.status == FeasibilityStatus.INFEASIBLE
    assert FeasibilityReasonCode.EDGE_TIME_REVERSED in {
        issue.code for issue in reversed_result.issues
    }


def test_graph_filters_infeasible_route_before_rendering_top_candidates():
    option = _flight_option(
        "CTU",
        "CJU",
        datetime(2026, 7, 31, 20, 0, tzinfo=CHINA),
        datetime(2026, 7, 31, 19, 0, tzinfo=KOREA),
    )
    class MustNotRank:
        def rank(self, **_kwargs):
            raise AssertionError("infeasible routes must be filtered before ranking")

    graph = build_travel_plan_graph(
        flight_tool=CachedFlightSearchTool(
            lambda requests: [
                _tool_result(request.request_id, option)
                for request in requests
            ]
        ),
        route_planner=MustNotRank(),
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="CJU",
                travel_date=date(2026, 7, 31),
            )
        }
    )

    assert state["candidate_routes"] == []
    assert len(state["rejected_candidate_routes"]) == 1
    assert (
        state["rejected_candidate_routes"][0].feasibility.status
        == FeasibilityStatus.INFEASIBLE
    )
    assert "infeasible_routes_filtered:1" in state["warnings"]
    assert "edge_time_reversed" in state["response"]


def test_graph_keeps_uncertain_route_and_renders_risk_reason():
    option = FlightOption(
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
        price=900.0,
        currency="CNY",
        departure_time=None,
        arrival_time=None,
        evidence=[],
        reliability="verified",
        warnings=[],
    )
    graph = build_travel_plan_graph(
        flight_tool=CachedFlightSearchTool(
            lambda requests: [
                _tool_result(request.request_id, option)
                for request in requests
            ]
        ),
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="CJU",
                travel_date=date(2026, 7, 31),
            )
        }
    )

    assert len(state["candidate_routes"]) == 1
    assert (
        state["candidate_routes"][0].feasibility.status
        == FeasibilityStatus.UNCERTAIN
    )
    assert "feasibility=uncertain" in state["response"]
    assert "missing_departure_time" in state["response"]
    assert "value=best_overall,lowest_price,lower_transfer_risk" in state["response"]
    assert "Only 1 sufficiently distinct usable routes" in state["response"]


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


def _train_edge(option: TrainOption) -> RouteEdge:
    return RouteEdge(
        edge_id="train:D638",
        mode="train",
        strategy="train_flight",
        origin=option.from_station,
        destination=option.to_station,
        travel_date=option.travel_date,
        price=option.lowest_price,
        departure_time=option.departure_at,
        arrival_time=option.arrival_at,
        duration_minutes=131,
        source="12306_mcp",
        hub_id="cn_重庆",
        leg_index=1,
        raw_option=option,
    )


def _flight_edge(
    origin: str,
    destination: str,
    departure: datetime,
    arrival: datetime,
    *,
    leg_index: int,
    departure_terminal: str | None = None,
    arrival_terminal: str | None = None,
) -> RouteEdge:
    option = _flight_option(
        origin,
        destination,
        departure,
        arrival,
        departure_terminal=departure_terminal,
        arrival_terminal=arrival_terminal,
    )
    return RouteEdge(
        edge_id=f"flight:{origin}:{destination}:{leg_index}",
        mode="flight",
        strategy="flight_flight" if leg_index == 1 else "train_flight",
        origin=origin,
        destination=destination,
        travel_date=departure.date(),
        price=500.0,
        departure_time=departure,
        arrival_time=arrival,
        duration_minutes=int((arrival - departure).total_seconds() // 60),
        source="flight_page_search",
        hub_id="hub",
        leg_index=leg_index,
        raw_option=option,
    )


def _flight_option(
    origin: str,
    destination: str,
    departure: datetime,
    arrival: datetime,
    *,
    departure_terminal: str | None = None,
    arrival_terminal: str | None = None,
) -> FlightOption:
    evidence = FlightEvidence(
        source_name="flights.ctrip.com",
        url="https://example.test/flight",
        price=500.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        captured_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        origin=origin,
        destination=destination,
        travel_date=departure.date(),
        metadata={
            "flight_no": "TEST100",
            "departure_airport_code": origin,
            "arrival_airport_code": destination,
            "departure_terminal": departure_terminal,
            "arrival_terminal": arrival_terminal,
        },
    )
    return FlightEvidenceVerifier().verify(
        [evidence],
        FlightSearchIntent(
            origin=origin,
            destination=destination,
            travel_date=departure.date(),
        ),
    )[0]


def _ground_edge(transfer) -> RouteEdge:
    return RouteEdge(
        edge_id="ground:test",
        mode="local_transfer",
        strategy="train_flight",
        origin=transfer.origin.canonical_id,
        destination=transfer.destination.canonical_id,
        travel_date=transfer.departure_at.date(),
        price=transfer.price,
        departure_time=transfer.departure_at,
        arrival_time=transfer.arrival_at,
        duration_minutes=transfer.duration_minutes,
        buffer_minutes=transfer.buffer_minutes,
        source=transfer.source,
        reliability=transfer.reliability,
        hub_id="cn_重庆",
        leg_index=1,
        raw_option=transfer,
    )


def _tool_result(request_id: str, option: FlightOption):
    return ToolResult(
        status=ToolStatus.SUCCESS,
        data=FlightSearchOutput(options=(option,), raw_state={}),
        metrics=ToolMetrics(
            request_id=request_id,
            started_at=datetime.now(timezone.utc),
            latency_ms=1,
            cache_hit=False,
            attempts=1,
            backend="fake",
        ),
    )
