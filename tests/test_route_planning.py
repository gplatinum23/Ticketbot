from __future__ import annotations

from datetime import date, datetime, timezone

from flight_watch_agent.models import FlightEvidence, FlightSearchIntent, SearchResult, TrainOption
from flight_watch_agent.travel_plan_graph import (
    CandidateRoute,
    LlmRoutePlanner,
    RouteDecision,
    build_travel_plan_graph,
)


class HubTrainProvider:
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        if intent.destination != "SHA":
            return []
        return [
            TrainOption(
                train_code="G700",
                from_station="Nanjing South",
                from_station_code="NKH",
                to_station="Shanghai",
                to_station_code="SHH",
                travel_date=intent.travel_date,
                start_time="06:00",
                arrive_time="08:10",
                duration="02:10",
                seats={"second_class": "5"},
                prices={"second_class": 180.0},
                train_class_name="high_speed",
            )
        ]


class HubSearchTool:
    def search(self, query: str) -> list[SearchResult]:
        if "SHA SIN" not in query:
            return []
        return [
            SearchResult(
                title="SHA to SIN",
                url="https://example.com/sha-sin",
                snippet="flight price",
                source_name="flights.ctrip.com",
            )
        ]


class HubExtractor:
    def extract(self, url: str) -> list[FlightEvidence]:
        if url != "https://example.com/sha-sin":
            return []
        return [
            FlightEvidence(
                source_name="flights.ctrip.com",
                url=url,
                price=900.0,
                currency="CNY",
                departure_time=datetime(2026, 7, 10, 11, 40, tzinfo=timezone.utc),
                arrival_time=datetime(2026, 7, 10, 16, 30, tzinfo=timezone.utc),
                captured_at=datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc),
                origin="SHA",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                metadata={"flight_no": "MU543"},
            )
        ]


class FakeStructuredLlm:
    def __init__(self, output: RouteDecision) -> None:
        self.output = output

    def invoke(self, _messages):
        return self.output


class FakeLlm:
    def __init__(self, output: RouteDecision) -> None:
        self.output = output

    def with_structured_output(self, _schema):
        return FakeStructuredLlm(self.output)


def test_travel_plan_graph_builds_train_flight_transfer_route():
    graph = build_travel_plan_graph(
        web_search=HubSearchTool(),
        page_extractor=HubExtractor(),
        train_provider=HubTrainProvider(),
        transfer_hubs=["SHA"],
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

    transfer_routes = [route for route in state["candidate_routes"] if route.route_type == "train_flight"]
    assert len(transfer_routes) == 1
    assert transfer_routes[0].total_price == 1080.0
    assert "Train+Flight via Shanghai" in transfer_routes[0].summary


def test_llm_route_planner_respects_llm_ranked_route_ids():
    planner = LlmRoutePlanner(
        FakeLlm(
            RouteDecision(
                ranked=[
                    {"route_id": "r2", "score": 93, "rationale": "best value"},
                    {"route_id": "r1", "score": 70, "rationale": "more expensive"},
                ],
                summary="Ranked by total value.",
            )
        )
    )
    routes = [
        CandidateRoute(route_id="r1", route_type="flight", total_price=1200, summary="direct"),
        CandidateRoute(route_id="r2", route_type="train_flight", total_price=980, summary="transfer"),
    ]

    ranked = planner.rank(
        intent=FlightSearchIntent(
            origin="NKG",
            destination="SIN",
            travel_date=date(2026, 7, 10),
            currency="CNY",
        ),
        routes=routes,
    )

    assert [route.route_id for route in ranked] == ["r2", "r1"]
    assert ranked[0].score == 93
