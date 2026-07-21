from __future__ import annotations

import threading
from datetime import date, datetime, timezone

from flight_watch_agent.models import FlightEvidence, FlightOption, FlightSearchIntent, SearchResult, TrainOption
from flight_watch_agent.travel_plan_graph import (
    CandidateRoute,
    HubEndpointDecision,
    HubPlanningBatch,
    HubSuggestion,
    LlmRoutePlanner,
    RouteDecision,
    _compute_wait_minutes,
    _flight_duration_minutes,
    _flight_segment_count,
    _minimum_transfer_minutes,
    _summarise_flight_option,
    build_travel_plan_graph,
)


class HubTrainProvider:
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        if intent.destination not in {"SHA", "上海"}:
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


def test_flight_summary_displays_next_day_arrival():
    departure = datetime(2026, 11, 15, 13, 30, tzinfo=timezone.utc)
    arrival = datetime(2026, 11, 16, 17, 15, tzinfo=timezone.utc)
    evidence = FlightEvidence(
        source_name="flights.ctrip.com",
        url="https://example.com/ctu-sin",
        price=1248.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        captured_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        origin="CTU",
        destination="SIN",
        travel_date=date(2026, 11, 15),
        metadata={"flight_no": "ZH9406+ZH229"},
    )
    option = FlightOption(
        origin="CTU",
        destination="SIN",
        travel_date=date(2026, 11, 15),
        price=1248.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        evidence=[evidence],
        reliability="verified",
        warnings=[],
    )

    assert "2026-11-15 13:30-17:15(+1d)" in _summarise_flight_option(option)


def test_flight_metrics_include_cross_day_duration_and_internal_segments():
    departure = datetime(2026, 8, 15, 16, 55, tzinfo=timezone.utc)
    arrival = datetime(2026, 8, 17, 0, 10, tzinfo=timezone.utc)
    evidence = FlightEvidence(
        source_name="flights.ctrip.com",
        url="https://example.com/ctu-sin",
        price=977.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        captured_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
        origin="CTU",
        destination="SIN",
        travel_date=date(2026, 8, 15),
        metadata={"flight_no": "SC8726+SC8061", "transfer_count": 1},
    )
    option = FlightOption(
        origin="CTU",
        destination="SIN",
        travel_date=date(2026, 8, 15),
        price=977.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        evidence=[evidence],
        reliability="verified",
        warnings=[],
    )

    assert _flight_duration_minutes(option) == 1875
    assert _flight_segment_count(option) == 2


def test_independently_booked_flight_transfer_requires_two_hours():
    assert _minimum_transfer_minutes("flight_flight") == 120
    assert _minimum_transfer_minutes("train_flight") == 120
    assert _minimum_transfer_minutes("train_train") == 60


def test_wait_minutes_uses_full_dates_for_overnight_transfer():
    arrival = datetime(2026, 11, 15, 23, 30, tzinfo=timezone.utc)
    departure = datetime(2026, 11, 16, 1, 15, tzinfo=timezone.utc)

    assert _compute_wait_minutes(arrival, departure) == 105


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


class MultiStrategyTrainProvider:
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        pairs = {
            ("CTU", "广通北"): ("K100", "Chengdu", "Guangtongbei", "07:00", "11:00", 120.0),
            ("广通北", "DLU"): ("D200", "Guangtongbei", "Dali", "12:10", "15:00", 80.0),
            ("昆明", "DLU"): ("D8701", "Kunming", "Dali", "14:20", "16:30", 145.0),
            ("昆明南", "DLU"): ("D8703", "Kunming South", "Dali", "14:30", "16:40", 150.0),
        }
        key = (intent.origin, intent.destination)
        if key not in pairs:
            return []
        train_code, from_station, to_station, start_time, arrive_time, price = pairs[key]
        return [
            TrainOption(
                train_code=train_code,
                from_station=from_station,
                from_station_code=None,
                to_station=to_station,
                to_station_code=None,
                travel_date=intent.travel_date,
                start_time=start_time,
                arrive_time=arrive_time,
                duration="02:00",
                seats={"second_class": "有"},
                prices={"second_class": price},
                train_class_name="rail",
            )
        ]


class MultiStrategySearchTool:
    def search(self, query: str) -> list[SearchResult]:
        routes = {
            "SIN KMG": "https://example.com/sin-kmg",
            "NKG CAN": "https://example.com/nkg-can",
            "CAN SIN": "https://example.com/can-sin",
        }
        for route, url in routes.items():
            if route in query:
                return [
                    SearchResult(
                        title=route,
                        url=url,
                        snippet="flight price",
                        source_name="flights.ctrip.com",
                    )
                ]
        return []


class CountingMultiStrategySearchTool(MultiStrategySearchTool):
    def __init__(self) -> None:
        self.routes: list[tuple[str, str]] = []

    def search(self, query: str) -> list[SearchResult]:
        pieces = query.split()
        if len(pieces) >= 2:
            self.routes.append((pieces[0], pieces[1]))
        return super().search(query)


class CountingMultiStrategyTrainProvider(MultiStrategyTrainProvider):
    def __init__(self) -> None:
        self.routes: list[tuple[str, str]] = []

    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        self.routes.append((intent.origin, intent.destination))
        return super().query_train_options(intent)


class MultiStrategyExtractor:
    def extract(self, url: str) -> list[FlightEvidence]:
        data = {
            "https://example.com/sin-kmg": ("SIN", "KMG", "TR100", 600.0, 9, 30, 12, 0),
            "https://example.com/nkg-can": ("NKG", "CAN", "CZ100", 400.0, 8, 0, 10, 0),
            "https://example.com/can-sin": ("CAN", "SIN", "CZ200", 500.0, 13, 0, 17, 0),
        }
        if url not in data:
            return []
        origin, destination, flight_no, price, dh, dm, ah, am = data[url]
        return [
            FlightEvidence(
                source_name="flights.ctrip.com",
                url=url,
                price=price,
                currency="CNY",
                departure_time=datetime(2026, 7, 10, dh, dm, tzinfo=timezone.utc),
                arrival_time=datetime(2026, 7, 10, ah, am, tzinfo=timezone.utc),
                captured_at=datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc),
                origin=origin,
                destination=destination,
                travel_date=date(2026, 7, 10),
                metadata={"flight_no": flight_no},
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


class FakeHubProposer:
    def __init__(self, suggestions) -> None:
        self.suggestions = suggestions
        self.calls = []

    def propose(self, **kwargs):
        self.calls.append(kwargs)
        return self.suggestions


class FakeHubPlanner:
    def __init__(self, result: HubPlanningBatch) -> None:
        self.result = result
        self.calls = []

    def plan(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeHubEndpointValidator:
    def __init__(self, decisions=None, *, error: Exception | None = None) -> None:
        self.decisions = decisions or []
        self.error = error
        self.calls = []

    def validate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.decisions


class FakeBatchEvidenceJudge:
    def __init__(self) -> None:
        self.batch_calls = []
        self.single_calls = []

    def judge_many(self, requests):
        self.batch_calls.append(requests)
        return {request.request_id: request.evidence for request in requests}

    def judge(self, *, intent, search_result, evidence):
        self.single_calls.append((intent, search_result, evidence))
        return evidence


class RejectingBatchEvidenceJudge(FakeBatchEvidenceJudge):
    def judge_many(self, requests):
        self.batch_calls.append(requests)
        return {request.request_id: [] for request in requests}


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


def test_travel_plan_graph_builds_train_train_route():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        transfer_hubs=["guangtongbei"],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="DLU",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    train_train_routes = [route for route in state["candidate_routes"] if route.route_type == "train_train"]
    assert len(train_train_routes) == 1
    assert train_train_routes[0].total_price == 200.0
    assert "Train+Train via Guangtongbei" in train_train_routes[0].summary


def test_travel_plan_graph_builds_flight_train_route():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        transfer_hubs=["kunming"],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="SIN",
                destination="DLU",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    flight_train_routes = [route for route in state["candidate_routes"] if route.route_type == "flight_train"]
    assert len(flight_train_routes) == 1
    assert flight_train_routes[0].total_price == 745.0
    assert "Flight+Train via Kunming" in flight_train_routes[0].summary


def test_travel_plan_graph_builds_flight_flight_route():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        transfer_hubs=["guangzhou"],
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

    flight_flight_routes = [route for route in state["candidate_routes"] if route.route_type == "flight_flight"]
    assert len(flight_flight_routes) == 1
    assert flight_flight_routes[0].total_price == 900.0
    assert "Flight+Flight via CAN" in flight_flight_routes[0].summary


def test_travel_plan_graph_reuses_same_flight_query_across_strategies():
    search = CountingMultiStrategySearchTool()
    graph = build_travel_plan_graph(
        web_search=search,
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        transfer_hubs=["guangzhou"],
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

    assert search.routes.count(("CAN", "SIN")) == 1
    can_to_sin_edges = [
        edge
        for edge in state["route_edges"]
        if edge.mode == "flight" and edge.origin == "CAN" and edge.destination == "SIN"
    ]
    assert {edge.strategy for edge in can_to_sin_edges} == {"train_flight", "flight_flight"}
    flight_stats = state["query_execution_stats"]
    assert flight_stats["planned_flight_queries"] > flight_stats["unique_flight_queries"]
    assert flight_stats["reused_flight_queries"] == (
        flight_stats["planned_flight_queries"] - flight_stats["unique_flight_queries"]
    )


def test_travel_plan_graph_batches_evidence_for_unique_flight_queries():
    judge = FakeBatchEvidenceJudge()
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        evidence_judge=judge,
        transfer_hubs=["guangzhou"],
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

    assert len(judge.batch_calls) == 1
    assert judge.single_calls == []
    request_routes = {
        (request.intent.origin, request.intent.destination)
        for request in judge.batch_calls[0]
    }
    assert ("NKG", "CAN") in request_routes
    assert ("CAN", "SIN") in request_routes
    assert any(route.route_type == "flight_flight" for route in state["candidate_routes"])


def test_travel_plan_graph_falls_back_when_batch_judge_rejects_route():
    judge = RejectingBatchEvidenceJudge()
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        evidence_judge=judge,
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="CAN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    assert len(judge.batch_calls) == 1
    assert len(judge.single_calls) == 1
    assert any(route.route_type == "flight" for route in state["candidate_routes"])


def test_travel_plan_graph_reuses_same_train_query_across_strategies():
    train_provider = CountingMultiStrategyTrainProvider()
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=train_provider,
        transfer_hubs=["kunming"],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="DLU",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    assert len(train_provider.routes) == 3
    assert len(set(train_provider.routes)) == 3
    assert sum(origin == "CTU" and destination != "DLU" for origin, destination in train_provider.routes) == 1
    assert sum(origin != "CTU" and destination == "DLU" for origin, destination in train_provider.routes) == 1
    assert state["query_execution_stats"]["planned_train_queries"] == 5
    assert state["query_execution_stats"]["unique_train_queries"] == 3
    assert state["query_execution_stats"]["reused_train_queries"] == 2


def test_travel_plan_graph_overlaps_train_and_flight_data_sources():
    train_started = threading.Event()
    flight_finished = threading.Event()

    class CoordinatedTrainProvider:
        def query_train_options(self, _intent: FlightSearchIntent) -> list[TrainOption]:
            train_started.set()
            assert flight_finished.wait(timeout=2)
            return []

    class CoordinatedSearchTool:
        def search(self, _query: str) -> list[SearchResult]:
            return [
                SearchResult(
                    title="CTU to BJS",
                    url="https://example.com/ctu-bjs",
                    snippet="flight price",
                    source_name="example.com",
                )
            ]

    class CoordinatedExtractor:
        def extract(self, url: str) -> list[FlightEvidence]:
            assert train_started.wait(timeout=2)
            departure = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
            flight_finished.set()
            return [
                FlightEvidence(
                    source_name="example.com",
                    url=url,
                    price=800.0,
                    currency="CNY",
                    departure_time=departure,
                    arrival_time=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
                    captured_at=departure,
                    origin="CTU",
                    destination="BJS",
                    travel_date=date(2026, 7, 10),
                )
            ]

    graph = build_travel_plan_graph(
        web_search=CoordinatedSearchTool(),
        page_extractor=CoordinatedExtractor(),
        train_provider=CoordinatedTrainProvider(),
        transfer_hubs=[],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    assert train_started.is_set()
    assert flight_finished.is_set()
    assert state["query_execution_stats"]["unique_train_queries"] == 1
    assert state["query_execution_stats"]["unique_flight_queries"] == 1


def test_travel_plan_graph_overlaps_direct_flight_prefetch_with_hub_planning():
    flight_started = threading.Event()
    hub_planning_started = threading.Event()

    class CoordinatedHubPlanner:
        def plan(self, **_kwargs):
            assert flight_started.wait(timeout=2)
            hub_planning_started.set()
            return HubPlanningBatch()

    class CoordinatedSearchTool:
        def search(self, _query: str) -> list[SearchResult]:
            return [
                SearchResult(
                    title="flight",
                    url="https://example.com/parallel",
                    snippet="flight price",
                    source_name="example.com",
                )
            ]

    class CoordinatedExtractor:
        def extract(self, url: str) -> list[FlightEvidence]:
            flight_started.set()
            assert hub_planning_started.wait(timeout=2)
            departure = datetime(2026, 7, 10, 9, 0, tzinfo=timezone.utc)
            return [
                FlightEvidence(
                    source_name="example.com",
                    url=url,
                    price=800.0,
                    currency="CNY",
                    departure_time=departure,
                    arrival_time=datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc),
                    captured_at=departure,
                    origin="CTU",
                    destination="SIN",
                    travel_date=date(2026, 7, 10),
                )
            ]

    graph = build_travel_plan_graph(
        web_search=CoordinatedSearchTool(),
        page_extractor=CoordinatedExtractor(),
        hub_planner=CoordinatedHubPlanner(),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            )
        }
    )

    assert flight_started.is_set()
    assert hub_planning_started.is_set()
    assert state["query_execution_stats"]["unique_flight_queries"] >= 1


def test_travel_plan_graph_uses_llm_supplemental_hub_after_index_hubs():
    proposer = FakeHubProposer(
        [
            HubSuggestion(
                official_airport_name="Guangzhou Baiyun International Airport",
                city="Guangzhou",
                country="CN",
                reason="Potential cheaper connection.",
            )
        ]
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_proposer=proposer,
    )

    state = graph.invoke(
        {
            "user_input": "Nanjing to Singapore, consider cheaper transfer hubs",
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert proposer.calls
    assert len(proposer.calls[0]["index_hubs"]) <= 5
    assert any("CAN" in hub.airport_codes for hub in state["llm_candidate_hubs"])
    flight_flight_routes = [route for route in state["candidate_routes"] if route.route_type == "flight_flight"]
    assert len(flight_flight_routes) == 1
    assert flight_flight_routes[0].total_price == 900.0


def test_travel_plan_graph_combines_hub_proposal_and_endpoint_validation_call():
    planner = FakeHubPlanner(
        HubPlanningBatch(
            suggestions=[
                HubSuggestion(
                    official_airport_name="Guangzhou Baiyun International Airport",
                    city="Guangzhou",
                    country="CN",
                    reason="Potential cheaper connection.",
                )
            ],
            decisions=[],
        )
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_planner=planner,
    )

    state = graph.invoke(
        {
            "user_input": "Nanjing to Singapore",
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert len(planner.calls) == 1
    assert "index_hubs" in planner.calls[0]
    assert "supplemental_hubs" in planner.calls[0]
    assert planner.calls[0]["supplemental_hubs"]
    assert "explicit_hubs" in planner.calls[0]
    assert len(state["llm_candidate_hubs"]) == 5
    assert any("CAN" in hub.airport_codes for hub in state["llm_candidate_hubs"])


def test_combined_hub_planner_suggestion_still_uses_local_endpoint_filter():
    planner = FakeHubPlanner(
        HubPlanningBatch(
            suggestions=[HubSuggestion(city="Nanjing", iata_if_explicit="NKG")],
            decisions=[],
        )
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_planner=planner,
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

    assert len(planner.calls) == 1
    assert all("NKG" not in hub.airport_codes for hub in state["candidate_hubs"])
    assert any(warning.startswith("hub_is_origin_or_destination:") for warning in state["warnings"])


def test_travel_plan_graph_warns_when_llm_hub_cannot_be_resolved():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_proposer=FakeHubProposer([HubSuggestion(official_airport_name="Imaginary Airport")]),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert "llm_hub:hub_place_unresolved:Imaginary Airport" in state["warnings"]


def test_travel_plan_graph_limits_llm_hubs_to_five_after_local_validation():
    proposer = FakeHubProposer(
        [
            HubSuggestion(official_airport_name="Guangzhou Baiyun International Airport", city="Guangzhou", country="CN"),
            HubSuggestion(official_airport_name="Shenzhen Bao'an International Airport", city="Shenzhen", country="CN"),
            HubSuggestion(official_airport_name="Xiamen Gaoqi International Airport", city="Xiamen", country="CN"),
            HubSuggestion(official_airport_name="Hangzhou Xiaoshan International Airport", city="Hangzhou", country="CN"),
            HubSuggestion(official_airport_name="Kunming Changshui International Airport", city="Kunming", country="CN"),
            HubSuggestion(official_airport_name="Chongqing Jiangbei International Airport", city="Chongqing", country="CN"),
        ]
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_proposer=proposer,
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="NKG",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert len(state["llm_candidate_hubs"]) == 5


def test_travel_plan_graph_filters_low_potential_llm_hub():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_proposer=FakeHubProposer(
            [
                HubSuggestion(
                    official_airport_name="Zigong Fengming Airport",
                    city="Zigong",
                    country="CN",
                )
            ]
        ),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="SIN",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert state["llm_candidate_hubs"] == []
    assert "llm_hub:hub_place_unresolved:Zigong Fengming Airport" in state["warnings"]


def test_endpoint_validator_filters_same_city_origin_hub_before_query_plan():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert all(hub.city != "成都" for hub in state["candidate_hubs"])
    assert all("TFU" not in hub.airport_codes for hub in state["candidate_hubs"])
    assert all("HZU" not in hub.airport_codes for hub in state["candidate_hubs"])
    assert all(item.hub_id != "cn_成都" for item in state["query_plan"].items)


def test_endpoint_validator_filters_llm_marked_endpoint_hub():
    validator = FakeHubEndpointValidator(
        [
            HubEndpointDecision(
                hub_id="cn_成都",
                is_origin_endpoint=True,
                reason="LLM decided this is the origin endpoint.",
            )
        ]
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_endpoint_validator=validator,
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert validator.calls
    assert all(hub.city != "成都" for hub in state["candidate_hubs"])


def test_explicit_endpoint_hub_is_filtered_by_local_index():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
            "explicit_hub_places": [HubSuggestion(city="Chengdu", station_name="成都")],
        }
    )

    assert all(hub.city != "成都" for hub in state["candidate_hubs"])
    assert "hub_is_origin_or_destination:成都" in state["warnings"]


def test_endpoint_validator_does_not_filter_when_llm_conflicts_with_local_index():
    validator = FakeHubEndpointValidator(
        [
            HubEndpointDecision(
                hub_id="cn_重庆",
                is_origin_endpoint=True,
                reason="Incorrect LLM endpoint judgment.",
            )
        ]
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_endpoint_validator=validator,
        transfer_hubs=["chongqing"],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert any(hub.city == "重庆" for hub in state["candidate_hubs"])
    assert "hub_is_origin_or_destination:重庆" not in state["warnings"]


def test_endpoint_validator_applies_corrected_hub_fields_after_local_validation():
    validator = FakeHubEndpointValidator(
        [
            HubEndpointDecision(
                hub_id="cn_重庆",
                corrected_city="重庆",
                corrected_airport_codes=["CKG"],
                corrected_train_places=["重庆"],
                reason="Normalize to Chongqing.",
            )
        ]
    )
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_endpoint_validator=validator,
        transfer_hubs=["chongqing"],
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    hub = next(hub for hub in state["candidate_hubs"] if hub.hub_id == "cn_重庆")
    assert hub.airport_codes == ["CKG"]
    assert hub.train_places == ["重庆"]


def test_endpoint_validator_failure_falls_back_to_local_filtering():
    graph = build_travel_plan_graph(
        web_search=MultiStrategySearchTool(),
        page_extractor=MultiStrategyExtractor(),
        train_provider=MultiStrategyTrainProvider(),
        hub_endpoint_validator=FakeHubEndpointValidator(error=RuntimeError("boom")),
    )

    state = graph.invoke(
        {
            "intent": FlightSearchIntent(
                origin="CTU",
                destination="BJS",
                travel_date=date(2026, 7, 10),
                currency="CNY",
            ),
        }
    )

    assert all(hub.city != "成都" for hub in state["candidate_hubs"])
    assert any(warning.startswith("llm_endpoint_validator_failed:") for warning in state["warnings"])


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


def test_llm_route_planner_corrects_objectively_dominated_order():
    planner = LlmRoutePlanner(
        FakeLlm(
            RouteDecision(
                ranked=[
                    {"route_id": "wuh", "score": 95, "rationale": "LLM mistake"},
                    {"route_id": "can", "score": 80, "rationale": "better route"},
                ],
                summary="Raw LLM ranking.",
            )
        )
    )
    routes = [
        CandidateRoute(
            route_id="wuh",
            route_type="flight_flight",
            total_price=1417.0,
            total_duration_minutes=1020,
            segment_count=3,
            summary="via WUH",
        ),
        CandidateRoute(
            route_id="can",
            route_type="flight_flight",
            total_price=1205.0,
            total_duration_minutes=970,
            segment_count=2,
            summary="via CAN",
        ),
    ]

    ranked = planner.rank(
        intent=FlightSearchIntent(
            origin="CTU",
            destination="SIN",
            travel_date=date(2026, 8, 15),
            currency="CNY",
        ),
        routes=routes,
    )

    assert [route.route_id for route in ranked] == ["can", "wuh"]
