from __future__ import annotations

from datetime import date, datetime, timezone

from flight_watch_agent.flight_react import (
    FlightEvidenceVerifier,
    build_react_flight_search_graph,
)
from flight_watch_agent.models import FlightEvidence, FlightSearchIntent, SearchResult
from flight_watch_agent.travel_plan_graph import build_travel_plan_graph


class FakeSearchTool:
    def __init__(self, results_by_query_count: list[list[SearchResult]]) -> None:
        self.results_by_query_count = results_by_query_count
        self.queries: list[str] = []

    def search(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        index = len(self.queries) - 1
        if index >= len(self.results_by_query_count):
            return []
        return self.results_by_query_count[index]


class FakeExtractor:
    def __init__(self, evidence_by_url: dict[str, list[FlightEvidence]]) -> None:
        self.evidence_by_url = evidence_by_url
        self.urls: list[str] = []

    def extract(self, url: str) -> list[FlightEvidence]:
        self.urls.append(url)
        return self.evidence_by_url.get(url, [])


def test_react_search_continues_until_second_source_is_found():
    intent = _intent()
    first = _result("ota-a", "https://a.example/flight")
    second = _result("ota-b", "https://b.example/flight")
    search = FakeSearchTool([[first], [second]])
    extractor = FakeExtractor(
        {
            first.url: [_evidence("ota-a", first.url, 980)],
            second.url: [_evidence("ota-b", second.url, 1010)],
        }
    )
    graph = build_react_flight_search_graph(web_search=search, page_extractor=extractor)

    state = graph.invoke({"intent": intent})

    assert len(search.queries) == 2
    assert len(state["verified_flight_options"]) == 1
    assert state["verified_flight_options"][0].evidence_count == 2


def test_react_search_stops_after_three_iterations_without_verified_option():
    intent = _intent()
    only = _result("ota-a", "https://a.example/flight")
    search = FakeSearchTool([[only], [], []])
    extractor = FakeExtractor({only.url: [_evidence("ota-a", only.url, 980)]})
    graph = build_react_flight_search_graph(web_search=search, page_extractor=extractor)

    state = graph.invoke({"intent": intent})

    assert len(search.queries) == 3
    assert state["verified_flight_options"] == []
    assert "insufficient_verified_flight_evidence" in state["warnings"]


def test_verifier_filters_single_source_and_date_mismatch():
    intent = _intent()
    verifier = FlightEvidenceVerifier()

    single_source = verifier.verify([_evidence("ota-a", "https://a.example/flight", 980)], intent)
    date_mismatch = verifier.verify(
        [
            _evidence("ota-a", "https://a.example/flight", 980, travel_date=date(2026, 7, 10)),
            _evidence("ota-b", "https://b.example/flight", 990, travel_date=date(2026, 7, 10)),
        ],
        intent,
    )

    assert single_source == []
    assert date_mismatch == []


def test_verifier_marks_large_price_spread_for_review():
    intent = _intent()
    verifier = FlightEvidenceVerifier()

    options = verifier.verify(
        [
            _evidence("ota-a", "https://a.example/flight", 700),
            _evidence("ota-b", "https://b.example/flight", 1400),
        ],
        intent,
    )

    assert len(options) == 1
    assert options[0].price == 700
    assert options[0].reliability == "price_volatility_review"
    assert options[0].warnings == ["price_variance_between_sources"]


def test_travel_plan_graph_uses_only_verified_flight_options():
    first = _result("ota-a", "https://a.example/flight")
    second = _result("ota-b", "https://b.example/flight")
    graph = build_travel_plan_graph(
        web_search=FakeSearchTool([[first, second]]),
        page_extractor=FakeExtractor(
            {
                first.url: [_evidence("ota-a", first.url, 980)],
                second.url: [_evidence("ota-b", second.url, 1010)],
            }
        ),
    )

    state = graph.invoke({"intent": _intent()})

    assert len(state["candidate_routes"]) == 1
    assert "evidence=2" in state["response"]


def _intent() -> FlightSearchIntent:
    return FlightSearchIntent(
        origin="BJS",
        destination="SHA",
        travel_date=date(2026, 7, 9),
        time_preference="morning",
        budget_threshold=1200,
        currency="CNY",
    )


def _result(source_name: str, url: str) -> SearchResult:
    return SearchResult(
        title=f"{source_name} flight",
        url=url,
        snippet="flight price",
        source_name=source_name,
    )


def _evidence(
    source_name: str,
    url: str,
    price: float,
    *,
    travel_date: date = date(2026, 7, 9),
) -> FlightEvidence:
    return FlightEvidence(
        source_name=source_name,
        url=url,
        price=price,
        currency="CNY",
        departure_time=datetime(2026, 7, 9, 8, 0, tzinfo=timezone.utc),
        arrival_time=datetime(2026, 7, 9, 10, 20, tzinfo=timezone.utc),
        captured_at=datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc),
        origin="BJS",
        destination="SHA",
        travel_date=travel_date,
    )
