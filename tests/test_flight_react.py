from __future__ import annotations

from datetime import date, datetime, timezone

from flight_watch_agent.flight_react import (
    FlightEvidenceBatchDecision,
    FlightEvidenceBatchDecisionResult,
    FlightEvidenceDecision,
    FlightEvidenceDecisionBatch,
    FlightEvidenceJudgeRequest,
    FlightEvidenceVerifier,
    FlightResponseDecision,
    FlightResponseDecisionBatch,
    LlmFlightEvidenceJudge,
    SkyscannerRouteSearchTool,
    build_react_flight_search_graph,
)
from flight_watch_agent.models import FlightEvidence, FlightSearchIntent, SearchResult
from flight_watch_agent.models import TrainOption
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


class FakeEvidenceJudge:
    def __init__(self, accepted_urls: set[str]) -> None:
        self.accepted_urls = accepted_urls
        self.calls: list[str] = []

    def judge(self, *, intent, search_result, evidence):
        self.calls.append(search_result.url)
        if search_result.url not in self.accepted_urls:
            return []
        return evidence


class FakeStructuredLlm:
    def __init__(self, batch: FlightEvidenceDecisionBatch) -> None:
        self.batch = batch
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return self.batch


class FakeLlm:
    def __init__(self, batch: FlightEvidenceDecisionBatch) -> None:
        self.schema = None
        self.structured_llm = FakeStructuredLlm(batch)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_llm


def test_llm_evidence_judge_batches_multiple_route_requests():
    llm = FakeLlm(
        FlightEvidenceBatchDecisionResult(
            decisions=[
                FlightEvidenceBatchDecision(
                    request_id="route-a",
                    candidate_index=0,
                    accept=True,
                    confidence=0.95,
                    price=800,
                    currency="CNY",
                    origin="BJS",
                    destination="SHA",
                    travel_date=date(2026, 7, 9),
                ),
                FlightEvidenceBatchDecision(
                    request_id="route-b",
                    candidate_index=0,
                    accept=True,
                    confidence=0.90,
                    price=900,
                    currency="CNY",
                    origin="BJS",
                    destination="CAN",
                    travel_date=date(2026, 7, 9),
                ),
            ]
        )
    )
    judge = LlmFlightEvidenceJudge(llm, max_batch_evidence=10)
    intent_a = _intent()
    intent_b = FlightSearchIntent(
        origin="BJS",
        destination="CAN",
        travel_date=date(2026, 7, 9),
        currency="CNY",
    )
    result_a = _result("ctrip", "https://example.com/a")
    result_b = _result("ctrip", "https://example.com/b")

    judged = judge.judge_many(
        [
            FlightEvidenceJudgeRequest(
                request_id="route-a",
                intent=intent_a,
                search_result=result_a,
                evidence=[_evidence("ctrip", result_a.url, 810)],
            ),
            FlightEvidenceJudgeRequest(
                request_id="route-b",
                intent=intent_b,
                search_result=result_b,
                evidence=[
                    FlightEvidence(
                        **{
                            **_evidence("ctrip", result_b.url, 910).__dict__,
                            "destination": "CAN",
                        }
                    )
                ],
            ),
        ]
    )

    assert len(llm.structured_llm.messages) == 1
    assert judge.last_batch_count == 1
    assert judged["route-a"][0].price == 800
    assert judged["route-b"][0].price == 900


def test_llm_evidence_judge_uses_one_response_decision_for_structured_ctrip_itineraries():
    llm = FakeLlm(
        FlightResponseDecisionBatch(
            decisions=[
                FlightResponseDecision(
                    request_id="ctrip-route",
                    accept=True,
                    confidence=0.95,
                )
            ]
        )
    )
    judge = LlmFlightEvidenceJudge(llm)
    intent = _intent()
    result = _result("flights.ctrip.com", "https://flights.ctrip.com/a")
    evidence = [
        FlightEvidence(
            **{
                **_evidence("flights.ctrip.com", result.url, price).__dict__,
                "metadata": {"segments": [{"flight_number": f"TEST{index}"}]},
            }
        )
        for index, price in enumerate((810, 820, 830), start=1)
    ]

    judged = judge.judge_many(
        [
            FlightEvidenceJudgeRequest(
                request_id="ctrip-route",
                intent=intent,
                search_result=result,
                evidence=evidence,
            )
        ]
    )

    assert len(llm.structured_llm.messages) == 1
    assert judge.last_batch_count == 1
    assert [item.price for item in judged["ctrip-route"]] == [810, 820, 830]


class FakeTrainProvider:
    def query_train_options(self, intent):
        return [
            TrainOption(
                train_code="G5",
                from_station="北京南",
                from_station_code="VNP",
                to_station="上海",
                to_station_code="SHH",
                travel_date=intent.travel_date,
                start_time="07:59",
                arrive_time="12:32",
                duration="04:33",
                seats={"second_class": "有"},
                prices={"二等座": 800.0},
                train_class_name="高速",
            )
        ]


def test_react_search_stops_after_single_source_is_found():
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

    assert len(search.queries) == 1
    assert len(state["verified_flight_options"]) == 1
    assert state["verified_flight_options"][0].evidence_count == 1


def test_react_search_stops_after_three_iterations_without_verified_option():
    intent = _intent()
    only = _result("ota-a", "https://a.example/flight")
    search = FakeSearchTool([[only], [], []])
    extractor = FakeExtractor({only.url: []})
    graph = build_react_flight_search_graph(web_search=search, page_extractor=extractor)

    state = graph.invoke({"intent": intent})

    assert len(search.queries) == 3
    assert state["verified_flight_options"] == []
    assert "insufficient_verified_flight_evidence" in state["warnings"]


def test_react_search_warns_when_web_search_returns_no_results():
    graph = build_react_flight_search_graph(
        web_search=FakeSearchTool([[], [], []]),
        page_extractor=FakeExtractor({}),
    )

    state = graph.invoke({"intent": _intent()})

    assert len([warning for warning in state["warnings"] if warning.startswith("web_search_no_results:")]) == 3
    assert "insufficient_verified_flight_evidence" in state["warnings"]


def test_react_search_warns_when_page_extracts_no_evidence():
    only = _result("ota-a", "https://a.example/flight")
    graph = build_react_flight_search_graph(
        web_search=FakeSearchTool([[only]]),
        page_extractor=FakeExtractor({only.url: []}),
        max_iterations=1,
    )

    state = graph.invoke({"intent": _intent()})

    assert f"page_extract_no_evidence:{only.url}" in state["warnings"]
    assert "insufficient_verified_flight_evidence" in state["warnings"]


def test_react_search_warns_when_evidence_misses_time_preference():
    only = _result("ota-a", "https://a.example/flight")
    graph = build_react_flight_search_graph(
        web_search=FakeSearchTool([[only]]),
        page_extractor=FakeExtractor(
            {
                only.url: [
                    _evidence(
                        "ota-a",
                        only.url,
                        980,
                        departure_time=datetime(2026, 7, 9, 20, 0, tzinfo=timezone.utc),
                    )
                ]
            }
        ),
        max_iterations=1,
    )

    state = graph.invoke({"intent": _intent()})

    assert "no_flight_options_match_time_preference:morning" in state["warnings"]
    assert "insufficient_verified_flight_evidence" in state["warnings"]


def test_skyscanner_route_search_constructs_route_url_from_airport_codes():
    results = SkyscannerRouteSearchTool().search("SIN TFU 2026-07-09 flight price")

    assert len(results) == 1
    assert results[0].url == (
        "https://www.skyscanner.com.sg/routes/sin/tfu/"
        "singapore-changi-to-chengdu-tianfu-international.html"
    )


def test_skyscanner_route_search_constructs_route_url_from_chinese_city_names():
    results = SkyscannerRouteSearchTool().search("北京 新加坡 2026-07-09 flight price")

    assert len(results) == 1
    assert results[0].url == "https://www.skyscanner.com.sg/routes/bjsa/sin/beijing-to-singapore-changi.html"


def test_react_search_uses_evidence_judge_before_verification():
    intent = _intent()
    first = _result("ota-a", "https://a.example/flight")
    second = _result("ota-b", "https://b.example/flight")
    rejected = _result("ota-c", "https://c.example/flight")
    judge = FakeEvidenceJudge({first.url, second.url})
    graph = build_react_flight_search_graph(
        web_search=FakeSearchTool([[first, second, rejected]]),
        page_extractor=FakeExtractor(
            {
                first.url: [_evidence("ota-a", first.url, 980)],
                second.url: [_evidence("ota-b", second.url, 1010)],
                rejected.url: [_evidence("ota-c", rejected.url, 1)],
            }
        ),
        evidence_judge=judge,
    )

    state = graph.invoke({"intent": intent})

    assert judge.calls == [first.url, second.url, rejected.url]
    assert len(state["verified_flight_options"]) == 1
    assert state["verified_flight_options"][0].evidence_count == 2
    assert {item.source_name for item in state["judged_evidence"]} == {"ota-a", "ota-b"}


def test_llm_evidence_judge_filters_and_normalises_extracted_evidence():
    intent = _intent()
    batch = FlightEvidenceDecisionBatch(
        decisions=[
            FlightEvidenceDecision(
                candidate_index=0,
                accept=True,
                confidence=0.91,
                price=990,
                currency="cny",
                origin="bjs",
                destination="sha",
                travel_date=intent.travel_date,
            ),
            FlightEvidenceDecision(
                candidate_index=1,
                accept=False,
                confidence=0.98,
                reason="hotel price",
            ),
        ]
    )
    llm = FakeLlm(batch)
    judge = LlmFlightEvidenceJudge(llm)

    judged = judge.judge(
        intent=intent,
        search_result=_result("ota-a", "https://a.example/flight"),
        evidence=[
            _evidence("ota-a", "https://a.example/flight", 980),
            _evidence("ota-a", "https://a.example/flight", 500),
        ],
    )

    assert llm.schema is FlightEvidenceDecisionBatch
    assert len(judged) == 1
    assert judged[0].price == 990
    assert judged[0].currency == "CNY"
    assert judged[0].origin == "BJS"
    assert judged[0].destination == "SHA"


def test_verifier_accepts_single_source_and_filters_date_mismatch():
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

    assert len(single_source) == 1
    assert single_source[0].price == 980
    assert date_mismatch == []


def test_verifier_filters_by_morning_time_preference():
    intent = _intent()
    verifier = FlightEvidenceVerifier()

    options = verifier.verify(
        [
            _evidence(
                "ota-a",
                "https://a.example/noon-flight",
                700,
                departure_time=datetime(2026, 7, 9, 12, 5, tzinfo=timezone.utc),
            ),
            _evidence(
                "ota-b",
                "https://b.example/morning-flight",
                800,
                departure_time=datetime(2026, 7, 9, 8, 40, tzinfo=timezone.utc),
            ),
        ],
        intent,
    )

    assert len(options) == 1
    assert options[0].price == 800
    assert options[0].departure_time.hour == 8


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


def test_travel_plan_graph_uses_single_source_flight_options():
    first = _result("ota-a", "https://a.example/flight")
    graph = build_travel_plan_graph(
        web_search=FakeSearchTool([[first]]),
        page_extractor=FakeExtractor(
            {
                first.url: [_evidence("ota-a", first.url, 980)],
            }
        ),
    )

    state = graph.invoke({"intent": _intent()})

    assert len(state["candidate_routes"]) == 1
    assert "Flight MU5100" in state["response"]
    assert "evidence=1" in state["response"]
    assert state["flight_search_debug"]["search_queries"]
    assert state["flight_search_debug"]["raw_results"] == [first]
    assert len(state["flight_search_debug"]["extracted_evidence"]) == 1
    assert len(state["flight_search_debug"]["verified_flight_options"]) == 1


def test_travel_plan_graph_includes_train_options_from_provider():
    graph = build_travel_plan_graph(
        web_search=FakeSearchTool([[]]),
        page_extractor=FakeExtractor({}),
        train_provider=FakeTrainProvider(),
    )

    state = graph.invoke({"intent": _intent()})

    assert len(state["train_options"]) == 1
    assert state["candidate_routes"][0].route_type == "train"
    assert "Train G5" in state["response"]


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
    departure_time: datetime = datetime(2026, 7, 9, 8, 0, tzinfo=timezone.utc),
) -> FlightEvidence:
    return FlightEvidence(
        source_name=source_name,
        url=url,
        price=price,
        currency="CNY",
        departure_time=departure_time,
        arrival_time=datetime(2026, 7, 9, 10, 20, tzinfo=timezone.utc),
        captured_at=datetime(2026, 7, 8, 10, 0, tzinfo=timezone.utc),
        origin="BJS",
        destination="SHA",
        travel_date=travel_date,
        metadata={"flight_no": "MU5100"},
    )
