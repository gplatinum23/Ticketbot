from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .models import FlightEvidence, FlightOption, FlightSearchIntent, SearchResult
from .progress import ProgressReporter, get_progress_reporter


@dataclass(frozen=True)
class SkyscannerPlace:
    code: str
    slug: str
    name: str


class WebSearchTool(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        """Search public pages for flight price evidence."""


class PageExtractor(Protocol):
    def extract(self, url: str) -> list[FlightEvidence]:
        """Extract flight evidence from a public page."""


class FlightEvidenceJudge(Protocol):
    def judge(
        self,
        *,
        intent: FlightSearchIntent,
        search_result: SearchResult,
        evidence: list[FlightEvidence],
    ) -> list[FlightEvidence]:
        """Judge and normalise extracted web evidence."""


@dataclass(frozen=True)
class FlightEvidenceJudgeRequest:
    request_id: str
    intent: FlightSearchIntent
    search_result: SearchResult
    evidence: list[FlightEvidence]


class FlightEvidenceDecision(BaseModel):
    candidate_index: int = Field(description="Index of the candidate evidence being judged.")
    accept: bool = Field(description="Whether this candidate is a plausible flight price.")
    confidence: float = Field(ge=0, le=1, description="Confidence that this is a matching flight price.")
    price: float | None = Field(default=None, description="Normalised ticket price.")
    currency: str | None = Field(default=None, description="ISO currency code.")
    origin: str | None = Field(default=None, description="Normalised origin city/airport/code if found.")
    destination: str | None = Field(default=None, description="Normalised destination city/airport/code if found.")
    travel_date: date | None = Field(default=None, description="Travel date if found.")
    reason: str | None = Field(default=None, description="Brief reason for acceptance or rejection.")


class FlightEvidenceDecisionBatch(BaseModel):
    decisions: list[FlightEvidenceDecision] = Field(default_factory=list)


class FlightEvidenceBatchDecision(FlightEvidenceDecision):
    request_id: str = Field(description="Request identifier supplied with the candidate batch.")


class FlightEvidenceBatchDecisionResult(BaseModel):
    decisions: list[FlightEvidenceBatchDecision] = Field(default_factory=list)


class FlightResponseDecision(BaseModel):
    request_id: str = Field(description="Request identifier supplied with the captured response.")
    accept: bool = Field(description="Whether the response plausibly matches the requested flight search.")
    confidence: float = Field(ge=0, le=1)


class FlightResponseDecisionBatch(BaseModel):
    decisions: list[FlightResponseDecision] = Field(default_factory=list)


class LlmFlightEvidenceJudge:
    def __init__(self, llm, *, min_confidence: float = 0.65, max_batch_evidence: int = 25) -> None:
        self.llm = llm
        self.min_confidence = min_confidence
        self.max_batch_evidence = max_batch_evidence
        self.last_batch_count = 0

    def judge(
        self,
        *,
        intent: FlightSearchIntent,
        search_result: SearchResult,
        evidence: list[FlightEvidence],
    ) -> list[FlightEvidence]:
        if not evidence:
            return []
        structured_llm = self.llm.with_structured_output(FlightEvidenceDecisionBatch)
        result = structured_llm.invoke(
            [
                ("system", _flight_evidence_judge_prompt()),
                (
                    "human",
                    _format_judge_input(
                        intent=intent,
                        search_result=search_result,
                        evidence=evidence,
                    ),
                ),
            ]
        )
        batch = (
            result
            if isinstance(result, FlightEvidenceDecisionBatch)
            else FlightEvidenceDecisionBatch.model_validate(result)
        )
        return _apply_evidence_decisions(
            evidence=evidence,
            decisions=batch.decisions,
            intent=intent,
            min_confidence=self.min_confidence,
        )

    def judge_many(
        self,
        requests: list[FlightEvidenceJudgeRequest],
    ) -> dict[str, list[FlightEvidence]]:
        judged = {request.request_id: [] for request in requests}
        non_empty = [request for request in requests if request.evidence]
        self.last_batch_count = 0
        compact_requests = [request for request in non_empty if _is_structured_ctrip_request(request)]
        detailed_requests = [request for request in non_empty if request not in compact_requests]

        for request_batch in _chunk_requests_by_count(compact_requests, self.max_batch_evidence):
            self.last_batch_count += 1
            structured_llm = self.llm.with_structured_output(FlightResponseDecisionBatch)
            result = structured_llm.invoke(
                [
                    ("system", _flight_response_batch_judge_prompt()),
                    ("human", _format_response_batch_judge_input(request_batch)),
                ]
            )
            batch = (
                result
                if isinstance(result, FlightResponseDecisionBatch)
                else FlightResponseDecisionBatch.model_validate(result)
            )
            decision_by_request = {decision.request_id: decision for decision in batch.decisions}
            for request in request_batch:
                decision = decision_by_request.get(request.request_id)
                if decision is None or not decision.accept or decision.confidence < self.min_confidence:
                    continue
                judged[request.request_id].extend(
                    item for item in request.evidence if _matches_intent(item, request.intent)
                )

        for request_batch in _chunk_judge_requests(detailed_requests, self.max_batch_evidence):
            self.last_batch_count += 1
            structured_llm = self.llm.with_structured_output(FlightEvidenceBatchDecisionResult)
            result = structured_llm.invoke(
                [
                    ("system", _flight_evidence_batch_judge_prompt()),
                    ("human", _format_batch_judge_input(request_batch)),
                ]
            )
            batch = (
                result
                if isinstance(result, FlightEvidenceBatchDecisionResult)
                else FlightEvidenceBatchDecisionResult.model_validate(result)
            )
            decisions_by_request: dict[str, list[FlightEvidenceDecision]] = defaultdict(list)
            for decision in batch.decisions:
                decisions_by_request[decision.request_id].append(
                    FlightEvidenceDecision.model_validate(
                        decision.model_dump(exclude={"request_id"})
                    )
                )
            for request in request_batch:
                judged[request.request_id].extend(
                    _apply_evidence_decisions(
                        evidence=request.evidence,
                        decisions=decisions_by_request.get(request.request_id, []),
                        intent=request.intent,
                        min_confidence=self.min_confidence,
                    )
                )
        return judged


class FlightEvidenceVerifier:
    def __init__(
        self,
        *,
        min_sources: int = 1,
        volatility_threshold: float = 0.25,
    ) -> None:
        self.min_sources = min_sources
        self.volatility_threshold = volatility_threshold

    def verify(
        self,
        evidence: list[FlightEvidence],
        intent: FlightSearchIntent,
    ) -> list[FlightOption]:
        valid_evidence = [
            item
            for item in evidence
            if _matches_intent(item, intent)
        ]

        grouped: dict[tuple[str, str, date], list[FlightEvidence]] = defaultdict(list)
        for item in valid_evidence:
            grouped[
                (
                    _normalise_place(item.origin or intent.origin),
                    _normalise_place(item.destination or intent.destination),
                    item.travel_date or intent.travel_date,
                )
            ].append(item)

        options: list[FlightOption] = []
        for (origin, destination, travel_date), items in grouped.items():
            if self.min_sources > 1 and len({_source_identity(item) for item in items}) < self.min_sources:
                continue

            sorted_items = sorted(items, key=lambda item: item.price)
            lowest = sorted_items[0]
            highest = sorted_items[-1]
            warnings: list[str] = []
            reliability = "verified"
            if highest.price and (highest.price - lowest.price) / highest.price > self.volatility_threshold:
                reliability = "price_volatility_review"
                warnings.append("price_variance_between_sources")

            options.append(
                FlightOption(
                    origin=origin,
                    destination=destination,
                    travel_date=travel_date,
                    price=lowest.price,
                    currency=lowest.currency or intent.currency,
                    departure_time=lowest.departure_time,
                    arrival_time=lowest.arrival_time,
                    evidence=sorted_items,
                    reliability=reliability,
                    warnings=warnings,
                )
            )

        return sorted(options, key=lambda option: option.price)


class DuckDuckGoHtmlSearchTool:
    def __init__(self, *, timeout_seconds: int = 10, max_results: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results

    def search(self, query: str) -> list[SearchResult]:
        results = self._search_html(query)
        if results:
            return results
        return self._search_lite(query)

    def _search_html(self, query: str) -> list[SearchResult]:
        params = urllib.parse.urlencode({"q": query})
        body = self._fetch(f"https://html.duckduckgo.com/html/?{params}")

        results: list[SearchResult] = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)">(?P<title>.*?)</a>',
            re.DOTALL,
        )
        for match in pattern.finditer(body):
            url = _normalise_duckduckgo_url(html.unescape(match.group("url")))
            title = _strip_tags(html.unescape(match.group("title")))
            source_name = urllib.parse.urlparse(url).netloc or "web"
            results.append(SearchResult(title=title, url=url, snippet="", source_name=source_name))
            if len(results) >= self.max_results:
                break
        return results

    def _search_lite(self, query: str) -> list[SearchResult]:
        params = urllib.parse.urlencode({"q": query})
        body = self._fetch(f"https://lite.duckduckgo.com/lite/?{params}")

        results: list[SearchResult] = []
        pattern = re.compile(
            r"<a[^>]+href=\"(?P<url>.*?)\"[^>]*class=['\"]result-link['\"][^>]*>(?P<title>.*?)</a>",
            re.DOTALL,
        )
        for match in pattern.finditer(body):
            title = _strip_tags(html.unescape(match.group("title")))
            if title.lower() == "more info":
                continue
            url = _normalise_duckduckgo_url(html.unescape(match.group("url")))
            source_name = urllib.parse.urlparse(url).netloc or "web"
            results.append(SearchResult(title=title, url=url, snippet="", source_name=source_name))
            if len(results) >= self.max_results:
                break
        return results

    def _fetch(self, url: str) -> str:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 flight-watch-agent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return response.read().decode("utf-8", errors="replace")


class CompositeWebSearchTool:
    def __init__(self, tools: list[WebSearchTool]) -> None:
        self.tools = tools

    def search(self, query: str) -> list[SearchResult]:
        results: list[SearchResult] = []
        for tool in self.tools:
            results.extend(tool.search(query))
        return _dedupe_results(results)


class CompositePageExtractor:
    def __init__(self, extractors: list[PageExtractor]) -> None:
        self.extractors = extractors

    def extract(self, url: str) -> list[FlightEvidence]:
        for extractor in self.extractors:
            supports = getattr(extractor, "supports", None)
            if supports is None or supports(url):
                return extractor.extract(url)
        return []


class SkyscannerRouteSearchTool:
    def search(self, query: str) -> list[SearchResult]:
        route = _skyscanner_route_from_query(query)
        if route is None:
            return []

        origin, destination = route
        url = (
            "https://www.skyscanner.com.sg/routes/"
            f"{origin.code}/{destination.code}/"
            f"{origin.slug}-to-{destination.slug}.html"
        )
        return [
            SearchResult(
                title=f"Skyscanner route {origin.name} to {destination.name}",
                url=url,
                snippet="Constructed Skyscanner route page.",
                source_name="www.skyscanner.com.sg",
            )
        ]


class RegexPageExtractor:
    _price_pattern = re.compile(
        r"(?:¥|￥|CNY|RMB|USD|\$)\s*(?P<price>\d{2,6}(?:\.\d{1,2})?)",
        re.IGNORECASE,
    )

    def __init__(self, *, timeout_seconds: int = 10) -> None:
        self.timeout_seconds = timeout_seconds

    def extract(self, url: str) -> list[FlightEvidence]:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "flight-watch-agent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")

        evidence: list[FlightEvidence] = []
        source_name = urllib.parse.urlparse(url).netloc or "web"
        captured_at = datetime.now(timezone.utc)
        for match in self._price_pattern.finditer(body):
            currency = _currency_from_match(match.group(0))
            evidence.append(
                FlightEvidence(
                    source_name=source_name,
                    url=url,
                    price=float(match.group("price")),
                    currency=currency,
                    departure_time=None,
                    arrival_time=None,
                    captured_at=captured_at,
                )
            )
            if len(evidence) >= 3:
                break
        return evidence


class ReactFlightSearchState(TypedDict, total=False):
    intent: FlightSearchIntent
    iteration: int
    max_iterations: int
    current_query: str
    search_queries: list[str]
    raw_results: list[SearchResult]
    extracted_evidence: list[FlightEvidence]
    _extracted_urls: list[str]
    judged_evidence: list[FlightEvidence]
    verified_flight_options: list[FlightOption]
    warnings: list[str]


def build_react_flight_search_graph(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    evidence_judge: FlightEvidenceJudge | None = None,
    verifier: FlightEvidenceVerifier | None = None,
    max_iterations: int = 3,
    progress_reporter: ProgressReporter | None = None,
):
    evidence_verifier = verifier or FlightEvidenceVerifier()
    progress = get_progress_reporter(progress_reporter)
    graph = StateGraph(ReactFlightSearchState)

    def generate_search_plan(state: ReactFlightSearchState) -> ReactFlightSearchState:
        iteration = state.get("iteration", 0) + 1
        max_iters = state.get("max_iterations", max_iterations)
        query = _build_search_query(state["intent"], iteration)
        progress.emit(f"生成机票搜索计划，第 {iteration}/{max_iters} 轮...")
        return {
            **state,
            "iteration": iteration,
            "max_iterations": max_iters,
            "current_query": query,
            "search_queries": state.get("search_queries", []) + [query],
        }

    def run_web_search(state: ReactFlightSearchState) -> ReactFlightSearchState:
        progress.emit(f"搜索公开机票页面: {state['current_query']}")
        warnings = list(state.get("warnings", []))
        raw_results = list(state.get("raw_results", []))
        try:
            query_results = web_search.search(state["current_query"])
            if not query_results:
                warnings.append(f"web_search_no_results:{state['current_query']}")
            raw_results.extend(query_results)
        except Exception as exc:
            warnings.append(f"web_search_failed:{exc}")
        return {**state, "raw_results": _dedupe_results(raw_results), "warnings": warnings}

    def extract_pages(state: ReactFlightSearchState) -> ReactFlightSearchState:
        progress.emit("抽取机票页面价格信息...")
        warnings = list(state.get("warnings", []))
        evidence = list(state.get("extracted_evidence", []))
        seen_urls = set(state.get("_extracted_urls", []))
        intent = state["intent"]

        for result in state.get("raw_results", []):
            if result.url in seen_urls:
                continue
            seen_urls.add(result.url)
            try:
                extracted = page_extractor.extract(result.url)
            except Exception as exc:
                warnings.append(f"page_extract_failed:{result.url}:{exc}")
                continue
            if not extracted:
                warnings.append(f"page_extract_no_evidence:{result.url}")
            evidence.extend(_normalise_extracted_evidence(extracted, result, intent))

        return {
            **state,
            "extracted_evidence": evidence,
            "_extracted_urls": list(seen_urls),
            "warnings": warnings,
        }

    def judge_evidence(state: ReactFlightSearchState) -> ReactFlightSearchState:
        progress.emit("判断机票证据是否匹配...")
        if evidence_judge is None:
            return {**state, "judged_evidence": state.get("extracted_evidence", [])}

        warnings = list(state.get("warnings", []))
        judged: list[FlightEvidence] = []
        intent = state["intent"]
        evidence_by_url: dict[str, list[FlightEvidence]] = defaultdict(list)
        for item in state.get("extracted_evidence", []):
            evidence_by_url[item.url].append(item)

        result_by_url = {result.url: result for result in state.get("raw_results", [])}
        for url, url_evidence in evidence_by_url.items():
            search_result = result_by_url.get(
                url,
                SearchResult(title="", url=url, snippet="", source_name=urllib.parse.urlparse(url).netloc or "web"),
            )
            try:
                judged.extend(
                    evidence_judge.judge(
                        intent=intent,
                        search_result=search_result,
                        evidence=url_evidence,
                    )
                )
            except Exception as exc:
                warnings.append(f"llm_evidence_judge_failed:{url}:{exc}")

        return {**state, "judged_evidence": judged, "warnings": warnings}

    def verify_options(state: ReactFlightSearchState) -> ReactFlightSearchState:
        progress.emit("聚合可用机票候选...")
        evidence = state.get("judged_evidence", state.get("extracted_evidence", []))
        options = evidence_verifier.verify(evidence, state["intent"])
        warnings = list(state.get("warnings", []))
        if not options and state.get("iteration", 0) >= state.get("max_iterations", max_iterations):
            if (
                state["intent"].time_preference
                and any(_matches_intent_without_time_preference(item, state["intent"]) for item in evidence)
            ):
                warnings.append(
                    f"no_flight_options_match_time_preference:{state['intent'].time_preference}"
                )
            warnings.append("insufficient_verified_flight_evidence")
        return {**state, "verified_flight_options": options, "warnings": warnings}

    def should_continue(state: ReactFlightSearchState) -> str:
        if state.get("verified_flight_options"):
            return "done"
        if state.get("iteration", 0) >= state.get("max_iterations", max_iterations):
            return "done"
        return "continue"

    graph.add_node("generate_search_plan", generate_search_plan)
    graph.add_node("web_search", run_web_search)
    graph.add_node("extract_pages", extract_pages)
    graph.add_node("judge_evidence", judge_evidence)
    graph.add_node("verify_options", verify_options)

    graph.add_edge(START, "generate_search_plan")
    graph.add_edge("generate_search_plan", "web_search")
    graph.add_edge("web_search", "extract_pages")
    graph.add_edge("extract_pages", "judge_evidence")
    graph.add_edge("judge_evidence", "verify_options")
    graph.add_conditional_edges(
        "verify_options",
        should_continue,
        {"continue": "generate_search_plan", "done": END},
    )

    return graph.compile()


def _build_search_query(intent: FlightSearchIntent, iteration: int) -> str:
    pieces = [
        intent.origin,
        intent.destination,
        intent.travel_date.isoformat(),
        "flight",
        "price",
    ]
    if iteration >= 2:
        pieces.extend(["airline", "OTA", intent.currency])
        if intent.time_preference:
            pieces.append(intent.time_preference)
    if iteration >= 3:
        pieces.extend(["cheap flights", "official", "booking"])
    return " ".join(piece for piece in pieces if piece)


def _normalise_extracted_evidence(
    evidence: list[FlightEvidence],
    result: SearchResult,
    intent: FlightSearchIntent,
) -> list[FlightEvidence]:
    normalised: list[FlightEvidence] = []
    for item in evidence:
        normalised.append(
            replace(
                item,
                source_name=item.source_name or result.source_name,
                origin=item.origin or intent.origin,
                destination=item.destination or intent.destination,
                travel_date=item.travel_date or intent.travel_date,
                currency=item.currency or intent.currency,
            )
        )
    return normalised


def _apply_evidence_decisions(
    *,
    evidence: list[FlightEvidence],
    decisions: list[FlightEvidenceDecision],
    intent: FlightSearchIntent,
    min_confidence: float,
) -> list[FlightEvidence]:
    accepted: list[FlightEvidence] = []
    for decision in decisions:
        if not decision.accept or decision.confidence < min_confidence:
            continue
        if decision.candidate_index < 0 or decision.candidate_index >= len(evidence):
            continue
        if decision.price is None or decision.price <= 0:
            continue
        original = evidence[decision.candidate_index]
        accepted.append(
            replace(
                original,
                price=decision.price,
                currency=(decision.currency or original.currency or intent.currency).upper(),
                origin=(decision.origin or original.origin or intent.origin).strip().upper(),
                destination=(decision.destination or original.destination or intent.destination).strip().upper(),
                travel_date=decision.travel_date or original.travel_date or intent.travel_date,
            )
        )
    return accepted


def _flight_evidence_judge_prompt() -> str:
    return """
You judge public web evidence for flight ticket prices.

Accept a candidate only when it is plausibly a flight ticket price for the requested route and date.
Reject hotel prices, ads, generic package prices, unrelated routes, wrong dates, and missing prices.
Use the user's requested route/date when the page evidence is clearly about that same query.
Return only structured decisions. Do not include hidden reasoning.
    """.strip()


def _flight_evidence_batch_judge_prompt() -> str:
    return """
You judge batches of public web evidence for flight ticket prices.

For every candidate, copy request_id and candidate_index into the decision.
Accept a candidate only when it plausibly matches that request's route and date.
Reject hotel prices, ads, packages, unrelated routes, wrong dates, and missing prices.
Use each request's requested_trip independently; never mix candidates between requests.
Return only structured decisions. Do not include hidden reasoning.
""".strip()


def _flight_response_batch_judge_prompt() -> str:
    return """
You judge captured Ctrip flight-search responses for requested routes.

Return exactly one decision per request_id with only request_id, accept, and confidence.
Accept when the structured response plausibly contains ticket options for that request's route and date.
Reject responses with a mismatched route/date, missing prices, or content that is not flight inventory.
The application will validate every itinerary's fields and time preference after this response-level decision.
Return only structured decisions. Do not include hidden reasoning.
""".strip()


def _chunk_judge_requests(
    requests: list[FlightEvidenceJudgeRequest],
    max_batch_evidence: int,
) -> list[list[FlightEvidenceJudgeRequest]]:
    limit = max(1, max_batch_evidence)
    batches: list[list[FlightEvidenceJudgeRequest]] = []
    current: list[FlightEvidenceJudgeRequest] = []
    current_size = 0
    for request in requests:
        request_size = len(request.evidence)
        if current and current_size + request_size > limit:
            batches.append(current)
            current = []
            current_size = 0
        current.append(request)
        current_size += request_size
    if current:
        batches.append(current)
    return batches


def _chunk_requests_by_count(
    requests: list[FlightEvidenceJudgeRequest],
    limit: int,
) -> list[list[FlightEvidenceJudgeRequest]]:
    size = max(1, limit)
    return [requests[index:index + size] for index in range(0, len(requests), size)]


def _format_batch_judge_input(requests: list[FlightEvidenceJudgeRequest]) -> str:
    payload = []
    for request in requests:
        request_payload = json.loads(
            _format_judge_input(
                intent=request.intent,
                search_result=request.search_result,
                evidence=request.evidence,
            )
        )
        request_payload["request_id"] = request.request_id
        payload.append(request_payload)
    return json.dumps({"requests": payload}, ensure_ascii=False)


def _format_response_batch_judge_input(requests: list[FlightEvidenceJudgeRequest]) -> str:
    payload: list[dict[str, object]] = []
    for request in requests:
        prices = [item.price for item in request.evidence if item.price is not None]
        payload.append(
            {
                "request_id": request.request_id,
                "requested_trip": {
                    "origin": request.intent.origin,
                    "destination": request.intent.destination,
                    "travel_date": request.intent.travel_date.isoformat(),
                    "currency": request.intent.currency,
                },
                "source_name": request.search_result.source_name,
                "url": request.search_result.url,
                "itinerary_count": len(request.evidence),
                "observed_origins": sorted({item.origin for item in request.evidence if item.origin}),
                "observed_destinations": sorted({item.destination for item in request.evidence if item.destination}),
                "observed_dates": sorted(
                    {item.travel_date.isoformat() for item in request.evidence if item.travel_date}
                ),
                "minimum_price": min(prices) if prices else None,
                "maximum_price": max(prices) if prices else None,
            }
        )
    return json.dumps({"requests": payload}, ensure_ascii=False)


def _is_structured_ctrip_request(request: FlightEvidenceJudgeRequest) -> bool:
    return bool(request.evidence) and all(
        item.source_name == "flights.ctrip.com"
        and isinstance(item.metadata, dict)
        and bool(item.metadata.get("segments"))
        for item in request.evidence
    )


def _format_judge_input(
    *,
    intent: FlightSearchIntent,
    search_result: SearchResult,
    evidence: list[FlightEvidence],
) -> str:
    candidates = [
        {
            "candidate_index": index,
            "source_name": item.source_name,
            "url": item.url,
            "price": item.price,
            "currency": item.currency,
            "origin": item.origin,
            "destination": item.destination,
            "travel_date": item.travel_date.isoformat() if item.travel_date else None,
        }
        for index, item in enumerate(evidence)
    ]
    return json.dumps(
        {
            "requested_trip": {
                "origin": intent.origin,
                "destination": intent.destination,
                "travel_date": intent.travel_date.isoformat(),
                "time_preference": intent.time_preference,
                "currency": intent.currency,
            },
            "search_result": {
                "title": search_result.title,
                "url": search_result.url,
                "snippet": search_result.snippet,
                "source_name": search_result.source_name,
            },
            "candidate_evidence": candidates,
        },
        ensure_ascii=False,
    )


def _matches_intent(item: FlightEvidence, intent: FlightSearchIntent) -> bool:
    if not _matches_intent_without_time_preference(item, intent):
        return False
    if not _matches_time_preference(item, intent.time_preference):
        return False
    return True


def _matches_intent_without_time_preference(item: FlightEvidence, intent: FlightSearchIntent) -> bool:
    if item.price <= 0:
        return False
    if item.travel_date is not None and item.travel_date != intent.travel_date:
        return False
    if item.origin is not None and _normalise_place(item.origin) != _normalise_place(intent.origin):
        return False
    if item.destination is not None and _normalise_place(item.destination) != _normalise_place(intent.destination):
        return False
    return True


def _matches_time_preference(item: FlightEvidence, time_preference: str | None) -> bool:
    if not time_preference:
        return True
    preference = time_preference.strip().lower()
    if preference not in {"morning", "afternoon", "evening"}:
        return True
    if item.departure_time is None:
        return False
    hour = item.departure_time.hour
    if preference == "morning":
        return hour < 12
    if preference == "afternoon":
        return 12 <= hour < 18
    return hour >= 18


def _dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: list[SearchResult] = []
    seen_urls: set[str] = set()
    for result in results:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        deduped.append(result)
    return deduped


def _source_identity(item: FlightEvidence) -> str:
    return item.source_name or urllib.parse.urlparse(item.url).netloc or item.url


_SKYSCANNER_PLACES: dict[str, SkyscannerPlace] = {
    "SIN": SkyscannerPlace("sin", "singapore-changi", "Singapore Changi"),
    "新加坡": SkyscannerPlace("sin", "singapore-changi", "Singapore Changi"),
    "SINGAPORE": SkyscannerPlace("sin", "singapore-changi", "Singapore Changi"),
    "CHANGI": SkyscannerPlace("sin", "singapore-changi", "Singapore Changi"),
    "樟宜": SkyscannerPlace("sin", "singapore-changi", "Singapore Changi"),
    "TFU": SkyscannerPlace("tfu", "chengdu-tianfu-international", "Chengdu Tianfu International"),
    "成都": SkyscannerPlace("tfu", "chengdu-tianfu-international", "Chengdu Tianfu International"),
    "CHENGDU": SkyscannerPlace("tfu", "chengdu-tianfu-international", "Chengdu Tianfu International"),
    "天府": SkyscannerPlace("tfu", "chengdu-tianfu-international", "Chengdu Tianfu International"),
    "BJS": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "BJSA": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "PEK": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "北京": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "BEIJING": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "PKX": SkyscannerPlace("bjsa", "beijing", "Beijing"),
    "SHA": SkyscannerPlace("csha", "shanghai", "Shanghai"),
    "PVG": SkyscannerPlace("csha", "shanghai", "Shanghai"),
    "上海": SkyscannerPlace("csha", "shanghai", "Shanghai"),
    "SHANGHAI": SkyscannerPlace("csha", "shanghai", "Shanghai"),
}


def _skyscanner_route_from_query(query: str) -> tuple[SkyscannerPlace, SkyscannerPlace] | None:
    pieces = [piece for piece in re.split(r"\s+", query.strip()) if piece]
    places: list[SkyscannerPlace] = []
    for piece in pieces:
        place = _SKYSCANNER_PLACES.get(_normalise_place(piece))
        if place is not None and place not in places:
            places.append(place)
        if len(places) >= 2:
            return places[0], places[1]
    return None


def _normalise_duckduckgo_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    for key in ("uddg", "u"):
        values = query.get(key)
        if values:
            return values[0]
    return url


def _normalise_place(value: str) -> str:
    return value.strip().upper()


def _currency_from_match(text: str) -> str:
    upper = text.upper()
    if "$" in text or "USD" in upper:
        return "USD"
    return "CNY"


def _strip_tags(value: str) -> str:
    return re.sub(r"<.*?>", "", value).strip()
