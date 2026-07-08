from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .models import FlightEvidence, FlightOption, FlightSearchIntent, SearchResult


class WebSearchTool(Protocol):
    def search(self, query: str) -> list[SearchResult]:
        """Search public pages for flight price evidence."""


class PageExtractor(Protocol):
    def extract(self, url: str) -> list[FlightEvidence]:
        """Extract flight evidence from a public page."""


class FlightEvidenceVerifier:
    def __init__(
        self,
        *,
        min_sources: int = 2,
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
            independent_sources = {_source_identity(item) for item in items}
            if len(independent_sources) < self.min_sources:
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
        params = urllib.parse.urlencode({"q": query})
        request = urllib.request.Request(
            f"https://html.duckduckgo.com/html/?{params}",
            headers={"User-Agent": "flight-watch-agent/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")

        results: list[SearchResult] = []
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)">(?P<title>.*?)</a>',
            re.DOTALL,
        )
        for match in pattern.finditer(body):
            url = html.unescape(match.group("url"))
            title = _strip_tags(html.unescape(match.group("title")))
            source_name = urllib.parse.urlparse(url).netloc or "web"
            results.append(SearchResult(title=title, url=url, snippet="", source_name=source_name))
            if len(results) >= self.max_results:
                break
        return results


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
    verified_flight_options: list[FlightOption]
    warnings: list[str]


def build_react_flight_search_graph(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    verifier: FlightEvidenceVerifier | None = None,
    max_iterations: int = 3,
):
    evidence_verifier = verifier or FlightEvidenceVerifier()
    graph = StateGraph(ReactFlightSearchState)

    def generate_search_plan(state: ReactFlightSearchState) -> ReactFlightSearchState:
        iteration = state.get("iteration", 0) + 1
        max_iters = state.get("max_iterations", max_iterations)
        query = _build_search_query(state["intent"], iteration)
        return {
            **state,
            "iteration": iteration,
            "max_iterations": max_iters,
            "current_query": query,
            "search_queries": state.get("search_queries", []) + [query],
        }

    def run_web_search(state: ReactFlightSearchState) -> ReactFlightSearchState:
        warnings = list(state.get("warnings", []))
        raw_results = list(state.get("raw_results", []))
        try:
            raw_results.extend(web_search.search(state["current_query"]))
        except Exception as exc:
            warnings.append(f"web_search_failed:{exc}")
        return {**state, "raw_results": _dedupe_results(raw_results), "warnings": warnings}

    def extract_pages(state: ReactFlightSearchState) -> ReactFlightSearchState:
        warnings = list(state.get("warnings", []))
        evidence = list(state.get("extracted_evidence", []))
        seen_urls = {item.url for item in evidence}
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
            evidence.extend(_normalise_extracted_evidence(extracted, result, intent))

        return {**state, "extracted_evidence": evidence, "warnings": warnings}

    def verify_options(state: ReactFlightSearchState) -> ReactFlightSearchState:
        options = evidence_verifier.verify(state.get("extracted_evidence", []), state["intent"])
        warnings = list(state.get("warnings", []))
        if not options and state.get("iteration", 0) >= state.get("max_iterations", max_iterations):
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
    graph.add_node("verify_options", verify_options)

    graph.add_edge(START, "generate_search_plan")
    graph.add_edge("generate_search_plan", "web_search")
    graph.add_edge("web_search", "extract_pages")
    graph.add_edge("extract_pages", "verify_options")
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


def _matches_intent(item: FlightEvidence, intent: FlightSearchIntent) -> bool:
    if item.price <= 0:
        return False
    if item.travel_date is not None and item.travel_date != intent.travel_date:
        return False
    if item.origin is not None and _normalise_place(item.origin) != _normalise_place(intent.origin):
        return False
    if item.destination is not None and _normalise_place(item.destination) != _normalise_place(intent.destination):
        return False
    return True


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


def _normalise_place(value: str) -> str:
    return value.strip().upper()


def _currency_from_match(text: str) -> str:
    upper = text.upper()
    if "$" in text or "USD" in upper:
        return "USD"
    return "CNY"


def _strip_tags(value: str) -> str:
    return re.sub(r"<.*?>", "", value).strip()
