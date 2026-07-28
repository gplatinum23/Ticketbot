from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
import uuid
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import Callable, Literal, Protocol, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from .models import (
    FlightEvidence,
    FlightOption,
    FlightPageAttemptResult,
    FlightSearchIntent,
    SearchResult,
)
from .places import (
    air_endpoint_matches,
    query_endpoint_matches,
    resolve_actual_airport,
)
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


FlightSearchAction = Literal[
    "search_primary",
    "search_secondary",
    "search_homepage",
    "refresh_capture",
    "await_human_verification",
    "relax_time_preference",
    "finish",
    "stop",
]


class FlightSearchObservation(BaseModel):
    action_id: str
    action: FlightSearchAction
    entrypoint: str
    status: str
    evidence_count: int = 0
    strict_match_count: int = 0
    fallback_match_count: int = 0
    warning: str | None = None


class FlightSearchActionDecision(BaseModel):
    action: FlightSearchAction
    reason: str = Field(default="", description="Short operational reason, not hidden reasoning.")


class FlightActionPlanner(Protocol):
    def plan(
        self,
        *,
        intent: FlightSearchIntent,
        observation: FlightSearchObservation,
        action_history: list[dict[str, object]],
        remaining_actions: int,
    ) -> FlightSearchActionDecision:
        """Choose the next bounded page-search action."""


class LlmFlightActionPlanner:
    def __init__(self, llm) -> None:
        self.llm = llm

    def plan(
        self,
        *,
        intent: FlightSearchIntent,
        observation: FlightSearchObservation,
        action_history: list[dict[str, object]],
        remaining_actions: int,
    ) -> FlightSearchActionDecision:
        structured = self.llm.with_structured_output(FlightSearchActionDecision)
        response = structured.invoke(
            [
                ("system", _flight_action_planner_prompt()),
                (
                    "human",
                    json.dumps(
                        {
                            "intent": {
                                "origin": intent.origin,
                                "destination": intent.destination,
                                "travel_date": intent.travel_date.isoformat(),
                                "time_preference": intent.time_preference,
                                "currency": intent.currency,
                            },
                            "observation": observation.model_dump(mode="json"),
                            "action_history": action_history,
                            "remaining_actions": remaining_actions,
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        if isinstance(response, FlightSearchActionDecision):
            return response
        return FlightSearchActionDecision.model_validate(response)


class CompiledReactFlightSearchGraph:
    def __init__(self, graph) -> None:
        self._graph = graph

    def invoke(self, state, config=None, **kwargs):
        actual_config = config or {
            "configurable": {"thread_id": f"flight-{uuid.uuid4().hex}"}
        }
        return self._graph.invoke(state, config=actual_config, **kwargs)

    def get_graph(self, *args, **kwargs):
        return self._graph.get_graph(*args, **kwargs)


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
            actual_origin = _evidence_airport_code(item, departure=True)
            actual_destination = _evidence_airport_code(item, departure=False)
            grouped[
                (
                    _normalise_place(actual_origin or item.origin or intent.origin),
                    _normalise_place(
                        actual_destination or item.destination or intent.destination
                    ),
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
                    requested_origin=intent.origin_place,
                    requested_destination=intent.destination_place,
                    actual_origin=resolve_actual_airport(origin),
                    actual_destination=resolve_actual_airport(destination),
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

    def extract_attempt(
        self,
        url: str,
        *,
        entrypoint: str,
        action_id: str,
        force_refresh: bool = False,
    ) -> FlightPageAttemptResult:
        for extractor in self.extractors:
            supports = getattr(extractor, "supports", None)
            if supports is not None and not supports(url):
                continue
            adaptive_extract = getattr(extractor, "extract_attempt", None)
            if callable(adaptive_extract):
                return adaptive_extract(
                    url,
                    entrypoint=entrypoint,
                    action_id=action_id,
                    force_refresh=force_refresh,
                )
            evidence = extractor.extract(url)
            return FlightPageAttemptResult(
                status="success" if evidence else "no_evidence",
                evidence=evidence,
                entrypoint=entrypoint,
                source_url=url,
            )
        return FlightPageAttemptResult(
            status="no_evidence",
            evidence=[],
            entrypoint=entrypoint,
            source_url=url,
            warning="no compatible page extractor",
        )


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
    observations: list[dict[str, object]]
    action_history: list[dict[str, object]]
    attempted_entrypoints: list[str]
    remaining_actions: int
    strict_evidence: list[FlightEvidence]
    fallback_evidence: list[FlightEvidence]
    current_action: dict[str, object]
    human_interrupt_count: int
    termination_reason: str | None
    _last_entrypoint: str
    _last_status: str
    _last_warning: str | None


def build_react_flight_search_graph(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    evidence_judge: FlightEvidenceJudge | None = None,
    action_planner: FlightActionPlanner | None = None,
    verifier: FlightEvidenceVerifier | None = None,
    max_iterations: int = 4,
    progress_reporter: ProgressReporter | None = None,
):
    evidence_verifier = verifier or FlightEvidenceVerifier()
    progress = get_progress_reporter(progress_reporter)
    graph = StateGraph(ReactFlightSearchState)

    def initialize_search(state: ReactFlightSearchState) -> ReactFlightSearchState:
        max_actions = state.get("max_iterations", max_iterations)
        return {
            **state,
            "iteration": state.get("iteration", 0),
            "max_iterations": max_actions,
            "search_queries": list(state.get("search_queries", [])),
            "raw_results": list(state.get("raw_results", [])),
            "extracted_evidence": list(state.get("extracted_evidence", [])),
            "judged_evidence": list(state.get("judged_evidence", [])),
            "verified_flight_options": list(state.get("verified_flight_options", [])),
            "warnings": list(state.get("warnings", [])),
            "observations": list(state.get("observations", [])),
            "action_history": list(state.get("action_history", [])),
            "attempted_entrypoints": list(state.get("attempted_entrypoints", [])),
            "remaining_actions": max(0, max_actions - state.get("iteration", 0)),
            "strict_evidence": list(state.get("strict_evidence", [])),
            "fallback_evidence": list(state.get("fallback_evidence", [])),
            "current_action": state.get("current_action")
            or FlightSearchActionDecision(
                action="search_primary",
                reason="initial deterministic fast path",
            ).model_dump(),
            "human_interrupt_count": state.get("human_interrupt_count", 0),
            "termination_reason": None,
        }

    def execute_action(state: ReactFlightSearchState) -> ReactFlightSearchState:
        decision = FlightSearchActionDecision.model_validate(state["current_action"])
        iteration = state.get("iteration", 0) + 1
        max_actions = state.get("max_iterations", max_iterations)
        entrypoint = _entrypoint_for_action(decision.action, state.get("_last_entrypoint"))
        action_id = f"action-{iteration}"
        query = _build_search_query(state["intent"], iteration, decision.action)
        progress.emit(f"执行机票搜索动作 {iteration}/{max_actions}: {decision.action} ({entrypoint})")
        progress.emit(f"搜索公开机票页面: {query}")
        warnings = list(state.get("warnings", []))
        raw_results = list(state.get("raw_results", []))
        extracted_evidence = list(state.get("extracted_evidence", []))
        strict_evidence = list(state.get("strict_evidence", []))
        fallback_evidence = list(state.get("fallback_evidence", []))
        status = "no_results"
        warning: str | None = None
        attempt_evidence: list[FlightEvidence] = []
        try:
            query_results = web_search.search(query)
            if not query_results:
                warning = f"web_search_no_results:{query}"
                warnings.append(warning)
            raw_results.extend(query_results)
        except Exception as exc:
            status = "tool_error"
            warning = f"web_search_failed:{exc}"
            warnings.append(warning)
            query_results = []

        intent = state["intent"]
        for result in query_results:
            try:
                attempt = _extract_page_attempt(
                    page_extractor,
                    result.url,
                    entrypoint=entrypoint,
                    action_id=action_id,
                    force_refresh=decision.action == "refresh_capture",
                )
            except Exception as exc:
                status = _status_from_exception(exc)
                warning = f"page_extract_failed:{result.url}:{exc}"
                warnings.append(warning)
                continue
            status = attempt.status
            if attempt.warning:
                warning = attempt.warning
                warnings.append(f"page_extract_{attempt.status}:{result.url}:{attempt.warning}")
            if not attempt.evidence and attempt.status == "no_evidence":
                warnings.append(f"page_extract_no_evidence:{result.url}")
            attempt_evidence.extend(_normalise_extracted_evidence(attempt.evidence, result, intent))

        hard_matches = [
            item for item in attempt_evidence if _matches_intent_without_time_preference(item, intent)
        ]
        strict_matches = [item for item in hard_matches if _matches_intent(item, intent)]
        if strict_matches:
            status = "success"
        elif hard_matches and intent.time_preference:
            status = "time_preference_mismatch"
        elif attempt_evidence:
            status = "hard_constraint_mismatch"
        extracted_evidence.extend(attempt_evidence)
        fallback_evidence.extend(item for item in hard_matches if item not in fallback_evidence)
        strict_evidence.extend(item for item in strict_matches if item not in strict_evidence)

        observation = FlightSearchObservation(
            action_id=action_id,
            action=decision.action,
            entrypoint=entrypoint,
            status=status,
            evidence_count=len(attempt_evidence),
            strict_match_count=len(strict_matches),
            fallback_match_count=len(hard_matches),
            warning=warning,
        )
        action_record = {
            "action_id": action_id,
            "action": decision.action,
            "entrypoint": entrypoint,
            "status": status,
            "reason": decision.reason,
        }
        attempted_entrypoints = list(state.get("attempted_entrypoints", []))
        if entrypoint not in attempted_entrypoints:
            attempted_entrypoints.append(entrypoint)

        return {
            **state,
            "iteration": iteration,
            "current_query": query,
            "search_queries": state.get("search_queries", []) + [query],
            "raw_results": _dedupe_results(raw_results),
            "extracted_evidence": extracted_evidence,
            "strict_evidence": strict_evidence,
            "fallback_evidence": fallback_evidence,
            "observations": state.get("observations", []) + [observation.model_dump(mode="json")],
            "action_history": state.get("action_history", []) + [action_record],
            "attempted_entrypoints": attempted_entrypoints,
            "remaining_actions": max(0, max_actions - iteration),
            "_last_entrypoint": entrypoint,
            "_last_status": status,
            "_last_warning": warning,
            "warnings": warnings,
        }

    def judge_and_verify(state: ReactFlightSearchState) -> ReactFlightSearchState:
        progress.emit("判断并验证机票证据...")
        strict_evidence = state.get("strict_evidence", [])
        if evidence_judge is None:
            judged = strict_evidence
            warnings = list(state.get("warnings", []))
        else:
            judged, warnings = _judge_evidence_by_url(
                evidence_judge,
                state["intent"],
                state.get("raw_results", []),
                strict_evidence,
                state.get("warnings", []),
            )
        options = evidence_verifier.verify(judged, state["intent"])
        return {
            **state,
            "judged_evidence": judged,
            "verified_flight_options": options,
            "warnings": warnings,
            "termination_reason": "verified" if options else state.get("termination_reason"),
        }

    def plan_next_action(state: ReactFlightSearchState) -> ReactFlightSearchState:
        remaining = state.get("remaining_actions", 0)
        fallback_available = bool(state.get("fallback_evidence")) and bool(state["intent"].time_preference)
        if remaining <= 0:
            decision = FlightSearchActionDecision(
                action="relax_time_preference" if fallback_available else "stop",
                reason="page action budget exhausted",
            )
        else:
            observation = FlightSearchObservation.model_validate(state["observations"][-1])
            if action_planner is not None:
                try:
                    proposed = action_planner.plan(
                        intent=state["intent"],
                        observation=observation,
                        action_history=state.get("action_history", []),
                        remaining_actions=remaining,
                    )
                except Exception as exc:
                    proposed = _deterministic_next_action(state, observation)
                    proposed = proposed.model_copy(
                        update={"reason": f"action planner failed; {proposed.reason}: {exc}"}
                    )
            else:
                proposed = _deterministic_next_action(state, observation)
            decision = _validate_action_decision(proposed, state)
        return {**state, "current_action": decision.model_dump()}

    def interrupt_for_human(state: ReactFlightSearchState) -> ReactFlightSearchState:
        resumed = interrupt(
            {
                "kind": "ctrip_manual_verification",
                "origin": state["intent"].origin,
                "destination": state["intent"].destination,
                "message": "Complete Ctrip verification in the open browser, then resume.",
            }
        )
        if not resumed:
            return {
                **state,
                "current_action": FlightSearchActionDecision(
                    action="stop",
                    reason="manual verification was not completed",
                ).model_dump(),
                "termination_reason": "human_verification_declined",
            }
        return {
            **state,
            "human_interrupt_count": state.get("human_interrupt_count", 0) + 1,
            "current_action": FlightSearchActionDecision(
                action="refresh_capture",
                reason="resume after manual verification",
            ).model_dump(),
        }

    def accept_fallback(state: ReactFlightSearchState) -> ReactFlightSearchState:
        intent = state["intent"]
        relaxed_intent = replace(intent, time_preference=None)
        evidence = state.get("fallback_evidence", [])
        if evidence_judge is None:
            judged = evidence
            warnings = list(state.get("warnings", []))
        else:
            judged, warnings = _judge_evidence_by_url(
                evidence_judge,
                relaxed_intent,
                state.get("raw_results", []),
                evidence,
                state.get("warnings", []),
            )
        marker = f"time_preference_not_met:{intent.time_preference}"
        options = [
            replace(option, warnings=[*option.warnings, marker])
            for option in evidence_verifier.verify(judged, relaxed_intent)
        ]
        if marker not in warnings:
            warnings.append(marker)
        return {
            **state,
            "judged_evidence": judged,
            "verified_flight_options": options,
            "warnings": warnings,
            "termination_reason": "time_preference_fallback" if options else "insufficient_evidence",
        }

    def finish_failure(state: ReactFlightSearchState) -> ReactFlightSearchState:
        warnings = list(state.get("warnings", []))
        if "insufficient_verified_flight_evidence" not in warnings:
            warnings.append("insufficient_verified_flight_evidence")
        return {
            **state,
            "verified_flight_options": [],
            "warnings": warnings,
            "termination_reason": state.get("termination_reason") or "insufficient_evidence",
        }

    def after_execute(state: ReactFlightSearchState) -> str:
        if state.get("_last_status") == "captcha_required":
            return "human"
        if state.get("strict_evidence"):
            return "judge"
        return "plan"

    def after_judge(state: ReactFlightSearchState) -> str:
        return "done" if state.get("verified_flight_options") else "plan"

    def route_action(state: ReactFlightSearchState) -> str:
        action = FlightSearchActionDecision.model_validate(state["current_action"]).action
        if action == "await_human_verification":
            return "human"
        if action == "relax_time_preference":
            return "fallback"
        if action in {"finish", "stop"}:
            return "fail"
        return "execute"

    def after_human(state: ReactFlightSearchState) -> str:
        action = FlightSearchActionDecision.model_validate(state["current_action"]).action
        return "fail" if action == "stop" else "execute"

    graph.add_node("initialize_search", initialize_search)
    graph.add_node("execute_action", execute_action)
    graph.add_node("judge_and_verify", judge_and_verify)
    graph.add_node("plan_next_action", plan_next_action)
    graph.add_node("interrupt_for_human", interrupt_for_human)
    graph.add_node("accept_fallback", accept_fallback)
    graph.add_node("finish_failure", finish_failure)

    graph.add_edge(START, "initialize_search")
    graph.add_edge("initialize_search", "execute_action")
    graph.add_conditional_edges(
        "execute_action",
        after_execute,
        {"human": "interrupt_for_human", "judge": "judge_and_verify", "plan": "plan_next_action"},
    )
    graph.add_conditional_edges(
        "judge_and_verify",
        after_judge,
        {"done": END, "plan": "plan_next_action"},
    )
    graph.add_conditional_edges(
        "plan_next_action",
        route_action,
        {
            "human": "interrupt_for_human",
            "fallback": "accept_fallback",
            "fail": "finish_failure",
            "execute": "execute_action",
        },
    )
    graph.add_conditional_edges(
        "interrupt_for_human",
        after_human,
        {"execute": "execute_action", "fail": "finish_failure"},
    )
    graph.add_edge("accept_fallback", END)
    graph.add_edge("finish_failure", END)

    return CompiledReactFlightSearchGraph(graph.compile(checkpointer=InMemorySaver()))


def invoke_react_flight_search(
    graph,
    state: dict[str, object],
    *,
    human_verification_handler: Callable[[dict[str, object]], bool] | None = None,
    thread_id: str | None = None,
) -> dict[str, object]:
    config = {"configurable": {"thread_id": thread_id or f"flight-{uuid.uuid4().hex}"}}
    result = graph.invoke(state, config=config)
    while result.get("__interrupt__"):
        payload = _interrupt_payload(result)
        resumed = bool(human_verification_handler and human_verification_handler(payload))
        result = graph.invoke(Command(resume=resumed), config=config)
    return result


def _interrupt_payload(state: dict[str, object]) -> dict[str, object]:
    interrupts = state.get("__interrupt__") or []
    if not interrupts:
        return {}
    value = getattr(interrupts[0], "value", {})
    return value if isinstance(value, dict) else {"message": str(value)}


def _extract_page_attempt(
    page_extractor: PageExtractor,
    url: str,
    *,
    entrypoint: str,
    action_id: str,
    force_refresh: bool,
) -> FlightPageAttemptResult:
    adaptive_extract = getattr(page_extractor, "extract_attempt", None)
    if callable(adaptive_extract):
        return adaptive_extract(
            url,
            entrypoint=entrypoint,
            action_id=action_id,
            force_refresh=force_refresh,
        )
    evidence = page_extractor.extract(url)
    return FlightPageAttemptResult(
        status="success" if evidence else "no_evidence",
        evidence=evidence,
        entrypoint=entrypoint,
        source_url=url,
    )


def _judge_evidence_by_url(
    evidence_judge: FlightEvidenceJudge,
    intent: FlightSearchIntent,
    raw_results: list[SearchResult],
    evidence: list[FlightEvidence],
    existing_warnings: list[str],
) -> tuple[list[FlightEvidence], list[str]]:
    warnings = list(existing_warnings)
    judged: list[FlightEvidence] = []
    evidence_by_url: dict[str, list[FlightEvidence]] = defaultdict(list)
    for item in evidence:
        evidence_by_url[item.url].append(item)
    result_by_url = {result.url: result for result in raw_results}
    for url, url_evidence in evidence_by_url.items():
        search_result = result_by_url.get(
            url,
            SearchResult(
                title="",
                url=url,
                snippet="",
                source_name=urllib.parse.urlparse(url).netloc or "web",
            ),
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
    return judged, warnings


def _entrypoint_for_action(action: FlightSearchAction, last_entrypoint: str | None) -> str:
    return {
        "search_primary": "international",
        "search_secondary": "online_list",
        "search_homepage": "homepage",
        "refresh_capture": last_entrypoint or "international",
    }.get(action, last_entrypoint or "international")


def _status_from_exception(exc: Exception) -> str:
    text = str(exc).casefold()
    if "captcha" in text or "manual verification" in text or "security verification" in text:
        return "captcha_required"
    if "login" in text or "account" in text or "password" in text:
        return "login_required"
    if "payload" in text or "batchsearch" in text or "timed out" in text:
        return "no_payload"
    if isinstance(exc, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return "parse_failed"
    return "tool_error"


def _deterministic_next_action(
    state: ReactFlightSearchState,
    observation: FlightSearchObservation,
) -> FlightSearchActionDecision:
    if observation.status == "captcha_required":
        return FlightSearchActionDecision(
            action="await_human_verification",
            reason="Ctrip requires manual verification",
        )
    attempted = set(state.get("attempted_entrypoints", []))
    for action, entrypoint in (
        ("search_secondary", "online_list"),
        ("search_homepage", "homepage"),
        ("search_primary", "international"),
    ):
        if entrypoint not in attempted:
            return FlightSearchActionDecision(
                action=action,
                reason=f"try unvisited Ctrip entrypoint after {observation.status}",
            )
    if state.get("fallback_evidence") and state["intent"].time_preference:
        return FlightSearchActionDecision(
            action="relax_time_preference",
            reason="strict time preference has no matching inventory",
        )
    refresh_count = sum(
        item.get("action") == "refresh_capture" for item in state.get("action_history", [])
    )
    if refresh_count == 0:
        return FlightSearchActionDecision(
            action="refresh_capture",
            reason="all entrypoints tried; perform one fresh capture",
        )
    return FlightSearchActionDecision(action="stop", reason="no recoverable action remains")


def _validate_action_decision(
    proposed: FlightSearchActionDecision,
    state: ReactFlightSearchState,
) -> FlightSearchActionDecision:
    observation = FlightSearchObservation.model_validate(state["observations"][-1])
    if observation.status == "captcha_required":
        return FlightSearchActionDecision(
            action="await_human_verification",
            reason="policy requires human verification for captcha",
        )
    if proposed.action == "await_human_verification":
        return _deterministic_next_action(state, observation)
    if proposed.action == "relax_time_preference":
        if state.get("fallback_evidence") and state["intent"].time_preference:
            return proposed
        return _deterministic_next_action(state, observation)
    if proposed.action in {"search_primary", "search_secondary", "search_homepage"}:
        entrypoint = _entrypoint_for_action(proposed.action, state.get("_last_entrypoint"))
        if entrypoint in state.get("attempted_entrypoints", []):
            return _deterministic_next_action(state, observation)
    if proposed.action == "refresh_capture":
        refresh_count = sum(
            item.get("action") == "refresh_capture" for item in state.get("action_history", [])
        )
        if refresh_count >= 1:
            return _deterministic_next_action(state, observation)
    return proposed


def _build_search_query(
    intent: FlightSearchIntent,
    iteration: int,
    action: FlightSearchAction = "search_primary",
) -> str:
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
    pieces.append(f"react_action={action}")
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


def _flight_action_planner_prompt() -> str:
    return """
You choose the next bounded browser action for one fixed flight route on Ctrip.

You may choose only one of:
- search_primary
- search_secondary
- search_homepage
- refresh_capture
- await_human_verification
- relax_time_preference
- finish
- stop

Rules:
- Never change the origin, destination, travel date, currency, or route scope.
- Use await_human_verification only when the observation reports a captcha.
- Use relax_time_preference only when matching-route evidence exists but misses the requested time preference.
- Prefer an untried entrypoint over repeating an entrypoint.
- Use refresh_capture at most once and only after entrypoints have been tried or verification was resumed.
- Stop when there is no useful recovery action.
- Return a short operational reason. Do not return private chain-of-thought.
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
    actual_origin = _evidence_airport_code(item, departure=True)
    actual_destination = _evidence_airport_code(item, departure=False)
    observed_origin = actual_origin or item.origin
    observed_destination = actual_destination or item.destination
    origin_matches = (
        True
        if observed_origin is None
        else air_endpoint_matches(intent.origin_place, observed_origin)
        if actual_origin is not None
        else query_endpoint_matches(intent.origin_place, observed_origin)
    )
    destination_matches = (
        True
        if observed_destination is None
        else air_endpoint_matches(intent.destination_place, observed_destination)
        if actual_destination is not None
        else query_endpoint_matches(intent.destination_place, observed_destination)
    )
    if observed_origin is not None and not origin_matches:
        return False
    if observed_destination is not None and not destination_matches:
        return False
    return True


def _evidence_airport_code(
    item: FlightEvidence,
    *,
    departure: bool,
) -> str | None:
    metadata = item.metadata or {}
    key = "departure_airport_code" if departure else "arrival_airport_code"
    value = metadata.get(key)
    text = str(value or "").strip().upper()
    return text or None


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
