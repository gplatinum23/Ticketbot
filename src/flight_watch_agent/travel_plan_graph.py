from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
import json
import time
from concurrent.futures import ThreadPoolExecutor
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
    generate_candidate_hubs_for_place_mentions,
    generate_candidate_hubs as generate_agent_candidate_hubs,
    select_strategies as select_agent_strategies,
)
from .flight_react import (
    FlightActionPlanner,
    FlightEvidenceJudge,
    FlightEvidenceJudgeRequest,
    FlightEvidenceVerifier,
    PageExtractor,
    WebSearchTool,
    build_react_flight_search_graph,
    invoke_react_flight_search,
)
from .models import FlightOption, FlightSearchIntent, SearchResult, TrainOption
from .places import (
    get_airport_index,
    get_station_index,
    normalise_airport_code,
    normalise_train_query_place,
    station_city_for_airport,
)
from .progress import ProgressReporter, get_progress_reporter
from .travel_tools import (
    CachedFlightSearchTool,
    CachedTrainSearchTool,
    FlightSearchOutput,
    FlightSearchRequest,
    FlightSearchTool,
    ToolError,
    InMemoryToolCache,
    ToolMetrics,
    ToolResult,
    ToolStatus,
    TrainSearchOutput,
    TrainSearchRequest,
    TrainSearchTool,
    classify_tool_error,
    flight_tool_result_from_state,
)


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


class HubProposer(Protocol):
    def propose(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        strategy_selection: StrategySelection,
        index_hubs: list[CandidateHub],
    ) -> list[object]:
        """Return structured hub suggestions for local index resolution."""


class HubEndpointValidator(Protocol):
    def validate(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        candidate_hubs: list[CandidateHub],
    ) -> list["HubEndpointDecision"]:
        """Return endpoint identity decisions and optional hub corrections."""


class HubPlanner(Protocol):
    def plan(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        strategy_selection: StrategySelection,
        index_hubs: list[CandidateHub],
        supplemental_hubs: list[CandidateHub],
        explicit_hubs: list[CandidateHub],
    ) -> "HubPlanningBatch":
        """Propose supplemental hubs and validate existing hubs in one LLM call."""


@dataclass(frozen=True)
class EndpointIdentity:
    label: str
    cities: frozenset[str]
    airport_codes: frozenset[str]
    train_places: frozenset[str]


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
    total_duration_minutes: int | None = None
    segment_count: int | None = None
    score: float | None = None


class TravelPlanState(TypedDict, total=False):
    user_input: str
    intent: FlightSearchIntent
    explicit_hub_places: list[object]
    region_info: RegionInfo
    strategy_selection: StrategySelection
    index_candidate_hubs: list[CandidateHub]
    llm_candidate_hubs: list[CandidateHub]
    candidate_hubs: list[CandidateHub]
    endpoint_validated_hubs: list[CandidateHub]
    hub_endpoint_decisions: list["HubEndpointDecision"]
    query_plan: QueryPlan
    route_edges: list[RouteEdge]
    train_options: list[TrainOption]
    verified_flight_options: list[FlightOption]
    transfer_train_options: list[TrainOption]
    transfer_flight_options: list[FlightOption]
    flight_search_debug: dict[str, object]
    transfer_search_debug: dict[str, object]
    prefetched_direct_flight_result: ToolResult[FlightSearchOutput]
    query_execution_stats: dict[str, int]
    candidate_routes: list[CandidateRoute]
    response: str
    warnings: list[str]


def build_react_flight_search_tool(
    *,
    web_search: WebSearchTool,
    page_extractor: PageExtractor,
    evidence_judge: FlightEvidenceJudge | None = None,
    action_planner: FlightActionPlanner | None = None,
    verifier: FlightEvidenceVerifier | None = None,
    progress_reporter: ProgressReporter | None = None,
    human_verification_handler=None,
    cache: InMemoryToolCache | None = None,
) -> FlightSearchTool:
    evidence_verifier = verifier or FlightEvidenceVerifier()
    batch_evidence_judge = (
        evidence_judge
        if callable(getattr(evidence_judge, "judge_many", None))
        else None
    )
    primary_graph = build_react_flight_search_graph(
        web_search=web_search,
        page_extractor=page_extractor,
        evidence_judge=None if batch_evidence_judge is not None else evidence_judge,
        action_planner=action_planner,
        verifier=evidence_verifier,
        progress_reporter=progress_reporter,
    )
    fallback_graph = (
        build_react_flight_search_graph(
            web_search=web_search,
            page_extractor=page_extractor,
            evidence_judge=evidence_judge,
            action_planner=action_planner,
            verifier=evidence_verifier,
            progress_reporter=progress_reporter,
        )
        if batch_evidence_judge is not None
        else primary_graph
    )

    def run_batch(
        requests: list[FlightSearchRequest] | tuple[FlightSearchRequest, ...],
    ) -> list[ToolResult[FlightSearchOutput]]:
        states: dict[str, dict[str, object]] = {}
        timings: dict[str, tuple[datetime, float]] = {}
        failed: dict[str, ToolResult[FlightSearchOutput]] = {}
        for request in requests:
            timings[request.request_id] = (datetime.now(timezone.utc), time.monotonic())
            try:
                states[request.request_id] = invoke_react_flight_search(
                    primary_graph,
                    {"intent": request.to_intent()},
                    human_verification_handler=human_verification_handler,
                )
            except Exception as exc:
                started_at, started_monotonic = timings[request.request_id]
                failed[request.request_id] = ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    error=classify_tool_error(exc),
                    metrics=ToolMetrics(
                        request_id=request.request_id,
                        started_at=started_at,
                        latency_ms=max(
                            0,
                            round((time.monotonic() - started_monotonic) * 1000),
                        ),
                        cache_hit=False,
                        attempts=1,
                        backend="ctrip_react",
                    ),
                )

        if batch_evidence_judge is not None and states:
            states = _batch_judge_flight_searches(
                states,
                batch_evidence_judge,
                evidence_verifier,
                fallback_graph,
                human_verification_handler=human_verification_handler,
            )

        results: list[ToolResult[FlightSearchOutput]] = []
        for request in requests:
            if request.request_id in failed:
                results.append(failed[request.request_id])
                continue
            state = _limit_flight_tool_state(states[request.request_id], request)
            started_at, started_monotonic = timings[request.request_id]
            results.append(
                flight_tool_result_from_state(
                    request,
                    state,
                    started_at=started_at,
                    started_monotonic=started_monotonic,
                )
            )
        return results

    return CachedFlightSearchTool(run_batch, cache=cache)


def build_travel_plan_graph(
    *,
    web_search: WebSearchTool | None = None,
    page_extractor: PageExtractor | None = None,
    train_provider: TrainProvider | None = None,
    flight_tool: FlightSearchTool | None = None,
    train_tool: TrainSearchTool | None = None,
    evidence_judge: FlightEvidenceJudge | None = None,
    flight_action_planner: FlightActionPlanner | None = None,
    verifier: FlightEvidenceVerifier | None = None,
    route_planner: RoutePlanner | None = None,
    hub_planner: HubPlanner | None = None,
    hub_proposer: HubProposer | None = None,
    hub_endpoint_validator: HubEndpointValidator | None = None,
    progress_reporter: ProgressReporter | None = None,
    human_verification_handler=None,
    transfer_hubs: list[str] | None = None,
):
    progress = get_progress_reporter(progress_reporter)
    evidence_verifier = verifier or FlightEvidenceVerifier()
    if flight_tool is None:
        if web_search is None or page_extractor is None:
            raise ValueError("flight_tool or both web_search and page_extractor are required.")
        flight_tool = build_react_flight_search_tool(
            web_search=web_search,
            page_extractor=page_extractor,
            evidence_judge=evidence_judge,
            action_planner=flight_action_planner,
            verifier=evidence_verifier,
            progress_reporter=progress,
            human_verification_handler=human_verification_handler,
        )
    if train_tool is None and train_provider is not None:
        train_tool = CachedTrainSearchTool(train_provider)
    graph = StateGraph(TravelPlanState)

    def classify_region(state: TravelPlanState) -> TravelPlanState:
        progress.emit("判断出发地和目的地区域...")
        return {**state, "region_info": classify_agent_region(state["intent"])}

    def select_strategies(state: TravelPlanState) -> TravelPlanState:
        progress.emit("选择可用出行策略...")
        return {
            **state,
            "strategy_selection": select_agent_strategies(
                state["intent"],
                state["region_info"],
            ),
        }

    def prefetch_direct_flight(state: TravelPlanState) -> TravelPlanState:
        if hub_planner is None or "direct_flight" not in state["strategy_selection"].enabled:
            return {}
        progress.emit("并行查询直达机票...")
        return {
            "prefetched_direct_flight_result": flight_tool.search(
                FlightSearchRequest.from_intent(state["intent"])
            )
        }

    def generate_candidate_hubs(state: TravelPlanState) -> TravelPlanState:
        progress.emit("生成候选中转城市...")
        candidate_budget = QueryBudget(max_hubs_per_strategy=30, max_flight_queries=50, max_train_queries=50)
        merge_budget = QueryBudget(max_hubs_per_strategy=10, max_flight_queries=50, max_train_queries=50)
        index_hubs = generate_agent_candidate_hubs(
            state["intent"],
            state["strategy_selection"],
            budget=QueryBudget(max_hubs_per_strategy=250, max_flight_queries=250, max_train_queries=250)
            if transfer_hubs is not None
            else candidate_budget,
        )
        rule_hubs = index_hubs[:5]
        supplemental_hubs = index_hubs[5:20]
        progress.emit(_format_hub_progress("规则候选 hub", rule_hubs, limit=5))
        progress.emit(_format_hub_progress("原始候选 hub", index_hubs, limit=10))
        explicit_places = list(state.get("explicit_hub_places", []))
        if transfer_hubs is not None:
            explicit_places.extend(transfer_hubs)

        explicit_hubs, explicit_warnings = generate_candidate_hubs_for_place_mentions(
            explicit_places,
            state["strategy_selection"],
        )
        if explicit_hubs:
            progress.emit(_format_hub_progress("显式候选 hub", explicit_hubs))
        if transfer_hubs is not None:
            index_hubs = []
            rule_hubs = []
            supplemental_hubs = []

        llm_hubs: list[CandidateHub] = []
        llm_warnings: list[str] = []
        endpoint_decisions: list[HubEndpointDecision] = []
        if hub_planner is not None and transfer_hubs is None:
            try:
                hub_plan = hub_planner.plan(
                    user_input=state.get("user_input", ""),
                    intent=state["intent"],
                    region_info=state["region_info"],
                    strategy_selection=state["strategy_selection"],
                    index_hubs=rule_hubs,
                    supplemental_hubs=supplemental_hubs,
                    explicit_hubs=explicit_hubs,
                )
                llm_hubs, llm_warnings = generate_candidate_hubs_for_place_mentions(
                    hub_plan.suggestions[:5],
                    state["strategy_selection"],
                )
                endpoint_decisions = list(hub_plan.decisions)
            except Exception as exc:
                llm_warnings.append(f"llm_hub_planning_failed:{exc}")
        elif hub_proposer is not None and transfer_hubs is None:
            try:
                suggestions = hub_proposer.propose(
                    user_input=state.get("user_input", ""),
                    intent=state["intent"],
                    region_info=state["region_info"],
                    strategy_selection=state["strategy_selection"],
                    index_hubs=rule_hubs,
                )
                llm_hubs, llm_warnings = generate_candidate_hubs_for_place_mentions(
                    suggestions[:5],
                    state["strategy_selection"],
                )
            except Exception as exc:
                llm_warnings.append(f"llm_hub_proposal_failed:{exc}")
        llm_hubs = llm_hubs[:5]
        progress.emit(_format_hub_progress("LLM 推荐 hub", llm_hubs, limit=5))
        if hub_planner is not None and transfer_hubs is None:
            backfilled = _backfill_supplemental_hubs(
                llm_hubs,
                supplemental_hubs,
                excluded_hub_ids={hub.hub_id for hub in [*explicit_hubs, *rule_hubs]},
                limit=5,
            )
            if len(backfilled) > len(llm_hubs):
                llm_warnings.append(f"supplemental_hubs_backfilled:{len(backfilled) - len(llm_hubs)}")
            llm_hubs = backfilled
        progress.emit(_format_hub_progress("LLM 补充 hub", llm_hubs, limit=5))

        hubs = _merge_hubs_with_budget(
            explicit_hubs,
            llm_hubs,
            rule_hubs,
            budget=merge_budget,
        )
        warnings = list(state.get("warnings", [])) + explicit_warnings + [
            f"llm_hub:{warning}" for warning in llm_warnings
        ]
        progress.emit(_format_hub_progress("进入查询计划的 hub", hubs, limit=10))
        return {
            **state,
            "index_candidate_hubs": rule_hubs,
            "llm_candidate_hubs": llm_hubs,
            "candidate_hubs": hubs,
            "hub_endpoint_decisions": endpoint_decisions,
            "warnings": warnings,
        }

    def validate_candidate_hubs(state: TravelPlanState) -> TravelPlanState:
        progress.emit("校验候选中转是否误判为起终点...")
        decisions = list(state.get("hub_endpoint_decisions", []))
        warnings = list(state.get("warnings", []))
        if hub_endpoint_validator is not None:
            try:
                decisions = hub_endpoint_validator.validate(
                    user_input=state.get("user_input", ""),
                    intent=state["intent"],
                    region_info=state["region_info"],
                    candidate_hubs=state.get("candidate_hubs", []),
                )
            except Exception as exc:
                warnings.append(f"llm_endpoint_validator_failed:{exc}")

        validated_hubs, validation_warnings = _validate_candidate_hubs_against_endpoints(
            state["intent"],
            state.get("candidate_hubs", []),
            decisions,
        )
        warnings.extend(validation_warnings)
        return {
            **state,
            "candidate_hubs": validated_hubs,
            "endpoint_validated_hubs": validated_hubs,
            "warnings": warnings,
        }

    def build_query_plan(state: TravelPlanState) -> TravelPlanState:
        progress.emit("构建火车和机票查询计划...")
        query_plan = build_agent_query_plan(
            state["intent"],
            state["strategy_selection"],
            state["candidate_hubs"],
        )
        warnings = list(state.get("warnings", [])) + query_plan.warnings
        return {**state, "query_plan": query_plan, "warnings": warnings}

    def execute_query_plan(state: TravelPlanState) -> TravelPlanState:
        progress.emit("执行查询计划...")
        query_plan = state["query_plan"]
        warnings = list(state.get("warnings", []))
        train_options: list[TrainOption] = []
        verified_flight_options: list[FlightOption] = []
        transfer_train_options: list[TrainOption] = []
        transfer_flight_options: list[FlightOption] = []
        route_edges: list[RouteEdge] = []
        flight_search_debug: dict[str, object] = _empty_flight_debug()
        transfer_search_debug: dict[str, object] = {"hubs": [], "searched": []}
        train_items = [
            item
            for item in query_plan.items
            if item.executable and item.mode == "train" and train_tool is not None
        ]
        flight_items = [
            item
            for item in query_plan.items
            if item.executable and item.mode == "flight"
        ]
        train_requests = [
            _train_request_for_query_item(item)
            for item in train_items
        ]
        flight_requests = [
            FlightSearchRequest.from_intent(_intent_for_query_item(state["intent"], item))
            for item in flight_items
        ]

        for item in [*flight_items, *train_items]:
            progress.emit(_progress_message_for_query_item(item))
        for item in query_plan.items:
            if not item.executable:
                warnings.append(f"query_not_implemented:{item.query_id}")

        train_executor = ThreadPoolExecutor(max_workers=1) if train_requests else None
        train_future = (
            train_executor.submit(train_tool.search_many, train_requests)
            if train_executor is not None and train_tool is not None
            else None
        )
        prefetched_result = state.get("prefetched_direct_flight_result")
        prefetched_request_id = (
            prefetched_result.metrics.request_id
            if prefetched_result is not None
            else None
        )
        pending_flight_requests = [
            request
            for request in flight_requests
            if request.request_id != prefetched_request_id
        ]
        pending_flight_results = flight_tool.search_many(pending_flight_requests)
        pending_results_by_id = {
            result.metrics.request_id: result
            for result in pending_flight_results
        }
        flight_results = [
            prefetched_result
            if request.request_id == prefetched_request_id and prefetched_result is not None
            else pending_results_by_id[request.request_id]
            for request in flight_requests
        ]
        for item, result in zip(flight_items, flight_results, strict=True):
            if result.metrics.cache_hit:
                progress.emit(_progress_reuse_message(item))
            if not result.ok:
                warnings.append(_tool_failure_warning("flight", item, result.error))
                continue
            output = result.data or FlightSearchOutput(options=(), raw_state={})
            react_state = output.raw_state
            current_flight_options = list(output.options)
            if item.strategy == "direct_flight":
                verified_flight_options.extend(current_flight_options)
                flight_search_debug = _react_debug(react_state)
                warnings.extend(result.warnings)
            elif item.strategy == "train_flight":
                transfer_flight_options.extend(current_flight_options)
                warnings.extend(
                    f"transfer_flight:{item.hub_id}:{warning}"
                    for warning in result.warnings
                )
                _append_transfer_debug(
                    transfer_search_debug,
                    item,
                    react_state,
                    current_flight_options,
                )
            else:
                warnings.extend(
                    f"{item.strategy}:{item.hub_id}:{warning}"
                    for warning in result.warnings
                )
                _append_transfer_debug(
                    transfer_search_debug,
                    item,
                    react_state,
                    current_flight_options,
                )
            route_edges.extend(_flight_edges_from_options(current_flight_options, item))

        train_results = train_future.result() if train_future is not None else []
        if train_executor is not None:
            train_executor.shutdown(wait=True)

        for item, result in zip(train_items, train_results, strict=True):
            if result.metrics.cache_hit:
                progress.emit(_progress_reuse_message(item))
            if not result.ok:
                warnings.append(_tool_failure_warning("train", item, result.error))
                continue
            output = result.data or TrainSearchOutput(options=())
            best_options = list(output.options[:3])
            if item.strategy == "direct_train":
                train_options.extend(best_options)
            elif item.strategy == "train_flight":
                transfer_train_options.extend(best_options)
            route_edges.extend(
                _train_edges_from_options(best_options, item, state["intent"].currency)
            )

        planned_train_queries = sum(item.executable and item.mode == "train" for item in query_plan.items)
        planned_flight_queries = sum(item.executable and item.mode == "flight" for item in query_plan.items)
        unique_train_queries = len({request.request_id for request in train_requests})
        unique_flight_queries = len({request.request_id for request in flight_requests})
        train_cache_hits = len({
            result.metrics.request_id
            for result in train_results
            if result.metrics.cache_hit
        })
        flight_cache_hits = len({
            result.metrics.request_id
            for result in flight_results
            if result.metrics.cache_hit
        })
        execution_stats = {
            "planned_train_queries": planned_train_queries,
            "unique_train_queries": unique_train_queries,
            "reused_train_queries": planned_train_queries - unique_train_queries,
            "train_cache_hits": train_cache_hits,
            "planned_flight_queries": planned_flight_queries,
            "unique_flight_queries": unique_flight_queries,
            "reused_flight_queries": planned_flight_queries - unique_flight_queries,
            "flight_cache_hits": flight_cache_hits,
            "flight_evidence_llm_batches": int(
                getattr(evidence_judge, "last_batch_count", 0)
            ),
        }
        progress.emit(
            "查询复用统计: "
            f"机票 {planned_flight_queries} 条计划/{unique_flight_queries} 个唯一请求，"
            f"火车 {planned_train_queries} 条计划/{unique_train_queries} 个唯一请求；"
            f"缓存命中 机票 {flight_cache_hits}/火车 {train_cache_hits}"
        )

        return {
            **state,
            "train_options": train_options,
            "verified_flight_options": verified_flight_options,
            "transfer_train_options": transfer_train_options,
            "transfer_flight_options": transfer_flight_options,
            "route_edges": route_edges,
            "flight_search_debug": flight_search_debug,
            "transfer_search_debug": transfer_search_debug,
            "query_execution_stats": execution_stats,
            "warnings": warnings,
        }

    def build_candidate_routes(state: TravelPlanState) -> TravelPlanState:
        progress.emit("构建候选路线...")
        routes: list[CandidateRoute] = []
        for option in state.get("train_options", []):
            routes.append(
                CandidateRoute(
                    route_id=f"train:{option.train_code}:{option.start_time}",
                    route_type="train",
                    train_option=option,
                    total_price=option.lowest_price,
                    summary=_summarise_train_option(option),
                    total_duration_minutes=_parse_duration_minutes(option.duration),
                    segment_count=1,
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
                    total_duration_minutes=_flight_duration_minutes(option),
                    segment_count=_flight_segment_count(option),
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
        progress.emit("排序候选路线...")
        routes = state.get("candidate_routes", [])
        if not routes or route_planner is None:
            return state
        warnings = list(state.get("warnings", []))
        try:
            ranked = route_planner.rank(intent=state["intent"], routes=routes)
        except Exception as exc:
            warnings.append(f"route_rank_failed:{exc}")
            return {**state, "warnings": warnings}
        return {
            **state,
            "candidate_routes": _enforce_dominance_order(ranked),
            "warnings": warnings,
        }

    def render_response(state: TravelPlanState) -> TravelPlanState:
        progress.emit("生成最终推荐结果...")
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
    graph.add_node("prefetch_direct_flight", prefetch_direct_flight)
    graph.add_node("generate_candidate_hubs", generate_candidate_hubs)
    graph.add_node("validate_candidate_hubs", validate_candidate_hubs)
    graph.add_node("build_query_plan", build_query_plan)
    graph.add_node("execute_query_plan", execute_query_plan)
    graph.add_node("build_candidate_routes", build_candidate_routes)
    graph.add_node("rank_routes", rank_routes)
    graph.add_node("render_response", render_response)

    graph.add_edge(START, "classify_region")
    graph.add_edge("classify_region", "select_strategies")
    graph.add_edge("select_strategies", "generate_candidate_hubs")
    graph.add_edge("select_strategies", "prefetch_direct_flight")
    graph.add_edge(["generate_candidate_hubs", "prefetch_direct_flight"], "validate_candidate_hubs")
    graph.add_edge("validate_candidate_hubs", "build_query_plan")
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


def _progress_message_for_query_item(item: QueryPlanItem) -> str:
    mode_text = "火车" if item.mode == "train" else "机票"
    strategy_text = {
        "direct_flight": "直达",
        "direct_train": "直达",
        "train_flight": "火车+飞机中转",
        "flight_train": "飞机+火车中转",
        "train_train": "火车+火车中转",
        "flight_flight": "飞机+飞机中转",
    }.get(item.strategy, item.strategy)
    return f"查询{strategy_text}{mode_text}: {item.origin} -> {item.destination}"


def _progress_reuse_message(item: QueryPlanItem) -> str:
    mode_text = "火车" if item.mode == "train" else "机票"
    return f"复用已有{mode_text}查询结果: {item.origin} -> {item.destination}"


def _format_hub_progress(label: str, hubs: list[CandidateHub], *, limit: int = 25) -> str:
    if not hubs:
        return f"{label}: 无"
    rendered = []
    for hub in hubs[:limit]:
        airport_text = "/".join(hub.airport_codes) if hub.airport_codes else "-"
        train_text = "/".join(hub.train_places) if hub.train_places else "-"
        strategy_text = "/".join(hub.strategies)
        tier_text = hub.flight_tier or "-"
        score_text = "-" if hub.flight_potential_score is None else f"{hub.flight_potential_score:.2f}"
        rendered.append(
            f"{hub.city}(机场:{airport_text}; tier:{tier_text}; score:{score_text}; "
            f"火车:{train_text}; 策略:{strategy_text})"
        )
    suffix = ""
    if len(hubs) > limit:
        suffix = f"; 另有 {len(hubs) - limit} 个"
    return f"{label}({len(hubs)}): " + " | ".join(rendered) + suffix


def _summarise_flight_option(option: FlightOption) -> str:
    lowest = option.evidence[0] if option.evidence else None
    metadata = lowest.metadata or {} if lowest else {}
    flight_no = metadata.get("flight_no") or "unknown flight"
    time_text = _format_datetime_range(
        option.departure_time,
        option.arrival_time,
        travel_date=option.travel_date,
    )
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
        "observations": [],
        "action_history": [],
        "attempted_entrypoints": [],
        "remaining_actions": 0,
        "human_interrupt_count": 0,
        "termination_reason": None,
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
        "observations": react_state.get("observations", []),
        "action_history": react_state.get("action_history", []),
        "attempted_entrypoints": react_state.get("attempted_entrypoints", []),
        "remaining_actions": react_state.get("remaining_actions", 0),
        "human_interrupt_count": react_state.get("human_interrupt_count", 0),
        "termination_reason": react_state.get("termination_reason"),
        "warnings": react_state.get("warnings", []),
    }


def _react_recovery_state(react_state: dict[str, object]) -> dict[str, object]:
    attempted = set(react_state.get("attempted_entrypoints", []))
    if "online_list" not in attempted:
        next_action = "search_secondary"
    elif "homepage" not in attempted:
        next_action = "search_homepage"
    else:
        next_action = "refresh_capture"
    return {
        **react_state,
        "judged_evidence": [],
        "verified_flight_options": [],
        "strict_evidence": [],
        "fallback_evidence": [],
        "current_action": {
            "action": next_action,
            "reason": "continue after batch evidence rejection",
        },
        "termination_reason": None,
    }


def _invoke_react_recovery(
    graph,
    react_state: dict[str, object],
    *,
    human_verification_handler=None,
) -> dict[str, object]:
    if int(react_state.get("remaining_actions", 0) or 0) <= 0:
        warnings = list(react_state.get("warnings", []))
        if "insufficient_verified_flight_evidence" not in warnings:
            warnings.append("insufficient_verified_flight_evidence")
        return {
            **react_state,
            "judged_evidence": [],
            "verified_flight_options": [],
            "warnings": warnings,
            "termination_reason": "batch_rejected_action_budget_exhausted",
        }
    return invoke_react_flight_search(
        graph,
        _react_recovery_state(react_state),
        human_verification_handler=human_verification_handler,
    )


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


def _train_request_for_query_item(
    item: QueryPlanItem,
) -> TrainSearchRequest:
    return TrainSearchRequest(
        origin=item.origin,
        destination=item.destination,
        travel_date=item.travel_date,
        max_results=3,
    )


def _tool_failure_warning(
    mode: str,
    item: QueryPlanItem,
    error: ToolError | None,
) -> str:
    if error is None:
        return f"{mode}_query_failed:{item.query_id}:unknown_error"
    return (
        f"{mode}_query_failed:{item.query_id}:"
        f"{error.code.value}:retryable={str(error.retryable).lower()}:{error.message}"
    )


def _limit_flight_tool_state(
    state: dict[str, object],
    request: FlightSearchRequest,
) -> dict[str, object]:
    options = list(state.get("verified_flight_options", []))
    if request.direct_only:
        direct_options: list[FlightOption] = []
        for option in options:
            direct_evidence = [
                evidence
                for evidence in option.evidence
                if isinstance(evidence.metadata, dict)
                and evidence.metadata.get("is_direct") is True
            ]
            if not direct_evidence:
                continue
            lowest = min(direct_evidence, key=lambda evidence: evidence.price)
            direct_options.append(
                replace(
                    option,
                    price=lowest.price,
                    departure_time=lowest.departure_time,
                    arrival_time=lowest.arrival_time,
                    evidence=direct_evidence,
                )
            )
        options = direct_options
    return {
        **state,
        "verified_flight_options": options[: request.max_results],
    }


def _batch_judge_flight_searches(
    search_cache: dict[str, dict],
    evidence_judge,
    verifier: FlightEvidenceVerifier,
    fallback_react_search,
    *,
    human_verification_handler=None,
) -> dict[str, dict]:
    requests: list[FlightEvidenceJudgeRequest] = []
    request_keys: dict[str, str] = {}
    for query_index, (cache_key, react_state) in enumerate(search_cache.items()):
        evidence_by_url: dict[str, list] = {}
        for evidence in react_state.get("extracted_evidence", []):
            evidence_by_url.setdefault(evidence.url, []).append(evidence)
        result_by_url = {
            result.url: result
            for result in react_state.get("raw_results", [])
            if isinstance(result, SearchResult)
        }
        for url_index, (url, evidence) in enumerate(evidence_by_url.items()):
            request_id = f"q{query_index}:u{url_index}"
            request_keys[request_id] = cache_key
            requests.append(
                FlightEvidenceJudgeRequest(
                    request_id=request_id,
                    intent=_verification_intent_for_react_state(react_state),
                    search_result=result_by_url.get(
                        url,
                        SearchResult(
                            title="",
                            url=url,
                            snippet="",
                            source_name=evidence[0].source_name if evidence else "web",
                        ),
                    ),
                    evidence=evidence,
                )
            )

    try:
        judged_by_request = evidence_judge.judge_many(requests)
    except Exception:
        return {
            cache_key: _invoke_react_recovery(
                fallback_react_search,
                react_state,
                human_verification_handler=human_verification_handler,
            )
            for cache_key, react_state in search_cache.items()
        }

    judged_by_query: dict[str, list] = {}
    for request_id, evidence in judged_by_request.items():
        cache_key = request_keys.get(request_id)
        if cache_key is not None:
            judged_by_query.setdefault(cache_key, []).extend(evidence)

    updated: dict[str, dict] = {}
    for cache_key, react_state in search_cache.items():
        judged_evidence = judged_by_query.get(cache_key, [])
        verification_intent = _verification_intent_for_react_state(react_state)
        options = verifier.verify(judged_evidence, verification_intent)
        if react_state.get("termination_reason") == "time_preference_fallback":
            marker = f"time_preference_not_met:{react_state['intent'].time_preference}"
            options = [
                replace(option, warnings=[*option.warnings, marker])
                for option in options
            ]
        if react_state.get("extracted_evidence") and not options:
            updated[cache_key] = _invoke_react_recovery(
                fallback_react_search,
                react_state,
                human_verification_handler=human_verification_handler,
            )
            continue
        updated[cache_key] = {
            **react_state,
            "judged_evidence": judged_evidence,
            "verified_flight_options": options,
        }
    return updated


def _verification_intent_for_react_state(react_state: dict[str, object]) -> FlightSearchIntent:
    intent = react_state["intent"]
    if react_state.get("termination_reason") == "time_preference_fallback":
        return replace(intent, time_preference=None)
    return intent


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
                duration_minutes=_parse_duration_minutes(option.duration),
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
                duration_minutes=_flight_duration_minutes(option),
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
                if (
                    wait_minutes is not None
                    and wait_minutes < _minimum_transfer_minutes(strategy)
                ):
                    continue
                total_price = first_edge.price + second_edge.price
                total_duration_minutes = _two_leg_duration_minutes(
                    first_edge,
                    second_edge,
                    wait_minutes,
                )
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
                        total_duration_minutes=total_duration_minutes,
                        segment_count=(
                            _edge_segment_count(first_edge)
                            + _edge_segment_count(second_edge)
                        ),
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
    if isinstance(edge.departure_time, datetime) and isinstance(edge.arrival_time, datetime):
        return _format_datetime_range(
            edge.departure_time,
            edge.arrival_time,
            travel_date=edge.travel_date,
        )
    start = _format_time_value(edge.departure_time)
    end = _format_time_value(edge.arrival_time)
    if start and end:
        return f" {start}-{end}"
    return ""


def _format_datetime_range(
    departure: datetime | None,
    arrival: datetime | None,
    *,
    travel_date: date,
) -> str:
    if departure is None or arrival is None:
        return ""
    departure_offset = (departure.date() - travel_date).days
    arrival_offset = (arrival.date() - travel_date).days
    departure_text = departure.strftime("%H:%M") + _day_offset_suffix(departure_offset)
    arrival_text = arrival.strftime("%H:%M") + _day_offset_suffix(arrival_offset)
    return f" {departure_text}-{arrival_text}"


def _day_offset_suffix(day_offset: int) -> str:
    if day_offset == 0:
        return ""
    sign = "+" if day_offset > 0 else ""
    return f"({sign}{day_offset}d)"


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
            if wait_minutes is not None and wait_minutes < _minimum_transfer_minutes("train_flight"):
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
                    total_duration_minutes=_sum_known_minutes(
                        _parse_duration_minutes(train.duration),
                        wait_minutes,
                        _flight_duration_minutes(flight),
                    ),
                    segment_count=1 + _flight_segment_count(flight),
                )
            )
    return sorted(routes, key=_route_sort_key)[:12]


def _route_sort_key(route: CandidateRoute) -> tuple[int, float]:
    if route.total_price is None:
        return (1, float("inf"))
    return (0, route.total_price)


def _minimum_transfer_minutes(strategy: str) -> int:
    # Query-plan legs are independently booked products, so airport and
    # intermodal transfers need more protection than a through train journey.
    if strategy == "train_train":
        return 60
    return 120


def _parse_duration_minutes(value: str | None) -> int | None:
    if not value:
        return None
    try:
        hours, minutes = value.split(":", 1)
        return int(hours) * 60 + int(minutes)
    except (AttributeError, TypeError, ValueError):
        return None


def _flight_duration_minutes(option: FlightOption) -> int | None:
    if option.departure_time is None or option.arrival_time is None:
        return None
    return max(
        0,
        int((option.arrival_time - option.departure_time).total_seconds() // 60),
    )


def _flight_segment_count(option: FlightOption) -> int:
    for evidence in option.evidence:
        metadata = evidence.metadata or {}
        transfer_count = metadata.get("transfer_count")
        if isinstance(transfer_count, int) and transfer_count >= 0:
            return transfer_count + 1
        segments = metadata.get("segments")
        if isinstance(segments, list) and segments:
            return len(segments)
        flight_no = metadata.get("flight_no")
        if isinstance(flight_no, str) and flight_no:
            return len([part for part in flight_no.split("+") if part]) or 1
    return 1


def _edge_segment_count(edge: RouteEdge) -> int:
    if isinstance(edge.raw_option, FlightOption):
        return _flight_segment_count(edge.raw_option)
    return 1


def _two_leg_duration_minutes(
    first_edge: RouteEdge,
    second_edge: RouteEdge,
    wait_minutes: int | None,
) -> int | None:
    return _sum_known_minutes(
        first_edge.duration_minutes,
        wait_minutes,
        second_edge.duration_minutes,
    )


def _sum_known_minutes(*values: int | None) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


def _route_dominates(first: CandidateRoute, second: CandidateRoute) -> bool:
    if (
        first.total_price is None
        or second.total_price is None
        or first.total_duration_minutes is None
        or second.total_duration_minutes is None
        or first.segment_count is None
        or second.segment_count is None
    ):
        return False
    no_worse = (
        first.total_price <= second.total_price
        and first.total_duration_minutes <= second.total_duration_minutes
        and first.segment_count <= second.segment_count
    )
    strictly_better = (
        first.total_price < second.total_price
        or first.total_duration_minutes < second.total_duration_minutes
        or first.segment_count < second.segment_count
    )
    return no_worse and strictly_better


def _enforce_dominance_order(routes: list[CandidateRoute]) -> list[CandidateRoute]:
    """Keep LLM preferences unless they violate an objective dominance relation."""
    remaining = list(routes)
    ordered: list[CandidateRoute] = []
    while remaining:
        next_index = next(
            (
                index
                for index, route in enumerate(remaining)
                if not any(
                    _route_dominates(other, route)
                    for other_index, other in enumerate(remaining)
                    if other_index != index
                )
            ),
            0,
        )
        ordered.append(remaining.pop(next_index))
    return ordered


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
    if isinstance(first_arrival, datetime) and isinstance(second_departure, datetime):
        return int((second_departure - first_arrival).total_seconds() // 60)
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


def _merge_hubs_with_budget(
    explicit_hubs: list[CandidateHub],
    llm_hubs: list[CandidateHub],
    index_hubs: list[CandidateHub],
    *,
    budget: QueryBudget,
) -> list[CandidateHub]:
    selected: list[CandidateHub] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    grouped_count: dict[str, int] = {}
    for hub in [*explicit_hubs, *llm_hubs, *index_hubs]:
        key = (hub.hub_id, tuple(hub.airport_codes), tuple(hub.train_places))
        if key in seen:
            continue
        usable_strategies = [
            strategy for strategy in hub.strategies
            if grouped_count.get(strategy, 0) < budget.max_hubs_per_strategy
        ]
        if not usable_strategies:
            continue
        seen.add(key)
        selected.append(
            CandidateHub(
                hub_id=hub.hub_id,
                city=hub.city,
                airport_codes=hub.airport_codes,
                train_places=hub.train_places,
                strategies=usable_strategies,
                priority=hub.priority,
                reason=hub.reason,
                flight_potential_score=hub.flight_potential_score,
                flight_tier=hub.flight_tier,
            )
        )
        for strategy in usable_strategies:
            grouped_count[strategy] = grouped_count.get(strategy, 0) + 1
    return selected


def _backfill_supplemental_hubs(
    llm_hubs: list[CandidateHub],
    supplemental_hubs: list[CandidateHub],
    *,
    excluded_hub_ids: set[str],
    limit: int,
) -> list[CandidateHub]:
    selected: list[CandidateHub] = []
    seen = set(excluded_hub_ids)
    for hub in [*llm_hubs, *supplemental_hubs]:
        if hub.hub_id in seen:
            continue
        seen.add(hub.hub_id)
        selected.append(hub)
        if len(selected) >= limit:
            break
    return selected


def _validate_candidate_hubs_against_endpoints(
    intent: FlightSearchIntent,
    hubs: list[CandidateHub],
    decisions: list["HubEndpointDecision"],
) -> tuple[list[CandidateHub], list[str]]:
    origin_identity = _endpoint_identity(intent.origin, label="origin")
    destination_identity = _endpoint_identity(intent.destination, label="destination")
    by_hub_id = {decision.hub_id: decision for decision in decisions}
    validated: list[CandidateHub] = []
    warnings: list[str] = []

    for hub in hubs:
        decision = by_hub_id.get(hub.hub_id)
        corrected_hub = _apply_hub_endpoint_correction(hub, decision)
        origin_match = _hub_matches_endpoint(corrected_hub, origin_identity)
        destination_match = _hub_matches_endpoint(corrected_hub, destination_identity)
        if origin_match or destination_match:
            warnings.append(f"hub_is_origin_or_destination:{corrected_hub.city or corrected_hub.hub_id}")
            continue
        validated.append(corrected_hub)

    return validated, warnings


def _apply_hub_endpoint_correction(
    hub: CandidateHub,
    decision: "HubEndpointDecision | None",
) -> CandidateHub:
    if decision is None:
        return hub

    airport_codes = _validated_airport_codes(decision.corrected_airport_codes) or hub.airport_codes
    train_places = _validated_train_places(decision.corrected_train_places) or hub.train_places
    flight_potential_score, flight_tier = _flight_potential_for_airport_codes(airport_codes)
    city = (decision.corrected_city or "").strip() or hub.city
    if city == hub.city and airport_codes == hub.airport_codes and train_places == hub.train_places:
        return hub
    return CandidateHub(
        hub_id=hub.hub_id,
        city=city,
        airport_codes=airport_codes,
        train_places=train_places,
        strategies=hub.strategies,
        priority=hub.priority,
        reason=hub.reason,
        flight_potential_score=flight_potential_score or hub.flight_potential_score,
        flight_tier=flight_tier or hub.flight_tier,
    )


def _validated_airport_codes(values: list[str]) -> list[str]:
    airport_index = get_airport_index()
    result: list[str] = []
    for value in values:
        airport = airport_index.resolve(value)
        if airport is None:
            continue
        if airport.iata not in result:
            result.append(airport.iata)
    return result


def _validated_train_places(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        place = normalise_train_query_place(value)
        if place is None:
            continue
        if place not in result:
            result.append(place)
    return result


def _flight_potential_for_airport_codes(codes: list[str]) -> tuple[float | None, str | None]:
    airport_index = get_airport_index()
    score: float | None = None
    tier: str | None = None
    tier_rank = {"T1": 4, "T2": 3, "T3": 2, "T4": 1}
    for code in codes:
        airport = airport_index.resolve(code)
        if airport is None:
            continue
        if airport.flight_potential_score is not None:
            score = airport.flight_potential_score if score is None else max(score, airport.flight_potential_score)
        if airport.flight_tier and tier_rank.get(airport.flight_tier, 0) > tier_rank.get(tier or "", 0):
            tier = airport.flight_tier
    return score, tier


def _endpoint_identity(value: str, *, label: str) -> EndpointIdentity:
    airport_index = get_airport_index()
    station_index = get_station_index()
    cities: set[str] = set()
    airport_codes: set[str] = set()
    train_places: set[str] = set()

    airport = airport_index.resolve(value)
    if airport is not None:
        airport_codes.add(airport.iata)
        if airport.city:
            cities.add(airport.city)
        station_city = station_city_for_airport(airport)
        if station_city:
            cities.add(station_city)
            primary_station = station_index.primary_station_for_city(station_city)
            train_places.add(primary_station or station_city)

    train_place = normalise_train_query_place(value)
    if train_place is not None:
        train_places.add(train_place)
        station = station_index.by_name.get(train_place)
        if station is not None and station.city_name:
            cities.add(station.city_name)
            primary_station = station_index.primary_station_for_city(station.city_name)
            train_places.add(primary_station or station.city_name)
        elif train_place in station_index.city_names:
            cities.add(train_place)
            primary_station = station_index.primary_station_for_city(train_place)
            train_places.add(primary_station or train_place)

    for airport_record in airport_index.airports:
        airport_city = station_city_for_airport(airport_record)
        if airport_city and airport_city in cities:
            airport_codes.add(airport_record.iata)
        elif airport_record.city and airport_record.city in cities:
            airport_codes.add(airport_record.iata)

    for city in list(cities):
        primary_station = station_index.primary_station_for_city(city)
        if primary_station:
            train_places.add(primary_station)

    return EndpointIdentity(
        label=label,
        cities=frozenset(cities),
        airport_codes=frozenset(airport_codes),
        train_places=frozenset(train_places),
    )


def _hub_matches_endpoint(hub: CandidateHub, endpoint: EndpointIdentity) -> bool:
    if hub.city in endpoint.cities:
        return True
    if set(hub.airport_codes) & set(endpoint.airport_codes):
        return True
    if set(hub.train_places) & set(endpoint.train_places):
        return True
    for airport_code in hub.airport_codes:
        airport = get_airport_index().resolve(airport_code)
        if airport is None:
            continue
        airport_city = station_city_for_airport(airport)
        if airport_city and airport_city in endpoint.cities:
            return True
        if airport.city and airport.city in endpoint.cities:
            return True
    for train_place in hub.train_places:
        station = get_station_index().by_name.get(train_place)
        if station is not None and station.city_name in endpoint.cities:
            return True
        if train_place in endpoint.cities:
            return True
    return False


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


class HubEndpointDecision(BaseModel):
    hub_id: str
    is_origin_endpoint: bool = False
    is_destination_endpoint: bool = False
    corrected_city: str | None = None
    corrected_airport_codes: list[str] = Field(default_factory=list)
    corrected_train_places: list[str] = Field(default_factory=list)
    reason: str | None = None


class HubEndpointDecisionBatch(BaseModel):
    decisions: list[HubEndpointDecision] = Field(default_factory=list)


class HubSuggestion(BaseModel):
    raw_text: str | None = None
    official_airport_name: str | None = None
    city: str | None = None
    country: str | None = None
    iata_if_explicit: str | None = None
    station_name: str | None = Field(
        default=None,
        description="12306 Chinese station name for railway hubs.",
    )
    station_pinyin: str | None = None
    strategies: list[str] = Field(default_factory=list)
    reason: str | None = None


class HubSuggestionBatch(BaseModel):
    suggestions: list[HubSuggestion] = Field(default_factory=list)


class HubPlanningBatch(BaseModel):
    suggestions: list[HubSuggestion] = Field(default_factory=list)
    decisions: list[HubEndpointDecision] = Field(default_factory=list)


class LlmHubPlanner:
    def __init__(self, llm, *, max_suggestions: int = 5) -> None:
        self.llm = llm
        self.max_suggestions = max_suggestions

    def plan(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        strategy_selection: StrategySelection,
        index_hubs: list[CandidateHub],
        supplemental_hubs: list[CandidateHub],
        explicit_hubs: list[CandidateHub],
    ) -> HubPlanningBatch:
        structured = self.llm.with_structured_output(HubPlanningBatch)
        response = structured.invoke(
            [
                (
                    "system",
                    _hub_planning_prompt(intent, region_info, strategy_selection, self.max_suggestions),
                ),
                (
                    "human",
                    json.dumps(
                        {
                            "user_input": user_input,
                            "index_hubs": _hub_payload(index_hubs),
                            "supplemental_candidates": _hub_payload(supplemental_hubs),
                            "explicit_hubs": _hub_payload(explicit_hubs),
                            "candidate_hubs": _hub_payload([*explicit_hubs, *index_hubs]),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        batch = response if isinstance(response, HubPlanningBatch) else HubPlanningBatch.model_validate(response)
        return HubPlanningBatch(
            suggestions=batch.suggestions[: self.max_suggestions],
            decisions=batch.decisions,
        )


class LlmHubProposer:
    def __init__(self, llm, *, max_suggestions: int = 5) -> None:
        self.llm = llm
        self.max_suggestions = max_suggestions

    def propose(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        strategy_selection: StrategySelection,
        index_hubs: list[CandidateHub],
    ) -> list[HubSuggestion]:
        structured = self.llm.with_structured_output(HubSuggestionBatch)
        response = structured.invoke(
            [
                ("system", _hub_suggestion_prompt(intent, region_info, strategy_selection, self.max_suggestions)),
                (
                    "human",
                    json.dumps(
                        {
                            "user_input": user_input,
                            "index_hubs": _hub_payload(index_hubs),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        batch = response if isinstance(response, HubSuggestionBatch) else HubSuggestionBatch.model_validate(response)
        return batch.suggestions[: self.max_suggestions]


class LlmHubEndpointValidator:
    def __init__(self, llm) -> None:
        self.llm = llm

    def validate(
        self,
        *,
        user_input: str,
        intent: FlightSearchIntent,
        region_info: RegionInfo,
        candidate_hubs: list[CandidateHub],
    ) -> list[HubEndpointDecision]:
        structured = self.llm.with_structured_output(HubEndpointDecisionBatch)
        response = structured.invoke(
            [
                ("system", _hub_endpoint_validation_prompt(intent, region_info)),
                (
                    "human",
                    json.dumps(
                        {
                            "user_input": user_input,
                            "origin": intent.origin,
                            "destination": intent.destination,
                            "candidate_hubs": _hub_payload(candidate_hubs),
                        },
                        ensure_ascii=False,
                    ),
                ),
            ]
        )
        batch = response if isinstance(response, HubEndpointDecisionBatch) else HubEndpointDecisionBatch.model_validate(response)
        return batch.decisions


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
        return _enforce_dominance_order(
            ordered + sorted(remaining, key=_route_sort_key)
        )


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
                "total_duration_minutes": route.total_duration_minutes,
                "segment_count": route.segment_count,
            }
        )
    return {"routes": serialized}


def _hub_payload(hubs: list[CandidateHub]) -> list[dict[str, object]]:
    return [
        {
            "hub_id": hub.hub_id,
            "city": hub.city,
            "airport_codes": hub.airport_codes,
            "train_places": hub.train_places,
            "strategies": hub.strategies,
            "priority": hub.priority,
            "reason": hub.reason,
        }
        for hub in hubs
    ]


def _hub_planning_prompt(
    intent: FlightSearchIntent,
    region_info: RegionInfo,
    strategy_selection: StrategySelection,
    max_suggestions: int,
) -> str:
    return f"""
You plan and validate transfer hubs for a multimodal travel search agent in one pass.

Trip:
- origin: {intent.origin}
- destination: {intent.destination}
- travel_date: {intent.travel_date.isoformat()}
- route_type: {region_info.route_type}
- enabled_strategies: {", ".join(strategy_selection.enabled)}

Tasks:
1. Return at most {max_suggestions} additional hub suggestions beyond index_hubs and explicit_hubs.
2. Return one endpoint decision for every item in candidate_hubs when possible.

Rules for suggestions:
- Return exactly {max_suggestions} suggestions when supplemental_candidates contains at least that many valid hubs.
- Prefer selecting diverse, practical hubs from supplemental_candidates by copying one airport code into raw_text.
- You may propose another high-value hub only when it is clearly better than the supplied candidates.
- Prefer hubs that may improve price or practical routing.
- Never suggest the origin or destination city as a transfer hub.
- For airports, provide official_airport_name, city, and country when possible.
- For rail hubs, provide station_name as the 12306 Chinese station name.
- Do not invent prices, schedules, or availability.

Rules for endpoint decisions:
- Same-city airports and railway stations count as the same endpoint.
- Mark whether each candidate is the origin or destination endpoint.
- Correct inconsistent city, airport code, or station fields when needed.
- Decisions and suggestions will be verified against local airport and station indexes.
""".strip()


def _hub_suggestion_prompt(
    intent: FlightSearchIntent,
    region_info: RegionInfo,
    strategy_selection: StrategySelection,
    max_suggestions: int,
) -> str:
    return f"""
You propose additional transfer hubs for a multimodal travel search agent.

Trip:
- origin: {intent.origin}
- destination: {intent.destination}
- travel_date: {intent.travel_date.isoformat()}
- route_type: {region_info.route_type}
- enabled_strategies: {", ".join(strategy_selection.enabled)}

Rules:
- Return at most {max_suggestions} suggestions.
- Prefer hubs that may improve price or practical routing beyond the provided index_hubs.
- For airports, provide official_airport_name, city, and country when possible.
- For rail hubs, provide station_name as the 12306 Chinese station name.
- Do not invent prices or final routes.
- Suggestions will be validated against local airport and station indexes; unverifiable hubs will be discarded.
""".strip()


def _hub_endpoint_validation_prompt(
    intent: FlightSearchIntent,
    region_info: RegionInfo,
) -> str:
    return f"""
You validate transfer hubs for a multimodal travel search agent.

Trip:
- origin: {intent.origin}
- destination: {intent.destination}
- travel_date: {intent.travel_date.isoformat()}
- route_type: {region_info.route_type}

Task:
- For each candidate hub, decide whether it is actually the origin endpoint or destination endpoint.
- Same-city airports count as the same endpoint. For example CTU, TFU, HZU, and Chengdu/成都 are all Chengdu-side endpoints.
- City airport groups count as the same endpoint. For example BJS, PEK, PKX, and Beijing/北京 are all Beijing-side endpoints.
- Same-city railway stations count as the same endpoint.
- If a hub has incorrect or incomplete city/airport/station fields, provide corrected_city, corrected_airport_codes, and corrected_train_places.
- Do not invent final routes, prices, or availability.
- The system will verify your decision against local airport and 12306 station indexes before using it.

Return one decision per candidate hub_id when possible.
""".strip()


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
2) Compare door-to-door total_duration_minutes, including transfer waits.
3) Route feasibility matters: independently booked transfers need sufficient time.
4) Fewer physical segments is better unless savings are significant.
5) Respect budget threshold and time preference when provided.

Never rank a route below another route that is simultaneously no more expensive,
no slower, and no more complex (segment_count), with at least one strict advantage.

Return `ranked` with best routes first.
Every chosen route_id must exist in provided data.
Do not invent routes.
""".strip()
