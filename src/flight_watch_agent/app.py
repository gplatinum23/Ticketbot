from __future__ import annotations

from .config import get_config, load_env_file
from .ctrip import CtripRouteSearchTool, CtripSeleniumWirePageExtractor
from .flight_react import (
    CompositePageExtractor,
    CompositeWebSearchTool,
    LlmFlightEvidenceJudge,
    build_react_flight_search_graph,
)
from .llm import build_default_llm
from .progress import ProgressReporter
from .request_graph import build_travel_plan_request_graph
from .trains import Mcp12306TrainProvider
from .travel_plan_graph import build_travel_plan_graph
from .travel_plan_graph import LlmHubPlanner
from .travel_plan_graph import LlmRoutePlanner


def build_default_request_agent(
    llm_model: str | None = None,
    *,
    include_train: bool = True,
    progress_reporter: ProgressReporter | None = None,
):
    load_env_file()
    llm = build_default_llm(llm_model)
    fast_llm = _build_fast_llm(llm)
    graph = build_travel_plan_request_graph(
        llm=fast_llm,
        travel_plan_graph=build_default_travel_plan_agent(
            llm=llm,
            fast_llm=fast_llm,
            include_train=include_train,
            progress_reporter=progress_reporter,
        ),
        progress_reporter=progress_reporter,
    )
    return graph


def build_default_travel_plan_agent(
    llm=None,
    *,
    fast_llm=None,
    include_train: bool = True,
    progress_reporter: ProgressReporter | None = None,
):
    load_env_file()
    llm = llm or build_default_llm()
    fast_llm = fast_llm or _build_fast_llm(llm)
    graph = build_travel_plan_graph(
        web_search=build_default_flight_web_search_tool(),
        page_extractor=build_default_flight_page_extractor(),
        train_provider=Mcp12306TrainProvider() if include_train else None,
        evidence_judge=LlmFlightEvidenceJudge(fast_llm),
        hub_planner=LlmHubPlanner(fast_llm),
        route_planner=LlmRoutePlanner(_build_route_llm(fast_llm)),
        progress_reporter=progress_reporter,
    )
    return graph


def build_default_flight_search_agent(
    *,
    llm=None,
    use_llm_judge: bool = True,
    max_iterations: int = 3,
    progress_reporter: ProgressReporter | None = None,
):
    load_env_file()
    evidence_judge = None
    if use_llm_judge:
        llm = llm or _build_fast_llm(build_default_llm())
        evidence_judge = LlmFlightEvidenceJudge(llm)

    return build_react_flight_search_graph(
        web_search=build_default_flight_web_search_tool(),
        page_extractor=build_default_flight_page_extractor(),
        evidence_judge=evidence_judge,
        max_iterations=max_iterations,
        progress_reporter=progress_reporter,
    )


def build_default_flight_web_search_tool():
    return CompositeWebSearchTool([CtripRouteSearchTool()])


def build_default_flight_page_extractor():
    accounts = _config_list("FLIGHT_WATCH_CTRIP_ACCOUNTS") or _config_list("FLIGHT_WATCH_CTRIP_USERNAME")
    passwords = _config_list("FLIGHT_WATCH_CTRIP_PASSWORDS") or _config_list("FLIGHT_WATCH_CTRIP_PASSWORD")
    return CompositePageExtractor(
        [
            CtripSeleniumWirePageExtractor(
                browser=get_config("FLIGHT_WATCH_CTRIP_BROWSER", "edge") or "edge",
                headless=_config_bool("FLIGHT_WATCH_CTRIP_HEADLESS", default=True),
                timeout_seconds=int(get_config("FLIGHT_WATCH_CTRIP_TIMEOUT", "30") or "30"),
                direct_only=_config_bool("FLIGHT_WATCH_CTRIP_DIRECT_ONLY", default=False),
                login_allowed=_config_bool("FLIGHT_WATCH_CTRIP_LOGIN_ALLOWED", default=bool(accounts and passwords)),
                accounts=accounts,
                passwords=passwords,
                cookies_file=get_config("FLIGHT_WATCH_CTRIP_COOKIES_FILE", "data/ctrip_cookies.json")
                or "data/ctrip_cookies.json",
                login_wait_seconds=int(get_config("FLIGHT_WATCH_CTRIP_LOGIN_WAIT_SECONDS", "300") or "300"),
                manual_verification_wait_seconds=int(
                    get_config("FLIGHT_WATCH_CTRIP_MANUAL_VERIFICATION_WAIT_SECONDS", "0") or "0"
                ),
                reuse_browser_session=_config_bool("FLIGHT_WATCH_CTRIP_REUSE_BROWSER", default=True),
            )
        ]
    )


def _config_bool(name: str, *, default: bool) -> bool:
    value = get_config(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _config_list(name: str) -> list[str]:
    value = get_config(name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_fast_llm(default_llm):
    model = get_config("FLIGHT_WATCH_FAST_LLM_MODEL")
    if not model:
        return default_llm
    return build_default_llm(model)


def _build_route_llm(fast_llm):
    model = get_config("FLIGHT_WATCH_ROUTE_LLM_MODEL")
    if not model:
        return fast_llm
    return build_default_llm(model)
