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
from .request_graph import build_travel_plan_request_graph
from .trains import Mcp12306TrainProvider
from .travel_plan_graph import build_travel_plan_graph
from .travel_plan_graph import LlmRoutePlanner


def build_default_request_agent(llm_model: str | None = None, *, include_train: bool = True):
    load_env_file()
    llm = build_default_llm(llm_model)
    graph = build_travel_plan_request_graph(
        llm=llm,
        travel_plan_graph=build_default_travel_plan_agent(llm=llm, include_train=include_train),
    )
    return graph


def build_default_travel_plan_agent(llm=None, *, include_train: bool = True):
    load_env_file()
    llm = llm or build_default_llm()
    graph = build_travel_plan_graph(
        web_search=build_default_flight_web_search_tool(),
        page_extractor=build_default_flight_page_extractor(),
        train_provider=Mcp12306TrainProvider() if include_train else None,
        evidence_judge=LlmFlightEvidenceJudge(llm),
        route_planner=LlmRoutePlanner(llm),
    )
    return graph


def build_default_flight_search_agent(
    *,
    llm=None,
    use_llm_judge: bool = True,
    max_iterations: int = 3,
):
    load_env_file()
    evidence_judge = None
    if use_llm_judge:
        llm = llm or build_default_llm()
        evidence_judge = LlmFlightEvidenceJudge(llm)

    return build_react_flight_search_graph(
        web_search=build_default_flight_web_search_tool(),
        page_extractor=build_default_flight_page_extractor(),
        evidence_judge=evidence_judge,
        max_iterations=max_iterations,
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
