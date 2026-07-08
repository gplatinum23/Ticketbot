from __future__ import annotations

import os
from pathlib import Path

from .config import get_config, load_env_file
from .graph import build_flight_watch_graph
from .flight_react import DuckDuckGoHtmlSearchTool, RegexPageExtractor
from .llm import build_default_llm
from .notifiers import ConsoleNotifier, MultiNotifier, WebhookNotifier
from .providers import MockFlightPriceProvider
from .request_graph import build_monitor_request_graph
from .storage import MonitorRepository
from .travel_plan_graph import build_travel_plan_graph


def default_db_path() -> Path:
    return Path(get_config("FLIGHT_WATCH_DB", "data/flight_watch.sqlite3") or "data/flight_watch.sqlite3")


def build_default_agent(db_path: str | Path | None = None):
    load_env_file()
    repository = MonitorRepository(db_path or default_db_path())
    provider = MockFlightPriceProvider()

    notifiers = [ConsoleNotifier()]
    webhook_url = get_config("FLIGHT_WATCH_WEBHOOK_URL")
    if webhook_url:
        notifiers.append(WebhookNotifier(webhook_url))

    notifier = MultiNotifier(notifiers)
    graph = build_flight_watch_graph(
        provider=provider,
        notifier=notifier,
        repository=repository,
    )
    return graph, repository


def build_default_request_agent(db_path: str | Path | None = None, llm_model: str | None = None):
    repository = MonitorRepository(db_path or default_db_path())
    llm = build_default_llm(llm_model)
    graph = build_monitor_request_graph(llm=llm, repository=repository)
    return graph, repository


def build_default_travel_plan_agent():
    graph = build_travel_plan_graph(
        web_search=DuckDuckGoHtmlSearchTool(),
        page_extractor=RegexPageExtractor(),
    )
    return graph
