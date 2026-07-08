from __future__ import annotations

import time
from datetime import datetime, timezone

from .graph import FlightWatchState
from .models import Monitor
from .storage import MonitorRepository


def run_once(graph, repository: MonitorRepository) -> list[FlightWatchState]:
    results: list[FlightWatchState] = []
    for monitor in repository.list_monitors(enabled_only=True):
        results.append(run_monitor_once(graph, monitor))
    return results


def run_monitor_once(graph, monitor: Monitor) -> FlightWatchState:
    return graph.invoke(
        {"monitor": monitor},
        config={"configurable": {"thread_id": monitor.id}},
    )


def watch_forever(graph, repository: MonitorRepository) -> None:
    while True:
        due_monitors = [m for m in repository.list_monitors(enabled_only=True) if _is_due(m)]
        for monitor in due_monitors:
            run_monitor_once(graph, monitor)
        sleep_seconds = _seconds_until_next_check(repository)
        time.sleep(max(sleep_seconds, 1))


def _is_due(monitor: Monitor) -> bool:
    if monitor.last_checked_at is None:
        return True
    elapsed = datetime.now(timezone.utc) - monitor.last_checked_at
    return elapsed.total_seconds() >= monitor.interval_seconds


def _seconds_until_next_check(repository: MonitorRepository) -> int:
    monitors = repository.list_monitors(enabled_only=True)
    if not monitors:
        return 60

    now = datetime.now(timezone.utc)
    waits: list[int] = []
    for monitor in monitors:
        if monitor.last_checked_at is None:
            return 1
        elapsed = (now - monitor.last_checked_at).total_seconds()
        waits.append(int(monitor.interval_seconds - elapsed))
    return max(min(waits), 1)
