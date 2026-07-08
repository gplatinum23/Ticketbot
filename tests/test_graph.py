from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from flight_watch_agent.graph import build_flight_watch_graph
from flight_watch_agent.models import FlightQuote
from flight_watch_agent.notifiers import build_notification
from flight_watch_agent.storage import MonitorRepository


class FakeProvider:
    name = "fake"

    def __init__(self, price: float) -> None:
        self.price = price

    def get_lowest_price(self, request):
        return FlightQuote(
            origin=request.origin,
            destination=request.destination,
            depart_date=request.depart_date,
            return_date=request.return_date,
            price=self.price,
            currency=request.currency,
            provider=self.name,
            deep_link="https://example.test/flight",
            fetched_at=datetime.now(timezone.utc),
        )


class SpyNotifier:
    name = "spy"

    def __init__(self) -> None:
        self.sent = []

    def send(self, monitor, quote):
        notification = build_notification(monitor, quote)
        self.sent.append(notification)
        return notification


def test_graph_sends_notification_when_price_is_below_threshold(tmp_path):
    repository = MonitorRepository(tmp_path / "test.sqlite3")
    monitor = repository.add_monitor(
        origin="SHA",
        destination="NRT",
        depart_date=date(2026, 9, 20),
        return_date=None,
        threshold_price=1800,
        currency="CNY",
        interval_seconds=3600,
    )
    notifier = SpyNotifier()
    graph = build_flight_watch_graph(
        provider=FakeProvider(price=1200),
        notifier=notifier,
        repository=repository,
    )

    state = graph.invoke({"monitor": monitor})

    assert state["decision"].should_notify is True
    assert len(notifier.sent) == 1


def test_graph_skips_notification_when_price_is_above_threshold(tmp_path):
    repository = MonitorRepository(tmp_path / "test.sqlite3")
    monitor = repository.add_monitor(
        origin="SHA",
        destination="NRT",
        depart_date=date(2026, 9, 20),
        return_date=None,
        threshold_price=1800,
        currency="CNY",
        interval_seconds=3600,
    )
    notifier = SpyNotifier()
    graph = build_flight_watch_graph(
        provider=FakeProvider(price=2200),
        notifier=notifier,
        repository=repository,
    )

    state = graph.invoke({"monitor": monitor})

    assert state["decision"].should_notify is False
    assert notifier.sent == []


def test_repository_rejects_invalid_monitor_input(tmp_path):
    repository = MonitorRepository(tmp_path / "test.sqlite3")

    with pytest.raises(ValueError):
        repository.add_monitor(
            origin="SHA",
            destination="SHA",
            depart_date=date(2026, 9, 20),
            return_date=None,
            threshold_price=1800,
            currency="CNY",
            interval_seconds=3600,
        )
