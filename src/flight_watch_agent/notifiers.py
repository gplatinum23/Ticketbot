from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Protocol

from .models import FlightQuote, Monitor


@dataclass(frozen=True)
class Notification:
    monitor_id: str
    title: str
    message: str
    quote: FlightQuote
    created_at: datetime


class Notifier(Protocol):
    name: str

    def send(self, monitor: Monitor, quote: FlightQuote) -> Notification:
        """Send a price alert."""


class ConsoleNotifier:
    name = "console"

    def send(self, monitor: Monitor, quote: FlightQuote) -> Notification:
        notification = build_notification(monitor, quote)
        print(f"[flight-watch] {notification.title}\n{notification.message}")
        return notification


class WebhookNotifier:
    name = "webhook"

    def __init__(self, url: str, timeout_seconds: int = 10) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def send(self, monitor: Monitor, quote: FlightQuote) -> Notification:
        notification = build_notification(monitor, quote)
        payload = json.dumps(_json_safe(asdict(notification))).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds):
            pass
        return notification


class MultiNotifier:
    name = "multi"

    def __init__(self, notifiers: list[Notifier]) -> None:
        self.notifiers = notifiers

    def send(self, monitor: Monitor, quote: FlightQuote) -> Notification:
        last_notification: Notification | None = None
        for notifier in self.notifiers:
            last_notification = notifier.send(monitor, quote)
        if last_notification is None:
            raise RuntimeError("No notifier configured.")
        return last_notification


def build_notification(monitor: Monitor, quote: FlightQuote) -> Notification:
    title = f"Flight price alert: {monitor.origin}->{monitor.destination}"
    message = (
        f"{monitor.origin}->{monitor.destination} on "
        f"{monitor.depart_date.isoformat()} is {quote.price:.2f} {quote.currency}, "
        f"threshold {monitor.threshold_price:.2f} {monitor.currency}."
    )
    return Notification(
        monitor_id=monitor.id,
        title=title,
        message=message,
        quote=quote,
        created_at=quote.fetched_at,
    )


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
