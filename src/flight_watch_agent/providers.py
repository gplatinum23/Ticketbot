from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Protocol

from .models import FlightQuote, FlightSearchRequest


class FlightPriceProvider(Protocol):
    name: str

    def get_lowest_price(self, request: FlightSearchRequest) -> FlightQuote:
        """Return the currently lowest known price for a route."""


class MockFlightPriceProvider:
    """Deterministic provider for local development and tests."""

    name = "mock"

    def get_lowest_price(self, request: FlightSearchRequest) -> FlightQuote:
        now = datetime.now(timezone.utc)
        hour_bucket = int(now.timestamp() // 3600)
        key = (
            f"{request.origin}:{request.destination}:"
            f"{request.depart_date.isoformat()}:{hour_bucket}"
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        base = 900 + int(digest[:6], 16) % 2400
        wave = 120 * math.sin(hour_bucket / 3)
        price = round(base + wave, 2)
        return FlightQuote(
            origin=request.origin,
            destination=request.destination,
            depart_date=request.depart_date,
            return_date=request.return_date,
            price=max(price, 1),
            currency=request.currency,
            provider=self.name,
            deep_link=None,
            fetched_at=now,
        )


class NotConfiguredFlightPriceProvider:
    name = "not_configured"

    def get_lowest_price(self, request: FlightSearchRequest) -> FlightQuote:
        raise RuntimeError(
            "No real flight price provider is configured. "
            "Implement FlightPriceProvider in providers.py and wire it in app.py."
        )
