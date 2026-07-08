from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal


@dataclass(frozen=True)
class FlightSearchRequest:
    origin: str
    destination: str
    depart_date: date
    return_date: date | None
    currency: str = "CNY"


@dataclass(frozen=True)
class FlightQuote:
    origin: str
    destination: str
    depart_date: date
    return_date: date | None
    price: float
    currency: str
    provider: str
    deep_link: str | None
    fetched_at: datetime


@dataclass(frozen=True)
class Monitor:
    id: str
    origin: str
    destination: str
    depart_date: date
    return_date: date | None
    threshold_price: float
    currency: str
    interval_seconds: int
    enabled: bool
    created_at: datetime
    updated_at: datetime
    last_checked_at: datetime | None = None
    last_price: float | None = None
    last_alert_at: datetime | None = None

    def to_search_request(self) -> FlightSearchRequest:
        return FlightSearchRequest(
            origin=self.origin,
            destination=self.destination,
            depart_date=self.depart_date,
            return_date=self.return_date,
            currency=self.currency,
        )


@dataclass(frozen=True)
class AlertDecision:
    should_notify: bool
    reason: str


@dataclass(frozen=True)
class FlightSearchIntent:
    origin: str
    destination: str
    travel_date: date
    time_preference: str | None = None
    budget_threshold: float | None = None
    currency: str = "CNY"
    max_segments: int = 3


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source_name: str


@dataclass(frozen=True)
class FlightEvidence:
    source_name: str
    url: str
    price: float
    currency: str
    departure_time: datetime | None
    arrival_time: datetime | None
    captured_at: datetime
    origin: str | None = None
    destination: str | None = None
    travel_date: date | None = None


@dataclass(frozen=True)
class FlightOption:
    origin: str
    destination: str
    travel_date: date
    price: float
    currency: str
    departure_time: datetime | None
    arrival_time: datetime | None
    evidence: list[FlightEvidence]
    reliability: Literal["verified", "price_volatility_review"]
    warnings: list[str]

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)
