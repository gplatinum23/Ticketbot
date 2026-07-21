from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Literal


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
class TrainOption:
    train_code: str
    from_station: str
    from_station_code: str | None
    to_station: str
    to_station_code: str | None
    travel_date: date
    start_time: str
    arrive_time: str
    duration: str
    seats: dict[str, str]
    prices: dict[str, float]
    train_class_name: str | None = None

    @property
    def lowest_price(self) -> float | None:
        if not self.prices:
            return None
        return min(self.prices.values())


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    source_name: str


@dataclass(frozen=True)
class FlightPageAttemptResult:
    status: Literal[
        "success",
        "no_payload",
        "no_evidence",
        "parse_failed",
        "captcha_required",
        "login_required",
        "time_preference_mismatch",
        "tool_error",
    ]
    evidence: list["FlightEvidence"]
    entrypoint: str
    source_url: str | None = None
    warning: str | None = None


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
    metadata: dict[str, object] | None = None


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
