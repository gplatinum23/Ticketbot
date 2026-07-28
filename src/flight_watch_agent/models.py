from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from .places import PlaceRef, resolve_actual_airport, resolve_air_query_place


@dataclass(frozen=True)
class FlightSearchIntent:
    origin: str
    destination: str
    travel_date: date
    time_preference: str | None = None
    budget_threshold: float | None = None
    currency: str = "CNY"
    max_segments: int = 3

    @property
    def origin_raw(self) -> str:
        return self.origin

    @property
    def destination_raw(self) -> str:
        return self.destination

    @property
    def origin_place(self) -> PlaceRef:
        return resolve_air_query_place(self.origin)

    @property
    def destination_place(self) -> PlaceRef:
        return resolve_air_query_place(self.destination)

    @property
    def travel_date_semantics(self) -> str:
        return "first_segment_departure_local_date"


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

    @property
    def departure_at(self) -> datetime:
        return _china_rail_datetime(self.travel_date, self.start_time)

    @property
    def arrival_at(self) -> datetime:
        duration_minutes = _duration_minutes(self.duration)
        if duration_minutes is not None:
            return self.departure_at + timedelta(minutes=duration_minutes)
        arrival = _china_rail_datetime(self.travel_date, self.arrive_time)
        if arrival < self.departure_at:
            arrival += timedelta(days=1)
        return arrival


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

    def __post_init__(self) -> None:
        _require_aware_datetime(self.departure_time, "departure_time")
        _require_aware_datetime(self.arrival_time, "arrival_time")
        _require_aware_datetime(self.captured_at, "captured_at")


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
    requested_origin: PlaceRef | None = None
    requested_destination: PlaceRef | None = None
    actual_origin: PlaceRef | None = None
    actual_destination: PlaceRef | None = None

    def __post_init__(self) -> None:
        _require_aware_datetime(self.departure_time, "departure_time")
        _require_aware_datetime(self.arrival_time, "arrival_time")
        if self.actual_origin is None:
            object.__setattr__(self, "actual_origin", resolve_actual_airport(self.origin))
        if self.actual_destination is None:
            object.__setattr__(
                self,
                "actual_destination",
                resolve_actual_airport(self.destination),
            )

    @property
    def evidence_count(self) -> int:
        return len(self.evidence)


_CHINA_RAIL_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def _china_rail_datetime(travel_date: date, value: str) -> datetime:
    try:
        hour_text, minute_text = value.split(":", 1)
        return datetime(
            travel_date.year,
            travel_date.month,
            travel_date.day,
            int(hour_text),
            int(minute_text),
            tzinfo=_CHINA_RAIL_TIMEZONE,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 12306 local time: {value!r}") from exc


def _duration_minutes(value: str | None) -> int | None:
    if not value:
        return None
    try:
        hours, minutes = value.split(":", 1)
        return int(hours) * 60 + int(minutes)
    except (AttributeError, TypeError, ValueError):
        return None


def _require_aware_datetime(value: datetime | None, field_name: str) -> None:
    if value is not None and (
        value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware.")
