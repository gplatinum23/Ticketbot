from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal


RouteRegionType = Literal[
    "china_to_abroad",
    "abroad_to_china",
    "china_domestic",
    "abroad_to_abroad",
]

TravelStrategy = Literal[
    "direct_flight",
    "direct_train",
    "train_flight",
    "flight_train",
    "train_train",
    "flight_flight",
]

QueryMode = Literal["train", "flight"]
EdgeMode = Literal["train", "flight", "local_transfer"]


@dataclass(frozen=True)
class RegionInfo:
    origin_country: str | None
    destination_country: str | None
    origin_is_china: bool
    destination_is_china: bool
    route_type: RouteRegionType


@dataclass(frozen=True)
class StrategySelection:
    enabled: list[TravelStrategy]
    disabled: dict[TravelStrategy, str]


@dataclass(frozen=True)
class CandidateHub:
    hub_id: str
    city: str
    airport_codes: list[str]
    train_places: list[str]
    strategies: list[TravelStrategy]
    priority: float
    reason: str
    flight_potential_score: float | None = None
    flight_tier: str | None = None


@dataclass(frozen=True)
class QueryPlanItem:
    query_id: str
    mode: QueryMode
    strategy: TravelStrategy
    origin: str
    destination: str
    travel_date: date
    leg_index: int
    hub_id: str | None = None
    executable: bool = True
    status: Literal["planned", "not_implemented"] = "planned"
    reason: str | None = None


@dataclass(frozen=True)
class QueryBudget:
    max_hubs_per_strategy: int = 10
    max_flight_queries: int = 50
    max_train_queries: int = 50
    max_total_routes: int = 150


@dataclass(frozen=True)
class QueryPlan:
    items: list[QueryPlanItem]
    budget: QueryBudget = field(default_factory=QueryBudget)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RouteEdge:
    edge_id: str
    mode: EdgeMode
    strategy: TravelStrategy
    origin: str
    destination: str
    travel_date: date
    price: float | None
    currency: str = "CNY"
    departure_time: datetime | str | None = None
    arrival_time: datetime | str | None = None
    duration_minutes: int | None = None
    source: str = ""
    confidence: float = 1.0
    hub_id: str | None = None
    leg_index: int = 1
    raw_option: object | None = None
