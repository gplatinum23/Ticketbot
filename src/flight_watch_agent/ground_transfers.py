from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Protocol

from .places import PlaceRef, resolve_place


TransferKind = Literal[
    "same_airport",
    "station_airport",
    "airport_station",
    "airport_airport",
]
ConnectionKind = Literal[
    "same_terminal",
    "same_airport",
    "station_airport",
    "airport_station",
    "cross_airport",
    "cross_city",
    "unsupported",
]
TransferReliability = Literal["curated_estimate", "city_default_estimate"]


@dataclass(frozen=True)
class GroundTransfer:
    origin: PlaceRef
    destination: PlaceRef
    kind: TransferKind
    duration_minutes: int
    buffer_minutes: int
    price: float
    currency: str
    source: str
    reliability: TransferReliability
    departure_at: datetime
    arrival_at: datetime

    @property
    def required_minutes(self) -> int:
        return self.duration_minutes + self.buffer_minutes


@dataclass(frozen=True)
class GroundTransferRule:
    origin_id: str
    destination_id: str
    duration_minutes: int
    buffer_minutes: int
    price: float
    source: str


@dataclass(frozen=True)
class ConnectionClassification:
    kind: ConnectionKind
    origin: PlaceRef
    destination: PlaceRef
    requires_ground_transfer: bool


class GroundTransferProvider(Protocol):
    def find(
        self,
        origin: str | PlaceRef,
        destination: str | PlaceRef,
        *,
        departure_at: datetime,
        currency: str = "CNY",
        origin_terminal: str | None = None,
        destination_terminal: str | None = None,
    ) -> GroundTransfer | None:
        """Return sourced transfer evidence, or None when no connection is known."""


class StaticGroundTransferProvider:
    """Conservative v1 transfer evidence for same-city intermodal connections."""

    def find(
        self,
        origin: str | PlaceRef,
        destination: str | PlaceRef,
        *,
        departure_at: datetime,
        currency: str = "CNY",
        origin_terminal: str | None = None,
        destination_terminal: str | None = None,
    ) -> GroundTransfer | None:
        origin_place = origin if isinstance(origin, PlaceRef) else resolve_place(origin)
        destination_place = (
            destination if isinstance(destination, PlaceRef) else resolve_place(destination)
        )
        classification = classify_connection(
            origin_place,
            destination_place,
            origin_terminal=origin_terminal,
            destination_terminal=destination_terminal,
        )
        if not classification.requires_ground_transfer:
            return None
        if classification.kind in {"cross_city", "unsupported"}:
            return None
        kind: TransferKind = (
            "airport_airport"
            if classification.kind == "cross_airport"
            else classification.kind
        )

        rule = _RULES.get(
            (origin_place.canonical_id, destination_place.canonical_id)
        )
        if rule is None:
            duration, buffer, price = _DEFAULTS[kind]
            source = "builtin_same_city_conservative_estimate:v1"
            reliability: TransferReliability = "city_default_estimate"
        else:
            duration = rule.duration_minutes
            buffer = rule.buffer_minutes
            price = rule.price
            source = rule.source
            reliability = "curated_estimate"

        return GroundTransfer(
            origin=origin_place,
            destination=destination_place,
            kind=kind,
            duration_minutes=duration,
            buffer_minutes=buffer,
            price=price,
            currency=currency,
            source=source,
            reliability=reliability,
            departure_at=departure_at,
            arrival_at=departure_at + timedelta(minutes=duration),
        )


def same_physical_endpoint(
    origin: str | PlaceRef,
    destination: str | PlaceRef,
) -> bool:
    origin_place = origin if isinstance(origin, PlaceRef) else resolve_place(origin)
    destination_place = (
        destination if isinstance(destination, PlaceRef) else resolve_place(destination)
    )
    return bool(
        origin_place.known
        and destination_place.known
        and origin_place.canonical_id == destination_place.canonical_id
    )


def classify_connection(
    origin: str | PlaceRef,
    destination: str | PlaceRef,
    *,
    origin_terminal: str | None = None,
    destination_terminal: str | None = None,
) -> ConnectionClassification:
    origin_place = origin if isinstance(origin, PlaceRef) else resolve_place(origin)
    destination_place = (
        destination if isinstance(destination, PlaceRef) else resolve_place(destination)
    )
    same_endpoint = (
        origin_place.known
        and destination_place.known
        and origin_place.canonical_id == destination_place.canonical_id
    )
    if same_endpoint and origin_place.kind == "airport":
        same_terminal = bool(
            origin_terminal
            and destination_terminal
            and origin_terminal.strip().casefold()
            == destination_terminal.strip().casefold()
        )
        return ConnectionClassification(
            kind="same_terminal" if same_terminal else "same_airport",
            origin=origin_place,
            destination=destination_place,
            requires_ground_transfer=not same_terminal,
        )

    if (
        origin_place.city_id
        and destination_place.city_id
        and origin_place.city_id != destination_place.city_id
    ):
        return ConnectionClassification(
            kind="cross_city",
            origin=origin_place,
            destination=destination_place,
            requires_ground_transfer=False,
        )

    pair = (origin_place.kind, destination_place.kind)
    if pair == ("station", "airport"):
        kind: ConnectionKind = "station_airport"
    elif pair == ("airport", "station"):
        kind = "airport_station"
    elif pair == ("airport", "airport"):
        kind = (
            "cross_airport"
            if origin_place.city_id
            and origin_place.city_id == destination_place.city_id
            else "cross_city"
        )
    elif same_endpoint:
        kind = "same_terminal"
    else:
        kind = "unsupported"
    return ConnectionClassification(
        kind=kind,
        origin=origin_place,
        destination=destination_place,
        requires_ground_transfer=kind
        in {"same_airport", "station_airport", "airport_station", "cross_airport"},
    )


def _rule(
    origin_id: str,
    destination_id: str,
    duration_minutes: int,
    buffer_minutes: int,
    price: float,
    source: str,
) -> GroundTransferRule:
    return GroundTransferRule(
        origin_id=origin_id,
        destination_id=destination_id,
        duration_minutes=duration_minutes,
        buffer_minutes=buffer_minutes,
        price=price,
        source=source,
    )


_DEFAULTS: dict[TransferKind, tuple[int, int, float]] = {
    "same_airport": (20, 20, 0.0),
    "station_airport": (90, 30, 80.0),
    "airport_station": (90, 20, 80.0),
    "airport_airport": (120, 30, 150.0),
}

_RULE_LIST = (
    _rule(
        "station:CUW",
        "airport:CKG",
        40,
        30,
        35.0,
        "builtin_curated_transfer_estimate:chongqingbei-ckg:v1",
    ),
    _rule(
        "station:BXP",
        "airport:PEK",
        75,
        30,
        35.0,
        "builtin_curated_transfer_estimate:beijingxi-pek:v1",
    ),
    _rule(
        "station:BXP",
        "airport:PKX",
        70,
        30,
        40.0,
        "builtin_curated_transfer_estimate:beijingxi-pkx:v1",
    ),
    _rule(
        "airport:PEK",
        "airport:PKX",
        120,
        30,
        180.0,
        "builtin_curated_transfer_estimate:pek-pkx:v1",
    ),
    _rule(
        "airport:PKX",
        "airport:PEK",
        120,
        30,
        180.0,
        "builtin_curated_transfer_estimate:pkx-pek:v1",
    ),
)

_RULES = {
    (rule.origin_id, rule.destination_id): rule
    for rule in _RULE_LIST
}
