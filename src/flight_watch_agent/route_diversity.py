from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import inf
from typing import Sequence

from .agent_models import RouteEdge
from .ground_transfers import GroundTransfer
from .models import FlightOption, TrainOption
from .places import resolve_place, resolve_station_place


class ValueProfile(str, Enum):
    BEST_OVERALL = "best_overall"
    LOWEST_PRICE = "lowest_price"
    SHORTEST_DURATION = "shortest_duration"
    LOWER_TRANSFER_RISK = "lower_transfer_risk"
    SHORTEST_WAIT = "shortest_wait"
    BALANCED_ALTERNATIVE = "balanced_alternative"


@dataclass(frozen=True)
class RouteSkeleton:
    route_type: str
    mode_sequence: tuple[str, ...]
    actual_endpoints: tuple[str, ...]
    core_services: tuple[str, ...]
    transfer_hubs: tuple[str, ...]

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.route_type,
            self.mode_sequence,
            self.actual_endpoints,
            self.core_services,
            self.transfer_hubs,
        )


@dataclass(frozen=True)
class DiversityCandidate:
    route_id: str
    skeleton: RouteSkeleton
    total_price: float | None
    total_duration_minutes: int | None
    transfer_wait_minutes: int | None
    transfer_count: int
    risk_score: float
    ranked_index: int
    shared_downstream_service: str | None = None


@dataclass(frozen=True)
class SelectedCandidate:
    route_id: str
    skeleton: RouteSkeleton
    value_profiles: tuple[ValueProfile, ...]


@dataclass(frozen=True)
class DiversitySelection:
    selected: tuple[SelectedCandidate, ...]
    exact_duplicates_removed: int
    family_variants_limited: int
    available_distinct_routes: int


def build_route_skeleton(
    *,
    route_type: str,
    edges: Sequence[RouteEdge],
    transfer_city: str | None = None,
    transfer_airport: str | None = None,
) -> RouteSkeleton:
    operational = [edge for edge in edges if edge.mode != "local_transfer"]
    endpoints = [
        endpoint
        for edge in edges
        for endpoint in (
            _edge_endpoint(edge, arrival=False),
            _edge_endpoint(edge, arrival=True),
        )
    ]
    services = tuple(
        _core_service_id(edge)
        for edge in operational
    )
    hubs = _transfer_hubs(
        operational,
        transfer_city=transfer_city,
        transfer_airport=transfer_airport,
    )
    return RouteSkeleton(
        route_type=route_type,
        mode_sequence=tuple(edge.mode for edge in edges),
        actual_endpoints=tuple(endpoints),
        core_services=services,
        transfer_hubs=hubs,
    )


def shared_downstream_flight_service(
    edges: Sequence[RouteEdge],
) -> str | None:
    operational = [edge for edge in edges if edge.mode != "local_transfer"]
    if len(operational) < 2 or operational[-1].mode != "flight":
        return None
    return _core_service_id(operational[-1])


def select_diverse_candidates(
    candidates: Sequence[DiversityCandidate],
    *,
    limit: int = 5,
    max_shared_downstream_service: int = 2,
) -> DiversitySelection:
    if limit <= 0:
        return DiversitySelection((), 0, 0, 0)

    distinct: list[DiversityCandidate] = []
    seen_skeletons: set[tuple[object, ...]] = set()
    for candidate in sorted(candidates, key=lambda item: item.ranked_index):
        identity = candidate.skeleton.identity
        if identity in seen_skeletons:
            continue
        seen_skeletons.add(identity)
        distinct.append(candidate)

    selected: list[DiversityCandidate] = []
    profiles_by_route: dict[str, list[ValueProfile]] = {}
    family_counts: dict[str, int] = {}
    family_limited_ids: set[str] = set()

    def can_select(candidate: DiversityCandidate) -> bool:
        if candidate.route_id in profiles_by_route:
            return True
        family = candidate.shared_downstream_service
        if family is None:
            return True
        return family_counts.get(family, 0) < max_shared_downstream_service

    def add(candidate: DiversityCandidate, profile: ValueProfile) -> None:
        if candidate.route_id in profiles_by_route:
            if profile not in profiles_by_route[candidate.route_id]:
                profiles_by_route[candidate.route_id].append(profile)
            return
        selected.append(candidate)
        profiles_by_route[candidate.route_id] = [profile]
        family = candidate.shared_downstream_service
        if family is not None:
            family_counts[family] = family_counts.get(family, 0) + 1

    profile_scorers = (
        (ValueProfile.BEST_OVERALL, _overall_key),
        (ValueProfile.LOWEST_PRICE, _price_key),
        (ValueProfile.SHORTEST_DURATION, _duration_key),
        (ValueProfile.LOWER_TRANSFER_RISK, _risk_key),
        (ValueProfile.SHORTEST_WAIT, _wait_key),
    )
    for profile, scorer in profile_scorers:
        if len(selected) >= limit:
            break
        eligible = [
            candidate
            for candidate in distinct
            if can_select(candidate)
            and scorer(candidate, distinct)[0] != inf
        ]
        if not eligible:
            continue
        add(min(eligible, key=lambda item: scorer(item, distinct)), profile)

    for candidate in distinct:
        if len(selected) >= limit:
            break
        if candidate.route_id in profiles_by_route:
            continue
        if not can_select(candidate):
            family_limited_ids.add(candidate.route_id)
            continue
        add(candidate, ValueProfile.BALANCED_ALTERNATIVE)

    return DiversitySelection(
        selected=tuple(
            SelectedCandidate(
                route_id=candidate.route_id,
                skeleton=candidate.skeleton,
                value_profiles=tuple(profiles_by_route[candidate.route_id]),
            )
            for candidate in selected
        ),
        exact_duplicates_removed=len(candidates) - len(distinct),
        family_variants_limited=len(family_limited_ids),
        available_distinct_routes=len(distinct),
    )


def _overall_key(
    candidate: DiversityCandidate,
    population: Sequence[DiversityCandidate],
) -> tuple[float, int]:
    price = _normalise(candidate.total_price, [item.total_price for item in population])
    duration = _normalise(
        candidate.total_duration_minutes,
        [item.total_duration_minutes for item in population],
    )
    wait = _normalise(
        candidate.transfer_wait_minutes or 0,
        [(item.transfer_wait_minutes or 0) for item in population],
    )
    transfers = _normalise(
        candidate.transfer_count,
        [item.transfer_count for item in population],
    )
    score = (
        0.30 * price
        + 0.25 * duration
        + 0.15 * wait
        + 0.15 * transfers
        + 0.15 * candidate.risk_score
    )
    return (score, candidate.ranked_index)


def _price_key(
    candidate: DiversityCandidate,
    _population: Sequence[DiversityCandidate],
) -> tuple[float, int]:
    return (
        candidate.total_price if candidate.total_price is not None else inf,
        candidate.ranked_index,
    )


def _duration_key(
    candidate: DiversityCandidate,
    _population: Sequence[DiversityCandidate],
) -> tuple[float, int]:
    return (
        (
            float(candidate.total_duration_minutes)
            if candidate.total_duration_minutes is not None
            else inf
        ),
        candidate.ranked_index,
    )


def _risk_key(
    candidate: DiversityCandidate,
    _population: Sequence[DiversityCandidate],
) -> tuple[float, int]:
    transfer_penalty = min(candidate.transfer_count, 4) * 0.10
    return (
        candidate.risk_score + transfer_penalty,
        candidate.ranked_index,
    )


def _wait_key(
    candidate: DiversityCandidate,
    _population: Sequence[DiversityCandidate],
) -> tuple[float, int]:
    return (
        (
            float(candidate.transfer_wait_minutes)
            if candidate.transfer_wait_minutes is not None
            else inf
        ),
        candidate.ranked_index,
    )


def _normalise(
    value: float | int | None,
    population: Sequence[float | int | None],
) -> float:
    known = [float(item) for item in population if item is not None]
    if value is None or not known:
        return 1.0
    low = min(known)
    high = max(known)
    if high == low:
        return 0.0
    return (float(value) - low) / (high - low)


def _edge_endpoint(edge: RouteEdge, *, arrival: bool) -> str:
    option = edge.raw_option
    if isinstance(option, FlightOption):
        place = option.actual_destination if arrival else option.actual_origin
        if place is not None:
            return place.canonical_id
    if isinstance(option, TrainOption):
        code = option.to_station_code if arrival else option.from_station_code
        name = option.to_station if arrival else option.from_station
        station = resolve_station_place(code) or resolve_station_place(name)
        if station is not None:
            return station.canonical_id
    if isinstance(option, GroundTransfer):
        place = option.destination if arrival else option.origin
        return place.canonical_id
    raw = edge.destination if arrival else edge.origin
    return resolve_place(raw).canonical_id


def _core_service_id(edge: RouteEdge) -> str:
    option = edge.raw_option
    departure = (
        edge.departure_time.isoformat()
        if edge.departure_time is not None
        else edge.travel_date.isoformat()
    )
    if isinstance(option, TrainOption):
        return f"train:{option.train_code}:{departure}"
    if isinstance(option, FlightOption):
        flight_number = _flight_number(option)
        origin = _edge_endpoint(edge, arrival=False)
        destination = _edge_endpoint(edge, arrival=True)
        return f"flight:{flight_number}:{origin}:{destination}:{departure}"
    return f"{edge.mode}:{_edge_endpoint(edge, arrival=False)}:{_edge_endpoint(edge, arrival=True)}:{departure}"


def _flight_number(option: FlightOption) -> str:
    for evidence in option.evidence:
        metadata = evidence.metadata or {}
        value = metadata.get("flight_no")
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    return "unknown"


def _transfer_hubs(
    operational: Sequence[RouteEdge],
    *,
    transfer_city: str | None,
    transfer_airport: str | None,
) -> tuple[str, ...]:
    explicit = [
        value.strip().upper()
        for value in (transfer_city, transfer_airport)
        if value and value.strip()
    ]
    edge_hubs = [
        edge.hub_id
        for edge in operational
        if edge.hub_id
    ]
    if explicit or edge_hubs:
        return tuple(dict.fromkeys([*explicit, *edge_hubs]))
    hubs: list[str] = []
    for first, second in zip(operational, operational[1:], strict=False):
        hubs.extend(
            (
                _edge_endpoint(first, arrival=True),
                _edge_endpoint(second, arrival=False),
            )
        )
    return tuple(dict.fromkeys(hubs))
