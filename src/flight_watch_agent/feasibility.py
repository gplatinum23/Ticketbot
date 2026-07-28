from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Literal, Sequence

from .agent_models import RouteEdge
from .ground_transfers import GroundTransfer, classify_connection
from .models import FlightOption, TrainOption
from .places import PlaceRef, resolve_place, resolve_station_place


class FeasibilityStatus(str, Enum):
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNCERTAIN = "uncertain"


class FeasibilityReasonCode(str, Enum):
    MISSING_ROUTE_EDGES = "missing_route_edges"
    MISSING_DEPARTURE_TIME = "missing_departure_time"
    MISSING_ARRIVAL_TIME = "missing_arrival_time"
    NAIVE_DATETIME = "naive_datetime"
    EDGE_TIME_REVERSED = "edge_time_reversed"
    ENDPOINT_DISCONNECTED = "endpoint_disconnected"
    MISSING_GROUND_TRANSFER = "missing_ground_transfer"
    CROSS_CITY_TRANSFER_UNSUPPORTED = "cross_city_transfer_unsupported"
    CONNECTION_MISSED = "connection_missed"
    INSUFFICIENT_CONNECTION_BUFFER = "insufficient_connection_buffer"
    GROUND_TRANSFER_EVIDENCE_UNCERTAIN = "ground_transfer_evidence_uncertain"
    TERMINAL_INFORMATION_MISSING = "terminal_information_missing"


IssueSeverity = Literal["error", "warning"]


@dataclass(frozen=True)
class FeasibilityIssue:
    code: FeasibilityReasonCode
    message: str
    edge_ids: tuple[str, ...] = ()
    severity: IssueSeverity = "error"
    available_minutes: int | None = None
    required_minutes: int | None = None


@dataclass(frozen=True)
class ConnectionCheck:
    first_edge_id: str
    second_edge_id: str
    available_minutes: int | None
    required_minutes: int
    ground_transfer_edge_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeasibilityResult:
    status: FeasibilityStatus
    issues: tuple[FeasibilityIssue, ...] = ()
    connections: tuple[ConnectionCheck, ...] = ()

    @property
    def usable(self) -> bool:
        return self.status != FeasibilityStatus.INFEASIBLE


@dataclass(frozen=True)
class FeasibilityPolicy:
    train_train_minutes: int = 60
    train_flight_domestic_minutes: int = 120
    train_flight_international_minutes: int = 180
    flight_train_domestic_minutes: int = 90
    flight_train_international_minutes: int = 120
    flight_flight_same_terminal_domestic_minutes: int = 90
    flight_flight_same_terminal_international_minutes: int = 150
    flight_flight_same_airport_domestic_minutes: int = 120
    flight_flight_same_airport_international_minutes: int = 180
    flight_flight_cross_airport_domestic_minutes: int = 180
    flight_flight_cross_airport_international_minutes: int = 240


class RouteFeasibilityEngine:
    def __init__(self, policy: FeasibilityPolicy | None = None) -> None:
        self.policy = policy or FeasibilityPolicy()

    def evaluate(
        self,
        *,
        route_type: str,
        edges: Sequence[RouteEdge],
    ) -> FeasibilityResult:
        if not edges:
            issue = FeasibilityIssue(
                code=FeasibilityReasonCode.MISSING_ROUTE_EDGES,
                message="Route has no structured edges to verify.",
                severity="warning",
            )
            return FeasibilityResult(
                status=FeasibilityStatus.UNCERTAIN,
                issues=(issue,),
            )

        issues: list[FeasibilityIssue] = []
        for edge in edges:
            issues.extend(_validate_edge_time(edge))
            if (
                edge.mode == "local_transfer"
                and isinstance(edge.raw_option, GroundTransfer)
                and edge.raw_option.reliability == "city_default_estimate"
            ):
                issues.append(
                    FeasibilityIssue(
                        code=FeasibilityReasonCode.GROUND_TRANSFER_EVIDENCE_UNCERTAIN,
                        message=(
                            "Ground transfer uses a conservative city-level estimate."
                        ),
                        edge_ids=(edge.edge_id,),
                        severity="warning",
                    )
                )

        issues.extend(_validate_edge_continuity(edges))
        connection_checks, connection_issues = self._validate_connections(
            edges=edges,
        )
        issues.extend(connection_issues)
        return FeasibilityResult(
            status=_status_from_issues(issues),
            issues=tuple(_dedupe_issues(issues)),
            connections=tuple(connection_checks),
        )

    def _validate_connections(
        self,
        *,
        edges: Sequence[RouteEdge],
    ) -> tuple[list[ConnectionCheck], list[FeasibilityIssue]]:
        operational = [
            (index, edge)
            for index, edge in enumerate(edges)
            if edge.mode != "local_transfer"
        ]
        checks: list[ConnectionCheck] = []
        issues: list[FeasibilityIssue] = []
        for position in range(len(operational) - 1):
            first_index, first = operational[position]
            second_index, second = operational[position + 1]
            transfers = [
                edge
                for edge in edges[first_index + 1 : second_index]
                if edge.mode == "local_transfer"
            ]
            available = _connection_minutes(first.arrival_time, second.departure_time)
            required = self._required_connection_minutes(
                first=first,
                second=second,
                transfers=transfers,
            )
            checks.append(
                ConnectionCheck(
                    first_edge_id=first.edge_id,
                    second_edge_id=second.edge_id,
                    available_minutes=available,
                    required_minutes=required,
                    ground_transfer_edge_ids=tuple(
                        edge.edge_id for edge in transfers
                    ),
                )
            )
            edge_ids = (
                first.edge_id,
                *(edge.edge_id for edge in transfers),
                second.edge_id,
            )
            if (
                first.mode == "flight"
                and second.mode == "flight"
                and _same_actual_airport(first, second)
                and not _same_known_terminal(first, second)
                and (
                    _flight_terminal(first, arrival=True) is None
                    or _flight_terminal(second, arrival=False) is None
                )
            ):
                issues.append(
                    FeasibilityIssue(
                        code=FeasibilityReasonCode.TERMINAL_INFORMATION_MISSING,
                        message=(
                            "Same-airport connection lacks complete terminal "
                            "information; conservative transfer time is used."
                        ),
                        edge_ids=edge_ids,
                        severity="warning",
                    )
                )
            if available is None:
                issues.append(
                    FeasibilityIssue(
                        code=FeasibilityReasonCode.MISSING_DEPARTURE_TIME,
                        message="Connection time cannot be proven from absolute timestamps.",
                        edge_ids=edge_ids,
                        severity="warning",
                        required_minutes=required,
                    )
                )
            elif available < 0:
                issues.append(
                    FeasibilityIssue(
                        code=FeasibilityReasonCode.CONNECTION_MISSED,
                        message="The next leg departs before the previous leg arrives.",
                        edge_ids=edge_ids,
                        available_minutes=available,
                        required_minutes=required,
                    )
                )
            elif available < required:
                issues.append(
                    FeasibilityIssue(
                        code=FeasibilityReasonCode.INSUFFICIENT_CONNECTION_BUFFER,
                        message=(
                            f"Connection has {available} minutes but requires "
                            f"at least {required} minutes."
                        ),
                        edge_ids=edge_ids,
                        available_minutes=available,
                        required_minutes=required,
                    )
                )
        return checks, issues

    def _required_connection_minutes(
        self,
        *,
        first: RouteEdge,
        second: RouteEdge,
        transfers: Sequence[RouteEdge],
    ) -> int:
        international = _connection_is_international(first, second)
        ground_duration = sum(
            edge.duration_minutes or 0
            for edge in transfers
        )
        ground_buffer = max(
            (edge.buffer_minutes for edge in transfers),
            default=0,
        )

        if first.mode == "train" and second.mode == "train":
            base = self.policy.train_train_minutes
        elif first.mode == "train" and second.mode == "flight":
            base = (
                self.policy.train_flight_international_minutes
                if international
                else self.policy.train_flight_domestic_minutes
            )
        elif first.mode == "flight" and second.mode == "train":
            base = (
                self.policy.flight_train_international_minutes
                if international
                else self.policy.flight_train_domestic_minutes
            )
        elif first.mode == "flight" and second.mode == "flight":
            transfer_kind = _ground_transfer_kind(transfers)
            same_terminal = _same_known_terminal(first, second)
            if same_terminal and not transfers:
                base = (
                    self.policy.flight_flight_same_terminal_international_minutes
                    if international
                    else self.policy.flight_flight_same_terminal_domestic_minutes
                )
            elif transfer_kind == "airport_airport":
                base = (
                    self.policy.flight_flight_cross_airport_international_minutes
                    if international
                    else self.policy.flight_flight_cross_airport_domestic_minutes
                )
            else:
                base = (
                    self.policy.flight_flight_same_airport_international_minutes
                    if international
                    else self.policy.flight_flight_same_airport_domestic_minutes
                )
        else:
            base = 0
        return ground_duration + max(base, ground_buffer)


def _validate_edge_time(edge: RouteEdge) -> list[FeasibilityIssue]:
    issues: list[FeasibilityIssue] = []
    if edge.departure_time is None:
        issues.append(
            FeasibilityIssue(
                code=FeasibilityReasonCode.MISSING_DEPARTURE_TIME,
                message="Edge is missing an absolute departure time.",
                edge_ids=(edge.edge_id,),
                severity="warning",
            )
        )
    elif not _is_aware(edge.departure_time):
        issues.append(
            FeasibilityIssue(
                code=FeasibilityReasonCode.NAIVE_DATETIME,
                message="Departure time has no timezone.",
                edge_ids=(edge.edge_id,),
            )
        )
    if edge.arrival_time is None:
        issues.append(
            FeasibilityIssue(
                code=FeasibilityReasonCode.MISSING_ARRIVAL_TIME,
                message="Edge is missing an absolute arrival time.",
                edge_ids=(edge.edge_id,),
                severity="warning",
            )
        )
    elif not _is_aware(edge.arrival_time):
        issues.append(
            FeasibilityIssue(
                code=FeasibilityReasonCode.NAIVE_DATETIME,
                message="Arrival time has no timezone.",
                edge_ids=(edge.edge_id,),
            )
        )
    if (
        edge.departure_time is not None
        and edge.arrival_time is not None
        and _is_aware(edge.departure_time)
        and _is_aware(edge.arrival_time)
        and edge.arrival_time < edge.departure_time
    ):
        issues.append(
            FeasibilityIssue(
                code=FeasibilityReasonCode.EDGE_TIME_REVERSED,
                message="Edge arrives before it departs.",
                edge_ids=(edge.edge_id,),
            )
        )
    return issues


def _validate_edge_continuity(
    edges: Sequence[RouteEdge],
) -> list[FeasibilityIssue]:
    issues: list[FeasibilityIssue] = []
    for first, second in zip(edges, edges[1:], strict=False):
        first_destination = _edge_place(first, arrival=True)
        second_origin = _edge_place(second, arrival=False)
        if (
            first_destination.known
            and second_origin.known
            and first_destination.canonical_id == second_origin.canonical_id
        ):
            continue
        classification = classify_connection(first_destination, second_origin)
        edge_ids = (first.edge_id, second.edge_id)
        if classification.kind == "cross_city":
            issues.append(
                FeasibilityIssue(
                    code=FeasibilityReasonCode.CROSS_CITY_TRANSFER_UNSUPPORTED,
                    message="Adjacent route edges require unsupported cross-city movement.",
                    edge_ids=edge_ids,
                )
            )
        elif classification.requires_ground_transfer:
            issues.append(
                FeasibilityIssue(
                    code=FeasibilityReasonCode.MISSING_GROUND_TRANSFER,
                    message="Adjacent endpoints require an explicit ground-transfer edge.",
                    edge_ids=edge_ids,
                )
            )
        else:
            issues.append(
                FeasibilityIssue(
                    code=FeasibilityReasonCode.ENDPOINT_DISCONNECTED,
                    message="Adjacent route endpoints do not connect.",
                    edge_ids=edge_ids,
                )
            )
    return issues


def _edge_place(edge: RouteEdge, *, arrival: bool) -> PlaceRef:
    if isinstance(edge.raw_option, GroundTransfer):
        return edge.raw_option.destination if arrival else edge.raw_option.origin
    if isinstance(edge.raw_option, FlightOption):
        actual = (
            edge.raw_option.actual_destination
            if arrival
            else edge.raw_option.actual_origin
        )
        if actual is not None:
            return actual
    if isinstance(edge.raw_option, TrainOption):
        code = (
            edge.raw_option.to_station_code
            if arrival
            else edge.raw_option.from_station_code
        )
        name = (
            edge.raw_option.to_station
            if arrival
            else edge.raw_option.from_station
        )
        station = resolve_station_place(code or name)
        if station is not None:
            return station
    value = edge.destination if arrival else edge.origin
    return resolve_place(value)


def _connection_is_international(
    first: RouteEdge,
    second: RouteEdge,
) -> bool:
    for edge in (first, second):
        if not isinstance(edge.raw_option, FlightOption):
            continue
        origin = edge.raw_option.actual_origin
        destination = edge.raw_option.actual_destination
        if (
            origin is not None
            and destination is not None
            and origin.country
            and destination.country
            and origin.country != destination.country
        ):
            return True
    return False


def _ground_transfer_kind(transfers: Sequence[RouteEdge]) -> str | None:
    for edge in transfers:
        if isinstance(edge.raw_option, GroundTransfer):
            return edge.raw_option.kind
    return None


def _same_known_terminal(first: RouteEdge, second: RouteEdge) -> bool:
    arrival_terminal = _flight_terminal(first, arrival=True)
    departure_terminal = _flight_terminal(second, arrival=False)
    return bool(
        arrival_terminal
        and departure_terminal
        and arrival_terminal.casefold() == departure_terminal.casefold()
    )


def _same_actual_airport(first: RouteEdge, second: RouteEdge) -> bool:
    first_destination = _edge_place(first, arrival=True)
    second_origin = _edge_place(second, arrival=False)
    return bool(
        first_destination.kind == "airport"
        and second_origin.kind == "airport"
        and first_destination.canonical_id == second_origin.canonical_id
    )


def _flight_terminal(edge: RouteEdge, *, arrival: bool) -> str | None:
    if not isinstance(edge.raw_option, FlightOption):
        return None
    key = "arrival_terminal" if arrival else "departure_terminal"
    for evidence in edge.raw_option.evidence:
        value = str((evidence.metadata or {}).get(key) or "").strip()
        if value:
            return value
    return None


def _connection_minutes(
    first_arrival: datetime | None,
    second_departure: datetime | None,
) -> int | None:
    if first_arrival is None or second_departure is None:
        return None
    if not _is_aware(first_arrival) or not _is_aware(second_departure):
        return None
    return int((second_departure - first_arrival).total_seconds() // 60)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _status_from_issues(
    issues: Sequence[FeasibilityIssue],
) -> FeasibilityStatus:
    if any(issue.severity == "error" for issue in issues):
        return FeasibilityStatus.INFEASIBLE
    if issues:
        return FeasibilityStatus.UNCERTAIN
    return FeasibilityStatus.FEASIBLE


def _dedupe_issues(
    issues: Sequence[FeasibilityIssue],
) -> list[FeasibilityIssue]:
    output: list[FeasibilityIssue] = []
    seen: set[tuple[object, ...]] = set()
    for issue in issues:
        key = (
            issue.code,
            issue.edge_ids,
            issue.available_minutes,
            issue.required_minutes,
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(issue)
    return output
