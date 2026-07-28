from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from typing import Callable, Generic, Protocol, Sequence, TypeVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .models import FlightOption, FlightSearchIntent, TrainOption


class ToolStatus(str, Enum):
    SUCCESS = "success"
    NO_RESULTS = "no_results"
    HUMAN_ACTION_REQUIRED = "human_action_required"
    ERROR = "error"


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    TIMEOUT = "timeout"
    CAPTCHA_REQUIRED = "captcha_required"
    LOGIN_REQUIRED = "login_required"
    RATE_LIMITED = "rate_limited"
    ROUTE_MISMATCH = "route_mismatch"
    PARSE_FAILED = "parse_failed"
    TOOL_UNAVAILABLE = "tool_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class ToolError:
    code: ToolErrorCode
    message: str
    retryable: bool
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolMetrics:
    request_id: str
    started_at: datetime
    latency_ms: int
    cache_hit: bool
    attempts: int
    backend: str


T = TypeVar("T")


@dataclass(frozen=True)
class ToolResult(Generic[T]):
    status: ToolStatus
    data: T | None
    metrics: ToolMetrics
    error: ToolError | None = None
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {ToolStatus.SUCCESS, ToolStatus.NO_RESULTS}


@dataclass(frozen=True)
class TrainSearchRequest:
    origin: str
    destination: str
    travel_date: date
    max_results: int = 20

    def __post_init__(self) -> None:
        _validate_route_request(
            self.origin,
            self.destination,
            self.travel_date,
            self.max_results,
        )

    @property
    def request_id(self) -> str:
        return _request_id("train", self.as_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            "origin": self.origin.strip(),
            "destination": self.destination.strip(),
            "travel_date": self.travel_date.isoformat(),
            "max_results": self.max_results,
        }

    def to_intent(self) -> FlightSearchIntent:
        return FlightSearchIntent(
            origin=self.origin,
            destination=self.destination,
            travel_date=self.travel_date,
        )


@dataclass(frozen=True)
class FlightSearchRequest:
    origin: str
    destination: str
    travel_date: date
    time_preference: str | None = None
    budget_threshold: float | None = None
    currency: str = "CNY"
    max_segments: int = 3
    max_results: int = 5
    direct_only: bool = False

    def __post_init__(self) -> None:
        _validate_route_request(
            self.origin,
            self.destination,
            self.travel_date,
            self.max_results,
        )
        if self.max_segments < 1:
            raise ValueError("max_segments must be positive.")
        if self.budget_threshold is not None and self.budget_threshold <= 0:
            raise ValueError("budget_threshold must be positive when supplied.")
        if not self.currency.strip():
            raise ValueError("currency must not be empty.")

    @property
    def request_id(self) -> str:
        return _request_id("flight", self.as_payload())

    def as_payload(self) -> dict[str, object]:
        return {
            "origin": self.origin.strip().upper(),
            "destination": self.destination.strip().upper(),
            "travel_date": self.travel_date.isoformat(),
            "time_preference": (self.time_preference or "").strip().casefold(),
            "budget_threshold": self.budget_threshold,
            "currency": self.currency.strip().upper(),
            "max_segments": self.max_segments,
            "max_results": self.max_results,
            "direct_only": self.direct_only,
        }

    def to_intent(self) -> FlightSearchIntent:
        return FlightSearchIntent(
            origin=self.origin,
            destination=self.destination,
            travel_date=self.travel_date,
            time_preference=self.time_preference,
            budget_threshold=self.budget_threshold,
            currency=self.currency,
            max_segments=self.max_segments,
        )

    @classmethod
    def from_intent(
        cls,
        intent: FlightSearchIntent,
        *,
        max_results: int = 5,
        direct_only: bool = False,
    ) -> "FlightSearchRequest":
        return cls(
            origin=intent.origin,
            destination=intent.destination,
            travel_date=intent.travel_date,
            time_preference=intent.time_preference,
            budget_threshold=intent.budget_threshold,
            currency=intent.currency,
            max_segments=intent.max_segments,
            max_results=max_results,
            direct_only=direct_only,
        )


@dataclass(frozen=True)
class TrainSearchOutput:
    options: tuple[TrainOption, ...]
    source: str = "12306_mcp"


@dataclass(frozen=True)
class FlightSearchOutput:
    options: tuple[FlightOption, ...]
    raw_state: dict[str, object]
    source: str = "flights.ctrip.com"


class TrainSearchTool(Protocol):
    def search(self, request: TrainSearchRequest) -> ToolResult[TrainSearchOutput]:
        """Search one train route without raising backend errors."""

    def search_many(
        self,
        requests: Sequence[TrainSearchRequest],
    ) -> list[ToolResult[TrainSearchOutput]]:
        """Search train routes in input order with duplicate suppression."""

    def clear_cache(self) -> None:
        """Invalidate all cached train results."""


class FlightSearchTool(Protocol):
    def search(self, request: FlightSearchRequest) -> ToolResult[FlightSearchOutput]:
        """Search one flight route without raising backend errors."""

    def search_many(
        self,
        requests: Sequence[FlightSearchRequest],
    ) -> list[ToolResult[FlightSearchOutput]]:
        """Search flight routes in input order with duplicate suppression."""

    def clear_cache(self) -> None:
        """Invalidate all cached flight results."""


@dataclass(frozen=True)
class ToolCachePolicy:
    ttl_seconds: int = 300
    max_entries: int = 512
    cache_no_results: bool = True

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive.")


class InMemoryToolCache:
    def __init__(self, policy: ToolCachePolicy | None = None) -> None:
        self.policy = policy or ToolCachePolicy()
        self._items: OrderedDict[str, tuple[float, ToolResult[object]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str):
        with self._lock:
            cached = self._items.get(key)
            if cached is None:
                return None
            expires_at, result = cached
            if expires_at <= time.monotonic():
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return deepcopy(result)

    def put(self, key: str, result: ToolResult[object]) -> None:
        if result.status == ToolStatus.ERROR or result.status == ToolStatus.HUMAN_ACTION_REQUIRED:
            return
        if result.status == ToolStatus.NO_RESULTS and not self.policy.cache_no_results:
            return
        with self._lock:
            self._items[key] = (
                time.monotonic() + self.policy.ttl_seconds,
                deepcopy(result),
            )
            self._items.move_to_end(key)
            while len(self._items) > self.policy.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class LegacyTrainProvider(Protocol):
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        """Legacy provider contract."""


class CachedTrainSearchTool:
    def __init__(
        self,
        provider: LegacyTrainProvider,
        *,
        cache: InMemoryToolCache | None = None,
        backend_name: str = "12306_mcp",
    ) -> None:
        self.provider = provider
        self.cache = cache or InMemoryToolCache()
        self.backend_name = backend_name

    def search(self, request: TrainSearchRequest) -> ToolResult[TrainSearchOutput]:
        return self.search_many([request])[0]

    def search_many(
        self,
        requests: Sequence[TrainSearchRequest],
    ) -> list[ToolResult[TrainSearchOutput]]:
        results: dict[str, ToolResult[TrainSearchOutput]] = {}
        for request in requests:
            if request.request_id in results:
                continue
            cached = self.cache.get(request.request_id)
            if cached is not None:
                results[request.request_id] = _as_cache_hit(cached)
                continue
            started_at = _utc_now()
            started = time.monotonic()
            try:
                options = tuple(
                    sorted(
                        self.provider.query_train_options(request.to_intent()),
                        key=_train_sort_key,
                    )[: request.max_results]
                )
                result = ToolResult(
                    status=ToolStatus.SUCCESS if options else ToolStatus.NO_RESULTS,
                    data=TrainSearchOutput(options=options),
                    metrics=_metrics(
                        request.request_id,
                        started_at,
                        started,
                        backend=self.backend_name,
                    ),
                )
            except Exception as exc:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    error=classify_tool_error(exc),
                    metrics=_metrics(
                        request.request_id,
                        started_at,
                        started,
                        backend=self.backend_name,
                    ),
                )
            self.cache.put(request.request_id, result)
            results[request.request_id] = result
        return [results[request.request_id] for request in requests]

    def clear_cache(self) -> None:
        self.cache.clear()


FlightBatchRunner = Callable[
    [Sequence[FlightSearchRequest]],
    list[ToolResult[FlightSearchOutput]],
]


class CachedFlightSearchTool:
    def __init__(
        self,
        runner: FlightBatchRunner,
        *,
        cache: InMemoryToolCache | None = None,
        backend_name: str = "ctrip_react",
    ) -> None:
        self.runner = runner
        self.cache = cache or InMemoryToolCache()
        self.backend_name = backend_name

    def search(self, request: FlightSearchRequest) -> ToolResult[FlightSearchOutput]:
        return self.search_many([request])[0]

    def search_many(
        self,
        requests: Sequence[FlightSearchRequest],
    ) -> list[ToolResult[FlightSearchOutput]]:
        results: dict[str, ToolResult[FlightSearchOutput]] = {}
        unique_misses: list[FlightSearchRequest] = []
        for request in requests:
            if request.request_id in results or any(
                item.request_id == request.request_id for item in unique_misses
            ):
                continue
            cached = self.cache.get(request.request_id)
            if cached is not None:
                results[request.request_id] = _as_cache_hit(cached)
            else:
                unique_misses.append(request)

        if unique_misses:
            try:
                fresh = self.runner(unique_misses)
                if len(fresh) != len(unique_misses):
                    raise RuntimeError(
                        "Flight tool runner returned a different number of results than requests."
                    )
            except Exception as exc:
                error = classify_tool_error(exc)
                fresh = [
                    ToolResult(
                        status=ToolStatus.ERROR,
                        data=None,
                        error=error,
                        metrics=ToolMetrics(
                            request_id=request.request_id,
                            started_at=_utc_now(),
                            latency_ms=0,
                            cache_hit=False,
                            attempts=1,
                            backend=self.backend_name,
                        ),
                    )
                    for request in unique_misses
                ]
            for request, result in zip(unique_misses, fresh, strict=True):
                self.cache.put(request.request_id, result)
                results[request.request_id] = result

        return [results[request.request_id] for request in requests]

    def clear_cache(self) -> None:
        self.cache.clear()


class TrainSearchArgs(BaseModel):
    origin: str = Field(description="12306 station or city name.")
    destination: str = Field(description="12306 station or city name.")
    travel_date: date
    max_results: int = Field(default=20, ge=1, le=100)


class FlightSearchArgs(BaseModel):
    origin: str = Field(description="Origin city or IATA code.")
    destination: str = Field(description="Destination city or IATA code.")
    travel_date: date
    time_preference: str | None = None
    budget_threshold: float | None = Field(default=None, gt=0)
    currency: str = "CNY"
    max_segments: int = Field(default=3, ge=1, le=3)
    max_results: int = Field(default=5, ge=1, le=20)
    direct_only: bool = False


def as_langchain_train_tool(tool: TrainSearchTool) -> StructuredTool:
    def invoke(**kwargs):
        request = TrainSearchRequest(**kwargs)
        return _public_train_result(tool.search(request))

    return StructuredTool.from_function(
        func=invoke,
        name="search_trains",
        description=(
            "Search verified 12306 train options for one route and date. "
            "Returns structured status, errors, metrics, and ticket options."
        ),
        args_schema=TrainSearchArgs,
    )


def as_langchain_flight_tool(tool: FlightSearchTool) -> StructuredTool:
    def invoke(**kwargs):
        request = FlightSearchRequest(**kwargs)
        return _public_flight_result(tool.search(request))

    return StructuredTool.from_function(
        func=invoke,
        name="search_flights",
        description=(
            "Search verified public flight options for one route and date. "
            "Returns structured status, errors, metrics, and concise options."
        ),
        args_schema=FlightSearchArgs,
    )


def classify_tool_error(exc: Exception) -> ToolError:
    text = str(exc)
    folded = text.casefold()
    if isinstance(exc, ValueError):
        return ToolError(ToolErrorCode.INVALID_INPUT, text, retryable=False)
    if isinstance(exc, TimeoutError) or "timed out" in folded or "timeout" in folded:
        return ToolError(ToolErrorCode.TIMEOUT, text, retryable=True)
    if "captcha" in folded or "manual verification" in folded or "安全访问" in text:
        return ToolError(ToolErrorCode.CAPTCHA_REQUIRED, text, retryable=False)
    if "login" in folded or "password" in folded or "account" in folded:
        return ToolError(ToolErrorCode.LOGIN_REQUIRED, text, retryable=False)
    if "rate limit" in folded or "too many requests" in folded:
        return ToolError(ToolErrorCode.RATE_LIMITED, text, retryable=True)
    if "route mismatch" in folded:
        return ToolError(ToolErrorCode.ROUTE_MISMATCH, text, retryable=False)
    if "parse" in folded or "non-json" in folded:
        return ToolError(ToolErrorCode.PARSE_FAILED, text, retryable=False)
    if "unavailable" in folded or "exited before responding" in folded:
        return ToolError(ToolErrorCode.TOOL_UNAVAILABLE, text, retryable=True)
    return ToolError(ToolErrorCode.INTERNAL_ERROR, text, retryable=False)


def tool_error_from_flight_state(state: dict[str, object]) -> ToolError | None:
    observations = state.get("observations", [])
    last_status = None
    last_warning = None
    if isinstance(observations, list) and observations:
        last = observations[-1]
        if isinstance(last, dict):
            last_status = str(last.get("status") or "")
            last_warning = str(last.get("warning") or "")
    termination = str(state.get("termination_reason") or "")
    warnings = [str(item) for item in state.get("warnings", [])]
    text = last_warning or " | ".join(warnings[-3:]) or termination

    if last_status == "captcha_required" or "human_verification" in termination:
        return ToolError(ToolErrorCode.CAPTCHA_REQUIRED, text, retryable=False)
    if last_status == "login_required":
        return ToolError(ToolErrorCode.LOGIN_REQUIRED, text, retryable=False)
    if last_status == "no_payload":
        return ToolError(ToolErrorCode.TIMEOUT, text, retryable=True)
    if last_status == "parse_failed":
        return ToolError(ToolErrorCode.PARSE_FAILED, text, retryable=False)
    if last_status == "tool_error":
        return ToolError(ToolErrorCode.TOOL_UNAVAILABLE, text, retryable=True)
    return None


def flight_tool_result_from_state(
    request: FlightSearchRequest,
    state: dict[str, object],
    *,
    started_at: datetime,
    started_monotonic: float,
    backend: str = "ctrip_react",
) -> ToolResult[FlightSearchOutput]:
    options = tuple(state.get("verified_flight_options", []))
    warnings = tuple(str(item) for item in state.get("warnings", []))
    error = tool_error_from_flight_state(state)
    if options:
        status = ToolStatus.SUCCESS
        error = None
    elif error is not None and error.code == ToolErrorCode.CAPTCHA_REQUIRED:
        status = ToolStatus.HUMAN_ACTION_REQUIRED
    elif error is not None:
        status = ToolStatus.ERROR
    else:
        status = ToolStatus.NO_RESULTS
    return ToolResult(
        status=status,
        data=FlightSearchOutput(options=options, raw_state=state),
        error=error,
        warnings=warnings,
        metrics=_metrics(
            request.request_id,
            started_at,
            started_monotonic,
            backend=backend,
        ),
    )


def _validate_route_request(
    origin: str,
    destination: str,
    travel_date: date,
    max_results: int,
) -> None:
    if not origin.strip() or not destination.strip():
        raise ValueError("origin and destination must not be empty.")
    if origin.strip().casefold() == destination.strip().casefold():
        raise ValueError("origin and destination must be different.")
    if not isinstance(travel_date, date):
        raise ValueError("travel_date must be a date.")
    if max_results <= 0:
        raise ValueError("max_results must be positive.")


def _request_id(namespace: str, payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"{namespace}:{digest}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _metrics(
    request_id: str,
    started_at: datetime,
    started_monotonic: float,
    *,
    backend: str,
) -> ToolMetrics:
    return ToolMetrics(
        request_id=request_id,
        started_at=started_at,
        latency_ms=max(0, round((time.monotonic() - started_monotonic) * 1000)),
        cache_hit=False,
        attempts=1,
        backend=backend,
    )


def _as_cache_hit(result):
    return replace(
        result,
        metrics=replace(
            result.metrics,
            cache_hit=True,
            latency_ms=0,
            attempts=0,
        ),
    )


def _train_sort_key(option: TrainOption) -> tuple[float, str]:
    price = option.lowest_price
    return (price if price is not None else float("inf"), option.start_time)


def _public_train_result(result: ToolResult[TrainSearchOutput]) -> dict[str, object]:
    options = result.data.options if result.data is not None else ()
    return {
        **_public_result_envelope(result),
        "options": [
            {
                "train_code": option.train_code,
                "from_station": option.from_station,
                "to_station": option.to_station,
                "travel_date": option.travel_date.isoformat(),
                "start_time": option.start_time,
                "arrive_time": option.arrive_time,
                "duration": option.duration,
                "lowest_price": option.lowest_price,
                "prices": option.prices,
                "seats": option.seats,
            }
            for option in options
        ],
    }


def _public_flight_result(result: ToolResult[FlightSearchOutput]) -> dict[str, object]:
    options = result.data.options if result.data is not None else ()
    return {
        **_public_result_envelope(result),
        "options": [
            {
                "origin": option.origin,
                "destination": option.destination,
                "travel_date": option.travel_date.isoformat(),
                "price": option.price,
                "currency": option.currency,
                "departure_time": (
                    option.departure_time.isoformat()
                    if option.departure_time is not None
                    else None
                ),
                "arrival_time": (
                    option.arrival_time.isoformat()
                    if option.arrival_time is not None
                    else None
                ),
                "reliability": option.reliability,
                "warnings": option.warnings,
                "evidence_count": option.evidence_count,
            }
            for option in options
        ],
    }


def _public_result_envelope(result: ToolResult[object]) -> dict[str, object]:
    error = result.error
    return {
        "status": result.status.value,
        "error": (
            {
                "code": error.code.value,
                "message": error.message,
                "retryable": error.retryable,
                "details": error.details,
            }
            if error is not None
            else None
        ),
        "warnings": list(result.warnings),
        "metrics": {
            "request_id": result.metrics.request_id,
            "started_at": result.metrics.started_at.isoformat(),
            "latency_ms": result.metrics.latency_ms,
            "cache_hit": result.metrics.cache_hit,
            "attempts": result.metrics.attempts,
            "backend": result.metrics.backend,
        },
    }
