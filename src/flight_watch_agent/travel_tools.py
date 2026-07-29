from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import threading
import time
import urllib.parse
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .models import FlightEvidence, FlightOption, FlightSearchIntent, TrainOption
from .places import PlaceRef, resolve_air_query_place


T = TypeVar("T")


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
    CANCELLED = "cancelled"
    INTERNAL_ERROR = "internal_error"


_RETRYABLE_ERROR_CODES = frozenset(
    {
        ToolErrorCode.TIMEOUT,
        ToolErrorCode.RATE_LIMITED,
        ToolErrorCode.TOOL_UNAVAILABLE,
    }
)


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
    termination_reason: str | None = None
    retry_delays_ms: tuple[int, ...] = ()


@dataclass(frozen=True)
class ToolExecutionContext:
    """Execution limits and correlation data supplied at the tool boundary.

    ``deadline_monotonic`` is deliberately absolute: one context can be shared by
    a batch without accidentally granting every item a fresh timeout budget.
    """

    deadline_monotonic: float | None = None
    cancel_event: threading.Event | None = None
    trace_id: str | None = None
    idempotency_key: str | None = None

    @classmethod
    def with_timeout(
        cls,
        timeout_seconds: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        cancel_event: threading.Event | None = None,
        trace_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> "ToolExecutionContext":
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        return cls(
            deadline_monotonic=clock() + timeout_seconds,
            cancel_event=cancel_event,
            trace_id=trace_id,
            idempotency_key=idempotency_key,
        )


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 3.0
    jitter_ratio: float = 0.1
    retryable_codes: frozenset[ToolErrorCode] = _RETRYABLE_ERROR_CODES

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if self.base_delay_seconds < 0 or self.max_delay_seconds < 0:
            raise ValueError("retry delays must not be negative.")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1.")
        if not self.retryable_codes <= _RETRYABLE_ERROR_CODES:
            raise ValueError("Only timeout, rate_limited, and tool_unavailable are retryable.")

    def should_retry(self, error: ToolError, attempts: int) -> bool:
        return attempts < self.max_attempts and error.code in self.retryable_codes

    def delay_seconds(self, completed_attempts: int, random_value: float) -> float:
        base = min(
            self.max_delay_seconds,
            self.base_delay_seconds * (2 ** max(0, completed_attempts - 1)),
        )
        jitter = (random_value * 2 - 1) * self.jitter_ratio
        return max(0.0, base * (1 + jitter))


@dataclass(frozen=True)
class RuntimeExecution(Generic[T]):
    value: T | None
    error: ToolError | None
    attempts: int
    termination_reason: str
    retry_delays_ms: tuple[int, ...] = ()


class ToolRuntime:
    """The only retry/deadline policy used by domain tools.

    It cannot preempt an already running synchronous backend call.  It does,
    however, prevent new attempts once cancellation or the shared deadline is
    observed, which lets batches return completed items safely.
    """

    def __init__(
        self,
        *,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] | None = None,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.clock = clock
        self.sleeper = sleeper
        self.random_value = random_value or random.random

    def execute(
        self,
        operation: Callable[[], T],
        *,
        context: ToolExecutionContext | None = None,
    ) -> RuntimeExecution[T]:
        attempts = 0
        delays: list[int] = []
        while True:
            stopped = self._stop_error(context)
            if stopped is not None:
                return RuntimeExecution(
                    value=None,
                    error=stopped,
                    attempts=attempts,
                    termination_reason=_termination_reason(stopped),
                    retry_delays_ms=tuple(delays),
                )
            attempts += 1
            try:
                value = operation()
            except Exception as exc:
                error = classify_tool_error(exc)
            else:
                return RuntimeExecution(
                    value=value,
                    error=None,
                    attempts=attempts,
                    termination_reason="completed",
                    retry_delays_ms=tuple(delays),
                )
            if not self.retry_policy.should_retry(error, attempts):
                return RuntimeExecution(
                    value=None,
                    error=error,
                    attempts=attempts,
                    termination_reason=(
                        "retry_exhausted"
                        if error.code in self.retry_policy.retryable_codes
                        else "non_retryable_error"
                    ),
                    retry_delays_ms=tuple(delays),
                )
            delay = self.retry_policy.delay_seconds(attempts, self.random_value())
            stopped = self._wait_or_stop(delay, context)
            if stopped is not None:
                return RuntimeExecution(
                    value=None,
                    error=stopped,
                    attempts=attempts,
                    termination_reason=_termination_reason(stopped),
                    retry_delays_ms=tuple(delays),
                )
            delays.append(round(delay * 1000))

    def retry_result(
        self,
        initial: ToolResult[T],
        operation: Callable[[], ToolResult[T]],
        *,
        context: ToolExecutionContext | None = None,
        initial_attempts: int = 1,
        initial_retry_delays_ms: tuple[int, ...] = (),
    ) -> RuntimeExecution[ToolResult[T]]:
        """Retry an already observed structured backend result when permitted."""

        result: ToolResult[T] | None = initial
        error = initial.error
        attempts = initial_attempts
        delays: list[int] = list(initial_retry_delays_ms)
        while error is not None:
            if not self.retry_policy.should_retry(error, attempts):
                return RuntimeExecution(
                    value=result,
                    error=error if result is None else None,
                    attempts=attempts,
                    termination_reason=(
                        "retry_exhausted"
                        if error.code in self.retry_policy.retryable_codes
                        else "non_retryable_error"
                    ),
                    retry_delays_ms=tuple(delays),
                )
            delay = self.retry_policy.delay_seconds(attempts, self.random_value())
            stopped = self._wait_or_stop(delay, context)
            if stopped is not None:
                return RuntimeExecution(
                    value=None,
                    error=stopped,
                    attempts=attempts,
                    termination_reason=_termination_reason(stopped),
                    retry_delays_ms=tuple(delays),
                )
            delays.append(round(delay * 1000))
            attempts += 1
            try:
                result = operation()
            except Exception as exc:
                result = None
                error = classify_tool_error(exc)
            else:
                error = result.error
        return RuntimeExecution(
            value=result,
            error=None,
            attempts=attempts,
            termination_reason="completed",
            retry_delays_ms=tuple(delays),
        )

    def _stop_error(self, context: ToolExecutionContext | None) -> ToolError | None:
        if context is None:
            return None
        if context.cancel_event is not None and context.cancel_event.is_set():
            return ToolError(ToolErrorCode.CANCELLED, "Tool execution was cancelled.", False)
        if (
            context.deadline_monotonic is not None
            and self.clock() >= context.deadline_monotonic
        ):
            return ToolError(ToolErrorCode.TIMEOUT, "Tool execution deadline exceeded.", True)
        return None

    def _wait_or_stop(
        self,
        delay_seconds: float,
        context: ToolExecutionContext | None,
    ) -> ToolError | None:
        stopped = self._stop_error(context)
        if stopped is not None:
            return stopped
        if (
            context is not None
            and context.deadline_monotonic is not None
            and self.clock() + delay_seconds >= context.deadline_monotonic
        ):
            return ToolError(ToolErrorCode.TIMEOUT, "Tool execution deadline exceeded.", True)
        if context is not None and context.cancel_event is not None:
            context.cancel_event.wait(delay_seconds)
        else:
            self.sleeper(delay_seconds)
        return self._stop_error(context)


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
        origin_place = resolve_air_query_place(self.origin)
        destination_place = resolve_air_query_place(self.destination)
        if not origin_place.known:
            raise ValueError(f"Unknown flight origin: {self.origin}")
        if not destination_place.known:
            raise ValueError(f"Unknown flight destination: {self.destination}")
        if (
            origin_place.city_id
            and origin_place.city_id == destination_place.city_id
        ):
            raise ValueError(
                "Flight origin and destination must not be in the same city."
            )

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
    def search(
        self,
        request: TrainSearchRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult[TrainSearchOutput]:
        """Search one train route without raising backend errors."""

    def search_many(
        self,
        requests: Sequence[TrainSearchRequest],
        *,
        context: ToolExecutionContext | None = None,
    ) -> list[ToolResult[TrainSearchOutput]]:
        """Search train routes in input order with duplicate suppression."""

    def clear_cache(self) -> None:
        """Invalidate all cached train results."""


class FlightSearchTool(Protocol):
    def search(
        self,
        request: FlightSearchRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult[FlightSearchOutput]:
        """Search one flight route without raising backend errors."""

    def search_many(
        self,
        requests: Sequence[FlightSearchRequest],
        *,
        context: ToolExecutionContext | None = None,
    ) -> list[ToolResult[FlightSearchOutput]]:
        """Search flight routes in input order with duplicate suppression."""

    def clear_cache(self) -> None:
        """Invalidate all cached flight results."""


@dataclass(frozen=True)
class ToolCachePolicy:
    ttl_seconds: int = 300
    no_result_ttl_seconds: int = 60
    max_entries: int = 512
    cache_no_results: bool = True

    def __post_init__(self) -> None:
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive.")
        if self.no_result_ttl_seconds <= 0:
            raise ValueError("no_result_ttl_seconds must be positive.")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive.")

    def ttl_for(self, result: ToolResult[object]) -> int | None:
        if result.status == ToolStatus.SUCCESS:
            return self.ttl_seconds
        if result.status == ToolStatus.NO_RESULTS and self.cache_no_results:
            return self.no_result_ttl_seconds
        return None


class ToolCache(Protocol):
    def get(self, key: str) -> ToolResult[object] | None:
        """Return a still-valid result, if present."""

    def put(self, key: str, result: ToolResult[object]) -> None:
        """Store only results allowed by the cache policy."""

    def clear(self) -> None:
        """Remove all entries in this cache namespace."""


class InMemoryToolCache:
    def __init__(self, policy: ToolCachePolicy | None = None) -> None:
        self.policy = policy or ToolCachePolicy()
        self._items: OrderedDict[str, tuple[float, ToolResult[object]]] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> ToolResult[object] | None:
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

    def put(
        self,
        key: str,
        result: ToolResult[object],
        *,
        expires_at_monotonic: float | None = None,
    ) -> None:
        ttl_seconds = self.policy.ttl_for(result)
        if ttl_seconds is None:
            return
        with self._lock:
            self._items[key] = (
                expires_at_monotonic or (time.monotonic() + ttl_seconds),
                deepcopy(result),
            )
            self._items.move_to_end(key)
            while len(self._items) > self.policy.max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class SqliteToolCache:
    """A safe L2 cache that stores only serialised domain results.

    Raw browser state is intentionally omitted from persisted flight outputs;
    cached results remain useful for planning but cannot restore a browser
    session, Cookie or captured response.
    """

    def __init__(
        self,
        path: str | Path,
        policy: ToolCachePolicy | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path)
        self.policy = policy or ToolCachePolicy()
        self.clock = clock
        self._memory = InMemoryToolCache(self.policy)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def get(self, key: str) -> ToolResult[object] | None:
        cached = self._memory.get(key)
        if cached is not None:
            return cached
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT expires_at, payload FROM tool_cache_entries WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            expires_at, payload = float(row[0]), str(row[1])
            if expires_at <= self.clock():
                connection.execute(
                    "DELETE FROM tool_cache_entries WHERE cache_key = ?", (key,)
                )
                return None
            try:
                result = _decode_cached_result(json.loads(payload))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                connection.execute(
                    "DELETE FROM tool_cache_entries WHERE cache_key = ?", (key,)
                )
                return None
            remaining = max(0.0, expires_at - self.clock())
            self._memory.put(
                key,
                result,
                expires_at_monotonic=time.monotonic() + remaining,
            )
            return deepcopy(result)

    def put(self, key: str, result: ToolResult[object]) -> None:
        ttl_seconds = self.policy.ttl_for(result)
        if ttl_seconds is None:
            return
        payload = json.dumps(
            _encode_cached_result(result),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        expires_at = self.clock() + ttl_seconds
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_cache_entries (cache_key, expires_at, payload, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    payload = excluded.payload,
                    created_at = excluded.created_at
                """,
                (key, expires_at, payload, self.clock()),
            )
            connection.execute(
                "DELETE FROM tool_cache_entries WHERE expires_at <= ?", (self.clock(),)
            )
            connection.execute(
                """
                DELETE FROM tool_cache_entries
                WHERE cache_key IN (
                    SELECT cache_key FROM tool_cache_entries
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.policy.max_entries,),
            )
        self._memory.put(key, result)

    def clear(self) -> None:
        self._memory.clear()
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM tool_cache_entries")

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_cache_entries (
                    cache_key TEXT PRIMARY KEY,
                    expires_at REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tool_cache_expiry ON tool_cache_entries(expires_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


_PERSISTED_SENSITIVE_MARKERS = (
    "cookie",
    "token",
    "password",
    "authorization",
    "session",
    "phone",
    "mobile",
    "identity",
    "passport",
    "order",
)


def _versioned_cache_key(
    *,
    tool_name: str,
    request: Mapping[str, object],
    backend_name: str,
    backend_version: str,
    parser_version: str,
    output_schema_version: str,
    cache_scope: str,
) -> str:
    payload = {
        "tool_name": tool_name,
        "request": request,
        "backend_name": backend_name,
        "backend_version": backend_version,
        "parser_version": parser_version,
        "output_schema_version": output_schema_version,
        "cache_scope": cache_scope,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{tool_name}:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _encode_cached_result(result: ToolResult[object]) -> dict[str, object]:
    data: dict[str, object] | None = None
    if isinstance(result.data, TrainSearchOutput):
        data = {
            "kind": "train",
            "source": result.data.source,
            "options": [_encode_train_option(option) for option in result.data.options],
        }
    elif isinstance(result.data, FlightSearchOutput):
        data = {
            "kind": "flight",
            "source": result.data.source,
            "options": [_encode_flight_option(option) for option in result.data.options],
            # raw_state may contain browser state or captured content and is never persisted.
            "raw_state": {},
        }
    elif result.data is not None:
        raise ValueError(f"Unsupported cached output type: {type(result.data)!r}")
    return {
        "status": result.status.value,
        "data": data,
        "error": (
            {
                "code": result.error.code.value,
                "message": result.error.message,
                "retryable": result.error.retryable,
                "details": _safe_persistent_value(result.error.details),
            }
            if result.error is not None
            else None
        ),
        "warnings": list(result.warnings),
        "metrics": {
            "request_id": result.metrics.request_id,
            "started_at": result.metrics.started_at.isoformat(),
            "latency_ms": result.metrics.latency_ms,
            "cache_hit": False,
            "attempts": result.metrics.attempts,
            "backend": result.metrics.backend,
            "termination_reason": result.metrics.termination_reason,
            "retry_delays_ms": list(result.metrics.retry_delays_ms),
        },
    }


def _decode_cached_result(payload: Mapping[str, object]) -> ToolResult[object]:
    raw_data = payload.get("data")
    data: object | None = None
    if isinstance(raw_data, Mapping):
        kind = raw_data.get("kind")
        options = raw_data.get("options", [])
        if not isinstance(options, list):
            raise ValueError("Cached options must be a list.")
        if kind == "train":
            data = TrainSearchOutput(
                options=tuple(_decode_train_option(item) for item in options),
                source=str(raw_data.get("source") or "12306_mcp"),
            )
        elif kind == "flight":
            data = FlightSearchOutput(
                options=tuple(_decode_flight_option(item) for item in options),
                raw_state={},
                source=str(raw_data.get("source") or "flights.ctrip.com"),
            )
        else:
            raise ValueError("Cached output kind is unknown.")
    raw_error = payload.get("error")
    error = None
    if isinstance(raw_error, Mapping):
        error = ToolError(
            code=ToolErrorCode(str(raw_error["code"])),
            message=str(raw_error.get("message") or ""),
            retryable=bool(raw_error.get("retryable")),
            details=dict(raw_error.get("details") or {}),
        )
    raw_metrics = payload["metrics"]
    if not isinstance(raw_metrics, Mapping):
        raise ValueError("Cached metrics are missing.")
    return ToolResult(
        status=ToolStatus(str(payload["status"])),
        data=data,
        error=error,
        warnings=tuple(str(item) for item in payload.get("warnings", [])),
        metrics=ToolMetrics(
            request_id=str(raw_metrics["request_id"]),
            started_at=datetime.fromisoformat(str(raw_metrics["started_at"])),
            latency_ms=int(raw_metrics["latency_ms"]),
            cache_hit=bool(raw_metrics.get("cache_hit")),
            attempts=int(raw_metrics["attempts"]),
            backend=str(raw_metrics["backend"]),
            termination_reason=(
                str(raw_metrics["termination_reason"])
                if raw_metrics.get("termination_reason") is not None
                else None
            ),
            retry_delays_ms=tuple(int(item) for item in raw_metrics.get("retry_delays_ms", [])),
        ),
    )


def _encode_train_option(option: TrainOption) -> dict[str, object]:
    return {
        "train_code": option.train_code,
        "from_station": option.from_station,
        "from_station_code": option.from_station_code,
        "to_station": option.to_station,
        "to_station_code": option.to_station_code,
        "travel_date": option.travel_date.isoformat(),
        "start_time": option.start_time,
        "arrive_time": option.arrive_time,
        "duration": option.duration,
        "seats": option.seats,
        "prices": option.prices,
        "train_class_name": option.train_class_name,
    }


def _decode_train_option(value: object) -> TrainOption:
    if not isinstance(value, Mapping):
        raise ValueError("Cached train option is invalid.")
    return TrainOption(
        train_code=str(value["train_code"]),
        from_station=str(value["from_station"]),
        from_station_code=_optional_text(value.get("from_station_code")),
        to_station=str(value["to_station"]),
        to_station_code=_optional_text(value.get("to_station_code")),
        travel_date=date.fromisoformat(str(value["travel_date"])),
        start_time=str(value["start_time"]),
        arrive_time=str(value["arrive_time"]),
        duration=str(value["duration"]),
        seats={str(key): str(item) for key, item in dict(value.get("seats") or {}).items()},
        prices={str(key): float(item) for key, item in dict(value.get("prices") or {}).items()},
        train_class_name=_optional_text(value.get("train_class_name")),
    )


def _encode_flight_option(option: FlightOption) -> dict[str, object]:
    return {
        "origin": option.origin,
        "destination": option.destination,
        "travel_date": option.travel_date.isoformat(),
        "price": option.price,
        "currency": option.currency,
        "departure_time": _encode_datetime(option.departure_time),
        "arrival_time": _encode_datetime(option.arrival_time),
        "evidence": [_encode_flight_evidence(item) for item in option.evidence],
        "reliability": option.reliability,
        "warnings": list(option.warnings),
        "requested_origin": _encode_place(option.requested_origin),
        "requested_destination": _encode_place(option.requested_destination),
        "actual_origin": _encode_place(option.actual_origin),
        "actual_destination": _encode_place(option.actual_destination),
    }


def _decode_flight_option(value: object) -> FlightOption:
    if not isinstance(value, Mapping):
        raise ValueError("Cached flight option is invalid.")
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        raise ValueError("Cached flight evidence is invalid.")
    return FlightOption(
        origin=str(value["origin"]),
        destination=str(value["destination"]),
        travel_date=date.fromisoformat(str(value["travel_date"])),
        price=float(value["price"]),
        currency=str(value["currency"]),
        departure_time=_decode_datetime(value.get("departure_time")),
        arrival_time=_decode_datetime(value.get("arrival_time")),
        evidence=[_decode_flight_evidence(item) for item in evidence],
        reliability=str(value["reliability"]),
        warnings=[str(item) for item in value.get("warnings", [])],
        requested_origin=_decode_place(value.get("requested_origin")),
        requested_destination=_decode_place(value.get("requested_destination")),
        actual_origin=_decode_place(value.get("actual_origin")),
        actual_destination=_decode_place(value.get("actual_destination")),
    )


def _encode_flight_evidence(evidence: FlightEvidence) -> dict[str, object]:
    return {
        "source_name": evidence.source_name,
        "url": _safe_source_url(evidence.url),
        "price": evidence.price,
        "currency": evidence.currency,
        "departure_time": _encode_datetime(evidence.departure_time),
        "arrival_time": _encode_datetime(evidence.arrival_time),
        "captured_at": evidence.captured_at.isoformat(),
        "origin": evidence.origin,
        "destination": evidence.destination,
        "travel_date": evidence.travel_date.isoformat() if evidence.travel_date else None,
        "metadata": _safe_persistent_value(evidence.metadata or {}),
    }


def _decode_flight_evidence(value: object) -> FlightEvidence:
    if not isinstance(value, Mapping):
        raise ValueError("Cached flight evidence is invalid.")
    travel_date = value.get("travel_date")
    return FlightEvidence(
        source_name=str(value["source_name"]),
        url=str(value["url"]),
        price=float(value["price"]),
        currency=str(value["currency"]),
        departure_time=_decode_datetime(value.get("departure_time")),
        arrival_time=_decode_datetime(value.get("arrival_time")),
        captured_at=datetime.fromisoformat(str(value["captured_at"])),
        origin=_optional_text(value.get("origin")),
        destination=_optional_text(value.get("destination")),
        travel_date=date.fromisoformat(str(travel_date)) if travel_date else None,
        metadata=dict(value.get("metadata") or {}),
    )


def _encode_place(place: PlaceRef | None) -> dict[str, object] | None:
    if place is None:
        return None
    return {
        "raw": place.raw,
        "kind": place.kind,
        "canonical_id": place.canonical_id,
        "display_name": place.display_name,
        "city_id": place.city_id,
        "city_name": place.city_name,
        "country": place.country,
        "query_code": place.query_code,
        "airport_codes": list(place.airport_codes),
        "airport_code": place.airport_code,
        "station_code": place.station_code,
    }


def _decode_place(value: object) -> PlaceRef | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("Cached place is invalid.")
    return PlaceRef(
        raw=str(value["raw"]),
        kind=str(value["kind"]),
        canonical_id=str(value["canonical_id"]),
        display_name=str(value["display_name"]),
        city_id=_optional_text(value.get("city_id")),
        city_name=_optional_text(value.get("city_name")),
        country=_optional_text(value.get("country")),
        query_code=_optional_text(value.get("query_code")),
        airport_codes=tuple(str(item) for item in value.get("airport_codes", [])),
        airport_code=_optional_text(value.get("airport_code")),
        station_code=_optional_text(value.get("station_code")),
    )


def _safe_persistent_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): (
                "[REDACTED]"
                if any(marker in str(key).casefold() for marker in _PERSISTED_SENSITIVE_MARKERS)
                else _safe_persistent_value(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe_persistent_value(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _safe_source_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _encode_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _decode_datetime(value: object) -> datetime | None:
    return datetime.fromisoformat(str(value)) if value else None


def _optional_text(value: object) -> str | None:
    return str(value) if value is not None else None


class LegacyTrainProvider(Protocol):
    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        """Legacy provider contract."""


class CachedTrainSearchTool:
    def __init__(
        self,
        provider: LegacyTrainProvider,
        *,
        cache: ToolCache | None = None,
        backend_name: str = "12306_mcp",
        backend_version: str = "12306-mcp-v1",
        parser_version: str = "train-option-v1",
        output_schema_version: str = "tool-result-v1",
        cache_scope: str = "public",
        runtime: ToolRuntime | None = None,
    ) -> None:
        self.provider = provider
        self.cache = cache or InMemoryToolCache()
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.parser_version = parser_version
        self.output_schema_version = output_schema_version
        self.cache_scope = cache_scope
        self.runtime = runtime or ToolRuntime()

    def search(
        self,
        request: TrainSearchRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult[TrainSearchOutput]:
        return self.search_many([request], context=context)[0]

    def search_many(
        self,
        requests: Sequence[TrainSearchRequest],
        *,
        context: ToolExecutionContext | None = None,
    ) -> list[ToolResult[TrainSearchOutput]]:
        results: dict[str, ToolResult[TrainSearchOutput]] = {}
        for request in requests:
            if request.request_id in results:
                continue
            cache_key = self._cache_key(request)
            cached = self.cache.get(cache_key)
            if cached is not None:
                results[request.request_id] = _as_cache_hit(cached)
                continue
            started_at = _utc_now()
            started = time.monotonic()
            execution = self.runtime.execute(
                lambda: tuple(
                    sorted(
                        self.provider.query_train_options(request.to_intent()),
                        key=_train_sort_key,
                    )[: request.max_results]
                ),
                context=context,
            )
            if execution.value is not None:
                options = execution.value
                result = ToolResult(
                    status=ToolStatus.SUCCESS if options else ToolStatus.NO_RESULTS,
                    data=TrainSearchOutput(options=options),
                    metrics=_metrics(
                        request.request_id,
                        started_at,
                        started,
                        backend=self.backend_name,
                        attempts=execution.attempts,
                        termination_reason=execution.termination_reason,
                        retry_delays_ms=execution.retry_delays_ms,
                    ),
                )
            else:
                result = ToolResult(
                    status=ToolStatus.ERROR,
                    data=None,
                    error=execution.error,
                    metrics=_metrics(
                        request.request_id,
                        started_at,
                        started,
                        backend=self.backend_name,
                        attempts=execution.attempts,
                        termination_reason=execution.termination_reason,
                        retry_delays_ms=execution.retry_delays_ms,
                    ),
                )
            self.cache.put(cache_key, result)
            results[request.request_id] = result
        return [results[request.request_id] for request in requests]

    def clear_cache(self) -> None:
        self.cache.clear()

    def _cache_key(self, request: TrainSearchRequest) -> str:
        return _versioned_cache_key(
            tool_name="train_search",
            request=request.as_payload(),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            parser_version=self.parser_version,
            output_schema_version=self.output_schema_version,
            cache_scope=self.cache_scope,
        )


FlightBatchRunner = Callable[
    [Sequence[FlightSearchRequest]],
    list[ToolResult[FlightSearchOutput]],
]


class CachedFlightSearchTool:
    def __init__(
        self,
        runner: FlightBatchRunner,
        *,
        cache: ToolCache | None = None,
        backend_name: str = "ctrip_react",
        backend_version: str = "ctrip-react-v1",
        parser_version: str = "flight-evidence-v1",
        output_schema_version: str = "tool-result-v1",
        cache_scope: str = "public",
        runtime: ToolRuntime | None = None,
    ) -> None:
        self.runner = runner
        self.cache = cache or InMemoryToolCache()
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.parser_version = parser_version
        self.output_schema_version = output_schema_version
        self.cache_scope = cache_scope
        self.runtime = runtime or ToolRuntime()

    def search(
        self,
        request: FlightSearchRequest,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult[FlightSearchOutput]:
        return self.search_many([request], context=context)[0]

    def search_many(
        self,
        requests: Sequence[FlightSearchRequest],
        *,
        context: ToolExecutionContext | None = None,
    ) -> list[ToolResult[FlightSearchOutput]]:
        results: dict[str, ToolResult[FlightSearchOutput]] = {}
        unique_misses: list[FlightSearchRequest] = []
        for request in requests:
            if request.request_id in results or any(
                item.request_id == request.request_id for item in unique_misses
            ):
                continue
            cached = self.cache.get(self._cache_key(request))
            if cached is not None:
                results[request.request_id] = _as_cache_hit(cached)
            else:
                unique_misses.append(request)

        if unique_misses:
            started_at = _utc_now()
            started = time.monotonic()
            execution = self.runtime.execute(
                lambda: self._run_batch(unique_misses),
                context=context,
            )
            if execution.value is None:
                fresh = [
                    self._runtime_error_result(
                        request,
                        execution.error,
                        started_at,
                        started,
                        execution,
                    )
                    for request in unique_misses
                ]
            else:
                fresh = []
                for request, initial in zip(
                    unique_misses,
                    execution.value,
                    strict=True,
                ):
                    retried = self.runtime.retry_result(
                        initial,
                        lambda request=request: self._run_batch([request])[0],
                        context=context,
                        initial_attempts=execution.attempts,
                        initial_retry_delays_ms=execution.retry_delays_ms,
                    )
                    if retried.value is None:
                        result = self._runtime_error_result(
                            request,
                            retried.error,
                            started_at,
                            started,
                            retried,
                        )
                    else:
                        result = replace(
                            retried.value,
                            metrics=_metrics(
                                request.request_id,
                                started_at,
                                started,
                                backend=self.backend_name,
                                attempts=retried.attempts,
                                termination_reason=retried.termination_reason,
                                retry_delays_ms=retried.retry_delays_ms,
                            ),
                        )
                    fresh.append(result)
            for request, result in zip(unique_misses, fresh, strict=True):
                self.cache.put(self._cache_key(request), result)
                results[request.request_id] = result

        return [results[request.request_id] for request in requests]

    def clear_cache(self) -> None:
        self.cache.clear()

    def _cache_key(self, request: FlightSearchRequest) -> str:
        return _versioned_cache_key(
            tool_name="flight_search",
            request=request.as_payload(),
            backend_name=self.backend_name,
            backend_version=self.backend_version,
            parser_version=self.parser_version,
            output_schema_version=self.output_schema_version,
            cache_scope=self.cache_scope,
        )

    def _run_batch(
        self,
        requests: Sequence[FlightSearchRequest],
    ) -> list[ToolResult[FlightSearchOutput]]:
        fresh = self.runner(requests)
        if len(fresh) != len(requests):
            raise RuntimeError(
                "Flight tool runner returned a different number of results than requests."
            )
        return fresh

    def _runtime_error_result(
        self,
        request: FlightSearchRequest,
        error: ToolError | None,
        started_at: datetime,
        started_monotonic: float,
        execution: RuntimeExecution[object],
    ) -> ToolResult[FlightSearchOutput]:
        return ToolResult(
            status=ToolStatus.ERROR,
            data=None,
            error=error
            or ToolError(
                ToolErrorCode.INTERNAL_ERROR,
                "Tool runtime ended without a result or error.",
                False,
            ),
            metrics=_metrics(
                request.request_id,
                started_at,
                started_monotonic,
                backend=self.backend_name,
                attempts=execution.attempts,
                termination_reason=execution.termination_reason,
                retry_delays_ms=execution.retry_delays_ms,
            ),
        )


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
    attempts: int = 1,
    termination_reason: str | None = None,
    retry_delays_ms: tuple[int, ...] = (),
) -> ToolMetrics:
    return ToolMetrics(
        request_id=request_id,
        started_at=started_at,
        latency_ms=max(0, round((time.monotonic() - started_monotonic) * 1000)),
        cache_hit=False,
        attempts=attempts,
        backend=backend,
        termination_reason=termination_reason,
        retry_delays_ms=retry_delays_ms,
    )


def _termination_reason(error: ToolError) -> str:
    if error.code == ToolErrorCode.CANCELLED:
        return "cancelled"
    if error.code == ToolErrorCode.TIMEOUT:
        return "deadline_exhausted"
    return "non_retryable_error"


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
                "requested_origin": _public_place(option.requested_origin),
                "requested_destination": _public_place(option.requested_destination),
                "actual_origin": _public_place(option.actual_origin),
                "actual_destination": _public_place(option.actual_destination),
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


def _public_place(place) -> dict[str, object] | None:
    if place is None:
        return None
    return {
        "raw": place.raw,
        "kind": place.kind,
        "canonical_id": place.canonical_id,
        "display_name": place.display_name,
        "city_id": place.city_id,
        "city_name": place.city_name,
        "country": place.country,
        "query_code": place.query_code,
        "airport_code": place.airport_code,
        "airport_codes": list(place.airport_codes),
        "station_code": place.station_code,
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
            "termination_reason": result.metrics.termination_reason,
            "retry_delays_ms": list(result.metrics.retry_delays_ms),
        },
    }
