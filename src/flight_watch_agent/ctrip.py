from __future__ import annotations

import gzip
import hashlib
import json
import os
import atexit
import threading
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from .models import FlightEvidence, FlightPageAttemptResult, FlightSearchIntent, SearchResult


CTRIP_SCHEME = "ctrip-selenium"
DEFAULT_CTRIP_COOKIES_FILE = "data/ctrip_cookies.json"
REQUIRED_CTRIP_COOKIES = ["AHeadUserInfo", "DUID", "IsNonUser", "_udl", "cticket", "login_type", "login_uid"]


class CtripManualVerificationRequired(RuntimeError):
    pass


class CtripRouteSearchTool:
    def search(self, query: str) -> list[SearchResult]:
        intent = _intent_from_query(query)
        if intent is None:
            return []
        url = build_ctrip_selenium_url(intent)
        return [
            SearchResult(
                title=f"Ctrip Selenium route {intent.origin} to {intent.destination}",
                url=url,
                snippet="Constructed Ctrip SeleniumWire crawl task.",
                source_name="flights.ctrip.com",
            )
        ]


@dataclass(frozen=True)
class RawResponseRef:
    capture_id: str
    source_url: str
    captured_at: datetime
    parser_version: str


@dataclass(frozen=True)
class CapturedCtripResponse:
    payload: dict[str, Any]
    source_url: str
    captured_at: datetime
    response_ref: RawResponseRef


@dataclass(frozen=True)
class ParsedCtripItinerary:
    itinerary_id: str | None
    segments: tuple[dict[str, object], ...]
    transfer_count: int
    price: float


class RawResponseStore(Protocol):
    def store(
        self,
        payload: Mapping[str, Any],
        *,
        source_url: str,
        captured_at: datetime,
        parser_version: str,
    ) -> RawResponseRef:
        """Store a redacted capture and return its stable reference."""


class CtripNavigatorProtocol(Protocol):
    def search_urls(self, intent: FlightSearchIntent) -> list[str]:
        """Return supported direct Ctrip search URLs in preference order."""

    def navigate(
        self,
        driver: object,
        intent: FlightSearchIntent,
        *,
        entrypoint: str,
        force_refresh: bool = False,
    ) -> str:
        """Navigate a browser to one Ctrip entrypoint and return its source URL."""


class BrowserSessionManagerProtocol(Protocol):
    def acquire(self) -> tuple[object, bool]:
        """Borrow a browser and indicate whether it was newly created."""

    def prepare(self, driver: object, *, is_new: bool) -> None:
        """Prepare a borrowed browser session for a query."""

    def release(self, driver: object) -> None:
        """Return or dispose a borrowed browser."""

    def close(self) -> None:
        """Dispose all managed browser resources."""


class CtripCaptureBackendProtocol(Protocol):
    def clear(self, driver: object) -> None:
        """Clear captured browser requests before navigation."""

    def capture(
        self,
        driver: object,
        *,
        source_url: str,
        timeout_seconds: int,
        manual_verification_wait_seconds: int,
    ) -> CapturedCtripResponse:
        """Capture one raw batchSearch response without parsing it."""


class InMemoryRawResponseStore:
    """Redacted in-process response store used until P1.5 adds durable recording."""

    _SENSITIVE_FIELD_MARKERS = (
        "cookie",
        "token",
        "password",
        "phone",
        "mobile",
        "identity",
        "passport",
        "order",
    )

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def store(
        self,
        payload: Mapping[str, Any],
        *,
        source_url: str,
        captured_at: datetime,
        parser_version: str,
    ) -> RawResponseRef:
        redacted = _redact_capture_value(dict(payload), self._SENSITIVE_FIELD_MARKERS)
        encoded = json.dumps(redacted, ensure_ascii=False, sort_keys=True, default=str)
        capture_id = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        reference = RawResponseRef(
            capture_id=capture_id,
            source_url=source_url,
            captured_at=captured_at,
            parser_version=parser_version,
        )
        with self._lock:
            self._records[capture_id] = redacted
        return reference

    def get(self, reference: RawResponseRef) -> dict[str, Any] | None:
        with self._lock:
            record = self._records.get(reference.capture_id)
            return json.loads(json.dumps(record)) if record is not None else None


class CtripPayloadParser:
    """Pure batchSearch payload parser: no browser, environment or network access."""

    version = "ctrip-payload-v1"

    def parse(self, payload: Mapping[str, Any]) -> list[ParsedCtripItinerary]:
        raw_data = payload.get("data")
        data = raw_data if isinstance(raw_data, Mapping) else {}
        raw_itineraries = data.get("flightItineraryList")
        if not isinstance(raw_itineraries, list):
            return []
        parsed: list[ParsedCtripItinerary] = []
        for raw_itinerary in raw_itineraries:
            if not isinstance(raw_itinerary, dict):
                continue
            raw_segments = raw_itinerary.get("flightSegments")
            segments = raw_segments if isinstance(raw_segments, list) else []
            itinerary_segments = _itinerary_segments(segments)
            if not itinerary_segments:
                continue
            raw_price_list = raw_itinerary.get("priceList")
            price = _lowest_price(raw_price_list if isinstance(raw_price_list, list) else [])
            if price is None:
                continue
            parsed.append(
                ParsedCtripItinerary(
                    itinerary_id=str(raw_itinerary.get("itineraryId") or "") or None,
                    segments=tuple(itinerary_segments),
                    transfer_count=_transfer_count(segments, itinerary_segments),
                    price=price,
                )
            )
        return parsed


class FlightEvidenceMapper:
    """Maps parsed Ctrip itineraries to domain evidence without browser access."""

    def map(
        self,
        itineraries: Sequence[ParsedCtripItinerary],
        intent: FlightSearchIntent,
        *,
        source_url: str,
        captured_at: datetime,
        direct_only: bool,
        max_results: int,
        response_ref: RawResponseRef | None = None,
    ) -> list[FlightEvidence]:
        evidence: list[FlightEvidence] = []
        for itinerary in itineraries:
            if direct_only and itinerary.transfer_count != 0:
                continue
            first_segment = itinerary.segments[0]
            last_segment = itinerary.segments[-1]
            if not _ctrip_airport_matches_request(
                first_segment.get("departure_airport_code"), intent.origin
            ):
                continue
            if not _ctrip_airport_matches_request(
                last_segment.get("arrival_airport_code"), intent.destination
            ):
                continue
            departure_time = _parse_ctrip_datetime(
                first_segment.get("departure_time"),
                airport_code=first_segment.get("departure_airport_code"),
            )
            metadata = _flight_metadata(
                itinerary_segments=list(itinerary.segments),
                transfer_count=itinerary.transfer_count,
                itinerary={"itineraryId": itinerary.itinerary_id},
            )
            if response_ref is not None:
                metadata["capture_ref"] = response_ref.capture_id
                metadata["capture_parser_version"] = response_ref.parser_version
            evidence.append(
                FlightEvidence(
                    source_name="flights.ctrip.com",
                    url=source_url,
                    price=itinerary.price,
                    currency=intent.currency or "CNY",
                    departure_time=departure_time,
                    arrival_time=_parse_ctrip_datetime(
                        last_segment.get("arrival_time"),
                        airport_code=last_segment.get("arrival_airport_code"),
                    ),
                    captured_at=captured_at,
                    origin=intent.origin,
                    destination=intent.destination,
                    travel_date=(
                        departure_time.date() if departure_time else intent.travel_date
                    ),
                    metadata=metadata,
                )
            )
        return _rank_ctrip_evidence(evidence, intent)[:max_results]


class CtripNavigator:
    """Browser navigation boundary for Ctrip search entrypoints."""

    HOMEPAGE_URL = "https://flights.ctrip.com/online/channel/domestic"

    def __init__(self, *, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def search_urls(self, intent: FlightSearchIntent) -> list[str]:
        return _build_ctrip_search_urls(intent)

    def navigate(
        self,
        driver: object,
        intent: FlightSearchIntent,
        *,
        entrypoint: str,
        force_refresh: bool = False,
    ) -> str:
        search_urls = self.search_urls(intent)
        if entrypoint == "international":
            source_url = search_urls[0]
            driver.get(source_url)
        elif entrypoint == "online_list":
            source_url = search_urls[1]
            driver.get(source_url)
        elif entrypoint == "homepage":
            source_url = self.HOMEPAGE_URL
            _drive_ctrip_homepage_search(
                driver,
                intent,
                timeout_seconds=self.timeout_seconds,
            )
        else:
            raise ValueError(f"Unsupported Ctrip entrypoint: {entrypoint}")
        if force_refresh and entrypoint != "homepage":
            driver.refresh()
        return source_url


class BrowserSessionManager:
    """Owns browser lifecycle and optional Ctrip authentication state."""

    def __init__(
        self,
        *,
        browser: str,
        headless: bool,
        reuse_browser_session: bool,
        login_allowed: bool,
        accounts: Sequence[str],
        passwords: Sequence[str],
        cookies_file: Path,
        timeout_seconds: int,
        login_wait_seconds: int,
        driver_factory: Callable[..., object] | None = None,
    ) -> None:
        self.browser = browser
        self.headless = headless
        self.reuse_browser_session = reuse_browser_session
        self.login_allowed = login_allowed
        self.accounts = list(accounts)
        self.passwords = list(passwords)
        self.cookies_file = cookies_file
        self.timeout_seconds = timeout_seconds
        self.login_wait_seconds = login_wait_seconds
        self._driver_factory = driver_factory or _init_seleniumwire_driver
        self._driver: object | None = None
        self._driver_ready = False
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        return self._driver_ready

    def acquire(self) -> tuple[object, bool]:
        if not self.reuse_browser_session:
            return self._driver_factory(browser=self.browser, headless=self.headless), True
        with self._lock:
            if self._driver is None:
                self._driver = self._driver_factory(
                    browser=self.browser,
                    headless=self.headless,
                )
                return self._driver, True
            return self._driver, False

    def prepare(self, driver: object, *, is_new: bool) -> None:
        if self.login_allowed and (is_new or not self._driver_ready):
            _CtripLoginSession(
                accounts=self.accounts,
                passwords=self.passwords,
                cookies_file=self.cookies_file,
                timeout_seconds=self.timeout_seconds,
                login_wait_seconds=self.login_wait_seconds,
            ).ensure_login(driver)
        if self.reuse_browser_session:
            self._driver_ready = True

    def release(self, driver: object) -> None:
        if self.reuse_browser_session:
            return
        _quit_driver(driver)

    def close(self) -> None:
        with self._lock:
            driver = self._driver
            self._driver = None
            self._driver_ready = False
        if driver is not None:
            _quit_driver(driver)


class CtripCaptureBackend:
    """Captures batchSearch responses and writes a redacted response reference."""

    def __init__(
        self,
        *,
        response_store: RawResponseStore | None = None,
        parser_version: str = CtripPayloadParser.version,
    ) -> None:
        self.response_store = response_store or InMemoryRawResponseStore()
        self.parser_version = parser_version

    def clear(self, driver: object) -> None:
        try:
            del driver.requests
        except AttributeError:
            pass

    def capture(
        self,
        driver: object,
        *,
        source_url: str,
        timeout_seconds: int,
        manual_verification_wait_seconds: int,
    ) -> CapturedCtripResponse:
        payload = _wait_for_ctrip_search_payload(
            driver,
            timeout_seconds=timeout_seconds,
            manual_verification_wait_seconds=manual_verification_wait_seconds,
        )
        captured_at = datetime.now(timezone.utc)
        response_ref = self.response_store.store(
            payload,
            source_url=source_url,
            captured_at=captured_at,
            parser_version=self.parser_version,
        )
        return CapturedCtripResponse(
            payload=payload,
            source_url=source_url,
            captured_at=captured_at,
            response_ref=response_ref,
        )


EvidenceFactory = Callable[..., list[FlightEvidence]]


class CtripFlightBackend:
    """Composes navigation, browser sessions, capture, parsing and evidence mapping."""

    def __init__(
        self,
        *,
        navigator: CtripNavigatorProtocol,
        session_manager: BrowserSessionManagerProtocol,
        capture_backend: CtripCaptureBackendProtocol,
        payload_parser: CtripPayloadParser | None = None,
        evidence_mapper: FlightEvidenceMapper | None = None,
        evidence_factory: EvidenceFactory | None = None,
        timeout_seconds: int = 30,
        direct_only: bool = False,
        max_results: int = 5,
        manual_verification_wait_seconds: int = 0,
    ) -> None:
        self.navigator = navigator
        self.session_manager = session_manager
        self.capture_backend = capture_backend
        self.payload_parser = payload_parser or CtripPayloadParser()
        self.evidence_mapper = evidence_mapper or FlightEvidenceMapper()
        self.evidence_factory = evidence_factory
        self.timeout_seconds = timeout_seconds
        self.direct_only = direct_only
        self.max_results = max_results
        self.manual_verification_wait_seconds = manual_verification_wait_seconds
        self._preferred_search_url_index: int | None = None

    def close(self) -> None:
        self.session_manager.close()

    def extract(self, intent: FlightSearchIntent) -> list[FlightEvidence]:
        driver, is_new_driver = self.session_manager.acquire()
        errors: list[str] = []
        fallback_evidence: list[FlightEvidence] = []
        try:
            self.session_manager.prepare(driver, is_new=is_new_driver)
            search_urls = self.navigator.search_urls(intent)
            indexed_entrypoints = list(enumerate(("international", "online_list")))
            if (
                self._preferred_search_url_index is not None
                and 0 <= self._preferred_search_url_index < len(indexed_entrypoints)
            ):
                preferred = self._preferred_search_url_index
                indexed_entrypoints.sort(
                    key=lambda item: 0 if item[0] == preferred else 1
                )
            for search_url_index, entrypoint in indexed_entrypoints:
                try:
                    evidence, source_url = self._capture_and_map(
                        driver,
                        intent,
                        entrypoint=entrypoint,
                    )
                    if evidence:
                        self._preferred_search_url_index = search_url_index
                        if _evidence_satisfies_requested_time(evidence, intent):
                            return evidence
                        if not fallback_evidence:
                            fallback_evidence = evidence
                        errors.append(
                            f"{source_url}:NoTimePreferenceMatch:{intent.time_preference}"
                        )
                    else:
                        errors.append(
                            f"{search_urls[search_url_index]}:NoFlightEvidence:"
                            "batchSearch returned no parsable itineraries"
                        )
                except Exception as exc:
                    errors.append(_attempt_error_text(entrypoint, exc))

            try:
                evidence, source_url = self._capture_and_map(
                    driver,
                    intent,
                    entrypoint="homepage",
                )
                if evidence:
                    if _evidence_satisfies_requested_time(evidence, intent):
                        return evidence
                    if not fallback_evidence:
                        fallback_evidence = evidence
                    errors.append(
                        f"{source_url}:NoTimePreferenceMatch:{intent.time_preference}"
                    )
                else:
                    errors.append(
                        "homepage_ui:NoFlightEvidence:"
                        "batchSearch returned no parsable itineraries"
                    )
            except Exception as exc:
                errors.append(_attempt_error_text("homepage_ui", exc))
            if fallback_evidence:
                return fallback_evidence
            if errors:
                raise RuntimeError(
                    "Ctrip SeleniumWire extraction failed; attempts="
                    + " | ".join(errors)
                )
            return []
        finally:
            self.session_manager.release(driver)

    def extract_attempt(
        self,
        intent: FlightSearchIntent,
        *,
        entrypoint: str,
        action_id: str,
        force_refresh: bool = False,
    ) -> FlightPageAttemptResult:
        del action_id
        driver, is_new_driver = self.session_manager.acquire()
        source_url: str | None = None
        try:
            self.session_manager.prepare(driver, is_new=is_new_driver)
            evidence, source_url = self._capture_and_map(
                driver,
                intent,
                entrypoint=entrypoint,
                force_refresh=force_refresh,
                manual_verification_wait_seconds=0,
            )
            if not evidence:
                return FlightPageAttemptResult(
                    status="no_evidence",
                    evidence=[],
                    entrypoint=entrypoint,
                    source_url=source_url,
                    warning="batchSearch returned no parsable itineraries",
                )
            status = (
                "success"
                if _evidence_satisfies_requested_time(evidence, intent)
                else "time_preference_mismatch"
            )
            return FlightPageAttemptResult(
                status=status,
                evidence=evidence,
                entrypoint=entrypoint,
                source_url=source_url,
            )
        except CtripManualVerificationRequired as exc:
            return FlightPageAttemptResult(
                status="captcha_required",
                evidence=[],
                entrypoint=entrypoint,
                source_url=source_url,
                warning=str(exc),
            )
        except Exception as exc:
            return FlightPageAttemptResult(
                status=_ctrip_attempt_status(exc),
                evidence=[],
                entrypoint=entrypoint,
                source_url=source_url,
                warning=f"{type(exc).__name__}:{str(exc).split('Stacktrace:')[0]}",
            )
        finally:
            self.session_manager.release(driver)

    def _capture_and_map(
        self,
        driver: object,
        intent: FlightSearchIntent,
        *,
        entrypoint: str,
        force_refresh: bool = False,
        manual_verification_wait_seconds: int | None = None,
    ) -> tuple[list[FlightEvidence], str]:
        self.capture_backend.clear(driver)
        source_url = self.navigator.navigate(
            driver,
            intent,
            entrypoint=entrypoint,
            force_refresh=force_refresh,
        )
        effective_manual_wait = (
            self.manual_verification_wait_seconds
            if manual_verification_wait_seconds is None
            else manual_verification_wait_seconds
        )
        capture = self.capture_backend.capture(
            driver,
            source_url=source_url,
            timeout_seconds=self.timeout_seconds,
            manual_verification_wait_seconds=effective_manual_wait,
        )
        evidence = self._map_capture(capture, intent)
        if (
            not evidence
            and effective_manual_wait > 0
            and _is_manual_verification_present(driver)
        ):
            _wait_for_manual_verification(
                driver,
                effective_manual_wait,
            )
            self.capture_backend.clear(driver)
            capture = self.capture_backend.capture(
                driver,
                source_url=source_url,
                timeout_seconds=self.timeout_seconds,
                manual_verification_wait_seconds=0,
            )
            evidence = self._map_capture(capture, intent)
        return evidence, source_url

    def _map_capture(
        self,
        capture: CapturedCtripResponse,
        intent: FlightSearchIntent,
    ) -> list[FlightEvidence]:
        if self.evidence_factory is not None:
            return self.evidence_factory(
                capture.payload,
                intent,
                source_url=capture.source_url,
                direct_only=self.direct_only,
                max_results=self.max_results,
                captured_at=capture.captured_at,
                response_ref=capture.response_ref,
            )
        return self.evidence_mapper.map(
            self.payload_parser.parse(capture.payload),
            intent,
            source_url=capture.source_url,
            captured_at=capture.captured_at,
            direct_only=self.direct_only,
            max_results=self.max_results,
            response_ref=capture.response_ref,
        )


class CtripSeleniumWirePageExtractor:
    """Compatibility PageExtractor adapter backed by composable Ctrip components."""

    def __init__(
        self,
        *,
        browser: str = "edge",
        headless: bool = True,
        timeout_seconds: int = 30,
        direct_only: bool = False,
        max_results: int = 5,
        login_allowed: bool = False,
        accounts: list[str] | None = None,
        passwords: list[str] | None = None,
        cookies_file: str | os.PathLike[str] = DEFAULT_CTRIP_COOKIES_FILE,
        login_wait_seconds: int = 300,
        manual_verification_wait_seconds: int = 0,
        reuse_browser_session: bool = True,
        backend: CtripFlightBackend | None = None,
    ) -> None:
        self.browser = browser
        self.headless = headless
        self.timeout_seconds = timeout_seconds
        self.direct_only = direct_only
        self.max_results = max_results
        self.login_allowed = login_allowed
        self.accounts = accounts or []
        self.passwords = passwords or []
        self.cookies_file = Path(cookies_file)
        self.login_wait_seconds = login_wait_seconds
        self.manual_verification_wait_seconds = manual_verification_wait_seconds
        self.reuse_browser_session = reuse_browser_session
        self.backend = backend or CtripFlightBackend(
            navigator=CtripNavigator(timeout_seconds=timeout_seconds),
            session_manager=BrowserSessionManager(
                browser=browser,
                headless=headless,
                reuse_browser_session=reuse_browser_session,
                login_allowed=login_allowed,
                accounts=self.accounts,
                passwords=self.passwords,
                cookies_file=self.cookies_file,
                timeout_seconds=timeout_seconds,
                login_wait_seconds=login_wait_seconds,
            ),
            capture_backend=CtripCaptureBackend(),
            # Keep the historical parser symbol patchable while it delegates
            # to the pure parser and mapper above in normal operation.
            evidence_factory=_legacy_payload_to_evidence,
            timeout_seconds=timeout_seconds,
            direct_only=direct_only,
            max_results=max_results,
            manual_verification_wait_seconds=manual_verification_wait_seconds,
        )
        atexit.register(self.close)

    def supports(self, url: str) -> bool:
        return urllib.parse.urlparse(url).scheme == CTRIP_SCHEME

    def close(self) -> None:
        self.backend.close()

    def extract(self, url: str) -> list[FlightEvidence]:
        return self.backend.extract(parse_ctrip_selenium_url(url))

    def extract_attempt(
        self,
        url: str,
        *,
        entrypoint: str,
        action_id: str,
        force_refresh: bool = False,
    ) -> FlightPageAttemptResult:
        return self.backend.extract_attempt(
            parse_ctrip_selenium_url(url),
            entrypoint=entrypoint,
            action_id=action_id,
            force_refresh=force_refresh,
        )


def _legacy_payload_to_evidence(
    payload: dict[str, Any],
    intent: FlightSearchIntent,
    **kwargs: object,
) -> list[FlightEvidence]:
    return parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url=str(kwargs["source_url"]),
        direct_only=bool(kwargs["direct_only"]),
        max_results=int(kwargs["max_results"]),
        captured_at=kwargs.get("captured_at") if isinstance(kwargs.get("captured_at"), datetime) else None,
        response_ref=kwargs.get("response_ref") if isinstance(kwargs.get("response_ref"), RawResponseRef) else None,
    )


def _attempt_error_text(entrypoint: str, exc: Exception) -> str:
    return f"{entrypoint}:{type(exc).__name__}:{str(exc).split('Stacktrace:')[0]}"


def _quit_driver(driver: object) -> None:
    try:
        driver.quit()
    except Exception:
        pass


def build_ctrip_selenium_url(intent: FlightSearchIntent) -> str:
    params = {
        "origin": intent.origin,
        "destination": intent.destination,
        "travel_date": intent.travel_date.isoformat(),
        "currency": intent.currency,
    }
    if intent.time_preference:
        params["time_preference"] = intent.time_preference
    query = urllib.parse.urlencode(params)
    return f"{CTRIP_SCHEME}://flight?{query}"


def parse_ctrip_selenium_url(url: str) -> FlightSearchIntent:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != CTRIP_SCHEME or parsed.netloc != "flight":
        raise ValueError(f"Unsupported Ctrip Selenium URL: {url}")
    query = urllib.parse.parse_qs(parsed.query)
    return FlightSearchIntent(
        origin=_required_query(query, "origin"),
        destination=_required_query(query, "destination"),
        travel_date=datetime.fromisoformat(_required_query(query, "travel_date")).date(),
        time_preference=query.get("time_preference", [None])[0] or None,
        currency=query.get("currency", ["CNY"])[0],
    )


def decode_ctrip_response_body(body: bytes) -> dict[str, Any]:
    try:
        text = gzip.decompress(body).decode("utf-8")
    except OSError:
        text = body.decode("utf-8")
    return json.loads(text)


def _redact_capture_value(
    value: object,
    sensitive_markers: Sequence[str],
) -> object:
    if isinstance(value, dict):
        output: dict[str, object] = {}
        for key, item in value.items():
            if any(marker in str(key).casefold() for marker in sensitive_markers):
                output[str(key)] = "[REDACTED]"
            else:
                output[str(key)] = _redact_capture_value(item, sensitive_markers)
        return output
    if isinstance(value, list):
        return [_redact_capture_value(item, sensitive_markers) for item in value]
    return value


def _wait_for_ctrip_search_payload(
    driver,
    *,
    timeout_seconds: int,
    manual_verification_wait_seconds: int,
) -> dict[str, Any]:
    if _is_manual_verification_present(driver):
        if manual_verification_wait_seconds > 0:
            _wait_for_manual_verification(driver, manual_verification_wait_seconds)
        else:
            raise CtripManualVerificationRequired("Ctrip manual verification is required.")

    try:
        request = driver.wait_for_request(
            "/international/search/api/search/batchSearch?.*",
            timeout=timeout_seconds,
        )
        if request.response is not None:
            return decode_ctrip_response_body(request.response.body)
    except Exception as exc:
        initial_error = exc
    else:
        initial_error = RuntimeError("batchSearch request had no response body.")

    payload = _extract_latest_ctrip_search_payload(driver)
    if payload is not None:
        return payload

    if manual_verification_wait_seconds <= 0:
        raise initial_error

    _wait_for_manual_verification(driver, manual_verification_wait_seconds)
    payload = _extract_latest_ctrip_search_payload(driver)
    if payload is not None:
        return payload

    request = driver.wait_for_request(
        "/international/search/api/search/batchSearch?.*",
        timeout=timeout_seconds,
    )
    if request.response is None:
        raise RuntimeError("batchSearch request had no response body after manual verification.")
    return decode_ctrip_response_body(request.response.body)


def _ctrip_attempt_status(exc: Exception) -> str:
    text = str(exc).casefold()
    if "login" in text or "account" in text or "password" in text:
        return "login_required"
    if "payload" in text or "batchsearch" in text or "timed out" in text:
        return "no_payload"
    if isinstance(exc, (ValueError, KeyError, TypeError, json.JSONDecodeError)):
        return "parse_failed"
    return "tool_error"


def _extract_latest_ctrip_search_payload(driver) -> dict[str, Any] | None:
    requests = list(getattr(driver, "requests", []) or [])
    for request in reversed(requests):
        url = getattr(request, "url", "")
        response = getattr(request, "response", None)
        if response is None or "batchSearch" not in url:
            continue
        try:
            return decode_ctrip_response_body(response.body)
        except Exception:
            continue
    return None


def _wait_for_manual_verification(driver, wait_seconds: int) -> None:
    if wait_seconds <= 0:
        return
    print(
        "Ctrip may require manual verification. "
        f"Please complete it in the browser within {wait_seconds} seconds..."
    )
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if not _is_manual_verification_present(driver):
            time.sleep(2)
            return
        time.sleep(1)


def _is_manual_verification_present(driver) -> bool:
    try:
        page_source = driver.page_source or ""
    except Exception:
        return False
    verification_markers = (
        "为保障您的安全访问",
        "请完成以下操作",
        "依次点击图标验证",
        "安全访问",
    )
    if any(marker in page_source for marker in verification_markers):
        return True
    try:
        return bool(
            driver.execute_script(
                """
                const text = document.body ? document.body.innerText : "";
                return text.includes("为保障您的安全访问")
                  || text.includes("请完成以下操作")
                  || text.includes("依次点击图标验证");
                """
            )
        )
    except Exception:
        return False


def parse_ctrip_batch_search_payload(
    payload: dict[str, Any],
    intent: FlightSearchIntent,
    *,
    source_url: str,
    direct_only: bool,
    max_results: int,
    captured_at: datetime | None = None,
    response_ref: RawResponseRef | None = None,
) -> list[FlightEvidence]:
    return FlightEvidenceMapper().map(
        CtripPayloadParser().parse(payload),
        intent,
        source_url=source_url,
        captured_at=captured_at or datetime.now(timezone.utc),
        direct_only=direct_only,
        max_results=max_results,
        response_ref=response_ref,
    )


def _intent_from_query(query: str) -> FlightSearchIntent | None:
    from .places import normalise_airport_code

    pieces = [piece for piece in query.split() if piece]
    if len(pieces) < 3:
        return None
    travel_date = None
    for piece in pieces:
        try:
            travel_date = datetime.fromisoformat(piece).date()
            break
        except ValueError:
            continue
    if travel_date is None:
        return None
    date_index = pieces.index(travel_date.isoformat())
    if date_index < 2:
        return None
    return FlightSearchIntent(
        origin=normalise_airport_code(pieces[0]) or pieces[0],
        destination=normalise_airport_code(pieces[1]) or pieces[1],
        travel_date=travel_date,
        time_preference=_time_preference_from_query_pieces(pieces),
    )


def _build_ctrip_search_urls(intent: FlightSearchIntent) -> list[str]:
    origin = intent.origin.lower()
    destination = intent.destination.lower()
    international_query = urllib.parse.urlencode(
        {
            "depdate": intent.travel_date.isoformat(),
            "cabin": "y_s",
            "adult": "1",
            "child": "0",
            "infant": "0",
        }
    )
    legacy_query = urllib.parse.urlencode({"depdate": intent.travel_date.isoformat()})
    return [
        f"https://flights.ctrip.com/international/search/oneway-{origin}-{destination}?{international_query}",
        f"https://flights.ctrip.com/online/list/oneway-{origin}-{destination}?{legacy_query}",
    ]


def _time_preference_from_query_pieces(pieces: list[str]) -> str | None:
    for piece in pieces:
        preference = piece.strip().lower()
        if preference in {"morning", "afternoon", "evening"}:
            return preference
    return None


def _init_seleniumwire_driver(
    *,
    browser: str,
    headless: bool,
):
    try:
        from seleniumwire import webdriver
    except ImportError as exc:
        raise RuntimeError(
            "Ctrip crawler requires optional dependency selenium-wire. "
            "Install with: python -m pip install -e \".[ctrip]\""
        ) from exc

    normalised_browser = browser.strip().lower()
    if normalised_browser == "chrome":
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless=new")
        _configure_browser_options(options)
        return webdriver.Chrome(options=options)
    if normalised_browser == "edge":
        options = webdriver.EdgeOptions()
        if headless:
            options.add_argument("--headless=new")
        _configure_browser_options(options)
        return webdriver.Edge(options=options)
    raise ValueError(f"Unsupported browser for Ctrip crawler: {browser}")


def _configure_browser_options(options) -> None:
    options.add_argument("--incognito")
    options.add_argument("--remote-debugging-port=0")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-extensions")
    options.add_argument("--pageLoadStrategy=eager")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])


class _CtripLoginSession:
    def __init__(
        self,
        *,
        accounts: list[str],
        passwords: list[str],
        cookies_file: Path,
        timeout_seconds: int,
        login_wait_seconds: int,
    ) -> None:
        self.accounts = [account for account in accounts if account]
        self.passwords = [password for password in passwords if password]
        self.cookies_file = cookies_file
        self.timeout_seconds = timeout_seconds
        self.login_wait_seconds = login_wait_seconds

    def ensure_login(self, driver) -> None:
        if not self.accounts:
            raise RuntimeError("Ctrip login is enabled but no account is configured.")
        if not self.passwords:
            raise RuntimeError("Ctrip login is enabled but no password is configured.")

        for index, account in enumerate(self.accounts):
            password = self.passwords[index % len(self.passwords)]
            if self._try_cookie_login(driver, account):
                return
            if self._try_password_login(driver, account, password):
                self._save_required_cookies(account, driver.get_cookies())
                return
        raise RuntimeError("Ctrip login failed for all configured accounts.")

    def _try_cookie_login(self, driver, account: str) -> bool:
        cookies = self._load_cookies(account)
        if not cookies:
            return False
        driver.get("https://www.ctrip.com/")
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                continue
        if self._is_logged_in(driver):
            return True
        self._delete_cookies(account)
        return False

    def _try_password_login(self, driver, account: str, password: str) -> bool:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        wait = WebDriverWait(driver, self.timeout_seconds)
        driver.get("https://flights.ctrip.com/online/channel/domestic")
        self._open_login_panel(driver, wait)
        account_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".r_input.bbz-js-iconable-input"))
        )
        _replace_input_value(account_input, account)
        password_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div[data-testid='accountPanel'] input[data-testid='passwordInput']"))
        )
        _replace_input_value(password_input, password)

        for selector in ('[for="checkboxAgreementInput"]',):
            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            if elements:
                elements[0].click()
                break

        submit = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".form_btn.form_btn--block")))
        submit.click()
        self._handle_double_auth_if_present(driver)
        return self._is_logged_in(driver)

    def _open_login_panel(self, driver, wait) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC

        if driver.find_elements(By.CLASS_NAME, "lg_loginbox_modal"):
            return
        login_buttons = driver.find_elements(By.CLASS_NAME, "tl_nfes_home_header_login_wrapper_siwkn")
        if login_buttons:
            login_buttons[0].click()
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "lg_loginwrap")))
            return
        driver.get("https://passport.ctrip.com/user/login")

    def _handle_double_auth_if_present(self, driver) -> None:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.support.ui import WebDriverWait

        wait = WebDriverWait(driver, self.timeout_seconds)
        selector = "[data-testid='doubleAuthSwitcherBox']"
        try:
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
        except Exception:
            return

        send_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} dl[data-testid='dynamicCodeInput'] a.btn-primary-s"))
        )
        send_button.click()
        code = self._wait_for_console_input("Please enter the Ctrip verification code: ")
        if not code:
            raise RuntimeError("Timed out waiting for Ctrip verification code.")
        code_input = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} input[data-testid='verifyCodeInput']"))
        )
        code_input.send_keys(code)
        verify_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, f"{selector} dl[data-testid='dynamicVerifyButton'] input[type='submit']"))
        )
        verify_button.click()

    def _wait_for_console_input(self, prompt: str) -> str | None:
        result: list[str | None] = [None]
        completed = threading.Event()

        def read_input() -> None:
            try:
                result[0] = input(prompt).strip()
            except EOFError:
                result[0] = None
            finally:
                completed.set()

        thread = threading.Thread(target=read_input, daemon=True)
        thread.start()
        thread.join(timeout=self.login_wait_seconds)
        if not completed.is_set():
            return None
        return result[0]

    def _is_logged_in(self, driver) -> bool:
        try:
            driver.get("https://my.ctrip.com/myinfo/home")
            deadline = time.time() + self.timeout_seconds
            while time.time() < deadline:
                if driver.current_url.startswith("https://my.ctrip.com/myinfo/home"):
                    return True
                time.sleep(0.25)
        except Exception:
            return False
        return False

    def _load_cookies(self, account: str) -> list[dict[str, Any]] | None:
        if not self.cookies_file.exists():
            return None
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            cookies = cookies_by_account.get(account)
            return cookies if isinstance(cookies, list) else None
        except Exception:
            return None

    def _save_required_cookies(self, account: str, cookies: list[dict[str, Any]]) -> None:
        self.cookies_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            if not isinstance(cookies_by_account, dict):
                cookies_by_account = {}
        except Exception:
            cookies_by_account = {}
        cookies_by_account[account] = [
            cookie for cookie in cookies if cookie.get("name") in REQUIRED_CTRIP_COOKIES
        ]
        self.cookies_file.write_text(
            json.dumps(cookies_by_account, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _delete_cookies(self, account: str) -> None:
        if not self.cookies_file.exists():
            return
        try:
            cookies_by_account = json.loads(self.cookies_file.read_text(encoding="utf-8"))
            if isinstance(cookies_by_account, dict) and account in cookies_by_account:
                del cookies_by_account[account]
                self.cookies_file.write_text(
                    json.dumps(cookies_by_account, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        except Exception:
            return


def _drive_ctrip_homepage_search(driver, intent: FlightSearchIntent, *, timeout_seconds: int) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    wait = WebDriverWait(driver, timeout_seconds)
    driver.get("https://flights.ctrip.com/online/channel/domestic")
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "pc_home-jipiao")))
    flight_tab = driver.find_element(By.CLASS_NAME, "pc_home-jipiao")
    _click_ctrip_element(driver, flight_tab)
    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "radio-label")))
    one_way = driver.find_elements(By.CLASS_NAME, "radio-label")[0]
    _click_ctrip_element(driver, one_way)

    wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "form-input-v3")))
    inputs = driver.find_elements(By.CLASS_NAME, "form-input-v3")
    _select_ctrip_place(
        driver,
        inputs[0],
        intent.origin,
        timeout_seconds=timeout_seconds,
    )
    _select_ctrip_place(
        driver,
        inputs[1],
        intent.destination,
        timeout_seconds=timeout_seconds,
    )

    date_inputs = driver.find_elements(By.CSS_SELECTOR, "[aria-label=请选择日期]")
    if date_inputs:
        _select_ctrip_travel_date(
            driver,
            date_inputs[0],
            intent.travel_date,
            timeout_seconds=timeout_seconds,
        )

    buttons = driver.find_elements(By.CLASS_NAME, "search-btn")
    if buttons:
        _click_ctrip_element(driver, buttons[0])
    else:
        inputs[1].send_keys(Keys.ENTER)


def _select_ctrip_place(driver, input_element, place: str, *, timeout_seconds: int) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait

    code = place.strip().upper()
    _replace_input_value(input_element, _ctrip_display_name(code))
    selector = (
        '.cflt-poi-selector-new .address'
        f'[data-u_remark*="Code:{code}"]'
    )
    wait = WebDriverWait(driver, timeout_seconds)

    def visible_match(current_driver):
        return next(
            (
                candidate
                for candidate in current_driver.find_elements(By.CSS_SELECTOR, selector)
                if candidate.is_displayed()
            ),
            False,
        )

    candidate = wait.until(visible_match)
    _click_ctrip_element(driver, candidate)
    wait.until(lambda _driver: f"({code})" in (input_element.get_attribute("value") or "").upper())


def _select_ctrip_travel_date(driver, date_input, travel_date, *, timeout_seconds: int) -> None:
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    expected_value = travel_date.isoformat()
    target_selector = f'[data-testid="date-day-{expected_value}"]'
    wait = WebDriverWait(driver, timeout_seconds)
    _click_ctrip_element(driver, date_input)
    wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, ".calendar-modal")))

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        targets = driver.find_elements(By.CSS_SELECTOR, target_selector)
        for target in targets:
            if target.is_displayed() and "date-disabled" not in (target.get_attribute("class") or ""):
                day_label = target.find_element(By.CLASS_NAME, "date-d")
                _click_ctrip_element(driver, day_label)
                wait.until(lambda _driver: date_input.get_attribute("value") == expected_value)
                return

        next_buttons = [
            button
            for button in driver.find_elements(By.CSS_SELECTOR, ".calendar-modal .next-ico")
            if button.is_displayed()
        ]
        if not next_buttons:
            break
        _click_ctrip_element(driver, next_buttons[-1])
        time.sleep(0.2)

    raise RuntimeError(f"Ctrip date picker could not select {expected_value}.")


def _click_ctrip_element(driver, element) -> None:
    from selenium.common.exceptions import ElementClickInterceptedException

    try:
        element.click()
        return
    except ElementClickInterceptedException:
        driver.execute_script(
            """
            document.querySelectorAll("iframe#stageFrame").forEach((frame) => {
                const style = window.getComputedStyle(frame);
                if (style.position === "fixed" && Number(style.opacity) === 0) {
                    frame.remove();
                }
            });
            """
        )
    try:
        element.click()
    except ElementClickInterceptedException:
        driver.execute_script("arguments[0].click();", element)


def _replace_input_value(element, value: str) -> None:
    from selenium.webdriver.common.keys import Keys

    element.click()
    element.send_keys(Keys.CONTROL + "a")
    element.send_keys(value)


_CTRIP_DISPLAY_NAMES = {
    "SIN": "新加坡",
    "CJU": "济州岛",
    "TFU": "成都",
    "CTU": "成都",
    "BJS": "北京",
    "PEK": "北京",
    "PKX": "北京",
    "SHA": "上海",
    "PVG": "上海",
}


def _ctrip_display_name(value: str) -> str:
    return _CTRIP_DISPLAY_NAMES.get(value.strip().upper(), value)


def _ctrip_airport_matches_request(observed_code: object, requested_code: str) -> bool:
    observed = str(observed_code or "").strip().upper()
    requested = requested_code.strip().upper()
    if not observed:
        return False
    from .places import air_endpoint_matches

    return air_endpoint_matches(requested, observed)


def _lowest_price(price_list: list[dict[str, Any]]) -> float | None:
    prices: list[float] = []
    for item in price_list:
        adult_price = item.get("adultPrice")
        if adult_price is None:
            continue
        adult_tax = item.get("adultTax")
        if adult_tax is None:
            sort_price = item.get("sortPrice", adult_price)
            adult_tax = 0 if item.get("freeOilFeeAndTax") else sort_price - adult_price
        prices.append(float(adult_price) + float(adult_tax))
    if not prices:
        return None
    return min(prices)


def _rank_ctrip_evidence(evidence: list[FlightEvidence], intent: FlightSearchIntent) -> list[FlightEvidence]:
    preference = (intent.time_preference or "").strip().lower()
    if preference not in {"morning", "afternoon", "evening"}:
        return sorted(evidence, key=lambda item: item.price)
    return sorted(
        evidence,
        key=lambda item: (
            0 if _matches_time_preference(item.departure_time, preference) else 1,
            item.price,
        ),
    )


def _evidence_satisfies_requested_time(evidence: list[FlightEvidence], intent: FlightSearchIntent) -> bool:
    preference = (intent.time_preference or "").strip().lower()
    if preference not in {"morning", "afternoon", "evening"}:
        return True
    return any(_matches_time_preference(item.departure_time, preference) for item in evidence)


def _matches_time_preference(departure_time: datetime | None, preference: str) -> bool:
    if departure_time is None:
        return False
    hour = departure_time.hour
    if preference == "morning":
        return hour < 12
    if preference == "afternoon":
        return 12 <= hour < 18
    return hour >= 18


def _flight_metadata(
    *,
    itinerary_segments: list[dict[str, object]],
    transfer_count: int,
    itinerary: dict[str, Any],
) -> dict[str, object]:
    first = itinerary_segments[0]
    last = itinerary_segments[-1]
    flight_no = "+".join(str(segment["flight_no"]) for segment in itinerary_segments if segment.get("flight_no"))
    airline_names = [str(segment["airline"]) for segment in itinerary_segments if segment.get("airline")]
    aircraft_names = [str(segment["aircraft"]) for segment in itinerary_segments if segment.get("aircraft")]
    return {
        "flight_no": flight_no or itinerary.get("itineraryId"),
        "airline": "+".join(airline_names) if airline_names else None,
        "aircraft": "+".join(aircraft_names) if aircraft_names else None,
        "departure_airport": first.get("departure_airport"),
        "departure_airport_code": first.get("departure_airport_code"),
        "departure_terminal": first.get("departure_terminal"),
        "arrival_airport": last.get("arrival_airport"),
        "arrival_airport_code": last.get("arrival_airport_code"),
        "arrival_terminal": last.get("arrival_terminal"),
        "transfer_count": transfer_count,
        "is_direct": transfer_count == 0,
        "itinerary_id": itinerary.get("itineraryId"),
        "segments": itinerary_segments,
    }


def _itinerary_segments(segments: list[dict[str, Any]]) -> list[dict[str, object]]:
    itinerary_segments: list[dict[str, object]] = []
    for segment in segments:
        for flight in segment.get("flightList", []):
            itinerary_segments.append(_segment_metadata(flight))
    return itinerary_segments


def _segment_metadata(flight: dict[str, Any]) -> dict[str, object]:
    return {
        "flight_no": flight.get("flightNo"),
        "operate_flight_no": flight.get("operateFlightNo"),
        "airline": flight.get("marketAirlineName"),
        "operate_airline": flight.get("operateAirlineName"),
        "aircraft": flight.get("aircraftName") or flight.get("planeType"),
        "departure_time": flight.get("departureDateTime"),
        "arrival_time": flight.get("arrivalDateTime"),
        "departure_airport": flight.get("departureAirportName"),
        "departure_airport_code": flight.get("departureAirportCode"),
        "departure_terminal": flight.get("departureTerminal"),
        "arrival_airport": flight.get("arrivalAirportName"),
        "arrival_airport_code": flight.get("arrivalAirportCode"),
        "arrival_terminal": flight.get("arrivalTerminal"),
        "duration": flight.get("duration"),
    }


def _transfer_count(segments: list[dict[str, Any]], itinerary_segments: list[dict[str, object]]) -> int:
    explicit_counts = [segment.get("transferCount") for segment in segments if segment.get("transferCount") is not None]
    if explicit_counts:
        return int(max(explicit_counts))
    return max(len(itinerary_segments) - 1, 0)


def _parse_ctrip_datetime(
    value: str | None,
    *,
    airport_code: object = None,
) -> datetime | None:
    if not value:
        return None
    from .places import timezone_for_airport

    airport_timezone = timezone_for_airport(str(airport_code or ""))
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=airport_timezone)
        except ValueError:
            continue
    return None


def _first(value: list[Any]) -> Any | None:
    if not value:
        return None
    return value[0]


def _required_query(query: dict[str, list[str]], name: str) -> str:
    value = query.get(name, [""])[0]
    if not value:
        raise ValueError(f"Missing required Ctrip Selenium URL query parameter: {name}")
    return value
