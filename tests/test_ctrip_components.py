from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from flight_watch_agent.ctrip import (
    CapturedCtripResponse,
    CtripCaptureBackend,
    CtripFlightBackend,
    CtripPayloadParser,
    CtripSeleniumWirePageExtractor,
    FlightEvidenceMapper,
    InMemoryRawResponseStore,
    RawResponseRef,
    build_ctrip_selenium_url,
)
from flight_watch_agent.models import FlightPageAttemptResult, FlightSearchIntent


def test_payload_parser_and_evidence_mapper_are_pure_and_browser_free():
    intent = FlightSearchIntent(
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
    )
    payload = _payload()
    captured_at = datetime(2026, 7, 29, tzinfo=timezone.utc)

    itineraries = CtripPayloadParser().parse(payload)
    evidence = FlightEvidenceMapper().map(
        itineraries,
        intent,
        source_url="https://example.test/batchSearch",
        captured_at=captured_at,
        direct_only=False,
        max_results=5,
        response_ref=RawResponseRef(
            capture_id="capture-1",
            source_url="https://example.test/batchSearch",
            captured_at=captured_at,
            parser_version="ctrip-payload-v1",
        ),
    )

    assert len(itineraries) == 1
    assert itineraries[0].transfer_count == 0
    assert evidence[0].metadata["flight_no"] == "3U8899"
    assert evidence[0].metadata["capture_ref"] == "capture-1"


def test_response_store_redacts_sensitive_fields_and_returns_reference():
    store = InMemoryRawResponseStore()
    captured_at = datetime(2026, 7, 29, tzinfo=timezone.utc)

    reference = store.store(
        {
            "token": "secret-token",
            "nested": {"cookieValue": "secret-cookie"},
            "data": {"flightItineraryList": []},
        },
        source_url="https://example.test/batchSearch",
        captured_at=captured_at,
        parser_version="ctrip-payload-v1",
    )
    stored = store.get(reference)

    assert reference.source_url == "https://example.test/batchSearch"
    assert stored == {
        "token": "[REDACTED]",
        "nested": {"cookieValue": "[REDACTED]"},
        "data": {"flightItineraryList": []},
    }


def test_capture_backend_only_captures_and_stores_raw_payload(monkeypatch):
    import flight_watch_agent.ctrip as ctrip_module

    payload = _payload()
    monkeypatch.setattr(
        ctrip_module,
        "_wait_for_ctrip_search_payload",
        lambda *_args, **_kwargs: payload,
    )
    backend = CtripCaptureBackend()
    driver = _CaptureDriver()

    backend.clear(driver)
    captured = backend.capture(
        driver,
        source_url="https://example.test/batchSearch",
        timeout_seconds=1,
        manual_verification_wait_seconds=0,
    )

    assert driver.requests_cleared is True
    assert captured.payload == payload
    assert captured.response_ref.parser_version == "ctrip-payload-v1"


def test_flight_backend_can_run_against_fake_navigation_session_and_capture():
    intent = FlightSearchIntent(
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
    )
    navigator = _FakeNavigator()
    session = _FakeSessionManager()
    capture = _FakeCaptureBackend(_payload())
    backend = CtripFlightBackend(
        navigator=navigator,
        session_manager=session,
        capture_backend=capture,
        timeout_seconds=1,
    )

    evidence = backend.extract(intent)

    assert len(evidence) == 1
    assert navigator.entrypoints == ["international"]
    assert session.prepared is True
    assert session.released is True
    assert capture.cleared is True


def test_legacy_page_extractor_delegates_to_injected_backend_without_selenium():
    backend = _FakeFlightBackend()
    extractor = CtripSeleniumWirePageExtractor(backend=backend)
    url = build_ctrip_selenium_url(
        FlightSearchIntent(
            origin="CTU",
            destination="CJU",
            travel_date=date(2026, 7, 31),
        )
    )

    assert extractor.extract(url) == []
    attempt = extractor.extract_attempt(
        url,
        entrypoint="international",
        action_id="attempt-1",
    )
    extractor.close()

    assert backend.intent is not None
    assert attempt.status == "no_evidence"
    assert backend.closed is True


class _CaptureDriver:
    def __init__(self) -> None:
        self.requests_cleared = False

    @property
    def requests(self):
        return []

    @requests.deleter
    def requests(self):
        self.requests_cleared = True


class _FakeNavigator:
    def __init__(self) -> None:
        self.entrypoints: list[str] = []

    def search_urls(self, _intent):
        return ["https://example.test/international", "https://example.test/list"]

    def navigate(self, _driver, _intent, *, entrypoint, force_refresh=False):
        assert force_refresh is False
        self.entrypoints.append(entrypoint)
        return f"https://example.test/{entrypoint}"


class _FakeSessionManager:
    def __init__(self) -> None:
        self.prepared = False
        self.released = False
        self.closed = False

    def acquire(self):
        return object(), True

    def prepare(self, _driver, *, is_new):
        assert is_new is True
        self.prepared = True

    def release(self, _driver):
        self.released = True

    def close(self):
        self.closed = True


class _FakeCaptureBackend:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.cleared = False

    def clear(self, _driver):
        self.cleared = True

    def capture(self, _driver, *, source_url, **_kwargs):
        captured_at = datetime(2026, 7, 29, tzinfo=timezone.utc)
        return CapturedCtripResponse(
            payload=self.payload,
            source_url=source_url,
            captured_at=captured_at,
            response_ref=RawResponseRef(
                capture_id="fake-capture",
                source_url=source_url,
                captured_at=captured_at,
                parser_version="ctrip-payload-v1",
            ),
        )


class _FakeFlightBackend:
    def __init__(self) -> None:
        self.intent = None
        self.closed = False

    def extract(self, intent):
        self.intent = intent
        return []

    def extract_attempt(self, intent, *, entrypoint, action_id, force_refresh=False):
        self.intent = intent
        assert entrypoint == "international"
        assert action_id == "attempt-1"
        assert force_refresh is False
        return FlightPageAttemptResult(
            status="no_evidence",
            evidence=[],
            entrypoint=entrypoint,
        )

    def close(self):
        self.closed = True


def _payload():
    fixture = Path(__file__).parent / "fixtures" / "ctrip" / "batch_search_direct.json"
    return json.loads(fixture.read_text(encoding="utf-8"))
