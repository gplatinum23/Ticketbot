from __future__ import annotations

from datetime import date, datetime, timezone

import flight_watch_agent.ctrip as ctrip_module

from flight_watch_agent.ctrip import (
    REQUIRED_CTRIP_COOKIES,
    CtripRouteSearchTool,
    CtripSeleniumWirePageExtractor,
    _CtripLoginSession,
    _configure_browser_options,
    _evidence_satisfies_requested_time,
    _is_manual_verification_present,
    build_ctrip_selenium_url,
    parse_ctrip_batch_search_payload,
    parse_ctrip_selenium_url,
)
from flight_watch_agent.app import build_default_flight_page_extractor, build_default_flight_web_search_tool
from flight_watch_agent.models import FlightEvidence, FlightSearchIntent


def test_ctrip_route_search_tool_builds_selenium_task_url():
    result = CtripRouteSearchTool().search("SIN TFU 2026-07-09 flight price")[0]

    assert result.source_name == "flights.ctrip.com"
    assert result.url == "ctrip-selenium://flight?origin=SIN&destination=TFU&travel_date=2026-07-09&currency=CNY"


def test_ctrip_route_search_tool_keeps_time_preference_from_query():
    result = CtripRouteSearchTool().search("NKG SIN 2026-07-10 flight price airline OTA CNY morning")[0]

    assert result.url == (
        "ctrip-selenium://flight?origin=NKG&destination=SIN&travel_date=2026-07-10"
        "&currency=CNY&time_preference=morning"
    )


def test_ctrip_selenium_url_round_trip():
    intent = FlightSearchIntent(
        origin="SIN",
        destination="TFU",
        travel_date=date(2026, 7, 9),
        time_preference="morning",
        currency="CNY",
    )

    parsed = parse_ctrip_selenium_url(build_ctrip_selenium_url(intent))

    assert parsed.origin == "SIN"
    assert parsed.destination == "TFU"
    assert parsed.travel_date == date(2026, 7, 9)
    assert parsed.time_preference == "morning"
    assert parsed.currency == "CNY"


def test_parse_ctrip_batch_search_payload_extracts_lowest_price_evidence():
    intent = FlightSearchIntent(origin="SIN", destination="TFU", travel_date=date(2026, 7, 9), currency="CNY")
    payload = {
        "data": {
            "flightItineraryList": [
                {
                    "flightSegments": [
                        {
                            "transferCount": 0,
                            "flightList": [
                                {
                                    "flightNo": "3U3920",
                                    "marketAirlineName": "四川航空",
                                    "aircraftName": "空客320",
                                    "departureDateTime": "2026-07-09 08:10:00",
                                    "arrivalDateTime": "2026-07-09 12:30:00",
                                    "departureAirportName": "樟宜机场",
                                    "departureAirportCode": "SIN",
                                    "departureTerminal": "T2",
                                    "arrivalAirportName": "天府国际机场",
                                    "arrivalAirportCode": "TFU",
                                    "arrivalTerminal": "T1",
                                    "duration": 270,
                                }
                            ],
                        }
                    ],
                    "priceList": [
                        {
                            "adultPrice": 1500,
                            "adultTax": 120,
                            "freeOilFeeAndTax": False,
                            "sortPrice": 1620,
                            "cabin": "Y",
                            "miseryIndex": 1,
                        },
                        {
                            "adultPrice": 1400,
                            "adultTax": 90,
                            "freeOilFeeAndTax": False,
                            "sortPrice": 1490,
                            "cabin": "Y",
                            "miseryIndex": 2,
                        },
                    ],
                }
            ]
        }
    }

    evidence = parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url="https://flights.ctrip.com/international/search/oneway-sin-tfu?depdate=2026-07-09",
        direct_only=True,
        max_results=5,
    )

    assert len(evidence) == 1
    assert evidence[0].source_name == "flights.ctrip.com"
    assert evidence[0].price == 1490
    assert evidence[0].origin == "SIN"
    assert evidence[0].destination == "TFU"
    assert evidence[0].travel_date == date(2026, 7, 9)
    assert evidence[0].metadata == {
        "flight_no": "3U3920",
        "airline": "四川航空",
        "aircraft": "空客320",
        "departure_airport": "樟宜机场",
        "departure_airport_code": "SIN",
        "departure_terminal": "T2",
        "arrival_airport": "天府国际机场",
        "arrival_airport_code": "TFU",
        "arrival_terminal": "T1",
        "transfer_count": 0,
        "is_direct": True,
        "itinerary_id": None,
        "segments": [
            {
                "flight_no": "3U3920",
                "operate_flight_no": None,
                "airline": "四川航空",
                "operate_airline": None,
                "aircraft": "空客320",
                "departure_time": "2026-07-09 08:10:00",
                "arrival_time": "2026-07-09 12:30:00",
                "departure_airport": "樟宜机场",
                "departure_airport_code": "SIN",
                "departure_terminal": "T2",
                "arrival_airport": "天府国际机场",
                "arrival_airport_code": "TFU",
                "arrival_terminal": "T1",
                "duration": 270,
            }
        ],
    }


def test_parse_ctrip_batch_search_payload_keeps_full_transfer_itinerary():
    intent = FlightSearchIntent(origin="SIN", destination="TFU", travel_date=date(2026, 7, 9), currency="CNY")
    payload = {
        "data": {
            "flightItineraryList": [
                {
                    "itineraryId": "MU5082_1783572000000,MU5855_1783603800000",
                    "flightSegments": [
                        {
                            "transferCount": 1,
                            "flightList": [
                                {
                                    "flightNo": "MU5082",
                                    "marketAirlineName": "东方航空",
                                    "aircraftName": "波音737",
                                    "departureDateTime": "2026-07-09 12:40:00",
                                    "arrivalDateTime": "2026-07-09 17:00:00",
                                    "departureAirportName": "樟宜机场",
                                    "departureAirportCode": "SIN",
                                    "departureTerminal": "T3",
                                    "arrivalAirportName": "长水国际机场",
                                    "arrivalAirportCode": "KMG",
                                },
                                {
                                    "flightNo": "MU5855",
                                    "marketAirlineName": "东方航空",
                                    "aircraftName": "波音737",
                                    "departureDateTime": "2026-07-09 20:50:00",
                                    "arrivalDateTime": "2026-07-09 22:25:00",
                                    "departureAirportName": "长水国际机场",
                                    "departureAirportCode": "KMG",
                                    "arrivalAirportName": "天府国际机场",
                                    "arrivalAirportCode": "TFU",
                                    "arrivalTerminal": "T2",
                                },
                            ],
                        }
                    ],
                    "priceList": [
                        {
                            "adultPrice": 1200,
                            "adultTax": 21,
                            "freeOilFeeAndTax": False,
                            "sortPrice": 1221,
                            "cabin": "Y",
                            "miseryIndex": 1,
                        },
                    ],
                }
            ]
        }
    }

    evidence = parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url="https://flights.ctrip.com/international/search/oneway-sin-tfu?depdate=2026-07-09",
        direct_only=False,
        max_results=5,
    )

    assert len(evidence) == 1
    assert evidence[0].departure_time.isoformat() == "2026-07-09T12:40:00+00:00"
    assert evidence[0].arrival_time.isoformat() == "2026-07-09T22:25:00+00:00"
    assert evidence[0].metadata["flight_no"] == "MU5082+MU5855"
    assert evidence[0].metadata["arrival_airport_code"] == "TFU"
    assert evidence[0].metadata["transfer_count"] == 1
    assert evidence[0].metadata["is_direct"] is False
    assert len(evidence[0].metadata["segments"]) == 2


def test_parse_ctrip_batch_search_payload_sorts_before_limiting_results():
    intent = FlightSearchIntent(origin="SIN", destination="TFU", travel_date=date(2026, 7, 9), currency="CNY")
    payload = {
        "data": {
            "flightItineraryList": [
                _ctrip_itinerary("MU5082", "MU5855", adult_price=200, adult_tax=1021),
                _ctrip_itinerary("HU748", "HU7085", adult_price=200, adult_tax=821),
            ]
        }
    }

    evidence = parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url="https://flights.ctrip.com/online/list/oneway-sin-tfu?depdate=2026-07-09",
        direct_only=False,
        max_results=1,
    )

    assert len(evidence) == 1
    assert evidence[0].price == 1021
    assert evidence[0].metadata["flight_no"] == "HU748+HU7085"


def test_parse_ctrip_batch_search_payload_prioritises_time_preference_before_price_limit():
    intent = FlightSearchIntent(
        origin="NKG",
        destination="SIN",
        travel_date=date(2026, 7, 10),
        time_preference="morning",
        currency="CNY",
    )
    payload = {
        "data": {
            "flightItineraryList": [
                _ctrip_itinerary(
                    "MU5878",
                    "MU9647",
                    adult_price=900,
                    adult_tax=126,
                    first_departure_time="2026-07-10 20:55:00",
                ),
                _ctrip_itinerary(
                    "3U6916",
                    "3U3919",
                    adult_price=1200,
                    adult_tax=118,
                    first_departure_time="2026-07-10 06:50:00",
                ),
            ]
        }
    }

    evidence = parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url="https://flights.ctrip.com/international/search/oneway-nkg-sin?depdate=2026-07-10",
        direct_only=False,
        max_results=1,
    )

    assert len(evidence) == 1
    assert evidence[0].price == 1318
    assert evidence[0].metadata["flight_no"] == "3U6916+3U3919"
    assert evidence[0].departure_time.hour == 6


def test_ctrip_evidence_without_requested_time_does_not_satisfy_preference():
    intent = FlightSearchIntent(
        origin="NKG",
        destination="SIN",
        travel_date=date(2026, 7, 10),
        time_preference="morning",
        currency="CNY",
    )
    payload = {
        "data": {
            "flightItineraryList": [
                _ctrip_itinerary(
                    "MU5878",
                    "MU9647",
                    adult_price=900,
                    adult_tax=126,
                    first_departure_time="2026-07-10 20:55:00",
                )
            ]
        }
    }

    evidence = parse_ctrip_batch_search_payload(
        payload,
        intent,
        source_url="https://flights.ctrip.com/international/search/oneway-nkg-sin?depdate=2026-07-10",
        direct_only=False,
        max_results=5,
    )

    assert _evidence_satisfies_requested_time(evidence, intent) is False


def test_ctrip_login_session_saves_only_required_cookies(tmp_path):
    cookie_file = tmp_path / "cookies.json"
    session = _CtripLoginSession(
        accounts=["account-a"],
        passwords=["password-a"],
        cookies_file=cookie_file,
        timeout_seconds=1,
        login_wait_seconds=1,
    )

    session._save_required_cookies(
        "account-a",
        [
            {"name": REQUIRED_CTRIP_COOKIES[0], "value": "keep"},
            {"name": "unrelated", "value": "drop"},
        ],
    )

    assert session._load_cookies("account-a") == [{"name": REQUIRED_CTRIP_COOKIES[0], "value": "keep"}]


def test_ctrip_login_session_deletes_invalid_account_cookie(tmp_path):
    cookie_file = tmp_path / "cookies.json"
    session = _CtripLoginSession(
        accounts=["account-a"],
        passwords=["password-a"],
        cookies_file=cookie_file,
        timeout_seconds=1,
        login_wait_seconds=1,
    )
    session._save_required_cookies("account-a", [{"name": REQUIRED_CTRIP_COOKIES[0], "value": "keep"}])

    session._delete_cookies("account-a")

    assert session._load_cookies("account-a") is None


def test_ctrip_login_is_enabled_when_account_and_password_are_configured(monkeypatch):
    monkeypatch.setenv("FLIGHT_WATCH_CTRIP_USERNAME", "account-a")
    monkeypatch.setenv("FLIGHT_WATCH_CTRIP_PASSWORD", "password-a")
    monkeypatch.delenv("FLIGHT_WATCH_CTRIP_LOGIN_ALLOWED", raising=False)

    extractor = build_default_flight_page_extractor().extractors[0]

    assert extractor.login_allowed is True
    assert extractor.accounts == ["account-a"]
    assert extractor.passwords == ["password-a"]


def test_ctrip_manual_verification_wait_is_configurable(monkeypatch):
    monkeypatch.setenv("FLIGHT_WATCH_CTRIP_MANUAL_VERIFICATION_WAIT_SECONDS", "120")

    extractor = build_default_flight_page_extractor().extractors[0]

    assert extractor.manual_verification_wait_seconds == 120


def test_ctrip_manual_verification_page_is_detected():
    driver = FakeDriver(
        """
        <div>为保障您的安全访问，请完成以下操作</div>
        <div>依次点击图标验证</div>
        """
    )

    assert _is_manual_verification_present(driver) is True


def test_ctrip_normal_page_is_not_manual_verification():
    driver = FakeDriver("<div>南京 到 新加坡 航班列表</div>")

    assert _is_manual_verification_present(driver) is False


def test_default_flight_search_uses_only_ctrip_source():
    search = build_default_flight_web_search_tool()

    assert [type(tool).__name__ for tool in search.tools] == ["CtripRouteSearchTool"]


def test_default_flight_page_extractor_uses_only_ctrip_extractor():
    extractor = build_default_flight_page_extractor()

    assert [type(item).__name__ for item in extractor.extractors] == ["CtripSeleniumWirePageExtractor"]


def test_ctrip_extractor_reuses_browser_between_route_queries(monkeypatch):
    drivers = []

    class ReusableFakeDriver:
        def __init__(self) -> None:
            self._requests = []
            self.urls = []
            self.quit_calls = 0

        @property
        def requests(self):
            return self._requests

        @requests.deleter
        def requests(self):
            self._requests = []

        def get(self, url: str) -> None:
            self.urls.append(url)

        def quit(self) -> None:
            self.quit_calls += 1

    def build_driver(**_kwargs):
        driver = ReusableFakeDriver()
        drivers.append(driver)
        return driver

    def parse_payload(_payload, intent, **_kwargs):
        departure = datetime.combine(intent.travel_date, datetime.min.time(), tzinfo=timezone.utc)
        return [
            FlightEvidence(
                source_name="flights.ctrip.com",
                url="https://flights.ctrip.com/test",
                price=500.0,
                currency="CNY",
                departure_time=departure,
                arrival_time=departure,
                captured_at=departure,
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
            )
        ]

    monkeypatch.setattr(ctrip_module, "_init_seleniumwire_driver", build_driver)
    monkeypatch.setattr(ctrip_module, "_wait_for_ctrip_search_payload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ctrip_module, "parse_ctrip_batch_search_payload", parse_payload)
    extractor = CtripSeleniumWirePageExtractor(reuse_browser_session=True)

    extractor.extract(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CTU", destination="SIN", travel_date=date(2026, 8, 1))
        )
    )
    extractor.extract(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CAN", destination="SIN", travel_date=date(2026, 8, 1))
        )
    )

    assert len(drivers) == 1
    assert len(drivers[0].urls) == 2
    assert drivers[0].quit_calls == 0
    extractor.close()
    assert drivers[0].quit_calls == 1


def test_ctrip_extractor_prioritises_search_url_that_succeeded_in_session(monkeypatch):
    class AdaptiveFakeDriver:
        def __init__(self) -> None:
            self._requests = []
            self.urls = []

        @property
        def requests(self):
            return self._requests

        @requests.deleter
        def requests(self):
            self._requests = []

        def get(self, url: str) -> None:
            self.urls.append(url)

        def quit(self) -> None:
            return None

    driver = AdaptiveFakeDriver()
    monkeypatch.setattr(ctrip_module, "_init_seleniumwire_driver", lambda **_kwargs: driver)
    monkeypatch.setattr(
        ctrip_module,
        "_wait_for_ctrip_search_payload",
        lambda current_driver, **_kwargs: {
            "legacy": "/online/list/" in current_driver.urls[-1]
        },
    )

    def parse_payload(payload, intent, **_kwargs):
        if not payload["legacy"]:
            return []
        timestamp = datetime.combine(intent.travel_date, datetime.min.time(), tzinfo=timezone.utc)
        return [
            FlightEvidence(
                source_name="flights.ctrip.com",
                url=driver.urls[-1],
                price=500.0,
                currency="CNY",
                departure_time=timestamp,
                arrival_time=timestamp,
                captured_at=timestamp,
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
            )
        ]

    monkeypatch.setattr(ctrip_module, "parse_ctrip_batch_search_payload", parse_payload)
    extractor = CtripSeleniumWirePageExtractor(reuse_browser_session=True)

    extractor.extract(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CTU", destination="SIN", travel_date=date(2026, 8, 1))
        )
    )
    first_query_url_count = len(driver.urls)
    extractor.extract(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CAN", destination="SIN", travel_date=date(2026, 8, 1))
        )
    )

    assert first_query_url_count == 2
    assert len(driver.urls) == 3
    assert "/online/list/" in driver.urls[-1]
    extractor.close()


def test_ctrip_adaptive_attempt_uses_requested_entrypoint(monkeypatch):
    class AttemptDriver:
        def __init__(self) -> None:
            self._requests = []
            self.urls = []

        @property
        def requests(self):
            return self._requests

        @requests.deleter
        def requests(self):
            self._requests = []

        def get(self, url: str) -> None:
            self.urls.append(url)

        def quit(self) -> None:
            return None

    driver = AttemptDriver()
    monkeypatch.setattr(ctrip_module, "_init_seleniumwire_driver", lambda **_kwargs: driver)
    monkeypatch.setattr(ctrip_module, "_wait_for_ctrip_search_payload", lambda *_args, **_kwargs: {})

    def parse_payload(_payload, intent, **kwargs):
        timestamp = datetime.combine(intent.travel_date, datetime.min.time(), tzinfo=timezone.utc)
        return [
            FlightEvidence(
                source_name="flights.ctrip.com",
                url=kwargs["source_url"],
                price=500.0,
                currency="CNY",
                departure_time=timestamp,
                arrival_time=timestamp,
                captured_at=timestamp,
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
            )
        ]

    monkeypatch.setattr(ctrip_module, "parse_ctrip_batch_search_payload", parse_payload)
    extractor = CtripSeleniumWirePageExtractor(reuse_browser_session=True)

    attempt = extractor.extract_attempt(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CTU", destination="SIN", travel_date=date(2026, 8, 1))
        ),
        entrypoint="online_list",
        action_id="action-2",
    )

    assert attempt.status == "success"
    assert attempt.entrypoint == "online_list"
    assert "/online/list/" in driver.urls[-1]
    extractor.close()


def test_ctrip_adaptive_attempt_returns_captcha_observation(monkeypatch):
    class CaptchaDriver:
        def __init__(self) -> None:
            self._requests = []

        @property
        def requests(self):
            return self._requests

        @requests.deleter
        def requests(self):
            self._requests = []

        def get(self, _url: str) -> None:
            return None

        def quit(self) -> None:
            return None

    monkeypatch.setattr(
        ctrip_module,
        "_init_seleniumwire_driver",
        lambda **_kwargs: CaptchaDriver(),
    )
    monkeypatch.setattr(
        ctrip_module,
        "_wait_for_ctrip_search_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ctrip_module.CtripManualVerificationRequired("verification required")
        ),
    )
    extractor = CtripSeleniumWirePageExtractor(reuse_browser_session=True)

    attempt = extractor.extract_attempt(
        build_ctrip_selenium_url(
            FlightSearchIntent(origin="CTU", destination="SIN", travel_date=date(2026, 8, 1))
        ),
        entrypoint="international",
        action_id="action-1",
    )

    assert attempt.status == "captcha_required"
    assert attempt.evidence == []
    extractor.close()


def test_configure_browser_options_uses_incognito_without_local_profile():
    options = FakeBrowserOptions()

    _configure_browser_options(options)

    assert "--incognito" in options.arguments


class FakeBrowserOptions:
    def __init__(self) -> None:
        self.arguments = []
        self.experimental_options = {}

    def add_argument(self, argument: str) -> None:
        self.arguments.append(argument)

    def add_experimental_option(self, name: str, value) -> None:
        self.experimental_options[name] = value


class FakeDriver:
    def __init__(self, page_source: str) -> None:
        self.page_source = page_source

    def execute_script(self, _script: str):
        return False


def _ctrip_itinerary(
    first_flight_no: str,
    second_flight_no: str,
    *,
    adult_price: int,
    adult_tax: int,
    first_departure_time: str = "2026-07-09 12:05:00",
):
    return {
        "itineraryId": f"{first_flight_no}_1783569900000,{second_flight_no}_1783598400000",
        "flightSegments": [
            {
                "transferCount": 1,
                "flightList": [
                    {
                        "flightNo": first_flight_no,
                        "marketAirlineName": "test-airline",
                        "departureDateTime": first_departure_time,
                        "arrivalDateTime": "2026-07-09 16:00:00",
                        "departureAirportCode": "SIN",
                        "arrivalAirportCode": "HAK",
                    },
                    {
                        "flightNo": second_flight_no,
                        "marketAirlineName": "test-airline",
                        "departureDateTime": "2026-07-09 20:20:00",
                        "arrivalDateTime": "2026-07-09 22:25:00",
                        "departureAirportCode": "HAK",
                        "arrivalAirportCode": "TFU",
                    },
                ],
            }
        ],
        "priceList": [
            {
                "adultPrice": adult_price,
                "adultTax": adult_tax,
                "freeOilFeeAndTax": False,
            }
        ],
    }
