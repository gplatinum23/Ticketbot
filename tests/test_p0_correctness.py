from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from flight_watch_agent.agent_models import RouteEdge
from flight_watch_agent.ctrip import _parse_ctrip_datetime
from flight_watch_agent.flight_react import FlightEvidenceVerifier
from flight_watch_agent.ground_transfers import (
    StaticGroundTransferProvider,
    classify_connection,
)
from flight_watch_agent.models import (
    FlightEvidence,
    FlightOption,
    FlightSearchIntent,
    TrainOption,
)
from flight_watch_agent.travel_plan_graph import _build_two_leg_routes_from_edges


CHINA = timezone(timedelta(hours=8), "Asia/Shanghai")
KOREA = timezone(timedelta(hours=9), "Asia/Seoul")


def test_verifier_preserves_city_request_and_uses_observed_physical_airports():
    intent = FlightSearchIntent(
        origin="CTU",
        destination="BJS",
        travel_date=date(2026, 7, 31),
    )
    evidence = _flight_evidence(
        origin="CTU",
        destination="BJS",
        departure_code="TFU",
        arrival_code="PEK",
        departure=datetime(2026, 7, 31, 8, 0, tzinfo=CHINA),
        arrival=datetime(2026, 7, 31, 10, 30, tzinfo=CHINA),
    )

    options = FlightEvidenceVerifier().verify([evidence], intent)

    assert len(options) == 1
    option = options[0]
    assert option.requested_origin is not None
    assert option.requested_origin.kind == "city"
    assert option.requested_origin.raw == "CTU"
    assert option.actual_origin is not None
    assert option.actual_origin.airport_code == "TFU"
    assert option.origin == "TFU"
    assert option.actual_destination is not None
    assert option.actual_destination.airport_code == "PEK"
    assert option.destination == "PEK"


def test_explicit_airport_request_rejects_different_same_city_airport():
    intent = FlightSearchIntent(
        origin="PEK",
        destination="CJU",
        travel_date=date(2026, 7, 31),
    )
    wrong_airport = _flight_evidence(
        origin="PEK",
        destination="CJU",
        departure_code="PKX",
        arrival_code="CJU",
        departure=datetime(2026, 7, 31, 8, 0, tzinfo=CHINA),
        arrival=datetime(2026, 7, 31, 12, 0, tzinfo=KOREA),
    )

    assert FlightEvidenceVerifier().verify([wrong_airport], intent) == []


def test_ctrip_local_times_keep_endpoint_timezones_and_absolute_order():
    departure = _parse_ctrip_datetime(
        "2026-07-31 20:15:00",
        airport_code="CTU",
    )
    arrival = _parse_ctrip_datetime(
        "2026-07-31 22:30:00",
        airport_code="CJU",
    )

    assert departure is not None and departure.utcoffset() == timedelta(hours=8)
    assert arrival is not None and arrival.utcoffset() == timedelta(hours=9)
    assert int((arrival - departure).total_seconds() // 60) == 75


def test_user_travel_date_is_the_first_segment_local_departure_date():
    intent = FlightSearchIntent(
        origin="CTU",
        destination="CJU",
        travel_date=date(2026, 7, 31),
    )
    evidence = _flight_evidence(
        origin="CTU",
        destination="CJU",
        departure_code="CTU",
        arrival_code="CJU",
        departure=datetime(2026, 7, 31, 23, 30, tzinfo=CHINA),
        arrival=datetime(2026, 8, 1, 4, 30, tzinfo=KOREA),
    )

    options = FlightEvidenceVerifier().verify([evidence], intent)

    assert intent.travel_date_semantics == "first_segment_departure_local_date"
    assert len(options) == 1
    assert options[0].travel_date == date(2026, 7, 31)
    assert options[0].arrival_time.date() == date(2026, 8, 1)


def test_train_option_exposes_absolute_cross_day_datetimes():
    option = TrainOption(
        train_code="K118",
        from_station="成都西",
        from_station_code="CMW",
        to_station="北京西",
        to_station_code="BXP",
        travel_date=date(2026, 7, 31),
        start_time="21:12",
        arrive_time="05:26",
        duration="32:14",
        seats={"硬座": "有"},
        prices={"硬座": 251.0},
    )

    assert option.departure_at == datetime(2026, 7, 31, 21, 12, tzinfo=CHINA)
    assert option.arrival_at == datetime(2026, 8, 2, 5, 26, tzinfo=CHINA)
    assert option.arrival_at > option.departure_at


def test_flight_evidence_rejects_naive_operational_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        _flight_evidence(
            origin="CTU",
            destination="CJU",
            departure_code="CTU",
            arrival_code="CJU",
            departure=datetime(2026, 7, 31, 20, 15),
            arrival=datetime(2026, 7, 31, 22, 30, tzinfo=KOREA),
        )


def test_train_flight_route_contains_sourced_station_airport_edge_and_cost():
    train = TrainOption(
        train_code="D638",
        from_station="成都东",
        from_station_code="ICW",
        to_station="重庆北",
        to_station_code="CUW",
        travel_date=date(2026, 7, 31),
        start_time="07:05",
        arrive_time="09:16",
        duration="02:11",
        seats={"二等座": "有"},
        prices={"二等座": 93.0},
    )
    flight = _flight_option(
        origin="CKG",
        destination="CJU",
        departure=datetime(2026, 7, 31, 20, 15, tzinfo=CHINA),
        arrival=datetime(2026, 8, 1, 22, 30, tzinfo=KOREA),
        price=1607.0,
    )
    edges = [
        _train_edge(train),
        _flight_edge(flight),
    ]

    routes = _build_two_leg_routes_from_edges(
        edges,
        ground_transfer_provider=StaticGroundTransferProvider(),
    )

    assert len(routes) == 1
    route = routes[0]
    assert route.total_price == 1735.0
    assert route.transfer_airport == "CKG"
    assert route.transfer_wait_minutes == 659
    assert route.route_edges is not None
    assert [edge.mode for edge in route.route_edges] == [
        "train",
        "local_transfer",
        "flight",
    ]
    transfer = route.route_edges[1]
    assert transfer.duration_minutes == 40
    assert transfer.buffer_minutes == 30
    assert transfer.price == 35.0
    assert transfer.source == "builtin_curated_transfer_estimate:chongqingbei-ckg:v1"
    assert transfer.reliability == "curated_estimate"
    assert "重庆北->Chongqing Jiangbei International Airport" in route.summary


def test_beijing_west_to_pek_and_pkx_have_distinct_transfer_evidence():
    provider = StaticGroundTransferProvider()
    departure = datetime(2026, 8, 2, 5, 26, tzinfo=CHINA)

    to_pek = provider.find("北京西", "PEK", departure_at=departure)
    to_pkx = provider.find("北京西", "PKX", departure_at=departure)

    assert to_pek is not None and to_pkx is not None
    assert to_pek.destination.airport_code == "PEK"
    assert to_pkx.destination.airport_code == "PKX"
    assert to_pek.source != to_pkx.source
    assert to_pek.duration_minutes == 75
    assert to_pkx.duration_minutes == 70


def test_connection_classification_distinguishes_terminal_airport_and_city_boundaries():
    same_terminal = classify_connection(
        "PEK",
        "PEK",
        origin_terminal="T3",
        destination_terminal="T3",
    )
    same_airport = classify_connection(
        "PEK",
        "PEK",
        origin_terminal="T2",
        destination_terminal="T3",
    )
    cross_airport = classify_connection("PEK", "PKX")
    station_airport = classify_connection("北京西", "PEK")
    cross_city = classify_connection("CKG", "PEK")

    assert same_terminal.kind == "same_terminal"
    assert not same_terminal.requires_ground_transfer
    assert same_airport.kind == "same_airport"
    assert same_airport.requires_ground_transfer
    assert cross_airport.kind == "cross_airport"
    assert cross_airport.requires_ground_transfer
    assert station_airport.kind == "station_airport"
    assert station_airport.requires_ground_transfer
    assert cross_city.kind == "cross_city"
    assert not cross_city.requires_ground_transfer

    provider = StaticGroundTransferProvider()
    departure = datetime(2026, 8, 2, 8, 0, tzinfo=CHINA)
    assert provider.find(
        "PEK",
        "PEK",
        departure_at=departure,
        origin_terminal="T3",
        destination_terminal="T3",
    ) is None
    terminal_change = provider.find(
        "PEK",
        "PEK",
        departure_at=departure,
        origin_terminal="T2",
        destination_terminal="T3",
    )
    assert terminal_change is not None
    assert terminal_change.kind == "same_airport"
    assert terminal_change.duration_minutes == 20


def test_cross_airport_transfer_has_sourced_edge_data():
    provider = StaticGroundTransferProvider()
    departure = datetime(2026, 8, 2, 8, 0, tzinfo=CHINA)

    transfer = provider.find("PEK", "PKX", departure_at=departure)

    assert transfer is not None
    assert transfer.kind == "airport_airport"
    assert transfer.duration_minutes == 120
    assert transfer.buffer_minutes == 30
    assert transfer.price == 180.0
    assert transfer.source == "builtin_curated_transfer_estimate:pek-pkx:v1"


def test_cross_city_connection_without_transfer_evidence_is_not_executable():
    train = TrainOption(
        train_code="D638",
        from_station="成都东",
        from_station_code="ICW",
        to_station="重庆北",
        to_station_code="CUW",
        travel_date=date(2026, 7, 31),
        start_time="07:05",
        arrive_time="09:16",
        duration="02:11",
        seats={"二等座": "有"},
        prices={"二等座": 93.0},
    )
    flight = _flight_option(
        origin="PEK",
        destination="CJU",
        departure=datetime(2026, 7, 31, 20, 15, tzinfo=CHINA),
        arrival=datetime(2026, 8, 1, 22, 30, tzinfo=KOREA),
        price=1607.0,
    )

    routes = _build_two_leg_routes_from_edges(
        [_train_edge(train), _flight_edge(flight)],
        ground_transfer_provider=StaticGroundTransferProvider(),
    )

    assert routes == []


def _flight_evidence(
    *,
    origin: str,
    destination: str,
    departure_code: str,
    arrival_code: str,
    departure: datetime,
    arrival: datetime,
    price: float = 1000.0,
) -> FlightEvidence:
    return FlightEvidence(
        source_name="flights.ctrip.com",
        url="https://example.test/flight",
        price=price,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        captured_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
        origin=origin,
        destination=destination,
        travel_date=departure.date(),
        metadata={
            "flight_no": "TEST100",
            "departure_airport_code": departure_code,
            "arrival_airport_code": arrival_code,
        },
    )


def _flight_option(
    *,
    origin: str,
    destination: str,
    departure: datetime,
    arrival: datetime,
    price: float,
) -> FlightOption:
    evidence = _flight_evidence(
        origin=origin,
        destination=destination,
        departure_code=origin,
        arrival_code=destination,
        departure=departure,
        arrival=arrival,
        price=price,
    )
    return FlightEvidenceVerifier().verify(
        [evidence],
        FlightSearchIntent(
            origin=origin,
            destination=destination,
            travel_date=departure.date(),
        ),
    )[0]


def _train_edge(option: TrainOption) -> RouteEdge:
    return RouteEdge(
        edge_id="train:D638",
        mode="train",
        strategy="train_flight",
        origin=option.from_station,
        destination=option.to_station,
        travel_date=option.travel_date,
        price=option.lowest_price,
        departure_time=option.departure_at,
        arrival_time=option.arrival_at,
        duration_minutes=131,
        source="12306_mcp",
        hub_id="cn_重庆",
        leg_index=1,
        raw_option=option,
    )


def _flight_edge(option: FlightOption) -> RouteEdge:
    return RouteEdge(
        edge_id="flight:TEST100",
        mode="flight",
        strategy="train_flight",
        origin=option.origin,
        destination=option.destination,
        travel_date=option.travel_date,
        price=option.price,
        departure_time=option.departure_time,
        arrival_time=option.arrival_time,
        duration_minutes=int(
            (option.arrival_time - option.departure_time).total_seconds() // 60
        ),
        source="flight_page_search",
        hub_id="cn_重庆",
        leg_index=2,
        raw_option=option,
    )
