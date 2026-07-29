from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from flight_watch_agent.agent_models import RouteEdge
from flight_watch_agent.flight_react import FlightEvidenceVerifier
from flight_watch_agent.models import (
    FlightEvidence,
    FlightOption,
    FlightSearchIntent,
    TrainOption,
)
from flight_watch_agent.route_diversity import (
    DiversityCandidate,
    RouteSkeleton,
    ValueProfile,
    build_route_skeleton,
    select_diverse_candidates,
    shared_downstream_flight_service,
)


CHINA = timezone(timedelta(hours=8))
KOREA = timezone(timedelta(hours=9))


def test_route_skeleton_uses_modes_actual_endpoints_services_and_hub():
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
        "CKG",
        "CJU",
        "9C8751+9C8625",
        datetime(2026, 7, 31, 20, 15, tzinfo=CHINA),
        datetime(2026, 8, 1, 22, 30, tzinfo=KOREA),
    )
    edges = [
        _train_edge(train),
        _flight_edge(flight, leg_index=2),
    ]

    skeleton = build_route_skeleton(
        route_type="train_flight",
        edges=edges,
        transfer_city="重庆",
        transfer_airport="CKG",
    )

    assert skeleton.mode_sequence == ("train", "flight")
    assert skeleton.actual_endpoints == (
        "station:ICW",
        "station:CUW",
        "airport:CKG",
        "airport:CJU",
    )
    assert skeleton.core_services[0].startswith("train:D638:")
    assert skeleton.core_services[1].startswith(
        "flight:9C8751+9C8625:airport:CKG:airport:CJU:"
    )
    assert skeleton.transfer_hubs == ("重庆", "CKG", "cn_重庆")
    assert shared_downstream_flight_service(edges) == skeleton.core_services[-1]


def test_exact_route_skeleton_duplicates_are_removed_after_ranking():
    skeleton = _skeleton("train:D638", "flight:9C8625", "重庆")
    selection = select_diverse_candidates(
        [
            _candidate("ranked-first", skeleton, rank=0),
            _candidate("same-skeleton", skeleton, rank=1),
        ]
    )

    assert [item.route_id for item in selection.selected] == ["ranked-first"]
    assert selection.exact_duplicates_removed == 1


def test_chengdu_to_jeju_top_five_limits_same_downstream_flight_variants():
    shared_ckg_flight = "flight:9C8751+9C8625:CKG:CJU:2026-07-31T20:15"
    candidates = [
        _candidate(
            "ckg-d638",
            _skeleton("train:D638", shared_ckg_flight, "重庆"),
            rank=0,
            price=1700,
            duration=2365,
            wait=659,
            risk=0.35,
            family=shared_ckg_flight,
        ),
        _candidate(
            "ckg-d620",
            _skeleton("train:D620", shared_ckg_flight, "重庆"),
            rank=1,
            price=1698,
            duration=2370,
            wait=680,
            risk=0.32,
            family=shared_ckg_flight,
        ),
        _candidate(
            "ckg-d3058",
            _skeleton("train:D3058", shared_ckg_flight, "重庆"),
            rank=2,
            price=1692,
            duration=2420,
            wait=701,
            risk=0.30,
            family=shared_ckg_flight,
        ),
        _candidate(
            "pek-k118",
            _skeleton("train:K118", "flight:OZ336", "北京"),
            rank=3,
            price=1612,
            duration=2700,
            wait=684,
            risk=0.42,
            family="flight:OZ336:PEK:CJU",
        ),
        _candidate(
            "can-flight",
            _skeleton("flight:CA430", "flight:CZ300", "广州"),
            rank=4,
            price=1850,
            duration=720,
            wait=150,
            risk=0.12,
            family="flight:CZ300:CAN:CJU",
        ),
        _candidate(
            "sha-flight",
            _skeleton("flight:MU5402", "flight:MU5059", "上海"),
            rank=5,
            price=1920,
            duration=650,
            wait=110,
            risk=0.18,
            family="flight:MU5059:PVG:CJU",
        ),
    ]

    selection = select_diverse_candidates(candidates)
    selected_ids = [item.route_id for item in selection.selected]
    profiles = {
        profile
        for item in selection.selected
        for profile in item.value_profiles
    }

    assert len(selected_ids) == 5
    assert len([route_id for route_id in selected_ids if route_id.startswith("ckg-")]) <= 2
    assert "pek-k118" in selected_ids
    assert "can-flight" in selected_ids
    assert "sha-flight" in selected_ids
    assert ValueProfile.BEST_OVERALL in profiles
    assert ValueProfile.LOWEST_PRICE in profiles
    assert ValueProfile.SHORTEST_DURATION in profiles
    assert ValueProfile.LOWER_TRANSFER_RISK in profiles


def test_selection_reports_when_fewer_than_five_distinct_routes_exist():
    selection = select_diverse_candidates(
        [
            _candidate(
                "only-route",
                _skeleton("flight:3U1", "", ""),
                rank=0,
            )
        ]
    )

    assert len(selection.selected) == 1
    assert selection.available_distinct_routes == 1


def test_lower_transfer_risk_profile_accounts_for_transfer_count():
    transfer = _candidate(
        "transfer",
        _skeleton("train:D1", "flight:CA1", "重庆"),
        rank=0,
        risk=0.1,
        transfer_count=1,
    )
    direct = _candidate(
        "direct",
        _skeleton("flight:3U1", "", ""),
        rank=1,
        risk=0.1,
        transfer_count=0,
    )

    selection = select_diverse_candidates([transfer, direct])
    lower_risk = next(
        item
        for item in selection.selected
        if ValueProfile.LOWER_TRANSFER_RISK in item.value_profiles
    )

    assert lower_risk.route_id == "direct"


def _candidate(
    route_id: str,
    skeleton: RouteSkeleton,
    *,
    rank: int,
    price: float = 1000,
    duration: int = 600,
    wait: int | None = 120,
    risk: float = 0.2,
    family: str | None = None,
    transfer_count: int = 1,
) -> DiversityCandidate:
    return DiversityCandidate(
        route_id=route_id,
        skeleton=skeleton,
        total_price=price,
        total_duration_minutes=duration,
        transfer_wait_minutes=wait,
        transfer_count=transfer_count,
        risk_score=risk,
        ranked_index=rank,
        shared_downstream_service=family,
    )


def _skeleton(
    first_service: str,
    second_service: str,
    hub: str,
) -> RouteSkeleton:
    services = tuple(
        service
        for service in (first_service, second_service)
        if service
    )
    return RouteSkeleton(
        route_type="train_flight" if first_service.startswith("train") else "flight_flight",
        mode_sequence=(
            ("train", "flight")
            if first_service.startswith("train")
            else ("flight",) if not second_service else ("flight", "flight")
        ),
        actual_endpoints=("city:CTU", f"hub:{hub}", "airport:CJU"),
        core_services=services,
        transfer_hubs=(hub,) if hub else (),
    )


def _flight_option(
    origin: str,
    destination: str,
    flight_no: str,
    departure: datetime,
    arrival: datetime,
) -> FlightOption:
    evidence = FlightEvidence(
        source_name="flights.ctrip.com",
        url="https://example.test/flight",
        price=1607.0,
        currency="CNY",
        departure_time=departure,
        arrival_time=arrival,
        captured_at=datetime(2026, 7, 29, tzinfo=timezone.utc),
        origin=origin,
        destination=destination,
        travel_date=departure.date(),
        metadata={"flight_no": flight_no},
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
        hub_id="cn_重庆",
        leg_index=1,
        raw_option=option,
    )


def _flight_edge(option: FlightOption, *, leg_index: int) -> RouteEdge:
    return RouteEdge(
        edge_id="flight:9C8751+9C8625",
        mode="flight",
        strategy="train_flight",
        origin=option.origin,
        destination=option.destination,
        travel_date=option.travel_date,
        price=option.price,
        departure_time=option.departure_time,
        arrival_time=option.arrival_time,
        hub_id="cn_重庆",
        leg_index=leg_index,
        raw_option=option,
    )
