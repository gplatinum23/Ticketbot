from __future__ import annotations

from datetime import date

from flight_watch_agent.agent_nodes import (
    build_query_plan,
    classify_region,
    generate_candidate_hubs,
    select_strategies,
)
from flight_watch_agent.models import FlightSearchIntent
from flight_watch_agent.places import get_airport_index, get_station_index


def test_region_classifier_handles_route_types():
    assert classify_region(_intent("NKG", "SIN")).route_type == "china_to_abroad"
    assert classify_region(_intent("SIN", "DLU")).route_type == "abroad_to_china"
    assert classify_region(_intent("CTU", "DLU")).route_type == "china_domestic"
    assert classify_region(_intent("SIN", "NRT")).route_type == "abroad_to_abroad"


def test_strategy_selector_matches_region_rules():
    china_to_abroad = select_strategies(_intent("NKG", "SIN"), classify_region(_intent("NKG", "SIN")))
    abroad_to_china = select_strategies(_intent("SIN", "DLU"), classify_region(_intent("SIN", "DLU")))
    china_domestic = select_strategies(_intent("CTU", "DLU"), classify_region(_intent("CTU", "DLU")))
    abroad_to_abroad = select_strategies(_intent("SIN", "NRT"), classify_region(_intent("SIN", "NRT")))

    assert china_to_abroad.enabled == ["direct_flight", "train_flight", "flight_flight"]
    assert abroad_to_china.enabled == ["direct_flight", "flight_train", "flight_flight"]
    assert set(china_domestic.enabled) == {
        "direct_flight",
        "direct_train",
        "train_flight",
        "flight_train",
        "train_train",
        "flight_flight",
    }
    assert abroad_to_abroad.enabled == ["direct_flight", "flight_flight"]


def test_candidate_hub_generator_returns_index_backed_hubs():
    intent = _intent("NKG", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs = generate_candidate_hubs(intent, selection)

    airport_index = get_airport_index()
    station_index = get_station_index()
    assert hubs
    assert all(hub.strategies for hub in hubs)
    assert all(hub.reason for hub in hubs)
    assert all(
        airport_index.resolve(code) is not None
        for hub in hubs
        for code in hub.airport_codes
    )
    assert all(
        station_index.resolve(place) is not None
        for hub in hubs
        for place in hub.train_places
    )


def test_candidate_hub_generator_models_train_train_hubs_from_station_index():
    intent = _intent("CTU", "DLU")
    selection = select_strategies(intent, classify_region(intent))

    hubs = generate_candidate_hubs(intent, selection)

    train_train_hubs = [hub for hub in hubs if "train_train" in hub.strategies]
    assert train_train_hubs
    assert all(hub.train_places for hub in train_train_hubs)
    assert all(
        get_station_index().resolve(place) is not None
        for hub in train_train_hubs
        for place in hub.train_places
    )


def test_query_planner_builds_direct_and_train_flight_queries_from_indexes():
    intent = _intent("NKG", "SIN")
    selection = select_strategies(intent, classify_region(intent))
    hubs = generate_candidate_hubs(intent, selection)

    plan = build_query_plan(intent, selection, hubs)

    executable = [item for item in plan.items if item.executable]
    assert any(item.query_id == "direct_flight:1" and item.mode == "flight" for item in executable)
    assert any(
        item.strategy == "train_flight"
        and item.mode == "train"
        and get_station_index().resolve(item.destination) is not None
        for item in executable
    )
    assert any(
        item.strategy == "train_flight"
        and item.mode == "flight"
        and get_airport_index().resolve(item.origin) is not None
        and item.destination == "SIN"
        for item in executable
    )
    assert any(item.strategy == "flight_flight" and item.mode == "flight" for item in executable)
    assert all(item.executable for item in plan.items)


def _intent(origin: str, destination: str) -> FlightSearchIntent:
    return FlightSearchIntent(
        origin=origin,
        destination=destination,
        travel_date=date(2026, 7, 10),
        currency="CNY",
    )
