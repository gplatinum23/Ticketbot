from __future__ import annotations

from datetime import date

from flight_watch_agent.agent_nodes import (
    build_query_plan,
    classify_region,
    generate_candidate_hubs,
    generate_candidate_hubs_for_place_mentions,
    select_strategies,
)
from flight_watch_agent.agent_models import CandidateHub, QueryBudget, StrategySelection
from flight_watch_agent.llm import PlaceMention
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


def test_query_budget_defaults_are_expanded_for_broader_search():
    budget = QueryBudget()

    assert budget.max_hubs_per_strategy == 10
    assert budget.max_flight_queries == 50
    assert budget.max_train_queries == 50
    assert budget.max_total_routes == 150


def test_query_budget_counts_unique_external_queries_not_strategy_references():
    intent = _intent("NKG", "SIN")
    selection = StrategySelection(
        enabled=["direct_flight", "train_flight", "flight_flight"],
        disabled={},
    )
    hub = CandidateHub(
        hub_id="cn_guangzhou",
        city="Guangzhou",
        airport_codes=["CAN"],
        train_places=["Guangzhou"],
        strategies=["train_flight", "flight_flight"],
        priority=1.0,
        reason="test",
    )

    plan = build_query_plan(
        intent,
        selection,
        [hub],
        budget=QueryBudget(max_flight_queries=2, max_train_queries=1),
    )

    can_to_sin = [
        item
        for item in plan.items
        if item.mode == "flight" and item.origin == "CAN" and item.destination == "SIN"
    ]
    assert {item.strategy for item in can_to_sin} == {"train_flight", "flight_flight"}
    assert not any(
        item.mode == "flight" and item.origin == "NKG" and item.destination == "CAN"
        for item in plan.items
    )
    assert plan.warnings == ["flight_query_budget_exhausted:flight_flight:cn_guangzhou:flight:1:CAN"]


def test_candidate_hub_generator_resolves_structured_airport_mention():
    intent = _intent("NKG", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs, warnings = generate_candidate_hubs_for_place_mentions(
        [
            PlaceMention(
                kind="airport",
                official_airport_name="Guangzhou Baiyun International Airport",
                city="Guangzhou",
                country="CN",
            )
        ],
        selection,
    )

    assert warnings == []
    assert len(hubs) == 1
    assert "CAN" in hubs[0].airport_codes
    assert "train_flight" in hubs[0].strategies


def test_candidate_hub_generator_warns_for_unresolved_llm_hub():
    intent = _intent("NKG", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs, warnings = generate_candidate_hubs_for_place_mentions(
        [PlaceMention(kind="airport", official_airport_name="Not A Real Airport")],
        selection,
    )

    assert hubs == []
    assert warnings == ["hub_place_unresolved:Not A Real Airport"]


def test_international_hub_generator_filters_low_potential_neighbor_airports():
    intent = _intent("CTU", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs = generate_candidate_hubs(
        intent,
        selection,
        budget=QueryBudget(max_hubs_per_strategy=5),
    )

    assert len(hubs) <= 5
    assert all(hub.flight_tier in {"T1", "T2"} for hub in hubs)
    assert all((hub.flight_potential_score or 0) >= 0.50 for hub in hubs)
    all_codes = {code for hub in hubs for code in hub.airport_codes}
    assert not (all_codes & {"CTU", "TFU", "HZU", "MIG", "ZKL", "YBP", "GYS", "BZX", "XIC"})
    assert {"CKG", "KMG", "CAN"} & all_codes


def test_llm_hub_suggestion_with_t4_airport_is_filtered_by_potential_rules():
    intent = _intent("CTU", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs, warnings = generate_candidate_hubs_for_place_mentions(
        [
            PlaceMention(
                kind="airport",
                official_airport_name="Zigong Fengming Airport",
                city="Zigong",
                country="CN",
            )
        ],
        selection,
    )

    assert hubs == []
    assert warnings == ["hub_place_unresolved:Zigong Fengming Airport"]


def test_llm_hub_suggestion_prefers_airport_code_over_station_short_name():
    intent = _intent("CTU", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs, warnings = generate_candidate_hubs_for_place_mentions(
        [PlaceMention(kind="airport", iata_if_explicit="SZX")],
        selection,
    )

    assert warnings == []
    assert len(hubs) == 1
    assert hubs[0].city == "\u6df1\u5733"
    assert hubs[0].airport_codes == ["SZX"]
    assert hubs[0].train_places == ["\u6df1\u5733"]


def test_llm_hub_suggestion_does_not_expand_same_city_t4_airport():
    intent = _intent("CTU", "SIN")
    selection = select_strategies(intent, classify_region(intent))

    hubs, warnings = generate_candidate_hubs_for_place_mentions(
        [PlaceMention(kind="airport", iata_if_explicit="WUH")],
        selection,
    )

    assert warnings == []
    assert len(hubs) == 1
    assert hubs[0].airport_codes == ["WUH"]


def _intent(origin: str, destination: str) -> FlightSearchIntent:
    return FlightSearchIntent(
        origin=origin,
        destination=destination,
        travel_date=date(2026, 7, 10),
        currency="CNY",
    )
