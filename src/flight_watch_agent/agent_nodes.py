from __future__ import annotations

from collections import defaultdict
from math import atan2, cos, radians, sin, sqrt

from .agent_models import (
    CandidateHub,
    QueryBudget,
    QueryPlan,
    QueryPlanItem,
    RegionInfo,
    StrategySelection,
    TravelStrategy,
)
from .models import FlightSearchIntent
from .places import get_airport_index, get_station_index, normalise_airport_code, station_city_for_airport


ALL_STRATEGIES: list[TravelStrategy] = [
    "direct_flight",
    "direct_train",
    "train_flight",
    "flight_train",
    "train_train",
    "flight_flight",
]

SUPPORTED_EXECUTABLE_STRATEGIES: set[TravelStrategy] = {
    "direct_flight",
    "direct_train",
    "train_flight",
    "flight_train",
    "train_train",
    "flight_flight",
}


def classify_region(intent: FlightSearchIntent) -> RegionInfo:
    origin_country = _place_country(intent.origin)
    destination_country = _place_country(intent.destination)
    origin_is_china = origin_country == "CN"
    destination_is_china = destination_country == "CN"
    if origin_is_china and destination_is_china:
        route_type = "china_domestic"
    elif origin_is_china:
        route_type = "china_to_abroad"
    elif destination_is_china:
        route_type = "abroad_to_china"
    else:
        route_type = "abroad_to_abroad"
    return RegionInfo(
        origin_country=origin_country,
        destination_country=destination_country,
        origin_is_china=origin_is_china,
        destination_is_china=destination_is_china,
        route_type=route_type,
    )


def select_strategies(intent: FlightSearchIntent, region_info: RegionInfo) -> StrategySelection:
    enabled_by_region: dict[str, list[TravelStrategy]] = {
        "china_to_abroad": ["direct_flight", "train_flight", "flight_flight"],
        "abroad_to_china": ["direct_flight", "flight_train", "flight_flight"],
        "china_domestic": ALL_STRATEGIES,
        "abroad_to_abroad": ["direct_flight", "flight_flight"],
    }
    enabled = list(enabled_by_region[region_info.route_type])
    disabled: dict[TravelStrategy, str] = {}
    for strategy in ALL_STRATEGIES:
        if strategy in enabled:
            continue
        disabled[strategy] = _disabled_reason(strategy, region_info)
    return StrategySelection(enabled=enabled, disabled=disabled)


def generate_candidate_hubs(
    intent: FlightSearchIntent,
    strategy_selection: StrategySelection,
    *,
    budget: QueryBudget | None = None,
) -> list[CandidateHub]:
    budget = budget or QueryBudget()
    hubs = _hub_templates_for_route(intent, strategy_selection)
    grouped_count: dict[TravelStrategy, int] = defaultdict(int)
    selected: list[CandidateHub] = []
    for hub in sorted(hubs, key=lambda item: item.priority, reverse=True):
        usable_strategies = [
            strategy for strategy in hub.strategies if strategy in strategy_selection.enabled
        ]
        if not usable_strategies:
            continue
        if all(grouped_count[strategy] >= budget.max_hubs_per_strategy for strategy in usable_strategies):
            continue
        selected.append(
            CandidateHub(
                hub_id=hub.hub_id,
                city=hub.city,
                airport_codes=hub.airport_codes,
                train_places=hub.train_places,
                strategies=usable_strategies,
                priority=hub.priority,
                reason=hub.reason,
            )
        )
        for strategy in usable_strategies:
            grouped_count[strategy] += 1
    return selected


def generate_candidate_hubs_for_places(
    places: list[str],
    strategy_selection: StrategySelection,
) -> list[CandidateHub]:
    route_type = _route_type_from_strategy_selection(strategy_selection)
    hubs: list[CandidateHub] = []
    seen: set[str] = set()
    for place in places:
        hub = _candidate_hub_from_place(place, route_type)
        if hub is None:
            continue
        usable_strategies = [
            strategy for strategy in hub.strategies if strategy in strategy_selection.enabled
        ]
        if not usable_strategies or hub.hub_id in seen:
            continue
        seen.add(hub.hub_id)
        hubs.append(
            CandidateHub(
                hub_id=hub.hub_id,
                city=hub.city,
                airport_codes=hub.airport_codes,
                train_places=hub.train_places,
                strategies=usable_strategies,
                priority=hub.priority,
                reason=hub.reason,
            )
        )
    return hubs


def build_query_plan(
    intent: FlightSearchIntent,
    strategy_selection: StrategySelection,
    candidate_hubs: list[CandidateHub],
    *,
    budget: QueryBudget | None = None,
) -> QueryPlan:
    budget = budget or QueryBudget()
    items: list[QueryPlanItem] = []
    warnings: list[str] = []
    train_count = 0
    flight_count = 0

    def add_item(item: QueryPlanItem) -> None:
        nonlocal train_count, flight_count
        if item.mode == "train":
            if train_count >= budget.max_train_queries:
                warnings.append(f"train_query_budget_exhausted:{item.query_id}")
                return
            train_count += 1
        if item.mode == "flight":
            if flight_count >= budget.max_flight_queries:
                warnings.append(f"flight_query_budget_exhausted:{item.query_id}")
                return
            flight_count += 1
        items.append(item)

    if "direct_flight" in strategy_selection.enabled:
        add_item(
            QueryPlanItem(
                query_id="direct_flight:1",
                mode="flight",
                strategy="direct_flight",
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
                leg_index=1,
            )
        )

    if "direct_train" in strategy_selection.enabled:
        add_item(
            QueryPlanItem(
                query_id="direct_train:1",
                mode="train",
                strategy="direct_train",
                origin=intent.origin,
                destination=intent.destination,
                travel_date=intent.travel_date,
                leg_index=1,
            )
        )

    for hub in candidate_hubs:
        if "train_flight" in hub.strategies:
            train_destination = hub.train_places[0] if hub.train_places else hub.airport_codes[0]
            add_item(
                QueryPlanItem(
                    query_id=f"train_flight:{hub.hub_id}:train",
                    mode="train",
                    strategy="train_flight",
                    origin=intent.origin,
                    destination=train_destination,
                    travel_date=intent.travel_date,
                    leg_index=1,
                    hub_id=hub.hub_id,
                )
            )
            for airport_code in hub.airport_codes[:2]:
                add_item(
                    QueryPlanItem(
                        query_id=f"train_flight:{hub.hub_id}:flight:{airport_code}",
                        mode="flight",
                        strategy="train_flight",
                        origin=airport_code,
                        destination=intent.destination,
                        travel_date=intent.travel_date,
                        leg_index=2,
                        hub_id=hub.hub_id,
                    )
                )

        if "flight_train" in hub.strategies:
            train_origin = hub.train_places[0] if hub.train_places else hub.airport_codes[0]
            for airport_code in hub.airport_codes[:2]:
                add_item(
                    QueryPlanItem(
                        query_id=f"flight_train:{hub.hub_id}:flight:{airport_code}",
                        mode="flight",
                        strategy="flight_train",
                        origin=intent.origin,
                        destination=airport_code,
                        travel_date=intent.travel_date,
                        leg_index=1,
                        hub_id=hub.hub_id,
                    )
                )
            add_item(
                QueryPlanItem(
                    query_id=f"flight_train:{hub.hub_id}:train",
                    mode="train",
                    strategy="flight_train",
                    origin=train_origin,
                    destination=intent.destination,
                    travel_date=intent.travel_date,
                    leg_index=2,
                    hub_id=hub.hub_id,
                )
            )

        if "train_train" in hub.strategies:
            train_place = hub.train_places[0] if hub.train_places else hub.city
            add_item(
                QueryPlanItem(
                    query_id=f"train_train:{hub.hub_id}:train:1",
                    mode="train",
                    strategy="train_train",
                    origin=intent.origin,
                    destination=train_place,
                    travel_date=intent.travel_date,
                    leg_index=1,
                    hub_id=hub.hub_id,
                )
            )
            add_item(
                QueryPlanItem(
                    query_id=f"train_train:{hub.hub_id}:train:2",
                    mode="train",
                    strategy="train_train",
                    origin=train_place,
                    destination=intent.destination,
                    travel_date=intent.travel_date,
                    leg_index=2,
                    hub_id=hub.hub_id,
                )
            )

        if "flight_flight" in hub.strategies:
            for airport_code in hub.airport_codes[:2]:
                add_item(
                    QueryPlanItem(
                        query_id=f"flight_flight:{hub.hub_id}:flight:1:{airport_code}",
                        mode="flight",
                        strategy="flight_flight",
                        origin=intent.origin,
                        destination=airport_code,
                        travel_date=intent.travel_date,
                        leg_index=1,
                        hub_id=hub.hub_id,
                    )
                )
                add_item(
                    QueryPlanItem(
                        query_id=f"flight_flight:{hub.hub_id}:flight:2:{airport_code}",
                        mode="flight",
                        strategy="flight_flight",
                        origin=airport_code,
                        destination=intent.destination,
                        travel_date=intent.travel_date,
                        leg_index=2,
                        hub_id=hub.hub_id,
                    )
                )

        for strategy in hub.strategies:
            if strategy in SUPPORTED_EXECUTABLE_STRATEGIES:
                continue
            mode = "flight" if strategy in {"flight_train", "flight_flight"} else "train"
            items.append(
                QueryPlanItem(
                    query_id=f"{strategy}:{hub.hub_id}:placeholder",
                    mode=mode,
                    strategy=strategy,
                    origin=intent.origin,
                    destination=intent.destination,
                    travel_date=intent.travel_date,
                    leg_index=1,
                    hub_id=hub.hub_id,
                    executable=False,
                    status="not_implemented",
                    reason=f"{strategy} is modelled but not executable in this MVP step.",
                )
            )

    return QueryPlan(items=items, budget=budget, warnings=warnings)


def _place_country(value: str) -> str | None:
    airport = get_airport_index().resolve(value)
    if airport is not None:
        return airport.country
    airport_code = normalise_airport_code(value)
    if airport_code and airport_code != value:
        airport = get_airport_index().resolve(airport_code)
        if airport is not None:
            return airport.country
    station = get_station_index().resolve(value)
    if station is not None:
        return "CN"
    return _COUNTRY_OVERRIDES.get(value.strip().upper()) or _COUNTRY_OVERRIDES.get(value.strip())


def _disabled_reason(strategy: TravelStrategy, region_info: RegionInfo) -> str:
    if strategy == "direct_train":
        return "direct train requires both origin and destination in China."
    if strategy == "train_flight":
        return "train_flight requires the first train leg to start in China."
    if strategy == "flight_train":
        return "flight_train requires the second train leg to end in China."
    if strategy == "train_train":
        return "train_train requires all train legs to stay in China."
    return f"{strategy} is not enabled for {region_info.route_type}."


def _hub_templates_for_route(
    intent: FlightSearchIntent,
    strategy_selection: StrategySelection,
) -> list[CandidateHub]:
    route_type = _route_type_from_strategy_selection(strategy_selection)
    return _generate_hubs_from_indexes(intent, route_type)


def _route_type_from_strategy_selection(strategy_selection: StrategySelection) -> str:
    if strategy_selection.enabled == ["direct_flight", "flight_flight"]:
        return "abroad_to_abroad"
    if "train_flight" in strategy_selection.enabled and "direct_train" not in strategy_selection.enabled:
        return "china_to_abroad"
    if "flight_train" in strategy_selection.enabled and "direct_train" not in strategy_selection.enabled:
        return "abroad_to_china"
    return "china_domestic"


def _generate_hubs_from_indexes(intent: FlightSearchIntent, route_type: str) -> list[CandidateHub]:
    airport_index = get_airport_index()
    station_index = get_station_index()
    origin_airport = airport_index.resolve(intent.origin)
    destination_airport = airport_index.resolve(intent.destination)
    origin_code = origin_airport.iata if origin_airport is not None else normalise_airport_code(intent.origin)
    destination_code = (
        destination_airport.iata if destination_airport is not None else normalise_airport_code(intent.destination)
    )

    grouped: dict[str, dict[str, object]] = {}
    for airport in airport_index.airports:
        if airport.iata in {origin_code, destination_code}:
            continue
        city_name = station_city_for_airport(airport)
        if airport.country == "CN" and city_name is None and route_type != "abroad_to_abroad":
            continue
        if route_type in {"china_to_abroad", "abroad_to_china", "china_domestic"} and airport.country != "CN":
            continue
        if route_type == "abroad_to_abroad" and airport.iata in {origin_code, destination_code}:
            continue

        hub_city = city_name or airport.city or airport.name
        key = f"{airport.country}:{hub_city}"
        entry = grouped.setdefault(
            key,
            {
                "city": hub_city,
                "country": airport.country,
                "airport_codes": [],
                "train_places": [],
                "priority": _hub_priority(airport, origin_airport, destination_airport, route_type),
                "reason": _hub_reason(route_type),
            },
        )
        entry["airport_codes"].append(airport.iata)  # type: ignore[index, union-attr]
        entry["priority"] = max(  # type: ignore[index]
            float(entry["priority"]),
            _hub_priority(airport, origin_airport, destination_airport, route_type),
        )
        if city_name:
            train_place = station_index.primary_station_for_city(city_name) or city_name
            train_places = entry["train_places"]  # type: ignore[index]
            if isinstance(train_places, list) and train_place not in train_places:
                train_places.append(train_place)

    hubs: list[CandidateHub] = []
    for key, entry in grouped.items():
        country = str(entry["country"])
        train_places = list(entry["train_places"]) if isinstance(entry["train_places"], list) else []
        airport_codes = _rank_airport_codes(list(entry["airport_codes"]))  # type: ignore[arg-type]
        strategies = _strategies_for_index_hub(route_type, country, airport_codes, train_places)
        if not strategies:
            continue
        hubs.append(
            CandidateHub(
                hub_id=_hub_id_from_key(key),
                city=str(entry["city"]),
                airport_codes=airport_codes,
                train_places=train_places,
                strategies=strategies,
                priority=float(entry["priority"]),
                reason=str(entry["reason"]),
            )
        )
    return sorted(hubs, key=lambda item: item.priority, reverse=True)


def _candidate_hub_from_place(place: str, route_type: str) -> CandidateHub | None:
    airport_index = get_airport_index()
    station_index = get_station_index()
    airport = airport_index.resolve(place)
    station_name = station_index.resolve(place)
    station_record = None
    if station_name is not None:
        station_record = station_index.by_name.get(station_name) or station_index.by_code.get(station_name.upper())

    city_name = station_record.city_name if station_record is not None else None
    if city_name is None and airport is not None:
        city_name = station_city_for_airport(airport)

    airport_codes: list[str] = []
    if airport is not None:
        airport_codes.append(airport.iata)
    if city_name is not None:
        for candidate in airport_index.airports:
            if station_city_for_airport(candidate) == city_name:
                airport_codes.append(candidate.iata)
    airport_codes = _rank_airport_codes(airport_codes)[:2]

    train_places: list[str] = []
    if station_name is not None:
        train_places.append(station_name)
    elif city_name is not None:
        primary_station = station_index.primary_station_for_city(city_name)
        if primary_station is not None:
            train_places.append(primary_station)

    country = airport.country if airport is not None else ("CN" if train_places else "")
    city = city_name or (airport.city if airport is not None else None) or station_name or place
    strategies = _strategies_for_index_hub(route_type, country, airport_codes, train_places)
    if not strategies:
        return None
    return CandidateHub(
        hub_id=_hub_id_from_key(f"{country}:{city}"),
        city=city,
        airport_codes=airport_codes,
        train_places=train_places,
        strategies=strategies,
        priority=10.0,
        reason=f"Resolved from user/LLM hub '{place}' using airport and station indexes.",
    )


def _strategies_for_index_hub(
    route_type: str,
    country: str,
    airport_codes: list[str],
    train_places: list[str],
) -> list[TravelStrategy]:
    has_airport = bool(airport_codes)
    has_train = bool(train_places)
    if route_type == "china_to_abroad":
        strategies: list[TravelStrategy] = []
        if has_train and has_airport:
            strategies.append("train_flight")
        if has_airport:
            strategies.append("flight_flight")
        return strategies
    if route_type == "abroad_to_china":
        strategies = []
        if has_airport and has_train:
            strategies.append("flight_train")
        if has_airport:
            strategies.append("flight_flight")
        return strategies
    if route_type == "china_domestic" and country == "CN":
        strategies = []
        if has_train:
            strategies.append("train_train")
        if has_train and has_airport:
            strategies.extend(["train_flight", "flight_train"])
        if has_airport:
            strategies.append("flight_flight")
        return strategies
    if route_type == "abroad_to_abroad" and has_airport:
        return ["flight_flight"]
    return []


def _hub_priority(airport, origin_airport, destination_airport, route_type: str) -> float:
    score = 1.0
    if "International" in airport.name:
        score += 0.25
    if airport.country == "CN":
        score += 0.1
    if route_type == "china_to_abroad":
        score += _distance_score(origin_airport, airport)
    elif route_type == "abroad_to_china":
        score += _distance_score(destination_airport, airport)
    elif route_type == "china_domestic":
        score += max(
            _distance_score(origin_airport, airport),
            _distance_score(destination_airport, airport),
        )
    else:
        score += max(
            _distance_score(origin_airport, airport),
            _distance_score(destination_airport, airport),
        )
    return score


def _distance_score(reference, airport) -> float:
    if reference is None:
        return 0.0
    distance = _distance_km(reference, airport)
    if distance is None:
        return 0.0
    return 1_000.0 / (distance + 300.0)


def _distance_km(a, b) -> float | None:
    if a.latitude is None or a.longitude is None or b.latitude is None or b.longitude is None:
        return None
    lat1 = radians(a.latitude)
    lat2 = radians(b.latitude)
    dlat = lat2 - lat1
    dlon = radians(b.longitude - a.longitude)
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * atan2(sqrt(h), sqrt(1 - h))


def _rank_airport_codes(codes: list[str]) -> list[str]:
    return list(dict.fromkeys(sorted(codes)))


def _hub_id_from_key(key: str) -> str:
    return (
        key.lower()
        .replace(":", "_")
        .replace(" ", "_")
        .replace("'", "")
        .replace("/", "_")
    )


def _hub_reason(route_type: str) -> str:
    return f"Generated from resources/airports.csv and resources/station_name.js for {route_type}."


_COUNTRY_OVERRIDES = {
    "SIN": "SG",
    "NRT": "JP",
    "HND": "JP",
    "TYO": "JP",
    "东京": "JP",
    "東京": "JP",
    "大理": "CN",
    "DLU": "CN",
}
