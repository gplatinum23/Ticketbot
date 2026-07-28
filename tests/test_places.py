from __future__ import annotations

from flight_watch_agent.places import (
    AirportQuery,
    get_airport_index,
    get_station_index,
    air_endpoint_matches,
    normalise_airport_code,
    normalise_train_query_place,
    query_endpoint_matches,
    resolve_actual_airport,
    resolve_air_query_place,
    resolve_place,
    resolve_station_place,
)


def test_station_index_parses_12306_station_file():
    index = get_station_index()

    assert len(index.stations) > 3000
    assert index.resolve("南京") == "南京"
    assert index.resolve("南京南") == "南京南"
    assert index.resolve("NKH") == "NKH"
    assert index.resolve("nanjingnan") == "南京南"


def test_airport_index_loads_csv_and_resolves_structured_queries():
    index = get_airport_index()

    assert len(index.airports) > 8000
    assert index.resolve(AirportQuery(iata="SIN")).name == "Singapore Changi Airport"
    assert index.resolve(AirportQuery(name="Nanjing Lukou", city="Nanjing", country="CN")).iata == "NKG"
    assert index.resolve(AirportQuery(iata="PVG")).name == "Shanghai Pudong International Airport"
    assert index.resolve(AirportQuery(iata="CTU")).flight_tier == "T1"
    assert index.resolve(AirportQuery(iata="CTU")).flight_potential_score == 0.86
    assert index.resolve(AirportQuery(iata="ZKL")).flight_tier == "T4"
    assert index.resolve(AirportQuery(iata="ZKL")).flight_potential_score == 0.16


def test_airport_index_normalises_names_to_iata():
    assert normalise_airport_code("南京") == "NKG"
    assert normalise_airport_code("南京南") == "NKG"
    assert normalise_airport_code("新加坡") == "SIN"
    assert normalise_airport_code("济州岛") == "CJU"
    assert normalise_airport_code("成都天府") == "TFU"
    assert normalise_airport_code("nkg") == "NKG"


def test_airport_index_prefers_high_potential_airport_for_city_query():
    airport = get_airport_index().resolve(AirportQuery(city="Chengdu", country="CN"))

    assert airport is not None
    assert airport.iata == "TFU"


def test_train_place_normalisation_uses_station_and_airport_tables_separately():
    assert normalise_train_query_place("南京南") == "南京南"
    assert normalise_train_query_place("NKH") == "NKH"
    assert normalise_train_query_place("NKG") == "南京"
    assert normalise_train_query_place("TFU") == "成都"
    assert normalise_train_query_place("SIN") is None
    assert normalise_train_query_place("新加坡") is None


def test_city_airport_and_station_identifiers_have_distinct_boundaries():
    beijing = resolve_air_query_place("BJS")
    capital = resolve_actual_airport("PEK")
    daxing = resolve_actual_airport("PKX")
    beijing_west = resolve_place("北京西")

    assert beijing.kind == "city"
    assert beijing.canonical_id == "city:CN:beijing"
    assert beijing.airport_codes == ("PEK", "PKX")
    assert capital is not None and capital.canonical_id == "airport:PEK"
    assert daxing is not None and daxing.canonical_id == "airport:PKX"
    assert beijing_west.kind == "station"
    assert beijing_west.canonical_id == "station:BXP"
    assert len({beijing.canonical_id, capital.canonical_id, daxing.canonical_id, beijing_west.canonical_id}) == 4


def test_city_query_accepts_member_airports_but_explicit_airport_is_exact():
    assert air_endpoint_matches("BJS", "PEK")
    assert air_endpoint_matches("BJS", "PKX")
    assert air_endpoint_matches("北京", "PEK")
    assert air_endpoint_matches("PEK", "PEK")
    assert not air_endpoint_matches("PEK", "PKX")
    assert not air_endpoint_matches("BJS", "CKG")
    assert query_endpoint_matches("BJS", "北京")


def test_unknown_airport_code_is_not_silently_treated_as_valid_airport():
    unknown = resolve_air_query_place("ZZZ")

    assert unknown.kind == "unknown"
    assert not unknown.known
    assert normalise_airport_code("ZZZ") is None


def test_station_telecode_and_iata_collision_requires_explicit_context():
    shanghai_station = resolve_station_place("SHH")
    shishmaref_airport = resolve_actual_airport("SHH")

    assert shanghai_station is not None
    assert shanghai_station.kind == "station"
    assert shanghai_station.canonical_id == "station:SHH"
    assert shishmaref_airport is not None
    assert shishmaref_airport.kind == "airport"
    assert shishmaref_airport.canonical_id == "airport:SHH"
