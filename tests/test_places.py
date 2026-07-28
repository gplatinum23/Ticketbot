from __future__ import annotations

from flight_watch_agent.places import (
    AirportQuery,
    get_airport_index,
    get_station_index,
    normalise_airport_code,
    normalise_train_query_place,
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
