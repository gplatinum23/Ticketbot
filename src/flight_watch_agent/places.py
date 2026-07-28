from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import timedelta, timezone, tzinfo
from functools import lru_cache
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATION_FILE = PROJECT_ROOT / "resources" / "station_name.js"
DEFAULT_AIRPORT_FILE = PROJECT_ROOT / "resources" / "airports_normalized_with_flight_potential.csv"
FALLBACK_AIRPORT_FILE = PROJECT_ROOT / "resources" / "airports.csv"
DEFAULT_AIRPORT_SUPPLEMENT_FILE = PROJECT_ROOT / "resources" / "airports.json"


@dataclass(frozen=True)
class StationRecord:
    name: str
    telecode: str
    pinyin: str
    short: str
    city_name: str


@dataclass(frozen=True)
class AirportRecord:
    iata: str
    name: str
    country: str
    region: str
    latitude: float | None = None
    longitude: float | None = None
    city: str | None = None
    flight_potential_score: float | None = None
    flight_tier: str | None = None


@dataclass(frozen=True)
class AirportQuery:
    name: str | None = None
    city: str | None = None
    iata: str | None = None
    country: str = ""
    raw_text: str | None = None


PlaceKind = Literal["city", "airport", "station", "unknown"]
PlaceResolutionRole = Literal["query", "actual", "any"]


@dataclass(frozen=True)
class CityAirportGroup:
    city_id: str
    query_code: str
    display_name: str
    english_name: str
    country: str
    airport_codes: tuple[str, ...]
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlaceRef:
    """A typed location identity; query cities and actual airports never share an id."""

    raw: str
    kind: PlaceKind
    canonical_id: str
    display_name: str
    city_id: str | None = None
    city_name: str | None = None
    country: str | None = None
    query_code: str | None = None
    airport_codes: tuple[str, ...] = ()
    airport_code: str | None = None
    station_code: str | None = None

    @property
    def known(self) -> bool:
        return self.kind != "unknown"


class StationIndex:
    def __init__(self, stations: list[StationRecord]) -> None:
        self.stations = stations
        self.by_name: dict[str, StationRecord] = {}
        self.by_code: dict[str, StationRecord] = {}
        self.by_pinyin: dict[str, StationRecord] = {}
        self.city_names: set[str] = set()
        self.city_by_pinyin: dict[str, str] = {}
        self.primary_station_by_city: dict[str, str] = {}
        for station in stations:
            self.by_name.setdefault(station.name, station)
            self.by_code.setdefault(station.telecode.upper(), station)
            self.by_pinyin.setdefault(station.pinyin.upper(), station)
            self.by_pinyin.setdefault(station.short.upper(), station)
            if station.city_name:
                self.city_names.add(station.city_name)
                self.city_by_pinyin.setdefault(station.pinyin.upper(), station.city_name)
                self.city_by_pinyin.setdefault(station.short.upper(), station.city_name)
                self.primary_station_by_city.setdefault(station.city_name, station.city_name)
                if station.name == station.city_name:
                    self.primary_station_by_city[station.city_name] = station.name

    def resolve(self, value: str) -> str | None:
        text = value.strip()
        if not text:
            return None
        upper = text.upper()
        if upper in self.by_code:
            return upper
        if text in self.by_name:
            return self.by_name[text].name
        if text in self.city_names:
            return text
        if upper in self.by_pinyin:
            return self.by_pinyin[upper].name
        return None

    def resolve_city_by_pinyin(self, value: str | None) -> str | None:
        if not value:
            return None
        return self.city_by_pinyin.get(value.strip().upper())

    def primary_station_for_city(self, city_name: str | None) -> str | None:
        if not city_name:
            return None
        return self.primary_station_by_city.get(city_name)


class AirportIndex:
    def __init__(self, airports: list[AirportRecord]) -> None:
        self.airports = airports
        self.by_iata = {airport.iata: airport for airport in airports}
        self.by_name: dict[str, AirportRecord] = {}
        for airport in airports:
            self.by_name.setdefault(_normalise_lookup_key(airport.name), airport)

    def resolve(self, value: str | AirportQuery) -> AirportRecord | None:
        query = _airport_query_from_value(value)
        if query is None:
            return None

        if query.iata:
            airport = self.by_iata.get(query.iata.upper())
            if airport is not None:
                return airport
            return None

        direct_keys = [
            query.name,
            query.city,
            query.raw_text,
            " ".join(part for part in (query.city, query.name) if part),
        ]
        for key in direct_keys:
            airport = self.by_name.get(_normalise_lookup_key(key or ""))
            if airport is not None and _airport_matches_country(airport, query.country):
                return airport

        return self._best_match(query)

    def _best_match(self, query: AirportQuery) -> AirportRecord | None:
        query_terms = _airport_query_terms(query)
        if not query_terms:
            return None

        best: tuple[tuple[int, float, int], AirportRecord] | None = None
        for airport in self.airports:
            if not _airport_matches_country(airport, query.country):
                continue
            airport_key = _normalise_lookup_key(airport.name)
            score = 0
            for term in query_terms:
                if term in airport_key:
                    score += 1
            if score == 0:
                continue
            if query.city and _normalise_lookup_key(query.city) in airport_key:
                score += 3
            if query.name and _normalise_lookup_key(query.name) in airport_key:
                score += 5
            if score < 2:
                continue
            rank = (
                score,
                airport.flight_potential_score or 0.0,
                _flight_tier_rank(airport.flight_tier),
            )
            if best is None or rank > best[0]:
                best = (rank, airport)
        return best[1] if best is not None else None


@lru_cache(maxsize=1)
def get_station_index() -> StationIndex:
    return StationIndex(_load_station_records(DEFAULT_STATION_FILE))


@lru_cache(maxsize=1)
def get_airport_index() -> AirportIndex:
    return AirportIndex(_load_airport_records(DEFAULT_AIRPORT_FILE))


def resolve_place(
    value: str,
    *,
    role: PlaceResolutionRole = "any",
) -> PlaceRef:
    """Resolve a legacy place string without collapsing a city into one airport.

    `query` interprets Ctrip city codes such as BJS/SHA/CTU as airport groups.
    `actual` interprets evidence endpoint codes as physical airports.
    """

    raw = value
    text = value.strip()
    if not text:
        return _unknown_place(raw)
    upper = text.upper()

    city_group = _city_group_for_alias(text)
    if city_group is not None and (
        role == "query"
        or upper == city_group.query_code
        and upper not in get_airport_index().by_iata
        or not re.fullmatch(r"[A-Za-z0-9]{3}", text)
    ):
        return _city_place(raw, city_group)

    airport = get_airport_index().resolve(text)
    if airport is not None and airport.iata != "BJS":
        group = _city_group_for_airport(airport)
        return PlaceRef(
            raw=raw,
            kind="airport",
            canonical_id=f"airport:{airport.iata}",
            display_name=airport.name,
            city_id=group.city_id if group else _generic_city_id(airport),
            city_name=group.display_name if group else airport.city,
            country=airport.country or None,
            query_code=airport.iata,
            airport_codes=(airport.iata,),
            airport_code=airport.iata,
        )

    station = _station_record_for_value(text)
    if station is not None:
        group = _city_group_for_station(station)
        return PlaceRef(
            raw=raw,
            kind="station",
            canonical_id=f"station:{station.telecode}",
            display_name=station.name,
            city_id=group.city_id if group else f"city:CN:{station.city_name}",
            city_name=group.display_name if group else station.city_name,
            country="CN",
            query_code=station.name,
            airport_codes=group.airport_codes if group else (),
            station_code=station.telecode,
        )

    if city_group is not None:
        return _city_place(raw, city_group)
    return _unknown_place(raw)


def resolve_air_query_place(value: str) -> PlaceRef:
    return resolve_place(value, role="query")


def resolve_actual_airport(value: str | None) -> PlaceRef | None:
    if not value:
        return None
    place = resolve_place(value, role="actual")
    return place if place.kind == "airport" else None


def resolve_station_place(value: str | None) -> PlaceRef | None:
    if not value:
        return None
    station = _station_record_for_value(value)
    if station is None:
        return None
    group = _city_group_for_station(station)
    return PlaceRef(
        raw=value,
        kind="station",
        canonical_id=f"station:{station.telecode}",
        display_name=station.name,
        city_id=group.city_id if group else f"city:CN:{station.city_name}",
        city_name=group.display_name if group else station.city_name,
        country="CN",
        query_code=station.name,
        airport_codes=group.airport_codes if group else (),
        station_code=station.telecode,
    )


def air_endpoint_matches(requested: str | PlaceRef, observed: str | PlaceRef) -> bool:
    requested_place = (
        requested if isinstance(requested, PlaceRef) else resolve_air_query_place(requested)
    )
    observed_place = (
        observed if isinstance(observed, PlaceRef) else resolve_place(observed, role="actual")
    )
    if requested_place.kind == "unknown" or observed_place.kind != "airport":
        return False
    if requested_place.kind == "airport":
        return requested_place.airport_code == observed_place.airport_code
    return bool(
        requested_place.city_id
        and observed_place.city_id
        and requested_place.city_id == observed_place.city_id
        and observed_place.airport_code in requested_place.airport_codes
    )


def query_endpoint_matches(
    requested: str | PlaceRef,
    observed: str | PlaceRef,
) -> bool:
    requested_place = (
        requested if isinstance(requested, PlaceRef) else resolve_air_query_place(requested)
    )
    observed_place = (
        observed if isinstance(observed, PlaceRef) else resolve_air_query_place(observed)
    )
    if not requested_place.known or not observed_place.known:
        return False
    if requested_place.kind == "airport":
        return (
            observed_place.kind == "airport"
            and requested_place.airport_code == observed_place.airport_code
        )
    if observed_place.kind == "airport":
        return bool(
            requested_place.city_id == observed_place.city_id
            and observed_place.airport_code in requested_place.airport_codes
        )
    return bool(
        requested_place.city_id
        and requested_place.city_id == observed_place.city_id
    )


def timezone_for_airport(value: str | None) -> tzinfo:
    airport = resolve_actual_airport(value)
    country = airport.country if airport is not None else None
    offset_hours, label = _TIMEZONE_BY_COUNTRY.get(
        country or "",
        (0, "UTC"),
    )
    return timezone(timedelta(hours=offset_hours), label)


def normalise_airport_code(value: str | None) -> str | None:
    if not value:
        return None
    place = resolve_air_query_place(value)
    if place.kind == "airport":
        return place.airport_code
    if place.kind == "city":
        return place.query_code

    station = get_station_index().by_name.get(value.strip())
    if station is not None and station.city_name:
        group = _city_group_for_station(station)
        if group is not None:
            return group.query_code
        city_airport = get_airport_index().resolve(station.city_name)
        return city_airport.iata if city_airport is not None else None
    return None


def normalise_train_query_place(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    upper = text.upper()
    airport = get_airport_index().resolve(text)
    if airport is not None and airport.country != "CN" and upper in _SKIP_TRAIN_FOR_AIRPORT_CODES:
        return None
    if airport is not None and airport.country == "CN":
        return _TRAIN_CITY_BY_IATA.get(airport.iata) or station_city_for_airport(airport)

    station = get_station_index().resolve(text)
    if station is not None:
        return station

    if airport is not None:
        if airport.country != "CN":
            return None
        return _TRAIN_CITY_BY_IATA.get(airport.iata)

    return text


def airport_prompt_hints() -> str:
    parts: list[str] = []
    for alias, query in _AIRPORT_QUERY_ALIASES.items():
        airport = get_airport_index().resolve(query)
        if airport is not None and not re.fullmatch(r"[A-Za-z]{3}", alias):
            parts.append(f"{alias} -> {airport.iata}")
    return "; ".join(parts)


def airport_alias_map() -> dict[str, str]:
    result: dict[str, str] = {}
    for airport in get_airport_index().airports:
        result[airport.iata] = airport.iata
        result[airport.iata.upper()] = airport.iata
    for alias, query in _AIRPORT_QUERY_ALIASES.items():
        airport = get_airport_index().resolve(query)
        if airport is not None:
            result[alias] = airport.iata
            result[alias.upper()] = airport.iata
    return result


def _load_station_records(path: Path) -> list[StationRecord]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"station_names\s*=\s*'(?P<data>.*)';?\s*$", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Could not parse station names from {path}")

    stations: list[StationRecord] = []
    for raw_entry in match.group("data").split("@"):
        if not raw_entry:
            continue
        fields = raw_entry.split("|")
        if len(fields) < 8:
            continue
        stations.append(
            StationRecord(
                name=fields[1],
                telecode=fields[2].upper(),
                pinyin=fields[3],
                short=fields[4],
                city_name=fields[7],
            )
        )
    return stations


def _load_airport_records(path: Path) -> list[AirportRecord]:
    if not path.exists():
        path = FALLBACK_AIRPORT_FILE
    airports: list[AirportRecord] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            iata = (row.get("iata_code") or "").strip().upper()
            name = (row.get("name") or "").strip()
            if not iata or not name or not re.fullmatch(r"[A-Z0-9]{3}", iata):
                continue
            airports.append(
                AirportRecord(
                    iata=iata,
                    name=name,
                    country=(row.get("iso_country") or "").strip().upper(),
                    region=(row.get("iso_region") or "").strip().upper(),
                    latitude=_parse_float(row.get("latitude_deg")),
                    longitude=_parse_float(row.get("longitude_deg")),
                    city=_derive_airport_city(name),
                    flight_potential_score=_parse_float(row.get("flight_potential_score")),
                    flight_tier=_normalise_flight_tier(row.get("flight_tier")),
                )
            )
    existing_codes = {airport.iata for airport in airports}
    for airport in _load_airport_supplements(DEFAULT_AIRPORT_SUPPLEMENT_FILE):
        if airport.iata not in existing_codes:
            airports.append(airport)
            existing_codes.add(airport.iata)
    for airport in _VIRTUAL_AIRPORT_RECORDS:
        if airport.iata not in existing_codes:
            airports.append(airport)
    return airports


def _load_airport_supplements(path: Path) -> list[AirportRecord]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    airports: list[AirportRecord] = []
    if not isinstance(rows, list):
        return airports
    for row in rows:
        if not isinstance(row, dict):
            continue
        iata = str(row.get("iata") or "").strip().upper()
        name = str(row.get("name") or "").strip()
        country = str(row.get("country") or "").strip().upper()
        if not iata or not name or not re.fullmatch(r"[A-Z0-9]{3}", iata):
            continue
        airports.append(
            AirportRecord(
                iata=iata,
                name=name,
                country=country,
                region="",
                city=str(row.get("city") or "").strip() or _derive_airport_city(name),
                flight_potential_score=_parse_float(row.get("flight_potential_score")),
                flight_tier=_normalise_flight_tier(row.get("flight_tier")),
            )
        )
    return airports


def station_city_for_airport(airport: AirportRecord) -> str | None:
    if airport.country != "CN":
        return None
    city = get_station_index().resolve_city_by_pinyin(airport.city)
    if city is not None:
        return city
    return _TRAIN_CITY_BY_IATA.get(airport.iata)


def _airport_query_from_value(value: str | AirportQuery) -> AirportQuery | None:
    if isinstance(value, AirportQuery):
        return value
    text = value.strip()
    if not text:
        return None
    alias = _AIRPORT_QUERY_ALIASES.get(text) or _AIRPORT_QUERY_ALIASES.get(text.upper())
    if alias is not None:
        return alias
    if re.fullmatch(r"[A-Za-z0-9]{3}", text):
        return AirportQuery(iata=text.upper(), country="", raw_text=text)
    return AirportQuery(name=text, city=text, country="", raw_text=text)


def _airport_query_terms(query: AirportQuery) -> list[str]:
    terms: list[str] = []
    for value in (query.name, query.city, query.raw_text):
        key = _normalise_lookup_key(value or "")
        if not key:
            continue
        terms.append(key)
        terms.extend(
            part
            for part in key.split(" ")
            if len(part) > 2 and part not in _AIRPORT_LOOKUP_STOPWORDS
        )
    return list(dict.fromkeys(terms))


def _airport_matches_country(airport: AirportRecord, country: str) -> bool:
    return not country or airport.country == country.upper()


def _normalise_lookup_key(value: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", value.casefold()).strip()


def _parse_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _normalise_flight_tier(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip().upper()
    return text if text in {"T1", "T2", "T3", "T4"} else None


def _flight_tier_rank(value: str | None) -> int:
    return {"T1": 4, "T2": 3, "T3": 2, "T4": 1}.get(value or "", 0)


def _derive_airport_city(name: str) -> str | None:
    cleaned = name.strip()
    if not cleaned:
        return None
    multi_word_cities = (
        "Hong Kong",
        "Kuala Lumpur",
        "New York",
        "Ho Chi Minh",
        "Los Angeles",
        "San Francisco",
        "Las Vegas",
        "Phnom Penh",
        "Abu Dhabi",
    )
    lowered = cleaned.casefold()
    for city in multi_word_cities:
        if lowered.startswith(city.casefold() + " "):
            return city
    first = cleaned.split()[0]
    if first.casefold() in {"airport", "international", "regional"}:
        return None
    return first


def _unknown_place(raw: str) -> PlaceRef:
    text = raw.strip()
    return PlaceRef(
        raw=raw,
        kind="unknown",
        canonical_id=f"unknown:{text.casefold()}",
        display_name=text,
    )


def _city_place(raw: str, group: CityAirportGroup) -> PlaceRef:
    return PlaceRef(
        raw=raw,
        kind="city",
        canonical_id=group.city_id,
        display_name=group.display_name,
        city_id=group.city_id,
        city_name=group.display_name,
        country=group.country,
        query_code=group.query_code,
        airport_codes=group.airport_codes,
    )


def _city_group_for_alias(value: str) -> CityAirportGroup | None:
    key = _normalise_lookup_key(value)
    return _CITY_GROUP_BY_ALIAS.get(key)


def _city_group_for_airport(airport: AirportRecord) -> CityAirportGroup | None:
    for group in _CITY_AIRPORT_GROUPS.values():
        if airport.iata in group.airport_codes:
            return group
    airport_city = _normalise_lookup_key(airport.city or "")
    for group in _CITY_AIRPORT_GROUPS.values():
        if airport.country == group.country and airport_city == _normalise_lookup_key(group.english_name):
            return group
    return None


def _city_group_for_station(station: StationRecord) -> CityAirportGroup | None:
    key = _normalise_lookup_key(station.city_name)
    return _CITY_GROUP_BY_ALIAS.get(key)


def _station_record_for_value(value: str) -> StationRecord | None:
    index = get_station_index()
    text = value.strip()
    upper = text.upper()
    if upper in index.by_code:
        return index.by_code[upper]
    if text in index.by_name:
        return index.by_name[text]
    if upper in index.by_pinyin:
        return index.by_pinyin[upper]
    if text in index.city_names:
        primary = index.primary_station_for_city(text)
        return index.by_name.get(primary or "")
    return None


def _generic_city_id(airport: AirportRecord) -> str | None:
    if airport.country == "CN":
        station_city = station_city_for_airport(airport)
        if station_city:
            return f"city:CN:{station_city}"
    if not airport.city:
        return None
    return f"city:{airport.country}:{_normalise_lookup_key(airport.city)}"


_CITY_AIRPORT_GROUPS: dict[str, CityAirportGroup] = {
    "BJS": CityAirportGroup(
        city_id="city:CN:beijing",
        query_code="BJS",
        display_name="北京",
        english_name="Beijing",
        country="CN",
        airport_codes=("PEK", "PKX"),
        aliases=("北京", "Beijing"),
    ),
    "SHA": CityAirportGroup(
        city_id="city:CN:shanghai",
        query_code="SHA",
        display_name="上海",
        english_name="Shanghai",
        country="CN",
        airport_codes=("SHA", "PVG"),
        aliases=("上海", "Shanghai"),
    ),
    "CTU": CityAirportGroup(
        city_id="city:CN:chengdu",
        query_code="CTU",
        display_name="成都",
        english_name="Chengdu",
        country="CN",
        airport_codes=("CTU", "TFU"),
        aliases=("成都", "Chengdu"),
    ),
    "CKG": CityAirportGroup(
        city_id="city:CN:chongqing",
        query_code="CKG",
        display_name="重庆",
        english_name="Chongqing",
        country="CN",
        airport_codes=("CKG",),
        aliases=("重庆", "Chongqing"),
    ),
    "CJU": CityAirportGroup(
        city_id="city:KR:jeju",
        query_code="CJU",
        display_name="济州岛",
        english_name="Jeju",
        country="KR",
        airport_codes=("CJU",),
        aliases=("济州", "济州岛", "Jeju"),
    ),
    "SIN": CityAirportGroup(
        city_id="city:SG:singapore",
        query_code="SIN",
        display_name="新加坡",
        english_name="Singapore",
        country="SG",
        airport_codes=("SIN",),
        aliases=("新加坡", "Singapore"),
    ),
}

_CITY_GROUP_BY_ALIAS: dict[str, CityAirportGroup] = {}
for _group in _CITY_AIRPORT_GROUPS.values():
    for _alias in (
        _group.query_code,
        _group.display_name,
        _group.english_name,
        *_group.aliases,
    ):
        _CITY_GROUP_BY_ALIAS[_normalise_lookup_key(_alias)] = _group


_TIMEZONE_BY_COUNTRY = {
    "CN": (8, "Asia/Shanghai"),
    "HK": (8, "Asia/Hong_Kong"),
    "MO": (8, "Asia/Macau"),
    "TW": (8, "Asia/Taipei"),
    "KR": (9, "Asia/Seoul"),
    "JP": (9, "Asia/Tokyo"),
    "SG": (8, "Asia/Singapore"),
    "TH": (7, "Asia/Bangkok"),
    "VN": (7, "Asia/Ho_Chi_Minh"),
    "MY": (8, "Asia/Kuala_Lumpur"),
}


_AIRPORT_QUERY_ALIASES: dict[str, AirportQuery] = {
    "BJS": AirportQuery(iata="BJS", city="Beijing", country="CN", raw_text="BJS"),
    "北京": AirportQuery(iata="BJS", city="Beijing", country="CN", raw_text="北京"),
    "北京首都": AirportQuery(name="Beijing Capital", city="Beijing", country="CN", raw_text="北京首都"),
    "北京大兴": AirportQuery(name="Beijing Daxing", city="Beijing", country="CN", raw_text="北京大兴"),
    "SHA": AirportQuery(iata="SHA", city="Shanghai", country="CN", raw_text="SHA"),
    "上海": AirportQuery(iata="SHA", city="Shanghai", country="CN", raw_text="上海"),
    "上海虹桥": AirportQuery(name="Shanghai Hongqiao", city="Shanghai", country="CN", raw_text="上海虹桥"),
    "上海浦东": AirportQuery(name="Shanghai Pudong", city="Shanghai", country="CN", raw_text="上海浦东"),
    "南京": AirportQuery(name="Nanjing Lukou", city="Nanjing", country="CN", raw_text="南京"),
    "南京禄口": AirportQuery(name="Nanjing Lukou", city="Nanjing", country="CN", raw_text="南京禄口"),
    "新加坡": AirportQuery(name="Singapore Changi", city="Singapore", country="SG", raw_text="新加坡"),
    "樟宜": AirportQuery(name="Singapore Changi", city="Singapore", country="SG", raw_text="樟宜"),
    "济州岛": AirportQuery(iata="CJU", city="Jeju", country="KR", raw_text="济州岛"),
    "济州": AirportQuery(iata="CJU", city="Jeju", country="KR", raw_text="济州"),
    "成都": AirportQuery(name="Chengdu Shuangliu", city="Chengdu", country="CN", raw_text="成都"),
    "成都双流": AirportQuery(name="Chengdu Shuangliu", city="Chengdu", country="CN", raw_text="成都双流"),
    "成都天府": AirportQuery(name="Chengdu Tianfu", city="Chengdu", country="CN", raw_text="成都天府"),
    "昆明": AirportQuery(name="Kunming Changshui", city="Kunming", country="CN", raw_text="昆明"),
    "昆明长水": AirportQuery(name="Kunming Changshui", city="Kunming", country="CN", raw_text="昆明长水"),
}

_TRAIN_CITY_BY_IATA = {
    "BJS": "北京",
    "PEK": "北京",
    "PKX": "北京",
    "SHA": "上海",
    "PVG": "上海",
    "NKG": "南京",
    "TFU": "成都",
    "CTU": "成都",
    "KMG": "昆明",
    "CAN": "广州",
    "SZX": "深圳",
}

_AIRPORT_CITY_CODES = {"BJS"}

_SKIP_TRAIN_FOR_AIRPORT_CODES = {"SIN"}

_VIRTUAL_AIRPORT_RECORDS = [
    AirportRecord(
        iata="BJS",
        name="Beijing city airports",
        country="CN",
        region="CN-11",
        city="Beijing",
    )
]

_AIRPORT_LOOKUP_STOPWORDS = {
    "airport",
    "international",
    "regional",
    "domestic",
    "city",
    "airports",
}
