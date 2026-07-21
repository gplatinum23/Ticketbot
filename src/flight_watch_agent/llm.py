from __future__ import annotations

import re
from datetime import date
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from .config import get_config, load_env_file
from .models import FlightSearchIntent
from .places import (
    AirportQuery,
    airport_alias_map,
    airport_prompt_hints,
    get_airport_index,
    normalise_airport_code,
)


DEFAULT_LLM_MODEL = "openai:gpt-4.1-mini"


class TravelPlanIntent(BaseModel):
    action: Literal["plan_trip", "unsupported"] = Field(
        description="The user's intended action."
    )
    origin: str | None = Field(
        default=None,
        description="Origin city, airport name, station name, or airport code as expressed by the user.",
    )
    destination: str | None = Field(
        default=None,
        description="Destination city, airport name, station name, or airport code as expressed by the user.",
    )
    origin_place: "PlaceMention | None" = Field(
        default=None,
        description="Structured origin place extracted from the user request.",
    )
    destination_place: "PlaceMention | None" = Field(
        default=None,
        description="Structured destination place extracted from the user request.",
    )
    hub_places: list["PlaceMention"] = Field(
        default_factory=list,
        description="Explicit transfer hubs or user-suggested intermediate places.",
    )
    travel_date: date | None = Field(
        default=None,
        description="Travel date in ISO format.",
    )
    time_preference: str | None = Field(
        default=None,
        description="Time preference such as morning, afternoon, evening, earliest, or latest.",
    )
    budget_threshold: float | None = Field(
        default=None,
        description="Preferred maximum total price.",
    )
    currency: str = Field(default="CNY", description="ISO currency code.")
    max_segments: int = Field(
        default=3,
        description="Maximum number of trip segments.",
    )
    missing_fields: list[str] = Field(default_factory=list)
    clarification: str | None = Field(
        default=None,
        description="Short question to ask the user when required information is missing.",
    )


class PlaceMention(BaseModel):
    raw_text: str | None = Field(default=None, description="Original place text from the user.")
    kind: Literal["airport", "station", "city", "unknown"] = Field(default="unknown")
    official_airport_name: str | None = Field(
        default=None,
        description="Official English airport name when the place is an airport.",
    )
    city: str | None = Field(default=None, description="City name in English or Chinese.")
    country: str | None = Field(default=None, description="ISO country code if known.")
    iata_if_explicit: str | None = Field(
        default=None,
        description="IATA code only if explicitly provided by the user.",
    )
    station_name: str | None = Field(
        default=None,
        description="12306 Chinese station name when the place is a railway station.",
    )
    station_pinyin: str | None = Field(default=None, description="Station pinyin if known.")


def build_default_llm(model: str | None = None):
    load_env_file()
    model_name = model or get_config("FLIGHT_WATCH_LLM_MODEL", DEFAULT_LLM_MODEL)
    return init_chat_model(model_name, temperature=0)


def parse_travel_plan_intent(
    user_input: str,
    llm,
    *,
    today: date | None = None,
) -> TravelPlanIntent:
    today = today or date.today()
    structured_llm = llm.with_structured_output(TravelPlanIntent)
    result = structured_llm.invoke(
        [
            (
                "system",
                _system_prompt(today),
            ),
            ("human", user_input),
        ]
    )
    if isinstance(result, TravelPlanIntent):
        intent = result
    else:
        intent = TravelPlanIntent.model_validate(result)
    return _normalize_intent(intent, user_input=user_input)


def required_missing_fields(intent: TravelPlanIntent) -> list[str]:
    if intent.action != "plan_trip":
        return ["supported_action"]

    missing: list[str] = []
    for field_name in ("origin", "destination", "travel_date"):
        value = getattr(intent, field_name)
        if value is None or value == "":
            missing.append(field_name)
    return missing


def to_flight_search_intent(intent: TravelPlanIntent) -> FlightSearchIntent:
    missing = required_missing_fields(intent)
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    return FlightSearchIntent(
        origin=_required(intent.origin),
        destination=_required(intent.destination),
        travel_date=_required(intent.travel_date),
        time_preference=intent.time_preference,
        budget_threshold=intent.budget_threshold,
        currency=intent.currency,
        max_segments=intent.max_segments,
    )


def _normalize_intent(intent: TravelPlanIntent, *, user_input: str = "") -> TravelPlanIntent:
    updates: dict[str, object] = {}
    origin = _normalise_endpoint(intent.origin, intent.origin_place, user_input)
    destination = _normalise_endpoint(intent.destination, intent.destination_place, user_input)
    if origin is None or destination is None:
        inferred_origin, inferred_destination = _infer_route_from_text(user_input)
        origin = origin or inferred_origin
        destination = destination or inferred_destination
    if origin:
        updates["origin"] = origin
    if destination:
        updates["destination"] = destination
    time_preference = _normalise_time_preference(intent.time_preference or _infer_time_preference(user_input))
    if time_preference:
        updates["time_preference"] = time_preference
    if intent.currency:
        updates["currency"] = intent.currency.strip().upper()
    if intent.max_segments < 1:
        updates["max_segments"] = 1
    elif intent.max_segments > 3:
        updates["max_segments"] = 3
    missing = required_missing_fields(intent.model_copy(update=updates))
    updates["missing_fields"] = sorted(set(intent.missing_fields + missing))
    return intent.model_copy(update=updates)


def _system_prompt(today: date) -> str:
    place_hints = airport_prompt_hints()
    time_hints = (
        "\u4e0a\u5348/\u65e9\u4e0a -> morning; "
        "\u4e0b\u5348 -> afternoon; \u665a\u4e0a -> evening."
    )
    return f"""
You extract real-time travel planning requests into structured data.

Today is {today.isoformat()}.

Rules:
- Return plan_trip when the user asks to find, compare, search, plan, or recommend travel options.
- Preserve the user's city, airport, station name, or explicit airport code; do not invent an airport code.
- Also populate origin_place and destination_place with official English airport names or 12306 Chinese station names when possible.
- Put explicit transfer hubs mentioned by the user in hub_places. Use 12306 Chinese station names for railway stations.
- For iata_if_explicit, only copy an IATA code that the user explicitly wrote.
- The application will resolve airport codes from its local airport CSV and station index after this step.
- Common airport lookup hints:
  {place_hints}
- Convert Chinese time preferences: {time_hints}
- Convert relative dates into absolute ISO dates based on today.
- If no currency is provided, use CNY.
- If no max_segments is provided, use 3.
- Put every missing required field in missing_fields.
- Required fields are origin, destination, and travel_date.
""".strip()


def _required(value):
    if value is None:
        raise ValueError("Required value is missing.")
    return value


_PLACE_ALIASES = {
    **airport_alias_map(),
}

_TIME_ALIASES = {
    "\u4e0a\u5348": "morning",
    "\u65e9\u4e0a": "morning",
    "\u65e9\u6668": "morning",
    "\u4e0a\u5348\u51fa\u53d1": "morning",
    "\u4e0b\u5348": "afternoon",
    "\u5348\u540e": "afternoon",
    "\u665a\u4e0a": "evening",
    "\u591c\u95f4": "evening",
    "\u6700\u65e9": "earliest",
    "\u5c3d\u65e9": "earliest",
    "\u6700\u665a": "latest",
    "morning": "morning",
    "afternoon": "afternoon",
    "evening": "evening",
    "earliest": "earliest",
    "latest": "latest",
}


def _normalise_place(value: str | None) -> str | None:
    return normalise_airport_code(value)


def _normalise_place_mention(value: PlaceMention | None) -> str | None:
    if value is None:
        return None
    for candidate in (
        value.iata_if_explicit,
        value.official_airport_name,
        value.station_name,
        value.city,
        value.raw_text,
        value.station_pinyin,
    ):
        resolved = _normalise_place(candidate)
        if resolved:
            return resolved
    return None


def _normalise_endpoint(
    raw_value: str | None,
    mention: PlaceMention | None,
    user_input: str,
) -> str | None:
    if mention is not None:
        explicit_iata = (mention.iata_if_explicit or "").strip().upper()
        if explicit_iata and _text_contains_token(user_input, explicit_iata):
            return _normalise_place(explicit_iata)

        # raw_text is the model's copy of what the user actually wrote. Resolve it
        # before accepting a model-selected airport for a city-level request.
        if mention.raw_text and _text_contains_value(user_input, mention.raw_text):
            resolved = _normalise_place(mention.raw_text)
            if resolved:
                return resolved

    if raw_value and _text_contains_value(user_input, raw_value):
        resolved = _normalise_place(raw_value)
        if resolved:
            return resolved

    if mention is not None and mention.city:
        airport = get_airport_index().resolve(
            AirportQuery(
                name=mention.official_airport_name if mention.kind == "airport" else None,
                city=mention.city,
                country=mention.country or "",
                raw_text=mention.raw_text,
            )
        )
        if airport is not None:
            return airport.iata

    return _normalise_place_mention(mention) or _normalise_place(raw_value)


def _text_contains_token(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text, re.IGNORECASE))


def _text_contains_value(text: str, value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if re.fullmatch(r"[A-Za-z0-9]{3}", candidate):
        return _text_contains_token(text, candidate)
    return candidate.casefold() in text.casefold()


def _infer_route_from_text(text: str) -> tuple[str | None, str | None]:
    matches: list[tuple[int, int, str]] = []
    for alias, code in _PLACE_ALIASES.items():
        index = text.find(alias)
        if index >= 0:
            matches.append((index, -len(alias), code))
    ordered_codes: list[str] = []
    for _, _, code in sorted(matches):
        if code not in ordered_codes:
            ordered_codes.append(code)
    if len(ordered_codes) < 2:
        return (None, None)
    return (ordered_codes[0], ordered_codes[1])


def _infer_time_preference(text: str) -> str | None:
    for alias, value in _TIME_ALIASES.items():
        if alias in text:
            return value
    return None


def _normalise_time_preference(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    return _TIME_ALIASES.get(text) or _TIME_ALIASES.get(text.lower()) or text
