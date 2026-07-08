from __future__ import annotations

from datetime import date
from typing import Literal

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from .config import get_config, load_env_file


DEFAULT_LLM_MODEL = "openai:gpt-4.1-mini"


class MonitorIntent(BaseModel):
    action: Literal["create_monitor", "unsupported"] = Field(
        description="The user's intended action."
    )
    origin: str | None = Field(
        default=None,
        description="Origin airport or city code, such as SHA, PVG, NRT, NYC.",
    )
    destination: str | None = Field(
        default=None,
        description="Destination airport or city code, such as SHA, PVG, NRT, NYC.",
    )
    depart_date: date | None = Field(
        default=None,
        description="Outbound date in ISO format.",
    )
    return_date: date | None = Field(
        default=None,
        description="Return date in ISO format. Null for one-way trips.",
    )
    threshold_price: float | None = Field(
        default=None,
        description="Notify when price is less than or equal to this value.",
    )
    currency: str = Field(default="CNY", description="ISO currency code.")
    interval_seconds: int = Field(
        default=3600,
        description="How often to check prices, in seconds.",
    )
    missing_fields: list[str] = Field(default_factory=list)
    clarification: str | None = Field(
        default=None,
        description="Short question to ask the user when required information is missing.",
    )


def build_default_llm(model: str | None = None):
    load_env_file()
    model_name = model or get_config("FLIGHT_WATCH_LLM_MODEL", DEFAULT_LLM_MODEL)
    return init_chat_model(model_name, temperature=0)


def parse_monitor_intent(user_input: str, llm, *, today: date | None = None) -> MonitorIntent:
    today = today or date.today()
    structured_llm = llm.with_structured_output(MonitorIntent)
    result = structured_llm.invoke(
        [
            (
                "system",
                _system_prompt(today),
            ),
            ("human", user_input),
        ]
    )
    if isinstance(result, MonitorIntent):
        intent = result
    else:
        intent = MonitorIntent.model_validate(result)
    return _normalize_intent(intent)


def required_missing_fields(intent: MonitorIntent) -> list[str]:
    if intent.action != "create_monitor":
        return ["supported_action"]

    missing: list[str] = []
    for field_name in ("origin", "destination", "depart_date", "threshold_price"):
        value = getattr(intent, field_name)
        if value is None or value == "":
            missing.append(field_name)
    return missing


def _normalize_intent(intent: MonitorIntent) -> MonitorIntent:
    updates: dict[str, object] = {}
    if intent.origin:
        updates["origin"] = intent.origin.strip().upper()
    if intent.destination:
        updates["destination"] = intent.destination.strip().upper()
    if intent.currency:
        updates["currency"] = intent.currency.strip().upper()
    missing = required_missing_fields(intent.model_copy(update=updates))
    updates["missing_fields"] = sorted(set(intent.missing_fields + missing))
    return intent.model_copy(update=updates)


def _system_prompt(today: date) -> str:
    return f"""
You extract flight price watch requests into structured data.

Today is {today.isoformat()}.

Rules:
- Only return create_monitor when the user asks to watch, monitor, track, or alert on flight prices.
- Use airport or city codes when the user provides them directly.
- If a city has multiple airports and the user did not give a code, use a common city code only when unambiguous; otherwise mark the field missing.
- Convert relative dates into absolute ISO dates based on today.
- If no check interval is provided, use 3600 seconds.
- If no currency is provided, use CNY.
- Put every missing required field in missing_fields.
- Required fields are origin, destination, depart_date, and threshold_price.
""".strip()
