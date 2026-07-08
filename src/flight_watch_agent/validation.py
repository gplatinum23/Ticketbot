from __future__ import annotations

import re
from datetime import date


_LOCATION_RE = re.compile(r"^[A-Za-z0-9]{2,8}$")


def validate_monitor_input(
    *,
    origin: str,
    destination: str,
    depart_date: date,
    return_date: date | None,
    threshold_price: float,
    interval_seconds: int,
) -> None:
    if not _LOCATION_RE.fullmatch(origin.strip()):
        raise ValueError("origin must be a 2-8 character airport or city code.")
    if not _LOCATION_RE.fullmatch(destination.strip()):
        raise ValueError("destination must be a 2-8 character airport or city code.")
    if origin.strip().upper() == destination.strip().upper():
        raise ValueError("origin and destination must be different.")
    if return_date is not None and return_date < depart_date:
        raise ValueError("return_date must be on or after depart_date.")
    if threshold_price <= 0:
        raise ValueError("threshold_price must be greater than 0.")
    if interval_seconds < 60:
        raise ValueError("interval_seconds must be at least 60.")
