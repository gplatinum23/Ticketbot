from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILE = ".env"


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> None:
    env_path = Path(path or os.getenv("FLIGHT_WATCH_ENV_FILE", DEFAULT_ENV_FILE))
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_value(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value


def get_config(name: str, default: str | None = None) -> str | None:
    load_env_file()
    return os.getenv(name, default)


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
