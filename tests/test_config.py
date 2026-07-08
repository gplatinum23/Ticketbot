from __future__ import annotations

import os

from flight_watch_agent.config import get_config, load_env_file


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key\nFLIGHT_WATCH_LLM_MODEL='openai:gpt-test'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FLIGHT_WATCH_LLM_MODEL", raising=False)

    load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "test-key"
    assert os.environ["FLIGHT_WATCH_LLM_MODEL"] == "openai:gpt-test"


def test_load_env_file_does_not_override_existing_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "shell-key")

    load_env_file(env_file)

    assert os.environ["OPENAI_API_KEY"] == "shell-key"


def test_get_config_loads_file_from_env_pointer(tmp_path, monkeypatch):
    env_file = tmp_path / "custom.env"
    env_file.write_text("FLIGHT_WATCH_DB=data/from-file.sqlite3\n", encoding="utf-8")
    monkeypatch.setenv("FLIGHT_WATCH_ENV_FILE", str(env_file))
    monkeypatch.delenv("FLIGHT_WATCH_DB", raising=False)

    assert get_config("FLIGHT_WATCH_DB") == "data/from-file.sqlite3"
