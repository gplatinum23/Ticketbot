from __future__ import annotations

import os

from flight_watch_agent import config
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
    env_file.write_text("FLIGHT_WATCH_LLM_MODEL=openai:gpt-from-file\n", encoding="utf-8")
    monkeypatch.setenv("FLIGHT_WATCH_ENV_FILE", str(env_file))
    monkeypatch.delenv("FLIGHT_WATCH_LLM_MODEL", raising=False)

    assert get_config("FLIGHT_WATCH_LLM_MODEL") == "openai:gpt-from-file"


def test_load_env_file_falls_back_to_project_root_env(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / ".env").write_text("FLIGHT_WATCH_LLM_MODEL=openai:gpt-root\n", encoding="utf-8")
    run_dir = tmp_path / "elsewhere"
    run_dir.mkdir()

    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.chdir(run_dir)
    monkeypatch.delenv("FLIGHT_WATCH_ENV_FILE", raising=False)
    monkeypatch.delenv("FLIGHT_WATCH_LLM_MODEL", raising=False)

    load_env_file()

    assert os.environ["FLIGHT_WATCH_LLM_MODEL"] == "openai:gpt-root"
