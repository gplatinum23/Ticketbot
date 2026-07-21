from __future__ import annotations

from flight_watch_agent import app
from flight_watch_agent.flight_react import LlmFlightEvidenceJudge
from flight_watch_agent.travel_plan_graph import LlmHubPlanner
from flight_watch_agent.travel_plan_graph import LlmRoutePlanner


def test_default_travel_agent_uses_combined_hub_planner_and_batch_judge(monkeypatch):
    captured = {}
    llm = object()

    monkeypatch.setattr(app, "load_env_file", lambda: None)
    monkeypatch.setattr(app, "build_default_llm", lambda *_args, **_kwargs: llm)
    monkeypatch.setattr(app, "build_default_flight_web_search_tool", lambda: object())
    monkeypatch.setattr(app, "build_default_flight_page_extractor", lambda: object())

    def capture_graph(**kwargs):
        captured.update(kwargs)
        return "graph"

    monkeypatch.setattr(app, "build_travel_plan_graph", capture_graph)

    graph = app.build_default_travel_plan_agent(include_train=False)

    assert graph == "graph"
    assert isinstance(captured["hub_planner"], LlmHubPlanner)
    assert isinstance(captured["evidence_judge"], LlmFlightEvidenceJudge)
    assert callable(captured["evidence_judge"].judge_many)
    assert "hub_proposer" not in captured
    assert "hub_endpoint_validator" not in captured


def test_default_travel_agent_uses_fast_model_for_structured_tasks_and_route_ranking(monkeypatch):
    captured = {}
    models = []

    monkeypatch.setattr(app, "load_env_file", lambda: None)
    monkeypatch.setattr(
        app,
        "get_config",
        lambda name, default=None: "openai:fast-model" if name == "FLIGHT_WATCH_FAST_LLM_MODEL" else default,
    )

    def build_llm(model=None):
        value = model or "openai:main-model"
        models.append(value)
        return value

    monkeypatch.setattr(app, "build_default_llm", build_llm)
    monkeypatch.setattr(app, "build_default_flight_web_search_tool", lambda: object())
    monkeypatch.setattr(app, "build_default_flight_page_extractor", lambda: object())
    monkeypatch.setattr(
        app,
        "build_travel_plan_graph",
        lambda **kwargs: captured.update(kwargs) or "graph",
    )

    app.build_default_travel_plan_agent(include_train=False)

    assert models == ["openai:main-model", "openai:fast-model"]
    assert isinstance(captured["route_planner"], LlmRoutePlanner)
    assert captured["route_planner"].llm == "openai:fast-model"
    assert captured["hub_planner"].llm == "openai:fast-model"
    assert captured["evidence_judge"].llm == "openai:fast-model"


def test_default_travel_agent_allows_dedicated_route_model(monkeypatch):
    captured = {}

    monkeypatch.setattr(app, "load_env_file", lambda: None)
    monkeypatch.setattr(
        app,
        "get_config",
        lambda name, default=None: {
            "FLIGHT_WATCH_FAST_LLM_MODEL": "openai:fast-model",
            "FLIGHT_WATCH_ROUTE_LLM_MODEL": "openai:route-model",
        }.get(name, default),
    )
    monkeypatch.setattr(app, "build_default_llm", lambda model=None: model or "openai:main-model")
    monkeypatch.setattr(app, "build_default_flight_web_search_tool", lambda: object())
    monkeypatch.setattr(app, "build_default_flight_page_extractor", lambda: object())
    monkeypatch.setattr(
        app,
        "build_travel_plan_graph",
        lambda **kwargs: captured.update(kwargs) or "graph",
    )

    app.build_default_travel_plan_agent(include_train=False)

    assert captured["route_planner"].llm == "openai:route-model"
