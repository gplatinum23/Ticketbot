from __future__ import annotations

from datetime import date

from flight_watch_agent.llm import PlaceMention, TravelPlanIntent
from flight_watch_agent.request_graph import build_travel_plan_request_graph


class FakeStructuredLlm:
    def __init__(self, intent: TravelPlanIntent) -> None:
        self.intent = intent

    def invoke(self, messages):
        return self.intent


class FakeLlm:
    def __init__(self, intent: TravelPlanIntent) -> None:
        self.intent = intent

    def with_structured_output(self, schema):
        return FakeStructuredLlm(self.intent)


class FakeTravelPlanGraph:
    def __init__(self) -> None:
        self.invocations = []

    def invoke(self, state):
        self.invocations.append(state)
        intent = state["intent"]
        return {
            "intent": intent,
            "response": f"planned {intent.origin}->{intent.destination} on {intent.travel_date.isoformat()}",
        }


def test_request_graph_plans_trip_from_structured_llm_output():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="bjs",
                destination="sha",
                travel_date=date(2026, 7, 9),
                time_preference="morning",
                budget_threshold=1200,
                currency="cny",
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke({"user_input": "tomorrow Beijing to Shanghai under 1200"})

    assert state["flight_search_intent"].origin == "BJS"
    assert state["flight_search_intent"].destination == "SHA"
    assert state["flight_search_intent"].travel_date == date(2026, 7, 9)
    assert state["response"] == "planned BJS->SHA on 2026-07-09"
    assert len(plan_graph.invocations) == 1


def test_request_graph_asks_for_missing_required_fields():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="BJS",
                destination=None,
                travel_date=date(2026, 7, 9),
                clarification="Where do you want to go?",
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke({"user_input": "plan from Beijing tomorrow"})

    assert state["response"] == "Where do you want to go?"
    assert state["errors"] == ["missing_fields:destination"]
    assert plan_graph.invocations == []


def test_request_graph_normalises_chinese_airport_names():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="\u65b0\u52a0\u5761",
                destination="\u6210\u90fd\u5929\u5e9c",
                travel_date=date(2026, 7, 9),
                time_preference="\u4e0a\u5348",
                budget_threshold=10000,
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke(
        {
            "user_input": (
                "\u5e2e\u6211\u67e5\u4e00\u4e0b 2026-07-09 "
                "\u65b0\u52a0\u5761\u5230\u6210\u90fd\u5929\u5e9c\uff0c"
                "\u4e0a\u5348\u51fa\u53d1"
            )
        }
    )

    assert state["flight_search_intent"].origin == "SIN"
    assert state["flight_search_intent"].destination == "TFU"
    assert state["flight_search_intent"].time_preference == "morning"
    assert state["response"] == "planned SIN->TFU on 2026-07-09"


def test_request_graph_infers_route_from_user_input_when_llm_misses_places():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin=None,
                destination=None,
                travel_date=date(2026, 7, 9),
                budget_threshold=10000,
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke(
        {
            "user_input": (
                "\u5e2e\u6211\u67e5\u4e00\u4e0b 2026-07-09 "
                "\u65b0\u52a0\u5761\u5230\u6210\u90fd\u5929\u5e9c\uff0c"
                "\u4e0a\u5348\u51fa\u53d1"
            )
        }
    )

    assert state["flight_search_intent"].origin == "SIN"
    assert state["flight_search_intent"].destination == "TFU"
    assert state["flight_search_intent"].time_preference == "morning"
    assert state["response"] == "planned SIN->TFU on 2026-07-09"


def test_request_graph_passes_structured_hub_places_to_plan_graph():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="CTU",
                destination="DLU",
                travel_date=date(2026, 7, 10),
                hub_places=[
                    PlaceMention(
                        kind="station",
                        raw_text="guangtongbei",
                        station_pinyin="guangtongbei",
                    )
                ],
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke({"user_input": "Chengdu to Dali via Guangtongbei"})

    assert state["flight_search_intent"].origin == "CTU"
    assert plan_graph.invocations[0]["explicit_hub_places"][0].station_pinyin == "guangtongbei"


def test_request_graph_uses_user_city_instead_of_model_selected_minor_airport():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="HZU",
                destination="SIN",
                origin_place=PlaceMention(
                    raw_text="成都",
                    kind="city",
                    city="Chengdu",
                    country="CN",
                ),
                destination_place=PlaceMention(
                    raw_text="新加坡",
                    kind="city",
                    city="Singapore",
                    country="SG",
                ),
                travel_date=date(2026, 11, 15),
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke({"user_input": "帮我查一下 2026-11-15 成都到新加坡的方案"})

    assert state["flight_search_intent"].origin == "CTU"
    assert state["flight_search_intent"].destination == "SIN"


def test_request_graph_preserves_explicit_minor_airport_code():
    plan_graph = FakeTravelPlanGraph()
    graph = build_travel_plan_request_graph(
        llm=FakeLlm(
            TravelPlanIntent(
                action="plan_trip",
                origin="HZU",
                destination="SIN",
                origin_place=PlaceMention(
                    raw_text="HZU",
                    kind="airport",
                    city="Chengdu",
                    country="CN",
                    iata_if_explicit="HZU",
                ),
                travel_date=date(2026, 11, 15),
            )
        ),
        travel_plan_graph=plan_graph,
    )

    state = graph.invoke({"user_input": "2026-11-15 HZU 到 SIN"})

    assert state["flight_search_intent"].origin == "HZU"
