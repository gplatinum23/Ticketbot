from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import date

from .app import (
    build_default_flight_search_agent,
    build_default_request_agent,
    build_default_travel_plan_agent,
)
from .models import FlightSearchIntent


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser(prog="flight-watch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser("ask", help="Plan travel from natural language.")
    ask_parser.add_argument("text", nargs="+", help="Natural language request.")
    ask_parser.add_argument("--model", help="LangChain model string, e.g. openai:gpt-4.1-mini.")
    ask_parser.add_argument("--flight-only", action="store_true", help="Skip train lookup and test flight planning only.")
    ask_parser.add_argument("--show-flight-raw", action="store_true", help="Print raw flight search debug output.")
    plan_parser = subparsers.add_parser("plan-flight", help="Search train and verified public flight options.")
    plan_parser.add_argument("--origin", required=True)
    plan_parser.add_argument("--destination", required=True)
    plan_parser.add_argument("--travel-date", required=True, type=date.fromisoformat)
    plan_parser.add_argument("--time-preference")
    plan_parser.add_argument("--budget", type=float)
    plan_parser.add_argument("--currency", default="CNY")
    plan_parser.add_argument("--show-flight-raw", action="store_true", help="Print raw flight search debug output.")
    debug_parser = subparsers.add_parser(
        "debug-flight-search",
        help="Debug only the public-page flight search pipeline.",
    )
    _add_flight_intent_arguments(debug_parser)
    debug_parser.add_argument("--max-iterations", type=int, default=3)
    debug_parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM evidence judging.")

    args = parser.parse_args()
    if args.command == "ask":
        graph = build_default_request_agent(llm_model=args.model, include_train=not args.flight_only)
        state = graph.invoke({"user_input": " ".join(args.text)})
        print(state["response"])
        if args.show_flight_raw:
            print(_format_flight_raw_output(state.get("plan_state", {})))
        return

    if args.command == "plan-flight":
        graph = build_default_travel_plan_agent()
        state = graph.invoke(
            {
                "intent": _flight_search_intent_from_args(args)
            }
        )
        print(state["response"])
        if args.show_flight_raw:
            print(_format_flight_raw_output(state))
        return

    if args.command == "debug-flight-search":
        graph = build_default_flight_search_agent(
            use_llm_judge=not args.no_llm_judge,
            max_iterations=args.max_iterations,
        )
        state = graph.invoke({"intent": _flight_search_intent_from_args(args)})
        print(_format_react_flight_raw_output(state))
        return


def _add_flight_intent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--origin", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--travel-date", required=True, type=date.fromisoformat)
    parser.add_argument("--time-preference")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--currency", default="CNY")


def _flight_search_intent_from_args(args) -> FlightSearchIntent:
    return FlightSearchIntent(
        origin=args.origin,
        destination=args.destination,
        travel_date=args.travel_date,
        time_preference=args.time_preference,
        budget_threshold=args.budget,
        currency=args.currency,
    )


def _format_flight_raw_output(state: dict) -> str:
    debug = state.get("flight_search_debug", {})
    return "\n".join(
        [
            "",
            "Flight search raw output:",
            json.dumps(_to_jsonable(debug), ensure_ascii=False, indent=2),
        ]
    )


def _format_react_flight_raw_output(state: dict) -> str:
    return "\n".join(
        [
            "Flight search raw output:",
            json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2),
        ]
    )


def _to_jsonable(value):
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
