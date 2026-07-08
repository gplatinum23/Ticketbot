from __future__ import annotations

import argparse
from datetime import date

from .app import build_default_agent, build_default_request_agent, build_default_travel_plan_agent
from .models import FlightSearchIntent
from .scheduler import run_once, watch_forever


def main() -> None:
    parser = argparse.ArgumentParser(prog="flight-watch")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a flight price monitor.")
    add_parser.add_argument("--origin", required=True, help="Origin airport or city code.")
    add_parser.add_argument("--destination", required=True, help="Destination airport or city code.")
    add_parser.add_argument("--depart-date", required=True, type=date.fromisoformat)
    add_parser.add_argument("--return-date", type=date.fromisoformat)
    add_parser.add_argument("--threshold", required=True, type=float)
    add_parser.add_argument("--currency", default="CNY")
    add_parser.add_argument("--interval", type=int, default=3600, help="Check interval in seconds.")

    subparsers.add_parser("list", help="List monitors.")
    subparsers.add_parser("run-once", help="Run one check for all enabled monitors.")
    subparsers.add_parser("watch", help="Run the long-lived watch loop.")
    ask_parser = subparsers.add_parser("ask", help="Create a monitor from natural language.")
    ask_parser.add_argument("text", nargs="+", help="Natural language request.")
    ask_parser.add_argument("--model", help="LangChain model string, e.g. openai:gpt-4.1-mini.")
    plan_parser = subparsers.add_parser("plan-flight", help="Search verified public flight options.")
    plan_parser.add_argument("--origin", required=True)
    plan_parser.add_argument("--destination", required=True)
    plan_parser.add_argument("--travel-date", required=True, type=date.fromisoformat)
    plan_parser.add_argument("--time-preference")
    plan_parser.add_argument("--budget", type=float)
    plan_parser.add_argument("--currency", default="CNY")

    args = parser.parse_args()
    if args.command == "ask":
        graph, _repository = build_default_request_agent(llm_model=args.model)
        state = graph.invoke({"user_input": " ".join(args.text)})
        print(state["response"])
        return

    if args.command == "plan-flight":
        graph = build_default_travel_plan_agent()
        state = graph.invoke(
            {
                "intent": FlightSearchIntent(
                    origin=args.origin,
                    destination=args.destination,
                    travel_date=args.travel_date,
                    time_preference=args.time_preference,
                    budget_threshold=args.budget,
                    currency=args.currency,
                )
            }
        )
        print(state["response"])
        return

    graph, repository = build_default_agent()

    if args.command == "add":
        monitor = repository.add_monitor(
            origin=args.origin,
            destination=args.destination,
            depart_date=args.depart_date,
            return_date=args.return_date,
            threshold_price=args.threshold,
            currency=args.currency,
            interval_seconds=args.interval,
        )
        print(f"Added monitor {monitor.id}: {monitor.origin}->{monitor.destination}")
        return

    if args.command == "list":
        for monitor in repository.list_monitors():
            last_price = "-" if monitor.last_price is None else f"{monitor.last_price:.2f}"
            print(
                f"{monitor.id} {monitor.origin}->{monitor.destination} "
                f"{monitor.depart_date.isoformat()} threshold={monitor.threshold_price:.2f} "
                f"{monitor.currency} last_price={last_price} enabled={monitor.enabled}"
            )
        return

    if args.command == "run-once":
        results = run_once(graph, repository)
        print(f"Checked {len(results)} monitor(s).")
        return

    if args.command == "watch":
        watch_forever(graph, repository)


if __name__ == "__main__":
    main()
