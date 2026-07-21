from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path

from flight_watch_agent.models import FlightSearchIntent
from flight_watch_agent.trains import Mcp12306TrainProvider, StdioMcpToolClient


class FakeMcpClient:
    def __init__(self) -> None:
        self.calls = []

    def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if name == "get-tickets":
            return [
                {
                    "train_no": "240000G53106",
                    "start_train_code": "G531",
                    "from_station": "北京南",
                    "from_station_telecode": "VNP",
                    "to_station": "上海虹桥",
                    "to_station_telecode": "AOH",
                    "start_time": "06:08",
                    "arrive_time": "12:04",
                    "lishi": "05:56",
                    "prices": [
                        {"seat_name": "商务座", "num": "8", "price": 2315, "discount": 84},
                        {"seat_name": "一等座", "num": "14", "price": 1005, "discount": 79},
                        {"seat_name": "二等座", "num": "有", "price": 598, "discount": 76},
                    ],
                }
            ]
        raise AssertionError(name)


def test_mcp_12306_provider_parses_get_tickets_prices_and_seats():
    client = FakeMcpClient()
    provider = Mcp12306TrainProvider(client)

    options = provider.query_train_options(
        FlightSearchIntent(
            origin="BJP",
            destination="SHH",
            travel_date=date(2026, 7, 10),
        )
    )

    assert len(options) == 1
    assert options[0].train_code == "G531"
    assert options[0].from_station == "北京南"
    assert options[0].from_station_code == "VNP"
    assert options[0].to_station == "上海虹桥"
    assert options[0].to_station_code == "AOH"
    assert options[0].duration == "05:56"
    assert options[0].seats["二等座"] == "有"
    assert options[0].prices["二等座"] == 598.0
    assert options[0].lowest_price == 598.0
    assert client.calls == [
        (
            "get-tickets",
            {
                "date": "2026-07-10",
                "fromStation": "BJP",
                "toStation": "SHH",
                "sortFlag": "startTime",
                "limitedNum": 20,
                "format": "json",
            },
        )
    ]


def test_stdio_mcp_client_defaults_to_12306_mcp_npx():
    client = StdioMcpToolClient()

    assert client.args == ["-y", "12306-mcp"]
    assert Path(client.command).name.lower() in {"npx", "npx.cmd"}


def test_stdio_mcp_client_reuses_initialized_process_for_multiple_calls():
    responses = [
        {"jsonrpc": "2.0", "id": 1, "result": {}},
        {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "[]"}]}},
        {"jsonrpc": "2.0", "id": 3, "result": {"content": [{"type": "text", "text": "[]"}]}},
    ]

    class FakeProcess:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO("".join(json.dumps(item) + "\n" for item in responses))
            self.stderr = io.StringIO()
            self.killed = False

        def poll(self):
            return 0 if self.killed else None

        def kill(self) -> None:
            self.killed = True

    class FakeStdioMcpToolClient(StdioMcpToolClient):
        def __init__(self) -> None:
            super().__init__(command="npx", args=["12306-mcp"])
            self.processes = []

        def _start_process(self):
            proc = FakeProcess()
            self.processes.append(proc)
            return proc

    client = FakeStdioMcpToolClient()

    assert client.call_tool("get-tickets", {"route": 1}) == []
    assert client.call_tool("get-tickets", {"route": 2}) == []

    assert len(client.processes) == 1
    sent_messages = [json.loads(line) for line in client.processes[0].stdin.getvalue().splitlines()]
    assert [message["method"] for message in sent_messages] == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
    ]
    client.close()
    assert client.processes[0].killed is True


def test_mcp_12306_provider_maps_airport_codes_to_train_places():
    client = FakeMcpClient()
    provider = Mcp12306TrainProvider(client)

    provider.query_train_options(
        FlightSearchIntent(
            origin="CTU",
            destination="KMG",
            travel_date=date(2026, 7, 10),
        )
    )

    assert client.calls[0][1]["fromStation"] == "成都"
    assert client.calls[0][1]["toStation"] == "昆明"


def test_mcp_12306_provider_skips_international_train_destination():
    client = FakeMcpClient()
    provider = Mcp12306TrainProvider(client)

    options = provider.query_train_options(
        FlightSearchIntent(
            origin="NKG",
            destination="SIN",
            travel_date=date(2026, 7, 10),
        )
    )

    assert options == []
    assert client.calls == []
