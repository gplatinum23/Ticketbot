from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from .models import FlightSearchIntent, TrainOption
from .places import normalise_train_query_place


class McpToolClient(Protocol):
    def call_tool(self, name: str, arguments: dict) -> Any:
        """Call an MCP tool and return the JSON object encoded in its text content."""


class StdioMcpToolClient:
    def __init__(
        self,
        *,
        command: str | None = None,
        args: list[str] | None = None,
        timeout_seconds: int = 60,
    ) -> None:
        self.command = command or _default_mcp_command()
        self.args = args or _default_mcp_args()
        self.timeout_seconds = timeout_seconds

    def call_tool(self, name: str, arguments: dict) -> Any:
        proc = self._start_process()
        stderr_lines: list[str] = []
        stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(proc, stderr_lines),
            daemon=True,
        )
        stderr_thread.start()

        try:
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "flight-watch-agent",
                            "version": "0.1.0",
                        },
                    },
                },
            )
            self._read_response(proc, stderr_lines)
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            )
            self._send(
                proc,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            response = self._read_response(proc, stderr_lines)
            return _extract_tool_json(response)
        finally:
            proc.kill()

    def _start_process(self) -> subprocess.Popen:
        env = os.environ.copy()
        env["DEBUG"] = os.getenv("FLIGHT_WATCH_12306_DEBUG", "false")
        env["PYTHONIOENCODING"] = "utf-8"
        node_dir = Path("C:/Program Files/nodejs")
        if node_dir.exists():
            env["PATH"] = f"{node_dir}{os.pathsep}{env.get('PATH', '')}"
        return subprocess.Popen(
            [self.command, *self.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

    def _send(self, proc: subprocess.Popen, payload: dict) -> None:
        if proc.stdin is None:
            raise RuntimeError("MCP server stdin is unavailable.")
        proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
        proc.stdin.flush()

    def _read_response(self, proc: subprocess.Popen, stderr_lines: list[str]) -> dict:
        if proc.stdout is None:
            raise RuntimeError("MCP server stdout is unavailable.")
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if proc.poll() is not None:
                tail = " | ".join(stderr_lines[-8:])
                raise RuntimeError(f"MCP server exited before responding. stderr={tail}")
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            return json.loads(line)
        tail = " | ".join(stderr_lines[-8:])
        raise TimeoutError(f"Timed out waiting for MCP response. stderr={tail}")


class Mcp12306TrainProvider:
    def __init__(self, client: McpToolClient | None = None, *, max_results: int = 20) -> None:
        self.client = client or StdioMcpToolClient()
        self.max_results = max_results

    def query_train_options(self, intent: FlightSearchIntent) -> list[TrainOption]:
        from_station = _normalise_train_place(intent.origin)
        to_station = _normalise_train_place(intent.destination)
        if from_station is None or to_station is None:
            return []

        arguments = {
            "date": intent.travel_date.isoformat(),
            "fromStation": from_station,
            "toStation": to_station,
            "sortFlag": "startTime",
            "limitedNum": self.max_results,
            "format": "json",
        }
        tickets = self.client.call_tool("get-tickets", arguments)
        if not isinstance(tickets, list):
            raise RuntimeError(f"12306 ticket query failed: {tickets!r}")

        options: list[TrainOption] = []
        for row in tickets[: self.max_results]:
            train_code = str(row.get("start_train_code") or row.get("train_no") or "")
            seats, prices = _parse_ticket_prices(row.get("prices") or [])
            options.append(
                TrainOption(
                    train_code=train_code,
                    from_station=str(row.get("from_station") or ""),
                    from_station_code=_optional_str(row.get("from_station_telecode")),
                    to_station=str(row.get("to_station") or ""),
                    to_station_code=_optional_str(row.get("to_station_telecode")),
                    travel_date=intent.travel_date,
                    start_time=str(row.get("start_time") or ""),
                    arrive_time=str(row.get("arrive_time") or ""),
                    duration=str(row.get("lishi") or row.get("duration") or ""),
                    seats=seats,
                    prices=prices,
                    train_class_name=_optional_str(_train_class_name(train_code)),
                )
            )

        return options


def _extract_tool_json(response: dict) -> Any:
    if "error" in response:
        raise RuntimeError(response["error"])
    content = response.get("result", {}).get("content", [])
    if not content:
        raise RuntimeError("MCP tool returned no content.")
    text = content[0].get("text", "")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"MCP tool returned non-JSON content: {text}") from exc


def _parse_ticket_prices(raw_prices: list[dict]) -> tuple[dict[str, str], dict[str, float]]:
    seats: dict[str, str] = {}
    parsed: dict[str, float] = {}
    for item in raw_prices:
        seat_name = str(item.get("seat_name") or "")
        if not seat_name:
            continue
        seats[seat_name] = str(item.get("num") or "")
        try:
            parsed[seat_name] = float(item.get("price"))
        except (TypeError, ValueError):
            continue
    return seats, parsed


def _train_class_name(train_code: str) -> str | None:
    if not train_code:
        return None
    prefix = train_code[0].upper()
    return {
        "G": "高铁/城际",
        "D": "动车",
        "Z": "直达特快",
        "T": "特快",
        "K": "快速",
    }.get(prefix)


def _normalise_train_place(value: str) -> str | None:
    return normalise_train_query_place(value)


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _default_mcp_command() -> str:
    configured = os.getenv("FLIGHT_WATCH_12306_MCP_COMMAND")
    if configured:
        return configured
    node_npx = Path("C:/Program Files/nodejs/npx.cmd")
    if node_npx.exists():
        return str(node_npx)
    return shutil.which("npx.cmd") or shutil.which("npx") or "npx"


def _default_mcp_args() -> list[str]:
    configured = os.getenv("FLIGHT_WATCH_12306_MCP_ARGS")
    if configured:
        return configured.split()
    return ["-y", "12306-mcp"]


def _drain_stderr(proc: subprocess.Popen, stderr_lines: list[str]) -> None:
    if proc.stderr is None:
        return
    for line in proc.stderr:
        stderr_lines.append(line.rstrip())
