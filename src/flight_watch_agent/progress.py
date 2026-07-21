from __future__ import annotations

import sys
import time
from typing import Protocol, TextIO


class ProgressReporter(Protocol):
    def emit(self, message: str) -> None:
        """Report a human-readable progress message."""


class NoopProgressReporter:
    def emit(self, message: str) -> None:
        return None


class ConsoleProgressReporter:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream or sys.stderr
        self.count = 0
        self.started_at = time.perf_counter()
        self.last_emitted_at = self.started_at

    def emit(self, message: str) -> None:
        now = time.perf_counter()
        self.count += 1
        step_seconds = now - self.last_emitted_at
        total_seconds = now - self.started_at
        self.last_emitted_at = now
        print(
            f"[{self.count}] {message} (+{step_seconds:.1f}s, total {total_seconds:.1f}s)",
            file=self.stream,
            flush=True,
        )


def get_progress_reporter(reporter: ProgressReporter | None) -> ProgressReporter:
    return reporter or NoopProgressReporter()
