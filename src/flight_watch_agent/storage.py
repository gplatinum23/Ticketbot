from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from .models import FlightQuote, Monitor
from .notifiers import Notification
from .validation import validate_monitor_input


class MonitorRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def add_monitor(
        self,
        *,
        origin: str,
        destination: str,
        depart_date: date,
        return_date: date | None,
        threshold_price: float,
        currency: str,
        interval_seconds: int,
    ) -> Monitor:
        validate_monitor_input(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            return_date=return_date,
            threshold_price=threshold_price,
            interval_seconds=interval_seconds,
        )
        now = datetime.now(timezone.utc)
        monitor = Monitor(
            id=str(uuid.uuid4()),
            origin=origin.upper(),
            destination=destination.upper(),
            depart_date=depart_date,
            return_date=return_date,
            threshold_price=threshold_price,
            currency=currency.upper(),
            interval_seconds=interval_seconds,
            enabled=True,
            created_at=now,
            updated_at=now,
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO monitors (
                    id, origin, destination, depart_date, return_date,
                    threshold_price, currency, interval_seconds, enabled,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    monitor.id,
                    monitor.origin,
                    monitor.destination,
                    monitor.depart_date.isoformat(),
                    _date_to_text(monitor.return_date),
                    monitor.threshold_price,
                    monitor.currency,
                    monitor.interval_seconds,
                    int(monitor.enabled),
                    _dt_to_text(monitor.created_at),
                    _dt_to_text(monitor.updated_at),
                ),
            )
        return monitor

    def get_monitor(self, monitor_id: str) -> Monitor:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM monitors WHERE id = ?",
                (monitor_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Monitor not found: {monitor_id}")
        return _row_to_monitor(row)

    def list_monitors(self, *, enabled_only: bool = False) -> list[Monitor]:
        sql = "SELECT * FROM monitors"
        params: tuple[object, ...] = ()
        if enabled_only:
            sql += " WHERE enabled = ?"
            params = (1,)
        sql += " ORDER BY created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_monitor(row) for row in rows]

    def record_quote(self, monitor: Monitor, quote: FlightQuote) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO price_observations (
                    id, monitor_id, price, currency, provider, deep_link, fetched_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    monitor.id,
                    quote.price,
                    quote.currency,
                    quote.provider,
                    quote.deep_link,
                    _dt_to_text(quote.fetched_at),
                ),
            )
            conn.execute(
                """
                UPDATE monitors
                SET last_checked_at = ?, last_price = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _dt_to_text(quote.fetched_at),
                    quote.price,
                    _dt_to_text(datetime.now(timezone.utc)),
                    monitor.id,
                ),
            )

    def record_notification(self, notification: Notification) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notifications (
                    id, monitor_id, title, message, price, currency, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    notification.monitor_id,
                    notification.title,
                    notification.message,
                    notification.quote.price,
                    notification.quote.currency,
                    _dt_to_text(notification.created_at),
                ),
            )
            conn.execute(
                """
                UPDATE monitors
                SET last_alert_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    _dt_to_text(notification.created_at),
                    _dt_to_text(datetime.now(timezone.utc)),
                    notification.monitor_id,
                ),
            )

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS monitors (
                    id TEXT PRIMARY KEY,
                    origin TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    depart_date TEXT NOT NULL,
                    return_date TEXT,
                    threshold_price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_checked_at TEXT,
                    last_price REAL,
                    last_alert_at TEXT
                );

                CREATE TABLE IF NOT EXISTS price_observations (
                    id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    deep_link TEXT,
                    fetched_at TEXT NOT NULL,
                    FOREIGN KEY (monitor_id) REFERENCES monitors(id)
                );

                CREATE TABLE IF NOT EXISTS notifications (
                    id TEXT PRIMARY KEY,
                    monitor_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    price REAL NOT NULL,
                    currency TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (monitor_id) REFERENCES monitors(id)
                );
                """
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _row_to_monitor(row: sqlite3.Row) -> Monitor:
    return Monitor(
        id=row["id"],
        origin=row["origin"],
        destination=row["destination"],
        depart_date=date.fromisoformat(row["depart_date"]),
        return_date=_text_to_date(row["return_date"]),
        threshold_price=row["threshold_price"],
        currency=row["currency"],
        interval_seconds=row["interval_seconds"],
        enabled=bool(row["enabled"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        last_checked_at=_text_to_dt(row["last_checked_at"]),
        last_price=row["last_price"],
        last_alert_at=_text_to_dt(row["last_alert_at"]),
    )


def _date_to_text(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _dt_to_text(value: datetime) -> str:
    return value.isoformat()


def _text_to_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def _text_to_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
