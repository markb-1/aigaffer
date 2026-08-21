"""Persistent record of pipeline runs.

A single SQLite table lets the bot answer "did I already run for this gameweek
and mode?" across process restarts, and keeps the rendered report plus the
machine-readable decision around for later inspection.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    ts TEXT,
    gw INTEGER,
    mode TEXT,
    report_md TEXT,
    decision_json TEXT
)
"""


class Store:
    """Run history kept in a SQLite file at ``db_path``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection that commits on success and always closes."""
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def save_run(self, gw: int, mode: str, report_md: str, decision: dict) -> None:
        """Record one completed run of ``mode`` for gameweek ``gw``."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (ts, gw, mode, report_md, decision_json)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    datetime.now(UTC).isoformat(),
                    gw,
                    mode,
                    report_md,
                    json.dumps(decision),
                ),
            )

    def has_run(self, gw: int, mode: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE gw = ? AND mode = ? LIMIT 1", (gw, mode)
            ).fetchone()
        return row is not None

    def last_runs(self, n: int = 5) -> list[dict]:
        """The ``n`` most recent runs, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT gw, mode, ts, decision_json FROM runs ORDER BY id DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [
            {"gw": gw, "mode": mode, "ts": ts, "decision": json.loads(decision_json)}
            for gw, mode, ts, decision_json in rows
        ]
