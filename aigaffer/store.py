"""Persistent record of pipeline runs, and of what the squad cost.

Three SQLite tables. ``runs`` lets the bot answer "did I already run for this
gameweek and mode?" across process restarts, and keeps the rendered report
plus the machine-readable decision around for later inspection. ``purchases``
is the purchase ledger: what we paid for each player we hold, which the public
API never says and which is the only way to know what he would sell for
(:mod:`aigaffer.ledger`). ``squads`` remembers the picks each gameweek served
— the bank and the fifteen — so that after a deadline the ledger's selling
estimates can be checked against the bank the game actually published.
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

# Prices are integer tenths of a million, exactly as the API quotes them.
# ``gw_seen`` is the gameweek the ledger first saw the player, which is the
# honesty label on his ``buy_price``: a row seeded from ``cost_change_start``
# on the ledger's first run is exact for a squad held since GW1, and a row
# written at first sighting is the price on the day the run noticed him, which
# can trail the owner's click by up to £0.1m if the price moved in between.
PURCHASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS purchases (
    player_id INTEGER PRIMARY KEY,
    buy_price INTEGER,
    gw_seen INTEGER
)
"""

# One row per gameweek the picks endpoint has served: the bank and the fifteen
# as the game published them. ``player_ids`` is a JSON list, because SQLite has
# no list and fifteen integers are not worth a join table.
SQUADS_SCHEMA = """
CREATE TABLE IF NOT EXISTS squads (
    gw INTEGER PRIMARY KEY,
    bank INTEGER,
    player_ids TEXT
)
"""


class Store:
    """Run history kept in a SQLite file at ``db_path``."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.execute(PURCHASES_SCHEMA)
            conn.execute(SQUADS_SCHEMA)

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

    def decision(self, gw: int, mode: str) -> dict | None:
        """The decision recorded for ``gw``'s ``mode`` run, or None for none.

        The newest row wins when there is more than one: a second row for the
        same gameweek and mode only exists because a person overruled the
        dedup with ``--force``, and what they forced is the record that
        stands. This is how the reminder reads back what the full report
        decided a day earlier — across process restarts and, in CI, across
        the state commit between two workflow runs.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT decision_json FROM runs WHERE gw = ? AND mode = ?"
                " ORDER BY id DESC LIMIT 1",
                (gw, mode),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def has_run(self, gw: int, mode: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE gw = ? AND mode = ? LIMIT 1", (gw, mode)
            ).fetchone()
        return row is not None

    def purchases(self) -> dict[int, int]:
        """The purchase ledger: player id to buy price, in tenths of a million.

        Every player the ledger believes we hold, and what we paid for him —
        the number the selling rule measures rises against
        (:func:`aigaffer.ledger.selling_price`). Empty on a store that has
        never run with a squad, which is how the ledger's first run knows to
        seed itself.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT player_id, buy_price FROM purchases").fetchall()
        return dict(rows)

    def record_purchase(self, player_id: int, buy_price: int, gw_seen: int) -> None:
        """One player entering the ledger, or re-entering it.

        REPLACE rather than INSERT, because a player sold and later bought back
        has a new purchase price and the old row would price his next sale off
        a deal that was already settled.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO purchases (player_id, buy_price, gw_seen)"
                " VALUES (?, ?, ?)",
                (player_id, buy_price, gw_seen),
            )

    def forget_purchase(self, player_id: int) -> None:
        """A player who has left the squad leaves the ledger with him.

        His sale is already settled — the money is in the bank the API
        publishes — so keeping the row would only mis-price him if he is ever
        bought back. Quiet for a player the ledger never held, because the
        caller is reconciling two lists and an absence is not an error.
        """
        with self._connect() as conn:
            conn.execute("DELETE FROM purchases WHERE player_id = ?", (player_id,))

    def record_squad(self, gw: int, bank: int, player_ids: list[int]) -> None:
        """What the picks endpoint said about ``gw``: the bank and the fifteen.

        REPLACE, because several runs serve one gameweek — scout, deadline,
        reminder — and each writes the same published facts; latest wins and
        they agree. This row existing is also how the ledger knows ``gw`` has
        already been reconciled, so the discrepancy line goes out once per
        gameweek and not once per run.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO squads (gw, bank, player_ids)"
                " VALUES (?, ?, ?)",
                (gw, bank, json.dumps(player_ids)),
            )

    def squad_record(self, gw: int) -> dict | None:
        """The squad and bank recorded for ``gw``, or None if no run saw it."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT gw, bank, player_ids FROM squads WHERE gw = ?", (gw,)
            ).fetchone()
        return None if row is None else self._squad_row(row)

    def last_squad_before(self, gw: int) -> dict | None:
        """The newest squad record from a gameweek before ``gw``, or None.

        What reconciliation compares the new gameweek's published bank
        against. "Newest before" rather than "``gw`` minus one" because a
        skipped week — an outage, a paused schedule — leaves a hole, and the
        last picks actually seen are still the last state the ledger priced.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT gw, bank, player_ids FROM squads WHERE gw < ?"
                " ORDER BY gw DESC LIMIT 1",
                (gw,),
            ).fetchone()
        return None if row is None else self._squad_row(row)

    @staticmethod
    def _squad_row(row: tuple) -> dict:
        gw, bank, player_ids = row
        return {"gw": gw, "bank": bank, "player_ids": json.loads(player_ids)}

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
