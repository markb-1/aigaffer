"""The what-if log: what each chip what-if said, kept where the inbox lives.

``$AIGAFFER_INBOX_DIR/whatifs.jsonl`` — beside the offset and the failure
marker, outside the checkout, never committed: the repo is public, and a
what-if is advice to one owner. Append-only, one JSON object a line, two lines
a what-if, both carrying the Telegram ``update_id`` that asked for it:

* ``numbers``, after message 1 — the band, the net and its parts, both paths,
  the squad and moves he would enter, where the minutes came from, and how
  long the solves took;
* ``opinion``, after the gaffer — his verdict, what he searched, and what it
  cost in tokens and list-price dollars.

Four readers. The daily cap counts today's ``numbers`` lines. A what-if killed
mid-run is resumed from what its lines say was done (spec §8). The recorder
warns when "Transfers made" follows a Play (Task 11). And one day the bands
are recalibrated from it: each line keeps both paths, so the realised points
can score them.

A line it cannot read — a torn last write, a hand edit — is skipped and never
raised: the inbox must not wedge on its own diary.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the handler imports both; this module needs only their shape
    from aigaffer.manager.chip_opinion import ChipOpinion
    from aigaffer.whatif import WhatIf

LOG_FILE = "whatifs.jsonl"
NUMBERS = "numbers"
OPINION = "opinion"
PLAY = "play"  # aigaffer.whatif.PLAY, repeated so this module imports nothing at runtime


class WhatIfLog:
    """The log file, read whole on every question: a few lines a day, so a
    scan is cheaper than any index would be to keep honest."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append_numbers(self, update_id: int, entry: dict) -> None:
        self._append(NUMBERS, update_id, entry)

    def append_opinion(self, update_id: int, entry: dict) -> None:
        self._append(OPINION, update_id, entry)

    def numbers_today(self, now: datetime) -> int:
        """``numbers`` lines stamped on ``now``'s UTC day — the cap's count."""
        today = now.astimezone(UTC).date()
        return sum(
            1
            for line in self._lines(NUMBERS)
            if (stamp := _when(line.get("ts"))) is not None and stamp.astimezone(UTC).date() == today
        )

    def numbers(self, update_id: int) -> dict | None:
        """The ``numbers`` line ``update_id`` wrote, or None."""
        return next((line for line in self._lines(NUMBERS) if line.get("update_id") == update_id), None)

    def has_opinion(self, update_id: int) -> bool:
        return any(line.get("update_id") == update_id for line in self._lines(OPINION))

    def latest_play(self, gw: int, after: str) -> dict | None:
        """The newest ``numbers`` line for gameweek ``gw`` whose band is Play,
        stamped after ``after`` (ISO-8601), or None. Newest by its timestamp,
        not its place in the file: a resumed what-if writes out of order."""
        since = _when(after)
        candidates = [
            (stamp, line)
            for line in self._lines(NUMBERS)
            if line.get("event") == gw
            and line.get("band") == PLAY
            and (stamp := _when(line.get("ts"))) is not None
            and (since is None or stamp > since)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda pair: pair[0])[1]

    def _append(self, kind: str, update_id: int, entry: dict) -> None:
        """One line, one write. A process killed mid-write leaves at most one
        torn line, which every reader skips."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({**entry, "kind": kind, "update_id": update_id}, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            # A torn line before this one would swallow it; start fresh.
            if handle.tell() and not _ends_in_newline(self.path):
                handle.write("\n")
            handle.write(line + "\n")

    def _lines(self, kind: str) -> list[dict]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        lines = []
        for raw in text.splitlines():
            try:
                line = json.loads(raw)
            except ValueError:
                continue
            if isinstance(line, dict) and line.get("kind") == kind and isinstance(line.get("update_id"), int):
                lines.append(line)
        return lines


def _ends_in_newline(path: Path) -> bool:
    with path.open("rb") as handle:
        handle.seek(-1, 2)
        return handle.read(1) == b"\n"


def _when(value: object) -> datetime | None:
    """An ISO timestamp as an aware datetime, or None for anything else."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=UTC)


def numbers_entry(whatif: "WhatIf", *, ts: str, total_seconds: float) -> dict:
    """Message 1's numbers as a log line: spec §8's fields and no others.

    ``squad_on`` is the fifteen he would enter — the free-hit team on a free
    hit, the standing squad otherwise — sorted, ids only, like every record.
    """
    on, off = whatif.on, whatif.off
    fielded = on.path.week1_freehit_squad if on.path.week1_freehit_squad else on.plan.squad
    return {
        "ts": ts,
        "event": whatif.event,
        "chip": whatif.kind,
        "band": whatif.band,
        "net": whatif.net,
        "gain": whatif.gain,
        "bars_diff": whatif.bars_diff,
        "margin": whatif.margin,
        "proven_on": on.path.proven,
        "proven_off": off.path.proven,
        "weekly_on": {str(w): xp for w, xp in on.path.weekly_xp.items()},
        "weekly_off": {str(w): xp for w, xp in off.path.weekly_xp.items()},
        "chips_on": {str(w): chip for w, chip in on.chips.items()},
        "chips_off": {str(w): chip for w, chip in off.chips.items()},
        "squad_on": sorted(fielded),
        "captain_on": on.lineup.captain,
        "vice_on": on.lineup.vice,
        "moves_on": {"in": list(on.plan.transfers_in), "out": list(on.plan.transfers_out)},
        "off_summary": {
            "in": list(off.plan.transfers_in),
            "out": list(off.plan.transfers_out),
            "hits": off.plan.hits,
        },
        "minutes_source": whatif.minutes_source,
        "solve_seconds": {"on": on.seconds, "off": off.seconds},
        "total_seconds": total_seconds,
    }


def opinion_entry(opinion: "ChipOpinion", *, ts: str, model: str) -> dict:
    """The gaffer's answer as a log line, priced at ``model``'s list rates."""
    usage = opinion.usage
    return {
        "ts": ts,
        "verdict": opinion.verdict,
        "better_week": opinion.better_week,
        "new_fact": opinion.new_fact,
        "searches": opinion.searches,
        "source": opinion.source,
        "turns": opinion.turns,
        "seconds": opinion.seconds,
        "tokens": {
            "input": usage.input,
            "cache_read": usage.cache_read,
            "cache_write": usage.cache_write,
            "output": usage.output,
        },
        "est_usd": usage.est_usd(model),
    }
