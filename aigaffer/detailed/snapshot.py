"""The run snapshot: what one run knew and weighed, kept as one compact file.

Every report-producing run — early, scout, deadline — ends with a decision,
and the reasons behind it are spread across the fetch, the projections, the
solver's shortlist, the gaffer's re-solves and the decision record. The GW
report prints the verdict and a little of why; the detailed report and the
"ask the gaffer" conversation both want the rest of it, and neither should
re-derive any of it. So the run writes it down once, here, in
``state/detailed/gw{n}-{mode}.json``, and both read only that.

What it holds is public FPL data and the bot's own numbers about it: for
every player the run projected, his fixtures and xP across the window; every
plan the gaffer could have finalized, under the id he saw it by; and the
decision. Nothing the owner said and nothing private.

The rows are the *projected set* — the history pool, the solver's candidate
pool and the squad — rather than all six hundred players: the projections
only mean anything for the players whose histories were fetched, and anyone
else would be stored at a confident-looking zero. A question about a player
outside the set is answered "not projected this week", which is the truth.

Keys are short (``n``, ``p``, ``f``, ``x``…) and xP is kept to two decimals:
there are about 180 rows a run and four runs a week, committed with the rest
of ``state/``. The ints JSON turns into strings on the way out are turned back
on the way in, so a snapshot read from disk is the snapshot that was built.
"""

import json
import os
import re
from pathlib import Path

from aigaffer.data.models import Bootstrap, Event, Fixture
from aigaffer.data.schedule import opponents as club_opponents
from aigaffer.detailed.skipped import skipped_transfers
from aigaffer.model.xp import PlayerProjection, projected_events
from aigaffer.report.render import POSITIONS
from aigaffer.solver.optimizer import Plan, resolve

# The order a gameweek's runs happen in. "The latest" snapshot is the last of
# these that exists for the highest gameweek, and "What changed" walks them in
# this order; a mode outside it would be written and never read.
MODES_ORDER = ("early", "scout", "deadline")

SNAPSHOT_DIR = "detailed"

_FILE = re.compile(r"gw(\d+)-([a-z]+)\.json")

# What a decision record from a run with no gaffer at all says about its
# source. The record carries ``decision_source`` only when the manager loop
# ran (a decision or a labelled fallback); a run with the manager switched
# off is the solver's week and says nothing, so the snapshot says it for it.
SOLVER = "solver"


def build_snapshot(
    *,
    event: Event,
    mode: str,
    ts: str,
    bootstrap: Bootstrap,
    fixtures: list[Fixture],
    projections: dict[int, PlayerProjection],
    xmins: dict[int, float],
    projected_ids: set[int],
    squad_ids: list[int],
    selling_prices: dict[int, int] | None,
    plans: dict[int, Plan],
    chosen_id: int,
    decision: dict,
    chip_evs: dict,
    calendar_lines: list[str],
    freehit: tuple[list[int], list[int]] | None,
    bank: int,
    shortlist: int,
    decay: float = 0.85,
    do_this: str | None = None,
) -> dict:
    """The run, as one JSON-ready dict.

    ``projections`` and ``xmins`` are the ones the decision was costed on —
    the gaffer's re-solve's when he re-solved — so the xP beside a player is
    the xP the plan was chosen on. ``plans`` is the registry he finalized
    against (the solver's shortlist on a run with no gaffer), and
    ``shortlist`` is how many of its ids the briefing printed: every id at or
    above it came from a re-solve, which is how a plan is marked ``resolved``
    with no field of its own on :class:`Plan`.

    ``decay`` is the window's per-gameweek discount, the run's ``cfg.decay``
    (the default is the configured 0.85); the skipped-transfer pass needs it to
    price a lost free-transfer step the way the solver does.

    ``do_this`` is the gameweek report's action block (chip banner, "working
    from", the "Do this" checklist) as text, built by the same function the
    report calls, so the detailed report opens with exactly what the phone was
    told. None for a draft, which has no checklist.

    ``decision`` is the run's decision record, read for what it settled: this
    week's free transfers, the armbands, the bench in substitution order, the
    chip and the gaffer's words. ``bank`` is in tenths, as the API serves it;
    the record does not carry it, so the caller hands it in from the inputs
    the run solved on.

    The eleven is not in the record, which keeps the bench and the armbands
    instead, so it is read back as the fielded fifteen less the bench: the
    chosen plan's squad on an ordinary week, the free hit's own on a free-hit
    week, since the standing squad is not the team that plays then.

    Raises ValueError for a squad player with no projection: the squad table
    is the report's spine, and a man missing from it should be named here
    rather than surface as a KeyError three modules later. Squad ⊆ the history
    pool makes it impossible today; the check is what keeps it so. A player
    in ``projected_ids`` with no projection or no bootstrap row is left out
    instead — a smaller pool is a smaller answer, not a broken one.
    """
    if chosen_id not in plans:
        raise ValueError(f"the chosen plan {chosen_id} is not among the plans {sorted(plans)}")
    missing = [pid for pid in squad_ids if pid not in projections]
    if missing:
        raise ValueError(f"squad players with no projection: {missing}")

    events = projected_events(projections)
    clubs = {team.id: team for team in bootstrap.teams}
    elements = {element.id: element for element in bootstrap.elements}
    opponents = club_opponents(fixtures, clubs, set(events))
    prices = selling_prices or {}

    players = {}
    for pid in sorted(set(projected_ids) | set(squad_ids)):
        element = elements.get(pid)
        projection = projections.get(pid)
        if element is None or projection is None:
            if pid in squad_ids:
                raise ValueError(f"squad player {pid} is not on the bootstrap")
            continue
        club = clubs[element.team]
        players[pid] = {
            "n": element.web_name,
            "p": POSITIONS.get(element.element_type, "?"),
            "c": club.short_name,
            "cf": club.name,
            "pr": element.now_cost,
            "sp": prices.get(pid),
            "st": element.status,
            "nw": element.news,
            "xm": round(xmins.get(pid, 0.0), 1),
            # Every window gameweek, a blank as an empty list: a missing key
            # would read as "not looked at" where the truth is "no game".
            "f": {gw: [list(o) for o in opponents.get((gw, element.team), [])] for gw in events},
            "x": {gw: round(projection.per_gw.get(gw, 0.0), 2) for gw in events},
            "t": round(projection.total, 2),
        }

    bench = list(decision.get("bench", []))
    fielded = freehit[0] if freehit is not None else plans[chosen_id].squad
    xi = list(freehit[1]) if freehit is not None else [p for p in fielded if p not in bench]

    skipped = skipped_transfers(
        players,
        list(squad_ids),
        {pid: _plan(plan, pid >= shortlist) for pid, plan in plans.items()},
        chosen_id,
        bank,
        decision.get("free_transfers"),
        decay,
        len(events),
        resolve(None),
        xi=xi,
    )

    return {
        "gw": event.id,
        "mode": mode,
        "ts": ts,
        "events": events,
        "bank": bank,
        "ft": decision.get("free_transfers"),
        "players": players,
        "squad": list(squad_ids),
        "xi": xi,
        "bench": bench,
        "captain": decision.get("captain"),
        "vice": decision.get("vice"),
        "plans": {pid: _plan(plan, pid >= shortlist) for pid, plan in sorted(plans.items())},
        "chosen": chosen_id,
        "do_this": do_this,
        "decision": {
            "rationale": decision.get("rationale", ""),
            "adjustments": list(decision.get("adjustments", [])),
            # The minutes he only wrote down, and how many searches it took:
            # both are the gaffer's alone, so a solver-only run carries
            # neither and the long read prints neither.
            **(
                {"unapplied": list(decision["unapplied"])}
                if "unapplied" in decision
                else {}
            ),
            **({"searches": decision["searches"]} if "searches" in decision else {}),
            "chip": decision.get("chip"),
            "chip_justification": decision.get("chip_justification", ""),
            "source": decision.get("decision_source", SOLVER),
        },
        "chip_evs": dict(chip_evs),
        "calendar": list(calendar_lines),
        "freehit": (
            None if freehit is None else {"squad": list(freehit[0]), "xi": list(freehit[1])}
        ),
        # The transfers the run passed over, and why each lost.
        "skipped": skipped,
    }


def snapshot_path(state_dir: Path, gw: int, mode: str) -> Path:
    """Where a gameweek's run of ``mode`` keeps its snapshot."""
    return state_dir / SNAPSHOT_DIR / f"gw{gw}-{mode}.json"


def write_snapshot(state_dir: Path, gw: int, mode: str, snapshot: dict) -> Path:
    """Write ``snapshot`` for gameweek ``gw``'s ``mode`` run, atomically.

    To a temp file and renamed over the old one, as the news ledger is: a
    crash leaves the last good snapshot, and a reader — the question handler
    reads while a tick may be writing — never sees half a file. A rerun of the
    same mode (``--force``) overwrites its own file, so each mode holds one
    snapshot, the latest. One line, with sorted keys: two hundred rows
    pretty-printed is thousands of lines a run in the state commit.
    """
    if mode not in MODES_ORDER:
        raise ValueError(f"unknown mode {mode!r}: a snapshot is one of {MODES_ORDER}")
    path = snapshot_path(state_dir, gw, mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def load_snapshots(state_dir: Path, gw: int) -> list[dict]:
    """Gameweek ``gw``'s snapshots in the order its runs happen; a mode that
    has not run (or never will this week) is simply not there."""
    snapshots = []
    for mode in MODES_ORDER:
        path = snapshot_path(state_dir, gw, mode)
        if path.exists():
            snapshots.append(_revive(json.loads(path.read_text(encoding="utf-8"))))
    return snapshots


def latest_snapshot(state_dir: Path) -> dict | None:
    """The newest snapshot there is: the highest gameweek, and within it the
    last mode to have run. None before the first run has written one.

    Read off the file names, so a stray file that is not a snapshot's is
    passed over rather than parsed.
    """
    directory = state_dir / SNAPSHOT_DIR
    if not directory.is_dir():
        return None
    found = []
    for path in directory.iterdir():
        match = _FILE.fullmatch(path.name)
        if match and match.group(2) in MODES_ORDER:
            found.append((int(match.group(1)), MODES_ORDER.index(match.group(2)), path))
    if not found:
        return None
    _, _, newest = max(found)
    return _revive(json.loads(newest.read_text(encoding="utf-8")))


def _plan(plan: Plan, resolved: bool) -> dict:
    """One plan as the options table reads it.

    The window figures live on the path, and a single-week plan has none: it
    reads as no path, no free transfers handed on (None, as the decision
    record keeps it) and nothing credited or charged.
    """
    path = plan.path
    return {
        "in": list(plan.transfers_in),
        "out": list(plan.transfers_out),
        "hits": plan.hits,
        "xp": round(plan.xp_total, 2),
        "net": round(plan.objective, 2),
        "path": (
            []
            if path is None
            else [
                {
                    "gw": move.event,
                    "in": list(move.transfers_in),
                    "out": list(move.transfers_out),
                    "chip": move.chip,
                    # What the move spends beyond the free transfers, for the
                    # road ahead's "1 FT, no hit".
                    "hits": move.hits,
                }
                for move in path.moves
            ]
        ),
        "ft_kept": path.ft_kept if path is not None else None,
        "ft_credit": round(path.ft_credit, 2) if path is not None else 0.0,
        "transfer_cost": round(path.transfer_cost, 2) if path is not None else 0.0,
        "bench_charge": round(plan.bench_charge, 2),
        "resolved": resolved,
    }


def _revive(snapshot: dict) -> dict:
    """Turn back the int keys JSON made strings of: player ids, plan ids, and
    each row's gameweeks. Nothing else in a snapshot is keyed by a number."""
    snapshot["players"] = {
        int(pid): {
            **row,
            "f": {int(gw): v for gw, v in row["f"].items()},
            "x": {int(gw): v for gw, v in row["x"].items()},
        }
        for pid, row in snapshot["players"].items()
    }
    snapshot["plans"] = {int(pid): plan for pid, plan in snapshot["plans"].items()}
    return snapshot
