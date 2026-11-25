"""What the last run of this gameweek told the owner, and whether this one agrees.

Several runs serve one gameweek and each decides afresh. Two of them can reach
different answers on numbers that hardly moved, and a text that quietly
replaces the one before it is worse than either answer. So a run is shown the
call made before it, has to say why when it changes that call
(:mod:`aigaffer.manager.keep_call` is the rule), and the report says in its
first line whether this is the same call or a changed one.

This module is the memory and the words: reading the earlier record, what has
moved in the API since, the briefing's section, and the report's line. It
decides nothing.

* **The record is the store's** (:meth:`aigaffer.store.Store.latest_verdict`),
  the same "latest verdict" "Transfers made" and the what-ifs read. The saved
  snapshot is read only for what the API said about each player then.
* **Calls are compared by the team they leave**, not by their move lists. Once
  the owner has entered a move, the next run rolls from the squad that has it,
  and that is the same call.
* **Nothing here may cost a run its report.** A record or a snapshot that is
  missing, half-written or an unfamiliar shape is no earlier call, or no facts
  about it, said on the log by the exception's class alone.
"""

import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from aigaffer.chips import FREE_HIT
from aigaffer.detailed.snapshot import snapshot_path
from aigaffer.executed import NO_CHIP, Verdict
from aigaffer.recording import MODE_NAMES, VERDICT_MODES
from aigaffer.report.render import chip_label, plural, price
from aigaffer.solver.optimizer import Plan
from aigaffer.store import Store

if TYPE_CHECKING:
    from aigaffer.news_ledger import Entry

# How far apart two plans' nets can be and still be the same number. The net is
# a decayed six-gameweek total. Measured on the saved runs of GW8 to GW12, where
# the squad and the free transfers did not change between two runs, a plan's gap
# over rolling moved by 0.01 at the median, 0.39 at the 90th percentile and 0.65
# at the most: 33 gaps from 8 pairs of runs, so not 33 independent draws. The one
# pair outside that is GW10's, where every move cost a hit and the gaps moved by
# 2.0, which this does not clear. Being a little high costs one sentence, since
# news always changes a call whatever the gap.
NOISE_POINTS = 1.5
# The same for the armband, on its own scale: one gameweek's expected points,
# which is what an armband is worth again. One man's projection moved by 0.19
# at the 90th percentile between those runs. Expected points and not the
# attacking EV the solver picks the armband on, because the saved runs keep no
# attacking EV to have measured.
CAPTAIN_NOISE = 0.5
# Minutes closer than this are the same minutes.
MINUTES_MOVED = 5.0

# The report's line is read on a phone above the checklist.
LINE_CHARS = 140
REASON_CHARS = 80
# What is worth quoting back of an old rationale, and how many facts.
WHY_CHARS = 240
MAX_FACTS = 12
# From this many days back a weekday would name the wrong week.
WEEKDAY_DAYS = 6

NEWS, FIXTURES, DEARER, CHEAPER = "news", "fixtures", "dearer", "cheaper"
SOLVERS_PICK = "this is the solver's pick"
GAFFER = "manager"

EARLIER_HEADING = "## Your earlier call this gameweek"
UNREADABLE = "earlier call: could not be read ({reason}) — running without it"
FACTS_SKIPPED = "earlier call: what has changed since is unknown ({reason})"

# An id in an old rationale is the old run's: this run's plan 4 is another plan.
_PLAN_ID = re.compile(r"\bplans?\s+\d+\b", re.IGNORECASE)
_MARKS = re.compile(r"[#*_`|<>\[\]]")


@dataclass(frozen=True)
class Earlier:
    """The call the last run made. ``squad`` is the fifteen it left: the
    temporary one on a free hit. ``minutes`` are the adjustments it was costed
    on. ``by_gaffer`` is False when that week was the solver's own."""

    mode: str
    ts: datetime
    squad: frozenset[int]
    squad_before: frozenset[int]
    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    captain: int
    vice: int
    chip: str
    reason: str
    by_gaffer: bool
    minutes: Mapping[int, float] = field(default_factory=dict)
    chip_evs: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Call:
    """A decision reduced to what makes two of them the same one."""

    squad: frozenset[int]
    chip: str
    captain: int
    vice: int


@dataclass(frozen=True)
class Continuity:
    """Everything a run holds about the call before it.

    ``since`` is what the API says differently about each player now, or None
    when that could not be worked out, which is not the same as nothing.
    ``used`` are the minutes the earlier run was costed on. ``known`` is the
    news ledger as it stood when this run began. ``chip_kept`` is whether the
    earlier call's chip can still be played, ``chip_moved`` whether the panel
    prices it differently now. ``section`` is the briefing's section, written."""

    earlier: Earlier
    since: Mapping[int, list[tuple[str, str]]] | None = None
    used: Mapping[int, float] = field(default_factory=dict)
    known: Mapping[int, "Entry"] = field(default_factory=dict)
    chip_kept: bool = True
    chip_moved: bool = True
    section: str = ""


# --- reading it back ---------------------------------------------------------


def read_earlier(store: Store, gw: int) -> Earlier | None:
    """The latest call made for ``gw``, or None. Never raises.

    A draft is no call, and neither is an early run the gaffer never decided:
    that one is shelved rather than sent, so the owner was told nothing. Both
    are passed over for the row before them."""
    try:
        before = None
        while True:
            verdict = store.latest_verdict(gw, VERDICT_MODES, at_or_before=before)
            if verdict is None:
                return None
            if not _unsent(verdict):
                return _parse(verdict)
            before = datetime.fromisoformat(verdict.ts) - timedelta(microseconds=1)
    except Exception as error:  # memory, not the week
        print(UNREADABLE.format(reason=type(error).__name__))
        return None


def _unsent(verdict: Verdict) -> bool:
    decision = verdict.decision
    source = decision.get("decision_source")
    shelved = verdict.mode == "early" and source is not None and source != GAFFER
    return bool(decision.get("initial_draft")) or shelved


def _parse(verdict: Verdict) -> Earlier:
    """The record as an :class:`Earlier`. Raises on any shape but the one a run
    writes; :func:`read_earlier` is the net."""
    decision = verdict.decision
    before = _ids(decision["squad_before"])
    came, went = _ids(decision["transfers_in"]), _ids(decision["transfers_out"])
    chip = decision["chip"]
    if not isinstance(chip, str):
        raise ValueError("chip")
    temporary = decision.get("freehit_squad") if chip == FREE_HIT else None
    ts = datetime.fromisoformat(verdict.ts)
    if ts.tzinfo is None:
        raise ValueError("ts")
    said = decision.get("change_reason") or why(decision.get("rationale"))
    return Earlier(
        mode=verdict.mode,
        ts=ts.astimezone(UTC),
        squad=frozenset(
            _ids(temporary) if temporary else (set(before) - set(went)) | set(came)
        ),
        squad_before=frozenset(before),
        transfers_in=came,
        transfers_out=went,
        captain=_id(decision["captain"]),
        vice=_id(decision["vice"]),
        chip=chip,
        reason=flat(said, WHY_CHARS),
        by_gaffer=decision.get("decision_source") == GAFFER,
        minutes={
            _id(entry["player_id"]): float(entry["expected_minutes"])
            for entry in decision.get("adjustments") or []
        },
        chip_evs={
            str(kind): float(value)
            for kind, value in (decision.get("chip_evs") or {}).items()
            if isinstance(value, (int, float))
        },
    )


def _id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("id")
    return value


def _ids(values: object) -> tuple[int, ...]:
    if not isinstance(values, list):
        raise ValueError("ids")
    return tuple(_id(value) for value in values)


def why(rationale: object) -> str:
    """The one paragraph of a rationale that argues for the plan: the first
    under WHY THIS PLAN, or the opening one when it has no such heading."""
    lines = rationale.strip().splitlines() if isinstance(rationale, str) else []
    start = next(
        (n + 1 for n, line in enumerate(lines) if line.strip().upper().startswith("WHY THIS PLAN")), 0
    )
    paragraph: list[str] = []
    for line in lines[start:]:
        if line.strip():
            paragraph.append(line.strip())
        elif paragraph:
            break
    return " ".join(paragraph)


def flat(text: object, limit: int) -> str:
    """One line of somebody's prose, unable to open a section, forge a column
    or name a plan id that means another plan here, and no longer than asked."""
    words = " ".join(_MARKS.sub(" ", text if isinstance(text, str) else "").split())
    return _cut(_PLAN_ID.sub("that plan", words), limit)


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def clean_reason(text: object) -> str | None:
    """His ``change_reason`` as the report prints it after "because", or None
    when there is nothing left to print."""
    said = re.sub(r"^because[\s,:]+", "", flat(text, 10_000), flags=re.IGNORECASE)
    return _cut(said.rstrip(" .…"), REASON_CHARS) or None


# --- comparing calls -----------------------------------------------------------


def fielded(plan: Plan, chip: str) -> frozenset[int]:
    """The fifteen a plan leaves with ``chip`` played: the temporary one on a
    free hit the plan was built for, else the standing squad."""
    if chip == FREE_HIT and plan.path is not None and plan.path.week1_freehit_squad:
        return frozenset(plan.path.week1_freehit_squad)
    return frozenset(plan.squad)


def differences(earlier: Earlier, call: Call) -> set[str]:
    """Which of squad, chip, captain and vice are not what they were."""
    return {
        part
        for part in ("squad", "chip", "captain", "vice")
        if getattr(earlier, part) != getattr(call, part)
    }


# --- what has moved since -------------------------------------------------------


def read_rows(state_dir: Path, gw: int, earlier: Earlier) -> dict[int, dict]:
    """The player rows the earlier run saved. Raises when there are none, or
    when the file is some other run's: the caller is the net."""
    path = snapshot_path(state_dir, gw, earlier.mode)
    saved = json.loads(path.read_text(encoding="utf-8"))
    if (saved["gw"], saved["mode"]) != (gw, earlier.mode):
        raise ValueError("another run's snapshot")
    if frozenset(saved["squad"]) != earlier.squad_before:
        raise ValueError("another squad's snapshot")
    return {int(pid): row for pid, row in saved["players"].items()}


def since(then: Mapping[int, dict], now: Mapping[int, dict]) -> dict[int, list[tuple[str, str]]]:
    """What the API says differently about each player held in both."""
    found: dict[int, list[tuple[str, str]]] = {}
    for pid in sorted(then.keys() & now.keys()):
        before, after = then[pid], now[pid]
        facts: list[tuple[str, str]] = []
        if before["nw"] != after["nw"] or before["st"] != after["st"]:
            told = flat(after["nw"], WHY_CHARS) or f"status {flat(after['st'], 20)}, no news"
            facts.append((NEWS, f"{told} (was: {flat(before['nw'], WHY_CHARS) or 'no news'})"))
        if before["pr"] != after["pr"]:
            kind = DEARER if after["pr"] > before["pr"] else CHEAPER
            facts.append((kind, f"price {price(after['pr'])} (was {price(before['pr'])})"))
        was, is_now = _games(before["f"]), _games(after["f"])
        if any(was[gw] != is_now[gw] for gw in was.keys() & is_now.keys()):
            facts.append((FIXTURES, "fixtures changed"))
        if facts:
            found[pid] = facts
    return found


def _games(by_week: Mapping) -> dict[int, list[tuple]]:
    """A row's fixtures however they were keyed: JSON made strings of them."""
    return {int(gw): sorted(tuple(game) for game in games) for gw, games in by_week.items()}


# --- the words -------------------------------------------------------------------


def label(earlier: Earlier, now: datetime) -> str:
    """``Tuesday's early look``, or ``9 Nov's early look`` from a week back.
    UTC, as the report's "Working from" line and the recording's reply are."""
    days = (now.astimezone(UTC).date() - earlier.ts.date()).days
    day = f"{earlier.ts:%A}" if days < WEEKDAY_DAYS else f"{earlier.ts.day} {earlier.ts:%b}"
    return f"{day}'s {MODE_NAMES.get(earlier.mode, earlier.mode)}"


def earlier_section(
    earlier: Earlier,
    facts: Mapping[int, list[tuple[str, str]]] | None,
    *,
    now: datetime,
    name: Callable[[int], str],
    plan_id: int | None,
    entered: bool,
    relevant: Iterable[int],
) -> str:
    """The briefing's section. ``plan_id`` is the plan on the board that is the
    earlier call, ``entered`` whether the owner has recorded it as made."""
    hours = max(0, round((now - earlier.ts).total_seconds() / 3600))
    who = lambda pid: f"{flat(name(pid), 60)} (id {pid})"  # noqa: E731
    moves = (
        "sell " + ", ".join(who(pid) for pid in earlier.transfers_out)
        + "; buy " + ", ".join(who(pid) for pid in earlier.transfers_in)
        if earlier.transfers_in or earlier.transfers_out
        else "roll"
    )
    chip = "none" if earlier.chip == NO_CHIP else chip_label(earlier.chip)
    lines = [
        EARLIER_HEADING,
        "",
        f"{label(earlier, now)} ({earlier.ts:%a %d %b %H:%M} UTC, {plural(hours, 'hour')} ago)"
        " told the owner:",
        f"- Transfers: {moves}",
        f"- Captain {who(earlier.captain)}, vice {who(earlier.vice)}. Chip: {chip}.",
    ]
    if earlier.reason:
        lines.append(f"- Why: {earlier.reason}")
    if not earlier.by_gaffer:
        lines.append("- That was the solver's own week: you were not available.")
    if entered:
        lines.append("- He has entered it: your squad above already has those moves in it.")
    lines.append(
        f"- On this board that call is plan {plan_id}."
        if plan_id is not None
        else "- That call is not among the plans on this board."
    )
    if facts is not None:
        wanted = set(relevant)
        shown = [
            f"- {who(pid)}: " + "; ".join(text for _, text in facts[pid])
            for pid in sorted(facts)
            if pid in wanted
        ]
        more = [f"- and {len(shown) - MAX_FACTS} more"] if len(shown) > MAX_FACTS else []
        lines += ["", "Since then, as the API has it:"]
        lines += [*shown[:MAX_FACTS], *more] or ["- nothing has changed for these players"]
    lines += [
        "",
        f"Keep that call unless something is new. Under {NOISE_POINTS} net over the window,"
        f" or {CAPTAIN_NOISE} xP for the captain, is noise: a change that small needs new"
        " evidence about a player, named in change_player_id. Real news is always enough.",
    ]
    return "\n".join(lines)


def report_line(
    earlier: Earlier,
    call: Call,
    *,
    transfers_out: list[int],
    transfers_in: list[int],
    name: Callable[[int], str],
    reason: str | None,
    now: datetime,
) -> str:
    """The report's first line: the same call as before, or what it is now.

    What it is now is read off the decision, so the line cannot disagree with
    the checklist under it; only the words after "because" are anybody's. Never
    over :data:`LINE_CHARS`: the vice goes first, then the captain, then the
    end of the reason."""
    run = label(earlier, now)
    differ = differences(earlier, call)
    if not differ:
        return f"Same call as {run}."
    who = lambda pid: flat(name(pid), 30)  # noqa: E731
    if differ == {"vice"}:
        return _cut(f"Same call as {run}; vice now {who(call.vice)}.", LINE_CHARS)

    parts: list[tuple[str, str]] = []
    if "squad" in differ:
        if call.chip == FREE_HIT:
            if "chip" not in differ:
                parts.append(("squad", f"a new {chip_label(FREE_HIT)} team"))
        elif not transfers_in and not transfers_out:
            parts.append(("squad", "roll"))
        elif len(transfers_in) == len(transfers_out) == 1:
            parts.append(("squad", f"sell {who(transfers_out[0])}, buy {who(transfers_in[0])}"))
        else:
            parts.append(("squad", plural(len(transfers_in), "transfer")))
    if "chip" in differ:
        parts.append(
            ("chip", f"play the {chip_label(call.chip)}")
            if call.chip != NO_CHIP
            else ("chip", f"no {chip_label(earlier.chip)}")
        )
    if "captain" in differ:
        parts.append(("captain", f"captain {who(call.captain)}"))
    if "vice" in differ:
        parts.append(("vice", f"vice {who(call.vice)}"))

    head = ""
    for dropped in ((), ("vice",), ("vice", "captain")):
        kept = [text for part, text in parts if part not in dropped] or [parts[0][1]]
        head = f"Changed from {run}: now {', '.join(kept)}"
        line = f"{head}, because {reason}." if reason else f"{head}."
        if len(line) <= LINE_CHARS:
            return line
    room = LINE_CHARS - len(head) - len(", because .")
    if reason and room >= 20:
        return f"{head}, because {_cut(reason, room)}."
    return _cut(f"{head}.", LINE_CHARS)
