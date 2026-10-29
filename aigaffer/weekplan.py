"""The Week plan: the detailed report's plan, as five texts on a phone.

The owner taps **Week plan** (or texts ``week plan`` or ``plan``) and gets,
within seconds, the newest detailed snapshot cut into five short messages,
each leading with its bottom line: what to do, the team news, why this plan,
the road ahead with the chips, and the fixtures. Nothing is solved and no
gaffer is asked, so it costs nothing and answers at once; the price is that
it is only as fresh as the last run, which the first line says
(``as of Wed scout run 23:35``).

It says what the report says. Both are pure readers of one dict, the run
snapshot (:mod:`aigaffer.detailed.snapshot`), so the action card is the
snapshot's own ``do_this`` word for word, the rationale blocks are the
gaffer's words as he wrote them, and the lines about moves, captains and the
road ahead come from the very helpers the detailed report prints them with
(:mod:`aigaffer.detailed.render` makes them public for this). There is no
shared layer between the two renderers on purpose: a layer would be a third
thing to keep in step, and two readers of one dict cannot disagree about it.

Two things are new here and nowhere else. The fixture strip of text 5 is
written for a phone (``BHA nfo FUL ars``), because the report's cell
(``LEE(h) 4.12``) is too wide for one. And there is a fallback action card,
for the snapshots written before the checklist was kept in them
(``do_this``): the same facts, laid out plainly from the snapshot, and
labelled as the lesser thing it is.

**Freshness is never hidden.** The snapshot's ``ts`` and ``deadline`` are
:data:`~aigaffer.report.render.DEADLINE_FORMAT` strings, minute precision and
UTC, and the owner's recorded week is ISO; both are parsed and compared in UTC
and shown in Irish time, as the chip forecast shows its day. Text 1 says so
when the owner has entered moves since the run, and when the gameweek's
deadline has gone, and a snapshot with no deadline says "deadline unknown"
and skips that one check rather than raising.

:func:`render_week_plan` is pure; :func:`week_plan` is the inbox's handler. It
reads the snapshots and the owner's recorded week under the state lock
(the hourly tick's ``git pull --rebase`` can rewrite ``state/detailed/``
under a reader), renders outside it, renders *all five* texts before sending
any, and replies for itself, so a failure while rendering sends nothing and
the inbox's one retry cannot double-send half a series. No network request is
made: the deadline comes from the snapshot, not the bootstrap.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.config import Config
from aigaffer.detailed.render import (
    NO_CHANGE,
    NO_GAFFER,
    RATIONALE_LABELS,
    REASON_UNAVAILABLE,
    captain_rivals,
    chip,
    name,
    planned,
    what_changed,
    window,
)
from aigaffer.detailed.skipped import BENCH
from aigaffer.detailed.snapshot import latest_snapshot, load_snapshots
from aigaffer.executed import Executed
from aigaffer.inbox import Reply
from aigaffer.report.render import (
    DEADLINE_FORMAT,
    HIT_POINTS,
    chip_label,
    noted,
    price,
    settled,
)
from aigaffer.statesync import REPORT_RUNNING, state_lock
from aigaffer.store import EXECUTED_DIR, read_executed

NO_PLAN = "I haven't written a plan yet — ask again after the next run."

# The owner's clock: a run stamped late on a Tuesday UTC night is Wednesday's
# to him, and the deadline is the time he sees on his phone.
LOCAL_ZONE = ZoneInfo("Europe/Dublin")

# The thresholds the brief left to the builder.
LOW_MINUTES = 65  # under this many projected minutes, a player is flagged
MAX_CHANGES = 6  # change bullets shown before "+N more"
MAX_SKIPPED = 3  # passed-over transfers shown
MAX_REASON = 240  # characters a flag line or reason may run to, ending on a sentence
SOFTER_FROM = 4  # window weeks from this index on are the softer ones

TEXTS = 5

UNDERWAY = (
    "⏱ GW{gw}'s deadline has passed — this is the plan that was made for it."
    " The next plan arrives with the next run."
)
ENTERED = (
    "✍️ You've entered moves since this run; the next report works from your"
    " squad."
)
NO_NEWS = "Nothing flagged: no news and no minutes worries in the squad or the targets."
SOLVER_NOTES = "No gaffer notes this run — it is the solver's pick, so there is no team-news research."
NOTHING_LEARNED = "No findings written up for this run."
NO_ROAD = "No moves planned after this week."
FALLBACK_LABEL = "Action card (the gameweek report has the full checklist)"

_POSITIONS = ("GKP", "DEF", "MID", "FWD")
# The line of the rationale that says which clubs were not searched.
_NOT_CHECKED = re.compile(
    r"^[-•*\s]*(?:what i (?:couldn.t|could not|didn.t|did not) (?:check|search)|not (?:checked|searched))\W*",
    re.IGNORECASE,
)


def render_week_plan(
    snapshots: list[dict], now: datetime, executed: Executed | None
) -> list[str]:
    """The week plan as five texts, in the order they are sent.

    ``snapshots`` are the gameweek's, oldest first, as
    :func:`~aigaffer.detailed.snapshot.load_snapshots` returns them; the last
    is the current one and the one before it is what "changed" is measured
    from. ``now`` is when the owner asked (timezone-aware), and ``executed``
    the position he has recorded for this gameweek, or None.
    """
    if not snapshots:
        raise ValueError("a week plan needs at least one snapshot")
    cur = snapshots[-1]
    prev = snapshots[-2] if len(snapshots) > 1 else None
    return [
        _this_week(cur, prev, now, executed),
        _team_news(cur),
        _why(cur),
        _road_and_chips(cur),
        _fixtures(cur),
    ]


# --- Times ------------------------------------------------------------------


def _utc(text: str | None) -> datetime | None:
    """A snapshot's ``ts`` or ``deadline`` as an aware UTC time, or None when
    it is absent or not in the report's format (an older snapshot kept an ISO
    ``ts``; that one is simply not comparable)."""
    try:
        return datetime.strptime(text or "", DEADLINE_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None


def _recorded(executed: Executed) -> datetime | None:
    """When the owner recorded his week (ISO), as an aware UTC time. A stamp
    with no zone is read as UTC, as the store writes it."""
    try:
        moment = datetime.fromisoformat(executed.recorded_at)
    except ValueError:
        return None
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)


def _irish(moment: datetime) -> datetime:
    return moment.astimezone(LOCAL_ZONE)


# --- Text 1: what to do -----------------------------------------------------


def _this_week(cur: dict, prev: dict | None, now: datetime, executed: Executed | None) -> str:
    ts, deadline = _utc(cur.get("ts")), _utc(cur.get("deadline"))
    ran = (
        f"{_irish(ts):%a} {cur['mode']} run {_irish(ts):%H:%M}"
        if ts
        else f"the {cur['mode']} run"
    )
    due = f"deadline {_irish(deadline):%a %d %b %H:%M} (Irish time)" if deadline else "deadline unknown"
    lines = [f"1/{TEXTS} · 📋 GW{cur['gw']} week plan · as of {ran} · {due}"]

    # Said first, under the header, because either one changes how the rest
    # should be read.
    if executed is not None and executed.gw == cur["gw"]:
        recorded = _recorded(executed)
        if recorded is not None and ts is not None and recorded > ts:
            lines += ["", ENTERED]
    if deadline is not None and deadline <= now.astimezone(UTC):
        lines += ["", UNDERWAY.format(gw=cur["gw"])]

    changes = _changes(cur, prev)
    if changes:
        lines += ["", *changes]
    lines += ["", cur["do_this"].strip() if cur.get("do_this") else _fallback_card(cur)]
    lines += ["", _summary(cur)]
    return "\n".join(lines)


def _changes(cur: dict, prev: dict | None) -> list[str]:
    """The detailed report's own "What changed" bullets, capped. Nothing when
    there is no earlier run to compare with or nothing differs: a section
    that says only "no change" is noise on a phone."""
    if prev is None:
        return []
    found = what_changed(prev, cur)
    if found == [NO_CHANGE]:
        return []
    shown = [f"• {line}" for line in found[:MAX_CHANGES]]
    if len(found) > MAX_CHANGES:
        shown.append(f"+{len(found) - MAX_CHANGES} more")
    return [f"Changed since the {prev['mode']} run:", *shown]


def _fallback_card(cur: dict) -> str:
    """The action card for a snapshot that has no ``do_this``: this week's
    moves priced, the chip, the armbands, the XI by position and the bench in
    order. Plain and complete enough to act on, and labelled as the lesser of
    the two so nobody takes it for the checklist."""
    plan = cur["plans"][cur["chosen"]]
    lines = [FALLBACK_LABEL]
    if plan["in"] or plan["out"]:
        lines.append(
            "Transfers: "
            + planned(cur, {"out": plan["out"], "in": plan["in"], "hits": plan["hits"]})
        )
    else:
        lines.append("Transfers: none — keep the free transfer.")
    played = chip(cur)
    if played != "none":
        lines.append(f"Chip: {chip_label(played)}")
    lines.append(f"Captain {name(cur, cur.get('captain'))} · Vice {name(cur, cur.get('vice'))}")
    groups = {pos: [] for pos in _POSITIONS}
    for pid in sorted(cur["xi"], key=lambda p: (-window(cur, p), p) if p in cur["players"] else (0, p)):
        row = cur["players"].get(pid)
        groups.setdefault(row["p"] if row else "?", []).append(name(cur, pid))
    shape = "-".join(str(len(groups[pos])) for pos in _POSITIONS[1:])
    lines.append(f"XI ({shape}):")
    lines += [f"{pos}: {', '.join(names)}" for pos, names in groups.items() if names]
    lines.append("Bench: " + (", ".join(name(cur, pid) for pid in _bench(cur)) or "none"))
    return "\n".join(lines)


def _bench(cur: dict) -> list[int]:
    """The bench in substitution order; on a free-hit week the temporary
    team's fifteen less its eleven, as the detailed report shows it."""
    free_hit = cur.get("freehit")
    if free_hit:
        return [pid for pid in free_hit["squad"] if pid not in free_hit["xi"]]
    return list(cur["bench"])


def _summary(cur: dict) -> str:
    """``1 FT used, 0 carried over · bank £3.7m · no hit`` — the week's
    transfer bookkeeping on one line. A wildcard or free hit week touches no
    free transfers, and says so instead of counting them."""
    plan = cur["plans"][cur["chosen"]]
    hits = plan["hits"]
    parts = []
    if chip(cur) in (WILDCARD, FREE_HIT):
        parts.append(f"{chip_label(chip(cur))}: free transfers untouched")
    elif cur.get("ft") is not None:
        used = max(0, len(plan["in"]) - hits)
        parts.append(f"{used} FT used, {max(0, cur['ft'] - used)} carried over")
    parts.append(f"bank {price(cur['bank'])}")
    parts.append(f"-{hits * HIT_POINTS} pts in hits" if hits else "no hit")
    return " · ".join(parts)


# --- Text 2: team news ------------------------------------------------------


def _team_news(cur: dict) -> str:
    """The research and the flags, for the players who matter only."""
    lines = [f"2/{TEXTS} · 📰 Team news"]
    rationale = cur["decision"].get("rationale") or ""
    learned = _block(rationale, "WHAT I LEARNED")
    caution = [line for line in learned if _NOT_CHECKED.match(line)]
    learned = [line for line in learned if not _NOT_CHECKED.match(line)]
    if learned:
        lines += ["", "What the gaffer learned:", *learned]
    elif not rationale:
        lines += ["", SOLVER_NOTES]
    else:
        lines += ["", NOTHING_LEARNED]

    flags = _flags(cur)
    lines += ["", "Flags:", *(flags or [NO_NEWS])]
    for line in caution:
        lines += ["", "⚠️ Not checked: " + _NOT_CHECKED.sub("", line).strip()]
    return "\n".join(lines)


def _block(rationale: str, label: str) -> list[str]:
    """The lines of ``rationale`` under ``label``, up to the next label, with
    blanks trimmed from both ends. His words, otherwise untouched."""
    out: list[str] = []
    inside = False
    for line in rationale.replace("\r\n", "\n").split("\n"):
        heading = line.strip().rstrip(":").strip()
        if heading in RATIONALE_LABELS:
            inside = heading == label
        elif inside:
            out.append(line.rstrip())
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _matters(cur: dict) -> list[int]:
    """Whose news is worth reading: the eleven, the bench, then this week's
    signings and the road ahead's, then the captain's rivals. Each once."""
    wanted = [*cur["xi"], *_bench(cur), *_targets(cur), *captain_rivals(cur)]
    seen: set[int] = set()
    return [pid for pid in wanted if not (pid in seen or seen.add(pid)) and pid in cur["players"]]


def _targets(cur: dict) -> list[int]:
    """Who the plan buys: this week's signings, then each later week's."""
    plan = cur["plans"][cur["chosen"]]
    bought = [*plan["in"]]
    for move in plan.get("path") or []:
        bought += move["in"]
    return bought


def _flags(cur: dict) -> list[str]:
    """One line for each player who matters with an FPL flag or fewer than
    :data:`LOW_MINUTES` projected minutes: ``Thomas (DEF) 35 min · Forced
    off``. The reason is the gaffer's own when he set one — the last number
    he set, or the last he only noted — and the FPL news string otherwise."""
    decision = cur["decision"]
    applied = settled(decision.get("adjustments") or [])
    spoken = {r["player_id"]: r["reason"] for r in applied}
    spoken.update(
        {r["player_id"]: r["reason"] for r in noted(decision.get("unapplied") or [], applied)}
    )
    out = []
    for pid in _matters(cur):
        row = cur["players"][pid]
        flagged = bool(row["nw"]) or row["st"] != "a" or row["xm"] < LOW_MINUTES
        if not flagged:
            continue
        reason = _short(spoken.get(pid) or row["nw"])
        line = f"{row['n']} ({row['p']}) {row['xm']:g} min"
        out.append(f"{line} · {reason}" if reason else line)
    return out


def _short(text: str | None) -> str:
    """One line, ending where a sentence ends.

    A cut at a fixed width left the gaffer's reasons stopping mid-word
    ("doesn't justi…"), which reads as a fault. So whole sentences are kept
    while the running total stays within :data:`MAX_REASON`. If even the first
    sentence is longer than that, it is cut at the last word boundary before
    the limit and closed with "…": a word is never split. Text that fits is
    returned as it came, apart from the whitespace being flattened to one line.
    """
    flat = " ".join((text or "").split())
    if len(flat) <= MAX_REASON:
        return flat
    kept = ""
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        joined = f"{kept} {sentence}" if kept else sentence
        if len(joined) > MAX_REASON:
            break
        kept = joined
    if kept:
        return kept
    # One long first sentence: leave room for the ellipsis, then back up to a
    # space so that no word is split. A single unbroken word longer than the
    # limit has no space to back up to, and is cut where it stands.
    room = flat[: MAX_REASON - 1]
    if not flat[MAX_REASON - 1].isspace() and " " in room:
        room = room.rsplit(" ", 1)[0]
    return room.rstrip() + "…"


# --- Text 3: why this plan --------------------------------------------------


def _why(cur: dict) -> str:
    lines = [f"3/{TEXTS} · 🤔 Why this plan"]
    reasons = cur.get("reasons") or {}
    why = _block(cur["decision"].get("rationale") or "", "WHY THIS PLAN")
    if why:
        lines += ["", *why]
    else:
        lines += ["", reasons.get("plan") or _no_words(cur)]

    runner = _runner_up(cur)
    if runner:
        lines += ["", runner]
    lines += ["", *_armbands(cur)]

    # A man passed over only because he would sit on the bench is a big number
    # nobody asks about; the ones the plan weighed and beat are the questions.
    # So the benched go last, and the rest lead by gain.
    skipped = sorted(
        cur.get("skipped") or [], key=lambda item: (item["tag"] == BENCH, -item["gap"])
    )[:MAX_SKIPPED]
    if skipped:
        notes = {item.get("id"): item.get("reason") for item in reasons.get("skipped") or []}
        lines += ["", "Passed over:"]
        lines += [
            f"• {name(cur, item['id'])} over {name(cur, item['replaces'])}"
            f" ({item['gap']:+.1f}): {_short(notes.get(item['id']) or item['tag'])}"
            for item in skipped
        ]

    triggers = _triggers(reasons.get("triggers"))
    if triggers:
        lines += ["", "What would change the plan:", *triggers]
    return "\n".join(lines)


def _no_words(cur: dict) -> str:
    source = cur["decision"].get("source") or ""
    return REASON_UNAVAILABLE if source.startswith("manager") else NO_GAFFER


def _runner_up(cur: dict) -> str:
    """``Next best: +Konsa for a hit, 1.6 behind · Rolling: 1.9 behind`` from
    the options table. Rolling is the plan that makes no move this week; the
    next best is the best of the others.

    Only plans costed on the same footing are compared. When the gaffer
    re-solves, his plans are costed on the minutes he set, and the solver's
    original shortlist on the old ones: their nets sit on different scales,
    and a shortlist plan can out-score the chosen one without being better.
    So the chosen plan is measured against the plans that share its
    ``resolved`` mark, which are the ones his own words compare. Margins are
    in net xP, as the options table has them.
    """
    chosen = cur["plans"][cur["chosen"]]
    others = [
        plan
        for pid, plan in cur["plans"].items()
        if pid != cur["chosen"] and bool(plan.get("resolved")) == bool(chosen.get("resolved"))
    ]
    rolling = [p for p in others if not p["in"] and not p["out"]]
    moving = [p for p in others if p["in"] or p["out"]]
    parts = []
    if moving:
        best = max(moving, key=lambda p: p["net"])
        parts.append(f"Next best: {_describe(cur, best, chosen)}, {_gap(chosen, best)}")
    if rolling and (chosen["in"] or chosen["out"]):
        parts.append(f"Rolling: {_gap(chosen, max(rolling, key=lambda p: p['net']))}")
    return " · ".join(parts)


def _describe(cur: dict, plan: dict, chosen: dict) -> str:
    """What ``plan`` does that the chosen plan does not, and the hits it costs.

    The men it buys that he does not. If it buys the same men (it differs only
    in what it sells, or in its later road ahead), the men it sells that he
    does not, written ``-Name``. If both are the same, the plans differ only
    in the moves planned for later weeks, and the line says so rather than
    naming a man the chosen plan already has.
    """
    extra = [pid for pid in plan["in"] if pid not in chosen["in"]]
    if not extra and set(plan["in"]) != set(chosen["in"]):
        extra = plan["in"]  # it buys only some of his men: say which
    if extra:
        bought = ", ".join(f"+{name(cur, pid)}" for pid in extra)
    else:
        sold = [pid for pid in plan["out"] if pid not in chosen["out"]]
        if not sold and set(plan["out"]) != set(chosen["out"]):
            sold = plan["out"]
        bought = (
            ", ".join(f"-{name(cur, pid)}" for pid in sold)
            or "same moves, a different road ahead"
        )
    if plan["hits"]:
        bought += " for a hit" if plan["hits"] == 1 else f" for {plan['hits']} hits"
    return bought


def _gap(chosen: dict, other: dict) -> str:
    margin = chosen["net"] - other["net"]
    return f"{margin:.1f} behind" if margin >= 0 else f"{-margin:.1f} ahead"


def _armbands(cur: dict) -> list[str]:
    """The captain, the vice and the rivals the armband could have gone to,
    each with the plain window total and this week's xP."""
    first = cur["events"][0] if cur.get("events") else None

    def line(label: str, pid: int | None) -> str | None:
        row = cur["players"].get(pid)
        if row is None:
            return None
        this_week = row["x"].get(first, 0.0)
        return f"{label} {row['n']} — {window(cur, pid):.1f} over the window, {this_week:.1f} this week"

    out = [line("Captain", cur.get("captain")), line("Vice", cur.get("vice"))]
    rivals = [line("", pid) for pid in captain_rivals(cur)]
    if any(rivals):
        out.append("Rivals:")
        out += [f"  {text.strip()}" for text in rivals if text]
    return [text for text in out if text]


def _triggers(value: object) -> list[str]:
    """The gaffer's decision points, when a later phase stores them in
    ``reasons["triggers"]``: a string, or a list of strings or of small dicts
    (their text fields joined with an arrow). The slot costs nothing today."""
    if not value:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out = []
    for item in items:
        if isinstance(item, dict):
            item = " → ".join(str(part) for part in item.values() if isinstance(part, str))
        if str(item).strip():
            out.append(f"• {_short(str(item))}")
    return out


# --- Text 4: the road ahead and the chips -----------------------------------


def _road_and_chips(cur: dict) -> str:
    lines = [f"4/{TEXTS} · 🛣 The road ahead and chips", "", "Current plan, re-planned every run:"]
    path = cur["plans"][cur["chosen"]].get("path") or []
    lines += [f"GW{move['gw']}: {planned(cur, move)}" for move in path] or [NO_ROAD]

    played = chip(cur)
    lines += ["", f"Chip this week: {chip_label(played)}" if played != "none" else "No chip this week."]
    lines += list(cur.get("calendar") or [])
    values = [
        f"{chip_label(kind)} {value:+.1f}" + (" (over the window)" if kind == WILDCARD else "")
        for kind, value in (cur.get("chip_evs") or {}).items()
    ]
    if values:
        lines += ["", "Worth this week: " + " · ".join(values)]
    return "\n".join(lines)


# --- Text 5: the fixtures ---------------------------------------------------


def _fixtures(cur: dict) -> str:
    events = cur["events"]
    total = {gw: sum(cur["players"][p]["x"].get(gw, 0.0) for p in cur["xi"] if p in cur["players"]) for gw in events}
    lines = [
        f"5/{TEXTS} · 📅 Upcoming gameweeks",
        "",
        "XI projected, before the armband: " + " · ".join(f"GW{gw} {total[gw]:.1f}" for gw in events),
    ]
    if len(events) > SOFTER_FROM:
        softer = events[SOFTER_FROM:]
        span = f"GW{softer[0]}" if len(softer) == 1 else f"GW{softer[0]}–GW{softer[-1]}"
        lines.append(f"{span} are softer: the fixtures are known, the form is a guess.")
    lines += [
        "",
        f"Opponents GW{events[0]} to GW{events[-1]}, home UPPER, away lower, — a blank, + a double:",
    ]

    order = {pos: index for index, pos in enumerate(_POSITIONS)}
    squad = sorted(
        (p for p in cur["xi"] if p in cur["players"]),
        key=lambda p: (order.get(cur["players"][p]["p"], 9), -window(cur, p), p),
    )
    lines += [_strip(cur, pid) for pid in squad]
    targets = []
    for pid in _targets(cur):
        if pid in cur["players"] and pid not in squad and pid not in targets:
            targets.append(pid)
    if targets:
        lines += ["", "Targets:", *(_strip(cur, pid) for pid in targets)]
    return "\n".join(lines)


def _strip(cur: dict, pid: int) -> str:
    """``Haaland BHA nfo FUL ars LEE bre — 32.9``: a man's opponents across the
    window and his total. A blank is ``—``, a double joins with ``+``."""
    row = cur["players"][pid]
    cells = []
    for gw in cur["events"]:
        games = row["f"].get(gw) or []
        cells.append(
            "+".join(opp if where == "h" else opp.lower() for opp, where in games) or "—"
        )
    return f"{row['n']} {' '.join(cells)} — {window(cur, pid):.1f}"


# --- The inbox's handler ----------------------------------------------------


def week_plan(cfg: Config, lock_path: Path, sent_at: datetime, reply: Reply) -> None:
    """The inbox's handler for "Week plan": read, render all five, send.

    ``lock_path`` is the state lock the hourly tick holds while it writes
    (the inbox binds it, as it does for the chip what-if). The snapshots and
    the owner's recorded week are read inside it and rendered outside it. If
    rendering raises, nothing has been sent and the inbox's apology and retry
    are exactly right; if it succeeds the five go out in order. Returns None:
    the handler replied for itself.
    """
    with state_lock(lock_path, on_wait=lambda: _say(reply, REPORT_RUNNING)):
        newest = latest_snapshot(cfg.state_dir)
        snapshots = [] if newest is None else load_snapshots(cfg.state_dir, newest["gw"])
        executed = (
            None
            if newest is None
            else read_executed(cfg.state_dir / EXECUTED_DIR, newest["gw"])
        )
    if not snapshots:
        reply(NO_PLAN)
        return
    for text in render_week_plan(snapshots, sent_at, executed):
        reply(text)


def _say(reply: Reply, text: str) -> None:
    """A courtesy message: one that fails is not the week plan failing."""
    try:
        reply(text)
    except Exception:
        pass
