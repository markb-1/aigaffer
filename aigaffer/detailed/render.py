"""The detailed report, as markdown: a pure function from run snapshots.

``reports/GW{n} Detailed Report.md`` is the owner-facing long form of a run:
the plan, the transfers side by side with their fixtures, the squad, the
armbands, what was passed over and what was weighed, and a history of what
changed between the week's runs. It reads only the snapshots
(:mod:`aigaffer.detailed.snapshot`), never the network or the solver, so it can
be re-rendered from the committed files alone and tested on hand-built dicts.

The last snapshot in the list is the current one; earlier ones are the week's
previous runs, oldest first, and feed "What changed".

Fixture cells read ``LEE(h) 4.12``: the opponent, where, and the player's xP
for that gameweek. A double joins its cells with ``+`` and one xP for the
gameweek (``LEE(h)+wol(a) 7.30``); a blank is ``— 0.00``. The away opponent is
lower case, as in FPL's own convention, so home and away read apart at a
glance without the bracket.

The prose slots (why this transfer, why this captain, why this chip) belong to
the gaffer, and a later phase stores them in the snapshot's ``reasons``. Until
then, and on any run that lacks them, a slot says honestly what it is: the
solver's pick when no gaffer ran, "reason unavailable" when one did and left
no words.
"""

from aigaffer.report.render import chip_label, plural, price

NO_GAFFER = "Solver's pick — no gaffer this run"
REASON_UNAVAILABLE = "Reason unavailable this run"

NO_CHANGE = "No change to the plan"
FIRST_LOOK = "First look"

# Said once under each table of players: the Weighted column is the one the
# planner decides on, and it is smaller than the sum of the cells beside it.
WEIGHTED_CAPTION = (
    "Weighted counts later gameweeks for less, as the planner does;"
    " it is the total decisions are made on."
)

_ATTACKERS = ("MID", "FWD")
_POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}
RIVALS = 2


def render_detailed(snapshots: list[dict], deadline_text: str) -> str:
    """The whole report for the current (last) snapshot, with the history of
    the earlier ones in "What changed"."""
    cur = snapshots[-1]
    lines = [
        f"# GW{cur['gw']} Detailed Report",
        "",
        f"Deadline: {deadline_text} · Last updated: {cur['mode']} run {cur['ts']}",
        "",
        "## What changed",
        "",
        *_history(snapshots),
        "",
        "## The plan",
        "",
        *_plan_section(cur),
        "",
        "## Transfers",
        "",
        *_transfers(cur),
        "",
        *_squad(cur),
        "",
        "## Captain and vice",
        "",
        *_armbands(cur),
        "",
        "## Transfers skipped",
        "",
        *_skipped(cur),
        "",
        "## The options weighed",
        "",
        *_options(cur),
        "",
        "## Chips",
        "",
        *_chips(cur),
    ]
    return "\n".join(lines) + "\n"


# --- What changed -----------------------------------------------------------


def what_changed(prev: dict | None, cur: dict) -> list[str]:
    """What differs from ``prev`` to ``cur``, one line each, old → new.

    No ``prev`` is the first look; no difference says so, rather than leaving
    the owner to wonder whether the section is empty because nothing changed or
    because it is broken.
    """
    if prev is None:
        return [FIRST_LOOK]
    lines = []
    old_moves, new_moves = _moves(prev), _moves(cur)
    if old_moves != new_moves:
        lines.append(f"Transfers: {_moves_text(cur)} (was: {_moves_text(prev)})")
    for label, key in (("Captain", "captain"), ("Vice", "vice")):
        if prev.get(key) != cur.get(key):
            lines.append(f"{label}: {_name(cur, cur.get(key))} (was: {_name(prev, prev.get(key))})")
    if _chip(prev) != _chip(cur):
        lines.append(f"Chip: {_chip(cur)} (was: {_chip(prev)})")
    if prev.get("bench") != cur.get("bench"):
        lines.append(f"Bench: {_names(cur, cur['bench'])} (was: {_names(prev, prev['bench'])})")
    for adjustment in cur["decision"].get("adjustments", []):
        pid = adjustment["player_id"]
        old = prev["players"].get(pid, {}).get("xm")
        new = cur["players"].get(pid, {}).get("xm")
        if old != new:
            shown = "not projected" if new is None else _minutes(new)
            was = "not projected" if old is None else _minutes(old)
            lines.append(f"{_name(cur, pid)}: xMins {was} → {shown}")
    for pid in cur["squad"]:
        before, after = prev["players"].get(pid), cur["players"].get(pid)
        if before is None or after is None:
            continue
        if before["nw"] != after["nw"]:
            now = f"news '{_clean(after['nw'])}'" if after["nw"] else "news cleared"
            was = f"'{_clean(before['nw'])}'" if before["nw"] else "none"
            lines.append(f"{after['n']}: {now} (was {was})")
        elif before["st"] != after["st"]:
            lines.append(f"{after['n']}: status {after['st']} (was {before['st']})")
    return lines or [NO_CHANGE]


def _history(snapshots: list[dict]) -> list[str]:
    """One entry per snapshot, newest first, each headed by its run."""
    out = []
    for index in range(len(snapshots) - 1, -1, -1):
        snapshot = snapshots[index]
        again = " (re-run)" if snapshot.get("reran") else ""
        out.append(f"**{snapshot['mode']} — {snapshot['ts']}{again}**")
        out.append("")
        prev = snapshots[index - 1] if index else None
        out.extend(f"- {line}" for line in what_changed(prev, snapshot))
        out.append("")
    return out[:-1]


def _pairs(snapshot: dict) -> list[tuple[int, int]]:
    """The chosen plan's transfers as ``(out, in)`` pairs.

    The plan keeps its ``out`` and ``in`` as sorted id lists, not matched
    pairs, so zipping them would pair strangers. FPL transfers are like for
    like by position, so each ``out`` is paired with an ``in`` of the same
    position, within a position strongest window total (``t``) first on both
    sides. Anything that cannot be paired (a free-hit style odd lot) is left
    out rather than invented.
    """
    plan = snapshot["plans"][snapshot["chosen"]]
    players = snapshot["players"]

    def by_position(ids):
        groups: dict[str, list[int]] = {}
        for pid in sorted(ids, key=lambda p: (-players[p]["t"], p) if p in players else (0, p)):
            groups.setdefault(players[pid]["p"] if pid in players else "?", []).append(pid)
        return groups

    gone, came = by_position(plan["out"]), by_position(plan["in"])
    pairs = []
    for pos in gone:
        pairs.extend(zip(gone[pos], came.get(pos, []), strict=False))
    return pairs


def _moves(snapshot: dict) -> tuple[list[int], list[int]]:
    plan = snapshot["plans"][snapshot["chosen"]]
    return sorted(plan["out"]), sorted(plan["in"])


def _moves_text(snapshot: dict) -> str:
    plan = snapshot["plans"][snapshot["chosen"]]
    if not plan["in"]:
        return "none"
    return ", ".join(f"{_name(snapshot, out)} → {_name(snapshot, ins)}" for out, ins in _pairs(snapshot))


def _chip(snapshot: dict) -> str:
    return snapshot["decision"].get("chip") or "none"


def _minutes(value: float) -> str:
    return f"{value:g}"


# --- Sections ---------------------------------------------------------------


def _plan_section(cur: dict) -> list[str]:
    plan = cur["plans"][cur["chosen"]]
    decision = cur["decision"]
    return [
        _clean(decision.get("rationale")) or _fallback(cur),
        "",
        f"Expected points over the window: {plan['xp']:.2f} xP;"
        f" net {plan['net']:.2f} (after hits and charges, plus the credit for"
        f" free transfers kept); {plural(plan['hits'], 'hit')}; bank {price(cur['bank'])}.",
    ]


def _table_head(cur: dict) -> list[str]:
    """The header and separator of a player table, one column a gameweek."""
    events = cur["events"]
    return [
        "| | Player | Price | " + " | ".join(f"GW{gw}" for gw in events) + " | Total | Weighted |",
        "|---|---|---|" + "---|" * len(events) + "---|---|",
    ]


def _transfers(cur: dict) -> list[str]:
    """Each move gets a table of its own.

    Markdown only reads rows as a table when a header and separator lead them,
    so a second pair's rows after a paragraph would show as raw pipes.
    """
    plan = cur["plans"][cur["chosen"]]
    if not plan["in"]:
        return ["No transfers this week."]
    reasons = {
        (item.get("out"), item.get("in")): item.get("reason")
        for item in _reasons(cur).get("transfers") or []
    }
    out: list[str] = []
    pairs = _pairs(cur)
    for gone, came in pairs:
        gap = cur["players"][came]["t"] - cur["players"][gone]["t"]
        plain_gap = _window(cur, came) - _window(cur, gone)
        out += [
            f"### {_name(cur, gone)} → {_name(cur, came)}",
            "",
            *_table_head(cur),
            _player_row(cur, gone, "Out"),
            _player_row(cur, came, "In"),
            "",
            WEIGHTED_CAPTION,
            "",
            f"{_name(cur, came)} over {_name(cur, gone)}: {gap:+.2f} weighted"
            f" ({plain_gap:+.2f} plain) over the window. "
            + _clean(reasons.get((gone, came)) or _fallback(cur)),
            "",
        ]
    if len(pairs) < max(len(plan["in"]), len(plan["out"])):
        # A legal plan swaps like for like, so this should never print; if it
        # does, say so rather than show fewer moves than were made.
        out += ["Note: a move could not be paired by position and is not shown.", ""]
    return out[:-1]


def _squad(cur: dict) -> list[str]:
    free_hit = cur.get("freehit")
    if free_hit:
        fielded, xi = free_hit["squad"], free_hit["xi"]
        out = [
            "## Your free-hit team",
            "",
            f"Standing squad: {_names(cur, cur['squad'])}",
            "",
        ]
    else:
        fielded, xi = cur["squad"], cur["xi"]
        out = ["## Your squad", ""]
    out += _table_head(cur)
    # The XI by position (goalkeeper to forward), the strongest weighted total
    # first within one; then the bench in the order the subs come on.
    lineup = sorted(
        xi, key=lambda p: (_POSITION_ORDER.get(cur["players"][p]["p"], 9), -cur["players"][p]["t"], p)
    )
    out += [_player_row(cur, pid, "XI") for pid in lineup]
    bench = [p for p in fielded if p not in xi] if free_hit else cur["bench"]
    out += [_player_row(cur, pid, f"Bench {i}") for i, pid in enumerate(bench, 1)]
    return [*out, "", WEIGHTED_CAPTION]


def captain_rivals(cur: dict) -> list[int]:
    """The men the armband could have gone to instead: the XI's other
    attackers, strongest plain window total first, at most :data:`RIVALS`.

    Public because the gaffer's writer is briefed on the same list the report
    prints, so a captain reason that weighs Saka against Haaland is weighing
    him against the man the reader sees beside him.
    """
    captain = cur["captain"]
    return sorted(
        (p for p in cur["xi"] if p != captain and cur["players"][p]["p"] in _ATTACKERS),
        key=lambda p: (-_window(cur, p), p),
    )[:RIVALS]


def _armbands(cur: dict) -> list[str]:
    reasons = _reasons(cur)
    captain, vice = cur["captain"], cur["vice"]
    totals = {pid: _window(cur, pid) for pid in cur["xi"]}
    rivals = captain_rivals(cur)
    out = []
    # The writer gives one reason for the armbands, about the captain; the vice
    # is shown without one.
    for label, pid, key in (("Captain", captain, "captain"), ("Vice", vice, None)):
        if pid is None:
            continue
        line = f"- **{label}: {_name(cur, pid)}** ({totals.get(pid, _window(cur, pid)):.2f} over the window)"
        if key:
            line += f" — {_clean(reasons.get(key)) or _fallback(cur)}"
        out.append(line)
    if rivals:
        out.append("- Rivals: " + ", ".join(f"{_name(cur, p)} ({totals[p]:.2f})" for p in rivals))
    return out or ["No armbands recorded."]


def _skipped(cur: dict) -> list[str]:
    rows = cur.get("skipped") or []
    if not rows:
        return ["Nothing to report."]
    notes = {
        item.get("id"): _clean(item.get("reason")) for item in _reasons(cur).get("skipped") or []
    }
    out = ["| Player | Instead of | Gap (weighted) | Price | Why not | Note |", "|---|---|---|---|---|---|"]
    for item in rows:
        out.append(
            f"| {_name(cur, item['id'])} | {_name(cur, item['replaces'])}"
            f" | {item['gap']:+.2f} | {item['price_diff'] / 10:+.1f}m | {item['tag']}"
            f" | {notes.get(item['id']) or ''} |"
        )
    return out


def _options(cur: dict) -> list[str]:
    out = ["| Plan | In | Out | Hits | xP | Net | |", "|---|---|---|---|---|---|---|"]
    for pid, plan in sorted(cur["plans"].items()):
        marks = []
        if pid == cur["chosen"]:
            marks.append("chosen")
        if plan.get("resolved"):
            marks.append("re-solved")
        out.append(
            f"| {pid} | {_names(cur, plan['in'])} | {_names(cur, plan['out'])}"
            f" | {plan['hits']} | {plan['xp']:.2f} | {plan['net']:.2f} | {', '.join(marks)} |"
        )
    return [*out, "", _clean(_reasons(cur).get("plan")) or _fallback(cur)]


def _chips(cur: dict) -> list[str]:
    decision = cur["decision"]
    chip = _chip(cur)
    out = [f"Chip played: {chip_label(chip)}." if chip != "none" else "No chip this week."]
    if chip != "none":
        out.append(
            _clean(_reasons(cur).get("chips") or decision.get("chip_justification"))
            or _fallback(cur)
        )
    out += ["", "Points each chip would add this week:"]
    for name, value in cur["chip_evs"].items():
        # The wildcard is priced over the whole horizon, and says so, as the
        # gameweek report's chip panel does.
        label = chip_label(name) + (" (over the window)" if name == "wildcard" else "")
        out.append(f"- {label}: {value:+.2f}")
    if cur.get("calendar"):
        out += ["", "Calendar:"]
        out += [f"- {line}" for line in cur["calendar"]]
    return out


# --- Cells and names --------------------------------------------------------


def _cell(row: dict, gw: int) -> str:
    """One gameweek's cell: ``LEE(h) 4.12``, a double joined with ``+``, a
    blank ``— 0.00``. Away opponents in lower case."""
    games = row["f"].get(gw) or []
    xp = row["x"].get(gw, 0.0)
    if not games:
        return f"— {xp:.2f}"
    text = "+".join(f"{opp if where == 'h' else opp.lower()}({where})" for opp, where in games)
    return f"{text} {xp:.2f}"


def _player_row(cur: dict, pid: int, tag: str) -> str:
    row = cur["players"][pid]
    cells = " | ".join(_cell(row, gw) for gw in cur["events"])
    total = sum(row["x"].values())
    return (
        f"| {tag} | {row['n']} ({row['p']}, {row['c']}) | {price(row['pr'])}"
        f" | {cells} | {total:.2f} | {row['t']:.2f} |"
    )


def _window(cur: dict, pid: int) -> float:
    return round(sum(cur["players"][pid]["x"].values()), 2)


def _name(snapshot: dict, pid: int | None) -> str:
    if pid is None:
        return "none"
    row = snapshot["players"].get(pid)
    return row["n"] if row else f"#{pid}"


def _names(snapshot: dict, pids: list[int]) -> str:
    return ", ".join(_name(snapshot, p) for p in pids) or "none"


def _clean(text: str | None) -> str:
    """Free text made safe for a table cell or a one-line bullet: newlines
    become spaces and a pipe is escaped, since the gaffer's words and the API's
    news are not ours to keep from breaking the table around them."""
    return (text or "").replace("\r", " ").replace("\n", " ").replace("|", "\\|")


def _reasons(cur: dict) -> dict:
    """The gaffer's per-slot words, ``{}`` where the run kept none."""
    return cur.get("reasons") or {}


def _fallback(cur: dict) -> str:
    """What a prose slot says with no words to put in it."""
    source = cur["decision"].get("source") or ""
    return REASON_UNAVAILABLE if source.startswith("manager") else NO_GAFFER
