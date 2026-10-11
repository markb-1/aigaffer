"""The gameweek review — "Looking back" — written once a week is final.

When FPL has finished a gameweek and checked its data, the bonus and the
automatic substitutions are confirmed and the points stop moving. That is
when the review runs: it reads what actually happened from the public API —
the score against the game's average and best, the rank's move, each
starter, the armband, the bench, the transfers net of hits, the chip — and
sets it against the verdict the week was played on. The gaffer then writes a
short account (what went well, what went wrong, lessons), which is appended
to the week's ``GW{n}.md`` under a marker and never touches anything above
it, and its lessons reach the next gameweek's briefings.

This half is the facts, and it is deterministic: no model, no clock. The
same :class:`WeekFacts` renders both the text the gaffer reads and the lines
the report prints, so the two cannot disagree.
"""

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from aigaffer.chips import CHIPS_BY_API_NAME, FREE_HIT
from aigaffer.data.models import Bootstrap, EntryPick, EntryWeek, HistoryRow, LiveStats
from aigaffer.executed import NO_CHIP, Verdict
from aigaffer.report.render import POSITIONS

REVIEW_MODE = "review"
# Failed attempts at one week's review before the schedule stops asking.
REVIEW_ATTEMPTS = 6
# A minutes call that missed by more than this is worth the gaffer's look.
MINUTES_MISS = 30
# The verdict's own words, as much of them as the writer is shown.
MAX_RATIONALE = 1500
# The manager's decision_source when he decided; anything else is the solver.
# Repeated rather than imported: the manager package is not this module's.
GAFFER_SOURCE = "manager"


class ReviewUnavailable(RuntimeError):
    """This tick cannot review the week, and the reason says why."""


@dataclass(frozen=True)
class Starter:
    player_id: int
    name: str
    club: str
    position: str
    minutes: int
    points: int
    multiplier: int


@dataclass(frozen=True)
class Scored:
    """A name and the points he scored this week."""

    name: str
    points: int


@dataclass(frozen=True)
class MinuteCall:
    """A minutes call the verdict made, against what he played."""

    name: str
    called: float
    played: int


@dataclass(frozen=True)
class Followed:
    """Whether the team fielded was the verdict's: the fifteen and the armband.

    Measured on the fifteen, never on the transfer lists. After "Transfers
    made" a verdict's moves are relative to the recorded squad, so a week the
    owner entered exactly can read as a hold on record while the API lists
    his transfers.
    """

    squad: bool
    captain: bool
    not_fielded: tuple[str, ...]
    fielded_instead: tuple[str, ...]
    captain_wanted: str | None


@dataclass(frozen=True)
class WeekFacts:
    gw: int
    points: int
    hits: int
    bench_points: int
    average: int
    highest: int | None
    rank_before: int | None
    rank_after: int | None
    starters: tuple[Starter, ...]
    captain: Starter | None
    best_in_xi: Starter | None
    auto_subs: tuple[tuple[Scored, Scored], ...]
    transfers_in: tuple[Scored, ...]
    transfers_out: tuple[Scored, ...]
    chip: str
    free_hit: bool
    verdict_mode: str | None
    gaffer_decided: bool | None
    followed: Followed | None
    minute_calls: tuple[MinuteCall, ...] | None
    rationale: str
    chip_justification: str

    @property
    def net_points(self) -> int:
        return self.points - self.hits

    @property
    def captain_points(self) -> int:
        return 0 if self.captain is None else self.captain.points * self.captain.multiplier

    @property
    def transfers_net(self) -> int:
        gained = sum(s.points for s in self.transfers_in)
        lost = sum(s.points for s in self.transfers_out)
        return gained - lost - self.hits

    def to_json(self) -> dict:
        return asdict(self)


def review_target(bootstrap: Bootstrap) -> int | None:
    """The latest gameweek whose points are final, or None for none yet."""
    final = [e.id for e in bootstrap.events if e.finished and e.data_checked]
    return max(final, default=None)


def build_facts(
    gw: int,
    bootstrap: Bootstrap,
    entry: EntryWeek,
    history: list[HistoryRow],
    live: dict[int, LiveStats],
    transfers: list[dict],
    verdict: Verdict | None,
) -> WeekFacts:
    """Everything the review says about ``gw``, from what the API served.

    ``transfers`` is the season's list (:meth:`FplClient.transfers`); only
    ``gw``'s rows count. A player the API no longer lists — sold abroad
    mid-season — is named by id and scores what ``live`` says, or nothing.
    """
    players = {p.id: p for p in bootstrap.elements}
    clubs = {t.id: t.short_name for t in bootstrap.teams}
    event = next(e for e in bootstrap.events if e.id == gw)

    def name(pid: int) -> str:
        player = players.get(pid)
        return player.web_name if player is not None else f"id {pid}"

    def points(pid: int) -> int:
        stats = live.get(pid)
        return 0 if stats is None else stats.total_points

    def starter(p: EntryPick) -> Starter:
        player = players.get(p.element)
        stats = live.get(p.element) or LiveStats()
        return Starter(
            player_id=p.element,
            name=name(p.element),
            club=clubs.get(player.team, "?") if player is not None else "?",
            position=POSITIONS.get(player.element_type, "?") if player is not None else "?",
            minutes=stats.minutes,
            points=stats.total_points,
            multiplier=p.multiplier,
        )

    ordered = sorted(entry.picks, key=lambda p: p.position)
    # The endpoint has applied the substitutions already: a multiplier above
    # zero is a player who counted. automatic_subs is only the story of it.
    starters = tuple(starter(p) for p in ordered if p.multiplier > 0)
    armband = next((p for p in ordered if p.multiplier >= 2), None)
    if armband is None:
        armband = next((p for p in ordered if p.is_captain), None)
    captain = None if armband is None else starter(armband)
    if captain is not None and armband.multiplier < 2:
        captain = Starter(**{**asdict(captain), "multiplier": 0})
    best = max(starters, key=lambda s: s.points, default=None)

    ranks = {row.event: row.overall_rank for row in history}
    week_moves = [row for row in transfers if row.get("event") == gw]
    chip = CHIPS_BY_API_NAME.get(entry.active_chip or "", NO_CHIP)
    fielded = {p.element for p in entry.picks}
    # The armband he chose, not the one that counted: a captain who did not
    # play passes it to the vice, and that is not a different choice.
    chosen = next((p.element for p in ordered if p.is_captain), None)

    record = verdict.decision if verdict is not None else None
    return WeekFacts(
        gw=gw,
        points=entry.points,
        hits=entry.event_transfers_cost,
        bench_points=entry.points_on_bench,
        average=event.average_entry_score,
        highest=event.highest_score,
        rank_before=ranks.get(gw - 1),
        rank_after=ranks.get(gw, entry.overall_rank),
        starters=starters,
        captain=captain,
        best_in_xi=best,
        auto_subs=tuple(
            (Scored(name(i), points(i)), Scored(name(o), points(o)))
            for i, o in entry.automatic_subs
        ),
        transfers_in=tuple(Scored(name(r["element_in"]), points(r["element_in"])) for r in week_moves),
        transfers_out=tuple(Scored(name(r["element_out"]), points(r["element_out"])) for r in week_moves),
        chip=chip,
        free_hit=chip == FREE_HIT,
        verdict_mode=None if verdict is None else verdict.mode,
        gaffer_decided=None if record is None else record.get("decision_source") == GAFFER_SOURCE,
        followed=None if record is None else _followed(record, fielded, chosen, name),
        minute_calls=None if record is None else _minute_calls(record, live, name),
        rationale=_cut(record.get("rationale")) if record else "",
        chip_justification=_cut(record.get("chip_justification")) if record else "",
    )


def _followed(record: dict, fielded: set[int], chosen: int | None, name) -> Followed | None:
    """The fifteen the verdict meant against the fifteen fielded — the rule
    :func:`aigaffer.orchestrator._recorded_lineup` uses — or None for a record
    that cannot say (written before its keys existed)."""
    try:
        if record.get("chip") == FREE_HIT and record.get("freehit_squad"):
            meant = set(record["freehit_squad"])
        else:
            # No fifteen solved from (a draft, or a record older than the
            # key): nothing to compare the fielded team with.
            if record.get("squad_before") is None:
                return None
            meant = (
                set(record["squad_before"])
                - set(record["transfers_out"])
                | set(record["transfers_in"])
            )
    except (KeyError, TypeError):
        return None
    wanted = record.get("captain")
    return Followed(
        squad=meant == fielded,
        captain=wanted == chosen,
        not_fielded=tuple(name(pid) for pid in sorted(meant - fielded)),
        fielded_instead=tuple(name(pid) for pid in sorted(fielded - meant)),
        captain_wanted=None if wanted is None else name(wanted),
    )


def _minute_calls(record: dict, live: dict[int, LiveStats], name) -> tuple[MinuteCall, ...] | None:
    """The verdict's minutes calls that missed by more than :data:`MINUTES_MISS`.

    The record keeps superseded calls as well as the last; the last per
    player is the one the week was played on.
    """
    adjustments = record.get("adjustments")
    if not isinstance(adjustments, list):
        return None
    last: dict[int, float] = {}
    for item in adjustments:
        try:
            last[int(item["player_id"])] = float(item["expected_minutes"])
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(
        MinuteCall(name(pid), called, live[pid].minutes)
        for pid, called in last.items()
        if pid in live and abs(called - live[pid].minutes) > MINUTES_MISS
    )


def _cut(text: object) -> str:
    """The verdict's own words, flattened and capped at MAX_RATIONALE."""
    flat = " ".join(text.split()) if isinstance(text, str) else ""
    return flat if len(flat) <= MAX_RATIONALE else flat[: MAX_RATIONALE - 1] + "…"


def _listing(items: Iterable[Scored]) -> str:
    return ", ".join(f"{s.name} ({s.points})" for s in items)


def review_facts(facts: WeekFacts) -> str:
    """The facts as plain text, for the writer to read."""
    hits = f" ({facts.net_points} after {facts.hits} in hits)" if facts.hits else ""
    highest = "" if facts.highest is None else f"; highest {facts.highest}"
    lines = [f"GW{facts.gw} final: {facts.points} points{hits}; average {facts.average}{highest}."]
    if facts.rank_after is not None:
        before = "" if facts.rank_before is None else f"{facts.rank_before:,} -> "
        lines.append(f"Overall rank: {before}{facts.rank_after:,}.")
    lines.append(f"Chip: {facts.chip}.")
    lines.append("Starters (points, minutes):")
    for s in facts.starters:
        armband = f", captain ×{s.multiplier}" if s.multiplier >= 2 else ""
        lines.append(f"- {s.name} ({s.club}, {s.position}): {s.points} pts, {s.minutes} mins{armband}")
    if facts.captain is not None:
        if facts.captain.multiplier == 0:
            lines.append(f"Captain: {facts.captain.name} did not play, and neither did the vice: 0.")
        else:
            lines.append(
                f"Captain: {facts.captain.name} {facts.captain.points} ×{facts.captain.multiplier}"
                f" = {facts.captain_points}."
            )
    if facts.best_in_xi is not None:
        lines.append(f"Best in the XI: {facts.best_in_xi.name} {facts.best_in_xi.points}.")
    subs = "; ".join(f"{i.name} ({i.points}) on for {o.name} ({o.points})" for i, o in facts.auto_subs)
    lines.append(f"Automatic substitutions: {subs or 'none'}.")
    lines.append(f"Bench points: {facts.bench_points}.")
    if facts.transfers_in or facts.transfers_out:
        temporary = " (free hit: the moves were for this week only)" if facts.free_hit else ""
        lines.append(
            f"Transfers{temporary}: in {_listing(facts.transfers_in)}; out {_listing(facts.transfers_out)};"
            f" net {facts.transfers_net:+d} after hits."
        )
    else:
        lines.append("Transfers: none.")
    lines.append(_followed_line(facts.followed))
    if facts.minute_calls is not None:
        misses = "; ".join(f"{c.name} called {c.called:g}, played {c.played}" for c in facts.minute_calls)
        lines.append(f"Minutes calls that missed by more than {MINUTES_MISS}: {misses or 'none'}.")
    return "\n".join(lines)


def _followed_line(followed: Followed | None) -> str:
    if followed is None:
        return "Verdict followed: no verdict on record to compare."
    if followed.squad and followed.captain:
        return "Verdict followed: yes."
    parts = []
    if not followed.squad:
        parts.append(
            f"the verdict's fifteen had {', '.join(followed.not_fielded) or 'nobody else'};"
            f" the team fielded {', '.join(followed.fielded_instead) or 'nobody else'} instead"
        )
    if not followed.captain:
        parts.append(f"the verdict's captain was {followed.captain_wanted}")
    return "Verdict followed: no — " + "; ".join(parts) + "."
