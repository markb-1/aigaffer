"""Chip what-ifs: what playing a chip at the coming deadline would do.

The owner taps "Wildcard?" or "Free hit?" and gets a number he can act on in a
minute. The number comes from two window solves on the same board, differing in
one binary — the held chip's in week 1:

* **on** — the chip pinned played this gameweek;
* **off** — the chip pinned *not* played this gameweek, free to be played later
  in the window against its bars.

Everything else is shared: the held chips (the other chips stay plannable in
both, against their calendar bars), the calendar's bars, the selling prices,
the free transfers, the week-1 lock from a recorded week, an unpinned opening
count (a pinned-on wildcard week is free and uncapped only in optimize_path's
unpinned branch), and one set of free-hit prices — priced once and handed to
both, as the sweep does, so the two solves cannot see different boards.

**net = on.objective − off.objective** is the only number the band reads. Both
objectives carry their hits and the bars of every chip each path plays, and the
bars price only weeks beyond the window, so a wildcard the off-path plays in GW8
anyway is compared timing against timing, and one it keeps past the window pays
its bar on the on side alone. **gain** is the same difference with the bars
taken back out — what the chip buys now — and **bars_diff** is the difference in
bars paid: gain − net.

The free hit is decided the same way, not on the one-week rebuild, because the
window sees what the one week cannot: the standing squad's planned moves wait a
week, the squad reverts, and no free transfer is added after a chip week. The
message shows it the way managers think — free-hit team against your eleven this
week, plus the knock-on, minus keeping it — and :class:`FreeHitTerms` is that sum.

Nothing here saves or sends anything: this module is arithmetic over two solves,
and the inbox's handler (:mod:`aigaffer.whatif_handler`) is what talks.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from time import monotonic

from aigaffer.chips import FREE_HIT, WILDCARD, HeldChip, held_for, playable_in
from aigaffer.config import Config
from aigaffer.model.xp import projected_events
from aigaffer.orchestrator import PreparedWeek, _held, _week1_lock
from aigaffer.recording import MODE_NAMES, VERDICT_MODES
from aigaffer.solver.multiweek import PlannedPath, _free_hit_prices, optimize_path
from aigaffer.solver.lineup import Lineup, attacking_evs, pick_lineup
from aigaffer.solver.optimizer import BENCH_WEIGHT, HIT_POINTS, Plan
from aigaffer.store import Store

# How far the net must be from nothing before it is called, per chip: the FPL
# expert's estimates of the noise in a six-week wildcard comparison and a
# one-week free-hit one (projection error that persists across weeks, and the
# optimiser's curse on the chip side). First guesses, to be recalibrated from
# the what-if log once twenty or more cases have been scored.
WHATIF_MARGIN = {WILDCARD: 8.0, FREE_HIT: 4.0}
# Seconds each of the two solves may take. CBC's limit is wall-clock, and the
# VM has one vCPU it may be sharing with an hourly tick. The first live
# wildcard what-if stopped both solves on a 60 s limit, so neither was proven
# and the band needed twice the margin (UNSURE_FACTOR): the common case on a
# wildcard board. A what-if is asked rarely and the owner accepted waiting
# minutes for it, so each solve gets two and a half; the hourly sweep keeps its
# own, shorter limits (plans.SWEEP_TIME_LIMIT). The same limit also caps each
# free-hit pricing sub-solve (_free_hit_prices), which proves in seconds in
# practice.
WHATIF_TIME_LIMIT = 150
PLAY, MARGINAL, HOLD = "play", "marginal", "hold"
# A week to hold for further out than this is marked provisional: projections
# five and more gameweeks ahead are the weakest the model makes.
PROVISIONAL_WEEKS = 4
# A solve that stopped on its time limit has a gap of unknown size, so an
# unsure answer is called only when the net clears the margin this many times.
UNSURE_FACTOR = 2


@dataclass(frozen=True)
class PathSide:
    """One of the two solves. ``chips`` are the chips the path plays, by
    gameweek, played weeks only; ``lineup`` is the eleven, armbands and bench
    the path fields in the gameweek being decided — the free-hit team's on a
    free-hit week, the plan's own squad otherwise (:func:`week1_lineup`), read
    by message 1, the log and the gaffer's briefing alike so all three name the
    same captain; ``seconds`` is how long the solve took."""

    plan: Plan
    path: PlannedPath
    chips: dict[int, str]
    lineup: Lineup
    seconds: float


@dataclass(frozen=True)
class HoldWeek:
    """The week a Hold (or a Marginal leaning hold) points at, and why.

    ``source`` says where it came from, and ``value`` means what that source
    can honestly say:

    * ``"off_path"`` — the path without the chip plays it in ``event`` inside
      the window; ``value`` is how much more that is worth than now (−net).
    * ``"calendar"`` — a free hit the calendar saves for ``event`` beyond the
      window; ``value`` is the calendar's (haircut) value there.
    * ``"past_window"`` — nothing names a week: a wildcard (the calendar values
      no wildcard week) or a chip the calendar left unassigned. ``event`` is
      None and ``value`` is the bar the chip faces this week, which is what
      keeping it is priced at — or None when the calendar has no bar for it.

    ``provisional`` marks a week more than :data:`PROVISIONAL_WEEKS` out.
    """

    event: int | None
    value: float | None
    source: str
    provisional: bool


@dataclass(frozen=True)
class FreeHitTerms:
    """A free hit's net, as a manager reads it: ``one_week`` (the free-hit team
    against the off-path's eleven this week, bench share stripped), plus
    ``knock_on`` (everything after this week), minus ``keeping`` (the bars)."""

    one_week: float
    knock_on: float
    keeping: float


@dataclass(frozen=True)
class WhatIf:
    """The answer, and everything the message and the log need to say it."""

    kind: str
    event: int
    window: list[int]
    on: PathSide
    off: PathSide
    net: float
    gain: float
    bars_diff: float
    weekly_gain: dict[int, float]
    band: str
    lean: str | None
    unsure: bool
    margin: float
    hold_week: HoldWeek | None
    fh_terms: FreeHitTerms | None
    refunded_hits: int
    minutes_source: str | None


def band_for(kind: str, net: float, unsure: bool) -> tuple[str, str | None]:
    """Play, Marginal or Hold for ``net`` against ``kind``'s margin, and the
    lean when Marginal.

    The edges belong to the call: a net exactly at +margin is a Play and at
    −margin a Hold. Inside, the lean is the net's sign, and a net of nothing
    leans hold — a chip cannot be taken back, so a coin-flip is a reason to
    wait. ``unsure`` (either solve stopped on time) doubles the margin.
    """
    margin = WHATIF_MARGIN[kind] * (UNSURE_FACTOR if unsure else 1)
    if net >= margin:
        return PLAY, None
    if net <= -margin:
        return HOLD, None
    return MARGINAL, (PLAY if net > 0 else HOLD)


def verdict_minutes(store: Store, gw: int) -> tuple[dict[int, float], str | None, dict | None]:
    """The minutes the latest report for ``gw`` was decided on, its name, and
    the record itself.

    The newest early/scout/deadline record supplies its *applied* adjustments —
    ``adjustments``, the ones a re-solve was actually run on — and never
    ``unapplied``, which he only wrote down. They are for the coming gameweek
    only, as in the run. The name is the record's day (UTC) and its mode,
    "Wednesday's scout", so message 1 can say whose minutes it inherited. No
    record: the model's own minutes, and nothing to name.
    """
    verdict = store.latest_verdict(gw, VERDICT_MODES)
    if verdict is None:
        return {}, None, None
    overrides = {
        int(entry["player_id"]): float(entry["expected_minutes"])
        for entry in verdict.decision.get("adjustments") or []
    }
    day = datetime.fromisoformat(verdict.ts).astimezone(UTC).strftime("%A")
    label = f"{day}'s {MODE_NAMES.get(verdict.mode, verdict.mode)}"
    return overrides, label, verdict.decision


def simulate(
    week: PreparedWeek,
    cfg: Config,
    kind: str,
    *,
    minutes_source: str | None = None,
    time_limit: int = WHATIF_TIME_LIMIT,
) -> WhatIf | None:
    """Solve the window with ``kind`` pinned on and off in week 1, and read it.

    None means no answer, not an error: no squad, the chip not held this
    gameweek (the inbox's gates catch both first), a board that cannot field a
    free-hit squad, or either solve returning nothing in ``time_limit``.

    A wildcard played after moves entered this week folds them in and refunds
    their hits, and the week's free transfers go back to the pre-move count:
    so the on side solves from ``executed.ft_before`` and the net gains
    :data:`HIT_POINTS` for every hit refunded. The recorded moves still bind
    both sides through the week-1 lock — a real wildcard could undo them, so
    this is conservative, and accepted.
    """
    effective = week.effective
    if effective.squad is None:
        return None
    events = projected_events(week.projections)
    held = _held(cfg, effective)
    chip = held_for(held, kind, events[0])
    if chip is None:
        return None
    bars = week.calendar.bars if week.calendar is not None else None
    lock = _week1_lock(effective)

    refunded, ft_on = 0, effective.free_transfers
    executed = week.executed
    if kind == WILDCARD and executed is not None and executed.transfers_in:
        refunded = max(0, len(executed.transfers_in) - executed.ft_before)
        ft_on = executed.ft_before

    freehit_prices = None
    if any(held_chip.chip == FREE_HIT for held_chip in playable_in(held, events)):
        freehit_prices = _free_hit_prices(
            effective.players, week.projections, effective.squad.player_ids,
            effective.squad.bank, events, time_limit,
            selling_prices=week.prices, held_chips=held,
        )
        if freehit_prices is None:
            return None

    def side(pinned: bool, free_transfers: int) -> PathSide | None:
        started = monotonic()
        answer = optimize_path(
            effective.players,
            week.projections,
            effective.squad.player_ids,
            effective.squad.bank,
            free_transfers,
            events,
            cfg.decay,
            forced_first_transfers=None,
            time_limit=time_limit,
            held_chips=held,
            freehit_prices=freehit_prices,
            selling_prices=week.prices,
            bars=bars,
            lock=lock,
            pin_chip=(kind, pinned),
        )
        if answer is None:
            return None
        plan, path = answer
        return PathSide(
            plan,
            path,
            _chips_played(path, events[0]),
            week1_lineup(plan, path, week, events[0]),
            monotonic() - started,
        )

    on = side(True, ft_on)
    off = side(False, effective.free_transfers) if on is not None else None
    if on is None or off is None:
        return None

    net = on.path.objective - off.path.objective + HIT_POINTS * refunded
    bars_diff = on.path.bars_paid - off.path.bars_paid
    gain = net + bars_diff
    weekly_gain = {
        event: _week_xp(on, event, events[0], week) - _week_xp(off, event, events[0], week)
        for event in events
    }
    unsure = not (on.path.proven and off.path.proven)
    band, lean = band_for(kind, net, unsure)
    hold_week = (
        _hold_week(kind, chip, net, off, events, week)
        if HOLD in (band, lean)
        else None
    )
    fh_terms = None
    if kind == FREE_HIT:
        one_week = weekly_gain[events[0]]
        fh_terms = FreeHitTerms(one_week=one_week, knock_on=gain - one_week, keeping=bars_diff)
    return WhatIf(
        kind=kind,
        event=events[0],
        window=list(events),
        on=on,
        off=off,
        net=net,
        gain=gain,
        bars_diff=bars_diff,
        weekly_gain=weekly_gain,
        band=band,
        lean=lean,
        unsure=unsure,
        margin=WHATIF_MARGIN[kind],
        hold_week=hold_week,
        fh_terms=fh_terms,
        refunded_hits=refunded,
        minutes_source=minutes_source,
    )


def _chips_played(path: PlannedPath, first: int) -> dict[int, str]:
    """The chips a path plays, by gameweek: week 1's off the path itself, the
    later weeks' off their moves."""
    played = {first: path.week1_chip} if path.week1_chip != "none" else {}
    played.update({move.event: move.chip for move in path.moves if move.chip != "none"})
    return played


def _week_xp(side: PathSide, event: int, first: int, week: PreparedWeek) -> float:
    """``side``'s points in ``event``, like for like with the other side.

    A free hit's week-1 figure is its one-week squad's score, which carries the
    bench at BENCH_WEIGHT; every other week's figure is the eleven and the
    armband alone. So a free hit played this week has its bench share taken
    back out, and the two sides' week-1 numbers compare the same thing. A free
    hit the off side plays later keeps its share — the fifteen is not on the
    path past week 1, and the asymmetry is a fraction of a point.
    """
    points = side.path.weekly_xp.get(event, 0.0)
    if event != first or side.path.week1_chip != FREE_HIT or side.path.week1_freehit_squad is None:
        return points
    bench = set(side.path.week1_freehit_squad) - set(side.path.week1_freehit_xi or [])
    return points - BENCH_WEIGHT * sum(
        week.projections[pid].per_gw.get(event, 0.0)
        for pid in bench
        if pid in week.projections
    )


def _hold_week(
    kind: str, chip: HeldChip, net: float, off: PathSide, events: list[int], week: PreparedWeek
) -> HoldWeek:
    """The week to hold for: the off-path's own week for the chip if it plays it
    in the window, else the calendar's saved-for week, else none at all."""
    later = sorted(event for event, played in off.chips.items() if played == kind)
    if later:
        return _held_until(later[0], -net, "off_path", events)
    calendar = week.calendar
    entry = None
    if calendar is not None:
        entry = next((e for e in calendar.entries if e.held.id == chip.id), None)
    if kind == FREE_HIT and entry is not None and entry.saved_for is not None:
        return _held_until(entry.saved_for, entry.value, "calendar", events)
    bar = None
    if calendar is not None:
        bar = calendar.bars.get(chip.id, {}).get(events[0])
    return HoldWeek(event=None, value=bar, source="past_window", provisional=False)


def _held_until(event: int, value: float | None, source: str, events: list[int]) -> HoldWeek:
    return HoldWeek(
        event=event,
        value=value,
        source=source,
        provisional=event - events[0] > PROVISIONAL_WEEKS,
    )


# A gap between two consecutive deadlines longer than this is an
# international break. Ten days: an ordinary week is seven, a midweek round
# fewer, and a break fourteen or more.
BREAK_DAYS = 10


def week1_lineup(plan: Plan, path: PlannedPath, week: PreparedWeek, event: int) -> Lineup:
    """The eleven, armbands and bench a path fields at ``event`` — the
    free-hit fifteen's on a free-hit week (the standing squad reverts and is
    not what he takes to the deadline), the plan's own squad otherwise. Picked
    the way :func:`~aigaffer.orchestrator.solve` picks the report's: on the
    gameweek's xP, the armband on the attacking ceiling."""
    squad = path.week1_freehit_squad if path.week1_chip == FREE_HIT and path.week1_freehit_squad else plan.squad
    players = week.effective.players
    positions = {pid: players[pid].element_type for pid in squad}
    gw_xp = {pid: week.projections[pid].per_gw.get(event, 0.0) for pid in squad if pid in week.projections}
    return pick_lineup(list(squad), positions, gw_xp, attacking_evs(week.projections, event))


def next_break(events, event: int) -> tuple[int, int] | None:
    """The first pair of consecutive gameweeks from ``event`` on whose
    deadlines are more than :data:`BREAK_DAYS` apart — ``(before, after)`` —
    or None. ``events`` is the bootstrap's list."""
    ordered = sorted((e for e in events if e.id >= event), key=lambda e: e.id)
    for before, after in zip(ordered, ordered[1:]):
        if after.deadline_time - before.deadline_time > timedelta(days=BREAK_DAYS):
            return before.id, after.id
    return None
