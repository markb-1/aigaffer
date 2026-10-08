"""The chip calendar: what each chip is being saved for, and what that costs.

The window solver sees six gameweeks. A chip it may play is worth playing
inside them only if no week beyond them is clearly better — and before the
calendar it judged that against a flat bar, which on a board where every week
looks alike meant never, and every first-set chip lapsing at GW19. So this
module looks past the window to the chip's expiry: it values each held chip in
every eligible week beyond the window, gives each one the week it is "saved
for" (one chip a week), and hands the window a bar per week — the saved-for
week's value, discounted for distance. The window plays a chip in week w only
where its gain there beats that bar.

Four consequences worth stating, so nobody files them as bugs:

* With nothing left beyond the window the bar is 0: the chip is played inside
  it. Every first-set bar is 0 once the window reaches GW19 — and with the
  window's own decay of 0.85 the solver then spends them early in that window
  (GW14–17 rather than GW19). That is the owner's "lean towards now".
* The bar for week w is v*·ρ^(w*−w), so between two window weeks it shrinks by
  (δ/ρ)^Δw against the window's decay δ: the in-window pick tilts slightly
  later than pure decay would. Accepted — it offsets δ's early tilt.
* The values are proxies, haircut by :data:`PROXY_SCALE`: they price a bench
  bought for the week, a captain bought in, and a free hit against today's
  squad, while the window's gain is what the reachable squad earns.
* On a flat board the triple captain and the bench boost may go as early as
  GW6. With PROXY_SCALE ≤ 0.85 and ρ^9 ≈ 0.76, a chip whose in-window gain
  equals its proxy faces a bar of about 0.65 × that gain, so the window plays
  it. That is the owner's "lean towards now", taken as the principal's
  decision. The free hit is held by its option floor until a week clearly
  beats it or expiry nears — the floor tapers over the chip's last five weeks
  to nothing at its expiry, so a first-set free hit still in hand in late
  December is readier to be played than one in October; the wildcard is held
  by its ramp.

The calendar is computed once per run from base-minute projections — never the
gaffer's minutes, which are for the coming gameweek — so his re-solves cannot
move it.
"""

from dataclasses import dataclass, field

from aigaffer.chips import BENCH_BOOST, CHIP_ORDER, FREE_HIT, TRIPLE_CAPTAIN, WILDCARD, HeldChip
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.multiweek import (
    FALLBACK_BARS,
    _projected,
    best_one_week_squads,
    squad_one_week,
)
from aigaffer.solver.optimizer import AVAILABLE, BENCH_WEIGHT

# How much less a week further out is worth: gentle, because a chip held is
# not a chip lost while its window is open. First guess; one place.
CHIP_DISCOUNT = 0.97
# What a free hit is worth before any week is priced: insurance against a
# blank or mass absences nobody can see yet. Added to its bar while a later
# week remains, so a flat board does not fire it in GW6 — and tapered over the
# chip's last FLOOR_TAPER_WEEKS weeks to 0 at expiry (see _floor), because by
# late December the alternative to playing a first-set free hit is losing it.
OPTION_FLOOR = {FREE_HIT: 10.0}
# Over how many weeks before its last the option floor falls to nothing: flat
# to GW14 and 0 at GW19 for the first set, GW33 to GW38 for the second.
FLOOR_TAPER_WEEKS = 5
PROXY_SCALE = {BENCH_BOOST: 0.7, TRIPLE_CAPTAIN: 0.85, FREE_HIT: 0.85}
# A triple captain is only worth earmarking on a player who plays the match.
TC_MIN_MINUTES = 80.0
WILDCARD_BAR = 35.0
WILDCARD_RAMP_WEEKS = 8
_VALUED = (BENCH_BOOST, TRIPLE_CAPTAIN, FREE_HIT)


@dataclass(frozen=True)
class CalendarEntry:
    held: HeldChip
    saved_for: int | None
    value: float | None
    bars: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ChipCalendar:
    entries: tuple[CalendarEntry, ...]
    discount: float
    proxy_scale: dict[str, float]
    horizon_end: int
    fell_back: bool = False

    @property
    def bars(self) -> dict[str, dict[int, float]]:
        return {entry.held.id: entry.bars for entry in self.entries if entry.bars}

    def record(self) -> dict:
        """The calendar as the decision record keeps it: JSON, ids and numbers."""
        return {
            "fell_back": self.fell_back,
            "discount": self.discount,
            "proxy_scale": dict(self.proxy_scale),
            "horizon_end": self.horizon_end,
            "entries": [
                {
                    "chip": entry.held.id,
                    "saved_for": entry.saved_for,
                    "value": entry.value,
                    "bars": {str(gw): bar for gw, bar in entry.bars.items()},
                }
                for entry in self.entries
            ],
        }


def calendar_chips(held: tuple[HeldChip, ...], window: list[int]) -> tuple[HeldChip, ...]:
    """The held chips the calendar plans: those the window can already reach or
    has passed the start of — the current set, plus the next once the window
    touches GW20."""
    return tuple(chip for chip in held if chip.start_event <= window[-1])


def horizon_end(held: tuple[HeldChip, ...], window: list[int]) -> int:
    """How far the calendar must project: the furthest expiry it plans for, and
    never short of the window itself."""
    return max([window[-1], *(chip.stop_event for chip in calendar_chips(held, window))])


def _beyond(chip: HeldChip, window: list[int]) -> list[int]:
    """The gameweeks after the window that ``chip`` may still be played in."""
    return list(range(max(window[-1] + 1, chip.start_event), chip.stop_event + 1))


def _kind(chip_id: str) -> str:
    """The kind a held chip's :attr:`~aigaffer.chips.HeldChip.id` names — the
    part before its ``@stop_event``."""
    return chip_id.partition("@")[0]


def assign(values: dict[str, dict[int, float]], anchors: dict[str, int]) -> dict[str, int]:
    """Each chip's saved-for week: the assignment of chips to distinct weeks
    that maximises Σ v·ρ^(w − anchor), a chip free to go unassigned.

    Brute force, because it is small: at most three valued chips a half over at
    most eighteen weeks is 18·17·16 ≈ 5k candidates. Candidates are tried
    chips in :data:`CHIP_ORDER`, weeks ascending, unassigned last, and only a
    strictly better total replaces the best so far — so ties go to the
    earlier week, then the earlier chip.
    """
    chips = sorted(values, key=lambda key: CHIP_ORDER.index(_kind(key)))
    options = {
        chip: [w for w in sorted(values[chip]) if values[chip][w] > 0] + [None]
        for chip in chips
    }
    best_total, best = 0.0, {}

    def walk(i: int, used: set[int], picks: dict[str, int], total: float) -> None:
        nonlocal best_total, best
        if i == len(chips):
            if total > best_total + 1e-9:
                best_total, best = total, dict(picks)
            return
        chip = chips[i]
        for week in options[chip]:
            if week is None:
                walk(i + 1, used, picks, total)
            elif week not in used:
                gain = values[chip][week] * CHIP_DISCOUNT ** (week - anchors[chip])
                walk(i + 1, used | {week}, {**picks, chip: week}, total + gain)

    walk(0, set(), {}, 0.0)
    return best


def _floor(chip: HeldChip, window: list[int], w: int) -> float:
    """The option floor ``chip`` carries in window week ``w``: the full floor
    while its expiry is :data:`FLOOR_TAPER_WEEKS` or more weeks off, falling a
    step a week to nothing at its last week — and nothing at all once no
    eligible week lies beyond the window, as before the taper."""
    if not _beyond(chip, window):
        return 0.0
    full = OPTION_FLOOR.get(chip.chip, 0.0)
    return full * min(1.0, (chip.stop_event - w) / FLOOR_TAPER_WEEKS)


def _bars_for(chip: HeldChip, window: list[int], saved_for: int | None, value: float | None) -> dict[int, float]:
    """The bar ``chip`` must clear in each window week inside its window."""
    weeks = [w for w in window if chip.allows(w)]
    if chip.chip == WILDCARD:
        left = len(_beyond(chip, window))
        bar = WILDCARD_BAR * min(1.0, left / WILDCARD_RAMP_WEEKS)
        return {w: bar for w in weeks}
    if saved_for is None or value is None:
        return {w: _floor(chip, window, w) for w in weeks}
    return {
        w: value * CHIP_DISCOUNT ** (saved_for - w) + _floor(chip, window, w)
        for w in weeks
    }


def _values(
    chip: str,
    weeks: list[int],
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    xmins: dict[int, float],
    current_squad: list[int],
    bank: int,
    selling_prices: dict[int, int] | None,
) -> dict[int, float]:
    """What ``chip`` would add in each of ``weeks``, haircut. A week that could
    not be priced is left out."""
    scale = PROXY_SCALE[chip]
    if chip == TRIPLE_CAPTAIN:
        starters = [
            pid for pid, minutes in xmins.items()
            if minutes >= TC_MIN_MINUTES and pid in projections
            and pid in players and players[pid].status == AVAILABLE
        ]
        values = {}
        for w in weeks:
            if not starters:
                break
            pick = max(starters, key=lambda pid: (projections[pid].attacking_per_gw.get(w, 0.0), -pid))
            values[w] = scale * _projected(projections, pick, w)
        return values
    if chip == BENCH_BOOST:
        priced = best_one_week_squads(
            players, projections, current_squad, bank, weeks,
            bench_weight=1.0, selling_prices=selling_prices,
        )
        # At bench weight 1 the solve's own eleven is arbitrary — every squad
        # point counts the same — so the fifteen it bought is lined up again
        # on the window's basis, and the bench is what that leaves. 0.9 is the
        # window's own bench-boost gain: the bench it was already scoring at
        # BENCH_WEIGHT is not new.
        values = {}
        for w, answer in priced.items():
            if answer is None:
                continue
            fifteen = answer[1]
            _, _, xi = squad_one_week(players, projections, fifteen, w)
            if not xi:  # no eleven, so no bench to tell from the rest: not priced
                continue
            bench = set(fifteen) - set(xi)
            values[w] = scale * (1 - BENCH_WEIGHT) * sum(
                _projected(projections, p, w) for p in bench
            )
        return values
    priced = best_one_week_squads(
        players, projections, current_squad, bank, weeks, selling_prices=selling_prices,
    )
    values = {}
    for w, answer in priced.items():
        if answer is None:
            continue
        held_score, _, xi = squad_one_week(players, projections, current_squad, w)
        if not xi:  # the squad held lines up no eleven, so its 0.0 is no baseline: not priced
            continue
        values[w] = scale * (answer[0] - held_score)
    return values


def build_calendar(
    held: tuple[HeldChip, ...],
    window: list[int],
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    xmins: dict[int, float],
    current_squad: list[int],
    bank: int,
    selling_prices: dict[int, int] | None = None,
) -> ChipCalendar:
    """The calendar for the chips the window can reach. ``projections`` must run
    to :func:`horizon_end` on base minutes; ``xmins`` are those base minutes."""
    planned = calendar_chips(held, window)
    values = {
        chip.id: _values(chip.chip, _beyond(chip, window), players, projections, xmins,
                         current_squad, bank, selling_prices)
        for chip in planned if chip.chip in _VALUED
    }
    # Each half is assigned on its own — its chips compete for its weeks, never
    # the other half's — and anchored where it may first be played in the
    # window, so a second set seen from GW15 discounts from GW20.
    saved: dict[str, int] = {}
    for stop in sorted({chip.stop_event for chip in planned}):
        half = [chip for chip in planned if chip.stop_event == stop and chip.id in values]
        anchors = {chip.id: max(window[0], chip.start_event) for chip in half}
        saved.update(assign({chip.id: values[chip.id] for chip in half}, anchors))
    # The haircut value at each chip's saved-for week; absent when unassigned.
    worth = {key: values[key][week] for key, week in saved.items()}
    entries = tuple(
        CalendarEntry(
            held=chip,
            saved_for=saved.get(chip.id),
            value=worth.get(chip.id),
            bars=_bars_for(chip, window, saved.get(chip.id), worth.get(chip.id)),
        )
        for chip in planned
    )
    return ChipCalendar(entries, CHIP_DISCOUNT, dict(PROXY_SCALE), horizon_end(held, window))


def fallback_calendar(held: tuple[HeldChip, ...], window: list[int]) -> ChipCalendar:
    """The calendar a failed run falls back on: the old flat bars, except that a
    chip with no week left beyond the window is played inside it (bar 0) — so
    a failure never costs a chip its expiry."""
    entries = []
    for chip in calendar_chips(held, window):
        weeks = [w for w in window if chip.allows(w)]
        bar = FALLBACK_BARS[chip.chip] if _beyond(chip, window) else 0.0
        entries.append(CalendarEntry(chip, None, None, {w: bar for w in weeks}))
    return ChipCalendar(tuple(entries), CHIP_DISCOUNT, dict(PROXY_SCALE), horizon_end(held, window), fell_back=True)
