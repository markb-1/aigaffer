"""Tests for the chip calendar.

The pure tests — :func:`assign`, the bars, the fallback, the horizon — need no
solver. The value tests run on one hand-built board, :func:`board_beyond_window`,
whose every number is worked out with a pencil below so a failure says which
arithmetic the calendar disagrees with.

**The board.** A window of GW6–7, three chips held to GW10 (triple captain,
bench boost and free hit), so the weeks beyond the window are GW8, 9 and 10.
The squad is the multi-week tests' spine — fifteen at 50, each on a club of
his own, the bank empty, so the budget is exactly 750 and only a 50 can come
in for a 50 — and its projections are the same every week:

==  ===  =====  ======================================================
id  pos     xP  role
==  ===  =====  ======================================================
1   GK     5.0  starts
2   GK     0.5  bench
3   DEF    4.4  starts
4   DEF    4.3  starts
5   DEF    4.2  starts
6   DEF    0.5  bench
7   DEF    0.5  bench
8   MID    5.6  starts
9   MID    5.7  starts
10  MID    5.8  starts
11  MID    5.9  starts
12  MID    6.0  starts — captain
13  FWD    4.1  starts
14  FWD    4.0  starts
15  FWD    3.9  bench
==  ===  =====  ======================================================

Outside it, at 50 and so buyable one-for-one:

* 16, 17 — defenders at 3.0 every week: a bench worth having.
* 18 — a midfielder worth nothing until GW10, then 15.0: the free hit's week.

Outside it at 200, which no fifteen can afford (14 × 50 + 200 = 900 > 750), so
they matter only to the triple captain, which prices a captain bought in:

* 20 — a striker, 85 minutes: 8.0 points (4.0 attacking) a week, and 10.0
  (6.0 attacking) in GW9. The armband's man.
* 21 — a striker, 70 minutes, more of everything (12.0 / 8.0, and 14.0 / 9.0
  in GW9): not trusted to play the match.
* 22 — a midfielder, 90 minutes, more points than 20 in GW9 (12.0) but less
  attacking EV (3.0): the armband is chosen on attacking EV.
* 23 — a striker, injured, the best of all (15.0 / 10.0): not available.

Everyone else plays 90 minutes and carries no attacking EV.

**The values.** ρ = 0.97, every anchor GW6.

*Triple captain*: 20 every week, at 0.85 × his points — GW8 6.8, GW9 8.5,
GW10 6.8.

*Bench boost*: at bench weight 1 the best fifteen sells 6 and 7 (0.5) for 16
and 17 (3.0) in every week, and 8 (5.6) for 18 in GW10 alone. Lined up on the
window's basis that fifteen benches 2 (0.5, the spare keeper), 16 and 17 (3.0)
and 15 (3.9) — the four worst legal substitutes, 10.4 — in every week, GW10
included (18 starts). So 0.7 × 0.9 × 10.4 = 6.552 in GW8, 9 and 10.

*Free hit*: the squad scores 55.0 + 6.0 + 0.1 × 5.4 = 61.54 a week. In GW8
and 9 the best fifteen adds only 16 and 17 to the bench, 0.1 × 2 × 2.5 = 0.5,
so 0.85 × 0.5 = 0.425. In GW10 it also buys 18 for 8: +9.4 in the eleven,
the armband 6.0 → 15.0 is +9.0, and the bench +0.5 — 18.9, so 0.85 × 18.9 =
16.065.

**The assignment.** The free hit takes GW10 (16.065 ρ⁴ = 14.22; nothing else
comes near). Of GW8–9, triple captain at 9 and bench boost at 8 is
8.5 ρ³ + 6.552 ρ² = 7.758 + 6.165 = 13.92, against the swap's
6.8 ρ² + 6.552 ρ³ = 6.398 + 5.980 = 12.38.
"""

import json

import pytest

from aigaffer.chips import BENCH_BOOST, FREE_HIT, TRIPLE_CAPTAIN, WILDCARD, HeldChip
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver import calendar
from aigaffer.solver.calendar import (
    CHIP_DISCOUNT,
    OPTION_FLOOR,
    PROXY_SCALE,
    WILDCARD_BAR,
    WILDCARD_RAMP_WEEKS,
    _bars_for,
    _values,
    assign,
    build_calendar,
    calendar_chips,
    fallback_calendar,
    horizon_end,
)
from aigaffer.solver.multiweek import FALLBACK_BARS
from aigaffer.solver.optimizer import BENCH_WEIGHT

TC19 = HeldChip(TRIPLE_CAPTAIN, 1, 19)
BB19 = HeldChip(BENCH_BOOST, 1, 19)
FH19 = HeldChip(FREE_HIT, 2, 19)
WC19 = HeldChip(WILDCARD, 2, 19)
TC38 = HeldChip(TRIPLE_CAPTAIN, 20, 38)
WINDOW = [6, 7, 8, 9, 10, 11]
FIRST_SET = (BB19, TC19, WC19, FH19)

GK, DEF, MID, FWD = 1, 2, 3, 4
RHO = CHIP_DISCOUNT


def test_the_calendar_covers_only_chips_the_window_can_reach():
    assert calendar_chips(FIRST_SET + (TC38,), WINDOW) == FIRST_SET
    assert TC38 in calendar_chips(FIRST_SET + (TC38,), [15, 16, 17, 18, 19, 20])


def test_the_horizon_runs_to_the_furthest_reachable_expiry_never_past_it():
    assert horizon_end(FIRST_SET + (TC38,), WINDOW) == 19
    assert horizon_end(FIRST_SET + (TC38,), [15, 16, 17, 18, 19, 20]) == 38
    assert horizon_end((HeldChip(TRIPLE_CAPTAIN, 20, 38),), [35, 36, 37, 38, 39, 40]) == 40  # the window itself


def test_assign_gives_each_chip_its_best_distinct_week():
    # TC best at 15 (12), BB best at 15 too (14) but next-best 17 (13).
    # Discounted from anchor 6: TC@15 12·ρ^9, BB@17 13·ρ^11 beats the swap.
    values = {"triple_captain@19": {15: 12.0, 17: 6.0}, "bench_boost@19": {15: 14.0, 17: 13.0}}
    anchors = {"triple_captain@19": 6, "bench_boost@19": 6}
    assert assign(values, anchors) == {"triple_captain@19": 15, "bench_boost@19": 17}


def test_assign_leaves_a_chip_out_when_weeks_run_short():
    # Only GW19 is beyond the window: the higher discounted value takes it.
    values = {"triple_captain@19": {19: 9.0}, "bench_boost@19": {19: 11.0}}
    anchors = {"triple_captain@19": 13, "bench_boost@19": 13}
    assert assign(values, anchors) == {"bench_boost@19": 19}


def test_assign_breaks_ties_on_the_earlier_week():
    values = {"triple_captain@19": {14: 10.0 / CHIP_DISCOUNT ** 8, 13: 10.0 / CHIP_DISCOUNT ** 7}}
    assert assign(values, {"triple_captain@19": 6}) == {"triple_captain@19": 13}


def test_assign_breaks_a_full_tie_on_chip_order():
    # Two chips worth the same in the one week left: bench boost (earlier in
    # CHIP_ORDER) takes it.
    values = {"triple_captain@19": {19: 10.0}, "bench_boost@19": {19: 10.0}}
    anchors = {"triple_captain@19": 13, "bench_boost@19": 13}
    assert assign(values, anchors) == {"bench_boost@19": 19}


def test_the_free_hit_carries_its_option_floor_until_expiry():
    assert set(_bars_for(FH19, WINDOW, saved_for=None, value=None).values()) == {OPTION_FLOOR[FREE_HIT]}
    assert _bars_for(FH19, WINDOW, saved_for=15, value=12.0)[6] == pytest.approx(12.0 * CHIP_DISCOUNT ** 9 + OPTION_FLOOR[FREE_HIT])
    assert set(_bars_for(FH19, [14, 15, 16, 17, 18, 19], saved_for=None, value=None).values()) == {0.0}


def test_assign_ignores_worthless_weeks():
    assert assign({"triple_captain@19": {15: 0.0, 16: -1.0}}, {"triple_captain@19": 6}) == {}


def test_a_saved_for_week_sets_a_bar_that_falls_with_distance():
    bars = _bars_for(TC19, WINDOW, saved_for=15, value=12.0)
    assert bars[15 - 9] == pytest.approx(12.0 * CHIP_DISCOUNT ** 9)   # GW6
    assert bars[11] == pytest.approx(12.0 * CHIP_DISCOUNT ** 4)
    assert bars[6] < bars[11]   # the within-window tilt towards later weeks: pinned


def test_no_saved_for_week_is_a_zero_bar():
    assert _bars_for(TC19, [14, 15, 16, 17, 18, 19], saved_for=None, value=None) == {
        w: 0.0 for w in [14, 15, 16, 17, 18, 19]
    }


def test_the_wildcard_ramps_to_zero_at_expiry():
    full = _bars_for(WC19, WINDOW, saved_for=None, value=None)            # k = 8 weeks after GW11
    assert set(full.values()) == {WILDCARD_BAR}
    half = _bars_for(WC19, [10, 11, 12, 13, 14, 15], saved_for=None, value=None)  # k = 4
    assert half == {w: pytest.approx(WILDCARD_BAR * 4 / WILDCARD_RAMP_WEEKS) for w in range(10, 16)}
    assert set(_bars_for(WC19, [14, 15, 16, 17, 18, 19], saved_for=None, value=None).values()) == {0.0}


def test_bars_cover_only_window_weeks_inside_the_chips_window():
    assert sorted(_bars_for(TC38, [17, 18, 19, 20, 21, 22], saved_for=30, value=10.0)) == [20, 21, 22]


def test_every_first_set_bar_is_zero_once_the_window_reaches_gw19():
    cal = fallback_calendar(FIRST_SET, [14, 15, 16, 17, 18, 19])
    assert all(v == 0.0 for entry in cal.entries for v in entry.bars.values())


def test_the_fallback_holds_chips_with_a_later_week_at_the_old_bars():
    cal = fallback_calendar(FIRST_SET, WINDOW)
    assert cal.fell_back
    for entry in cal.entries:
        assert set(entry.bars.values()) == {FALLBACK_BARS[entry.held.chip]}  # no option floor on the fallback


def test_the_record_round_trips_through_json():
    cal = fallback_calendar(FIRST_SET, WINDOW)
    assert json.loads(json.dumps(cal.record()))["fell_back"] is True


# --- the board -------------------------------------------------------------

BOARD_WINDOW = [6, 7]
BOARD_WEEKS = range(6, 11)
TC10 = HeldChip(TRIPLE_CAPTAIN, 1, 10)
BB10 = HeldChip(BENCH_BOOST, 1, 10)
FH10 = HeldChip(FREE_HIT, 2, 10)

SPINE = [
    (1, GK, 5.0), (2, GK, 0.5),
    (3, DEF, 4.4), (4, DEF, 4.3), (5, DEF, 4.2), (6, DEF, 0.5), (7, DEF, 0.5),
    (8, MID, 5.6), (9, MID, 5.7), (10, MID, 5.8), (11, MID, 5.9), (12, MID, 6.0),
    (13, FWD, 4.1), (14, FWD, 4.0), (15, FWD, 3.9),
]
SQUAD = [pid for pid, _, _ in SPINE]
# 0.85 × 8.0 in GW8 and GW10, 0.85 × 10.0 in GW9 — striker 20, every week.
TC_VALUES = {8: 6.8, 9: 8.5, 10: 6.8}
# 0.7 × 0.9 × (0.5 + 3.0 + 3.0 + 3.9): the bench of 2, 16, 17 and 15.
BB_VALUE = 0.7 * 0.9 * 10.4
# 0.85 × 0.5 — 16 and 17 on the bench — and 0.85 × 18.9 with 18 bought too.
FH_VALUES = {8: 0.425, 9: 0.425, 10: 0.85 * 18.9}


class Board:
    def __init__(self, players, projections, xmins, striker_points):
        self.players = players
        self.projections = projections
        self.xmins = xmins
        self.striker_points = striker_points
        self.squad = list(SQUAD)
        self.bank = 0
        self.held = (BB10, TC10, FH10)


def _player(pid: int, position: int, cost: int, status: str = "a") -> Player:
    """Every player on his own club, so only price and projection decide."""
    return Player(
        id=pid,
        web_name=f"P{pid}",
        team=pid,
        element_type=position,
        now_cost=cost,
        status=status,
        minutes=900,
        starts=10,
        total_points=0,
        bonus=0,
        saves=0,
    )


@pytest.fixture
def board_beyond_window() -> Board:
    rows = [(pid, pos, 50, {w: xp for w in BOARD_WEEKS}, {}, 90.0, "a") for pid, pos, xp in SPINE]
    rows += [
        (16, DEF, 50, {w: 3.0 for w in BOARD_WEEKS}, {}, 90.0, "a"),
        (17, DEF, 50, {w: 3.0 for w in BOARD_WEEKS}, {}, 90.0, "a"),
        (18, MID, 50, {w: 15.0 if w == 10 else 0.0 for w in BOARD_WEEKS}, {}, 90.0, "a"),
    ]

    def week9(other: float, gw9: float) -> dict[int, float]:
        return {w: gw9 if w == 9 else other for w in BOARD_WEEKS}

    rows += [
        (20, FWD, 200, week9(8.0, 10.0), week9(4.0, 6.0), 85.0, "a"),
        (21, FWD, 200, week9(12.0, 14.0), week9(8.0, 9.0), 70.0, "a"),
        (22, MID, 200, week9(7.0, 12.0), week9(2.0, 3.0), 90.0, "a"),
        (23, FWD, 200, week9(15.0, 15.0), week9(10.0, 10.0), 90.0, "i"),
    ]
    players = {pid: _player(pid, pos, cost, status) for pid, pos, cost, _, _, _, status in rows}
    projections = {
        pid: PlayerProjection(
            player_id=pid, per_gw=dict(per_gw), total=sum(per_gw.values()), attacking_per_gw=dict(attacking)
        )
        for pid, _, _, per_gw, attacking, _, _ in rows
    }
    xmins = {pid: minutes for pid, _, _, _, _, minutes, _ in rows}
    return Board(players, projections, xmins, striker_points=projections[20].per_gw)


def _build(board: Board, held=None) -> calendar.ChipCalendar:
    return build_calendar(
        board.held if held is None else held,
        BOARD_WINDOW,
        board.players,
        board.projections,
        board.xmins,
        board.squad,
        board.bank,
    )


def test_build_calendar_values_and_saves_each_chip(board_beyond_window):
    b = board_beyond_window   # window GW6-7, held TC/BB/FH to GW10, projections to GW10
    cal = build_calendar(b.held, [6, 7], b.players, b.projections, b.xmins, b.squad, b.bank)
    by_id = {e.held.id: e for e in cal.entries}
    # TC: the 85-minute striker with the higher attacking EV in GW9, not the
    # 70-minute one with more points.
    assert by_id["triple_captain@10"].saved_for == 9
    assert by_id["triple_captain@10"].value == pytest.approx(PROXY_SCALE[TRIPLE_CAPTAIN] * b.striker_points[9])
    assert not cal.fell_back
    # The rest of the assignment worked in the module docstring: the free hit
    # at GW10 (18's week), the bench boost at GW8 (flat, so the soonest).
    assert by_id["bench_boost@10"].saved_for == 8
    assert by_id["bench_boost@10"].value == pytest.approx(BB_VALUE, abs=1e-6)
    assert by_id["free_hit@10"].saved_for == 10
    assert by_id["free_hit@10"].value == pytest.approx(FH_VALUES[10], abs=1e-6)
    assert cal.horizon_end == 10
    assert cal.discount == CHIP_DISCOUNT and cal.proxy_scale == PROXY_SCALE


def test_the_window_bars_are_the_saved_for_values_discounted_back(board_beyond_window):
    # TC: 8.5 ρ^(9-w). BB: 6.552 ρ^(8-w). FH: 16.065 ρ^(10-w) plus its
    # option floor, since GW8-10 are still to come.
    cal = _build(board_beyond_window)
    floor = OPTION_FLOOR[FREE_HIT]
    expected = {
        "triple_captain@10": {6: 8.5 * RHO ** 3, 7: 8.5 * RHO ** 2},
        "bench_boost@10": {6: BB_VALUE * RHO ** 2, 7: BB_VALUE * RHO},
        "free_hit@10": {6: FH_VALUES[10] * RHO ** 4 + floor, 7: FH_VALUES[10] * RHO ** 3 + floor},
    }
    assert cal.bars == {
        chip: {w: pytest.approx(bar, abs=1e-6) for w, bar in bars.items()}
        for chip, bars in expected.items()
    }


def test_the_triple_captain_is_priced_on_attacking_ev_among_available_80_minute_men(board_beyond_window):
    # Every week the armband goes to 20 (85 minutes, the most attacking EV of
    # the men who qualify), never to 21 (70 minutes), 22 (more points in GW9,
    # less attacking EV) or 23 (injured): 0.85 × 20's points.
    b = board_beyond_window
    values = _values(TRIPLE_CAPTAIN, [8, 9, 10], b.players, b.projections, b.xmins, b.squad, b.bank, None)
    assert values == {w: pytest.approx(v) for w, v in TC_VALUES.items()}


def test_a_triple_captain_with_no_80_minute_man_is_worth_nothing(board_beyond_window):
    b = board_beyond_window
    short = {pid: min(minutes, 79.0) for pid, minutes in b.xmins.items()}
    assert _values(TRIPLE_CAPTAIN, [8, 9, 10], b.players, b.projections, short, b.squad, b.bank, None) == {}


def test_the_bench_boost_values_the_four_worst_of_the_fifteen(board_beyond_window, monkeypatch):
    # The weight-1 fifteen, lined up on the window's basis: its bench is its
    # four lowest-scoring legal substitutes, never four of its best. The
    # solve's own eleven is replaced here by one that benches the four best
    # midfielders — at weight 1 any eleven scores the same, so the calendar
    # must not read it. The bench is still 2, 16, 17 and 15: 10.4, × 0.7 × 0.9.
    real = calendar.best_one_week_squads

    def scrambled(*args, **kwargs):
        assert kwargs["bench_weight"] == 1.0
        return {
            w: (value, fifteen, [p for p in fifteen if p not in {9, 10, 11, 12}])
            for w, (value, fifteen, _) in real(*args, **kwargs).items()
        }

    monkeypatch.setattr(calendar, "best_one_week_squads", scrambled)
    b = board_beyond_window
    cal = _build(b, held=(BB10,))
    (entry,) = cal.entries
    # Flat across GW8-10, so the soonest week wins.
    assert entry.saved_for == 8
    assert entry.value == pytest.approx(BB_VALUE, abs=1e-6)
    assert entry.value == pytest.approx(PROXY_SCALE[BENCH_BOOST] * (1 - BENCH_WEIGHT) * 10.4, abs=1e-6)


def test_a_week_that_cannot_be_priced_is_skipped(board_beyond_window, monkeypatch):
    # GW10's sub-solve fails. The free hit loses the only week it was worth
    # anything in, and the bench boost one of three identical ones. With
    # GW8-10 left: TC@9 + BB@8 = 13.92 beats TC@10 + BB@8 + FH@9 =
    # 6.8 ρ⁴ + 6.552 ρ² + 0.425 ρ³ = 6.020 + 6.165 + 0.388 = 12.57, so the
    # free hit goes unassigned and holds on its option floor alone.
    real = calendar.best_one_week_squads

    def gw10_fails(*args, **kwargs):
        return {w: (None if w == 10 else answer) for w, answer in real(*args, **kwargs).items()}

    monkeypatch.setattr(calendar, "best_one_week_squads", gw10_fails)
    b = board_beyond_window
    assert _values(FREE_HIT, [8, 9, 10], b.players, b.projections, b.xmins, b.squad, b.bank, None) == {
        8: pytest.approx(0.425, abs=1e-6),
        9: pytest.approx(0.425, abs=1e-6),
    }
    cal = _build(b)
    by_id = {e.held.id: e for e in cal.entries}
    assert by_id["triple_captain@10"].saved_for == 9
    assert by_id["bench_boost@10"].saved_for == 8
    fh = by_id["free_hit@10"]
    assert (fh.saved_for, fh.value) == (None, None)
    assert fh.bars == {6: OPTION_FLOOR[FREE_HIT], 7: OPTION_FLOOR[FREE_HIT]}


def _cannot_line_up(week, monkeypatch):
    """Make ``squad_one_week`` find no eleven in ``week``, as it does for a
    squad with too few players at some position."""
    real = calendar.squad_one_week

    def empty_in_week(players, projections, squad, event):
        if event == week:
            return 0.0, squad, []
        return real(players, projections, squad, event)

    monkeypatch.setattr(calendar, "squad_one_week", empty_in_week)


def test_a_bench_boost_week_the_fifteen_cannot_line_up_is_skipped(board_beyond_window, monkeypatch):
    # No eleven means no bench: counting all fifteen would be a huge, silent
    # value, and the one week the boost would be saved for.
    _cannot_line_up(9, monkeypatch)
    b = board_beyond_window
    values = _values(BENCH_BOOST, [8, 9, 10], b.players, b.projections, b.xmins, b.squad, b.bank, None)
    assert values == {8: pytest.approx(BB_VALUE, abs=1e-6), 10: pytest.approx(BB_VALUE, abs=1e-6)}
    cal = _build(b, held=(BB10,))
    assert cal.entries[0].saved_for == 8


def test_a_free_hit_week_the_squad_cannot_line_up_is_skipped(board_beyond_window, monkeypatch):
    # No eleven for the squad held scores 0, so the free hit would be worth
    # the whole best squad: the week is left out, the others still price.
    _cannot_line_up(10, monkeypatch)
    b = board_beyond_window
    values = _values(FREE_HIT, [8, 9, 10], b.players, b.projections, b.xmins, b.squad, b.bank, None)
    assert values == {8: pytest.approx(FH_VALUES[8], abs=1e-6), 9: pytest.approx(FH_VALUES[9], abs=1e-6)}


def test_the_free_hit_is_the_best_squad_less_the_one_held(board_beyond_window):
    # Both sides on the window's basis: 0.85 × (62.04 - 61.54) in GW8 and 9,
    # 0.85 × (80.44 - 61.54) in GW10.
    b = board_beyond_window
    values = _values(FREE_HIT, [8, 9, 10], b.players, b.projections, b.xmins, b.squad, b.bank, None)
    assert values == {w: pytest.approx(v, abs=1e-6) for w, v in FH_VALUES.items()}


def test_a_built_calendar_round_trips_through_json(board_beyond_window):
    record = _build(board_beyond_window).record()
    assert json.loads(json.dumps(record)) == record
    assert record["fell_back"] is False
    assert record["horizon_end"] == 10
    entry = next(e for e in record["entries"] if e["chip"] == "triple_captain@10")
    assert entry["saved_for"] == 9
    assert entry["value"] == pytest.approx(8.5)
    assert set(entry["bars"]) == {"6", "7"}


def test_a_window_across_the_halves_saves_each_set_on_its_own(monkeypatch):
    # Window GW15-20, both triple captains held. No projection needs a solve:
    # the triple captain is priced off one 85-minute striker, worth 8.0 (4.0
    # attacking) every week and 10.0 (6.0 attacking) in GW25.
    #
    # The first set's (to GW19) has no week after the window left — the window
    # runs past its expiry — so nothing to save it for: unassigned, and a bar
    # of 0 in every window week it may go in, GW15-19.
    #
    # The second set's (from GW20) is valued over GW21-38 at 0.85 x 8.0 = 6.8,
    # and 0.85 x 10.0 = 8.5 in GW25, anchored at GW20 — the first window week
    # it may be played in, not the window's GW15. GW25 wins:
    # 8.5 ρ⁵ = 7.299 against GW21's 6.8 ρ = 6.596. Its one window week is
    # GW20, where the bar is 8.5 ρ^(25-20) = 7.299 — no option floor, it is a
    # triple captain.
    window = [15, 16, 17, 18, 19, 20]
    weeks = range(15, 39)
    points = {w: 10.0 if w == 25 else 8.0 for w in weeks}
    attacking = {w: 6.0 if w == 25 else 4.0 for w in weeks}
    players = {20: _player(20, FWD, 200)}
    projections = {
        20: PlayerProjection(
            player_id=20, per_gw=points, total=sum(points.values()),
            attacking_per_gw=attacking,
        )
    }
    asked: list[tuple[dict, dict]] = []
    real = calendar.assign

    def spy(values, anchors):
        asked.append((values, anchors))
        return real(values, anchors)

    monkeypatch.setattr(calendar, "assign", spy)

    cal = build_calendar(
        (TC19, TC38), window, players, projections, {20: 85.0}, [], 0
    )

    by_id = {entry.held.id: entry for entry in cal.entries}
    first, second = by_id["triple_captain@19"], by_id["triple_captain@38"]
    assert (first.saved_for, first.value) == (None, None)
    assert first.bars == {w: 0.0 for w in range(15, 20)}
    assert second.saved_for == 25
    assert second.value == pytest.approx(8.5)
    assert second.bars == {20: pytest.approx(8.5 * RHO ** 5)}
    assert cal.horizon_end == 38
    # One assignment a half, each anchored where it may first be played.
    assert [anchors for _, anchors in asked] == [
        {"triple_captain@19": 15},
        {"triple_captain@38": 20},
    ]
    assert asked[0][0] == {"triple_captain@19": {}}
    assert set(asked[1][0]["triple_captain@38"]) == set(range(21, 39))
