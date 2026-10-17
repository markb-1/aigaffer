"""Tests for the replay's pure core: a week's score, a week's step, the gate.

Nothing here touches the network or the solver. The scoring boards are one
hand-built fifteen, positions by id — 1-2 keepers, 3-7 defenders, 8-12
midfielders, 13-15 forwards — fielded 1-3-5-2:

==  ===  ==========================================
id  pos  role
==  ===  ==========================================
1   GK   starts
2   GK   bench[0], the substitute keeper
3   DEF  starts
4   DEF  starts
5   DEF  starts
6   DEF  bench[1], first outfielder on
7   DEF  bench[2]
8   MID  starts
9   MID  starts
10  MID  starts
11  MID  starts
12  MID  starts, and captains
13  FWD  starts, and is vice-captain
14  FWD  starts
15  FWD  bench[3]
==  ===  ==========================================

Every player scores 2 in 90 minutes unless a test says otherwise, so a full
turnout is eleven 2s and the captain's second helping: 24.

The gate boards are two squads over three gameweeks (2, 3, 4), and every
difference is worked by hand in the test that uses it.
"""

from math import sqrt

import pytest

from aigaffer.replay import (
    State,
    apply_moves,
    autosub_points,
    gate,
    next_free_transfers,
    render_summary,
)
from aigaffer.solver.lineup import Lineup

GK, DEF, MID, FWD = 1, 2, 3, 4


def _positions():
    # 1,2 GK; 3-7 DEF; 8-12 MID; 13-15 FWD
    return {**{p: GK for p in (1, 2)}, **{p: DEF for p in range(3, 8)},
            **{p: MID for p in range(8, 13)}, **{p: FWD for p in range(13, 16)}}


LINEUP = Lineup(xi=[1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14], captain=12, vice=13, bench=[2, 6, 7, 15])


# --- scoring a gameweek ------------------------------------------------------


def test_every_starter_plays_and_the_captain_doubles():
    actual = {p: (90, 2) for p in range(1, 16)}
    # Eleven at 2 = 22, plus the captain's second 2 = 24.
    assert autosub_points(LINEUP, _positions(), actual) == 24


def test_a_keeper_who_did_not_play_is_replaced_by_the_backup_keeper():
    actual = {p: (90, 2) for p in range(1, 16)} | {1: (0, 0), 2: (90, 6)}
    assert autosub_points(LINEUP, _positions(), actual) == 24 - 2 + 6


def test_an_outfield_sub_comes_on_in_bench_order_keeping_a_legal_shape():
    # Forward 14 does not play; 13 still plays, so one forward is enough and the
    # first bench outfielder (defender 6, 5 points) comes on before forward 15.
    actual = {p: (90, 2) for p in range(1, 16)} | {14: (0, 0), 6: (90, 5), 15: (90, 9)}
    assert autosub_points(LINEUP, _positions(), actual) == 24 - 2 + 5


def test_a_sub_that_would_break_the_shape_is_skipped():
    # Both forwards 13 and 14 miss. Bench order is 6, 7, 15:
    # - defender 6 replaces 13 (first non-player in XI order), leaving
    #   4 DEF, 5 MID, 1 FWD — 14 still counts until he is replaced: legal;
    # - defender 7 would replace 14, leaving no forward: illegal, so 7 sits;
    # - forward 15 replaces 14, leaving 4 DEF, 5 MID, 1 FWD: legal.
    actual = {p: (90, 2) for p in range(1, 16)} | {13: (0, 0), 14: (0, 0), 6: (90, 5), 15: (90, 9)}
    # The vice (13) did not play, the captain (12) did: 12 doubles.
    # 24 − 2 − 2 + 5 + 9 = 34.
    assert autosub_points(LINEUP, _positions(), actual) == 24 - 4 + 5 + 9


def test_the_vice_doubles_when_the_captain_does_not_play():
    actual = {p: (90, 2) for p in range(1, 16)} | {12: (0, 0), 6: (90, 2)}
    # 12 out, 6 on (2): eleven still 22; captain absent, vice 13 doubles: 24.
    assert autosub_points(LINEUP, _positions(), actual) == 24


def test_nobody_doubles_when_neither_captain_nor_vice_plays():
    # 12 and 13 both miss. Defender 6 (3) replaces 12: 4 DEF, 4 MID, 2 FWD.
    # Defender 7 (4) replaces 13: 5 DEF, 4 MID, 1 FWD — legal. Forward 15 finds
    # no one left to replace. Nine starters at 2 = 18, plus 3 + 4 = 25, and no
    # armband: doubling the man who came on for the captain would make it 28.
    actual = {p: (90, 2) for p in range(1, 16)} | {
        12: (0, 0), 13: (0, 0), 6: (90, 3), 7: (90, 4),
    }
    assert autosub_points(LINEUP, _positions(), actual) == 25


def test_an_outfielder_never_comes_on_for_a_keeper():
    # Both keepers miss. Defender 6 has played, but an eleven without a keeper
    # is not a legal shape, so the empty goal stays empty: ten starters at 2 =
    # 20, plus the captain's 2 = 22. Putting 6 in goal would make it 27.
    actual = {p: (90, 2) for p in range(1, 16)} | {1: (0, 0), 2: (0, 0), 6: (90, 5)}
    assert autosub_points(LINEUP, _positions(), actual) == 22


def test_a_player_with_no_history_for_the_week_did_not_play():
    # Forward 14's team blanked, so he has no row at all; he counts as 0
    # minutes and defender 6 (5) comes on for him: 24 − 2 + 5 = 27.
    actual = {p: (90, 2) for p in range(1, 16) if p != 14} | {6: (90, 5)}
    assert autosub_points(LINEUP, _positions(), actual) == 27


# --- a week's step ------------------------------------------------------------


def test_free_transfers_carry_by_the_game_rule():
    assert next_free_transfers(1, 0) == 2
    assert next_free_transfers(2, 1) == 2
    assert next_free_transfers(1, 3) == 1
    assert next_free_transfers(5, 0) == 5


def test_a_transfer_is_paid_for_at_that_weeks_prices():
    state = State(squad=tuple(range(1, 16)), bank=10, free_transfers=1)
    after = apply_moves(state, [20], [8], {8: 60, 20: 65}, hits=0)
    assert set(after.squad) == set(range(1, 16)) - {8} | {20}
    assert after.bank == 5
    assert after.free_transfers == 1


def test_moves_beyond_the_free_ones_carry_their_hits():
    # Two moves on one free transfer is one hit. 8 (60) and 9 (55) out, 20 (65)
    # and 21 (40) in: the bank goes 10 + 115 − 105 = 20, and the FTs go
    # max(1 − 2, 0) + 1 = 1.
    state = State(squad=tuple(range(1, 16)), bank=10, free_transfers=1)
    after = apply_moves(state, [20, 21], [8, 9], {8: 60, 9: 55, 20: 65, 21: 40}, hits=1)
    assert set(after.squad) == set(range(1, 16)) - {8, 9} | {20, 21}
    assert after.bank == 20
    assert after.free_transfers == 1


def test_hits_that_do_not_match_the_moves_are_refused():
    # One move on one free transfer owes no hit; a caller claiming one (or
    # claiming none for two moves) would score the week at the wrong price.
    state = State(squad=tuple(range(1, 16)), bank=10, free_transfers=1)
    with pytest.raises(ValueError, match="hits"):
        apply_moves(state, [20], [8], {8: 60, 20: 65}, hits=1)
    with pytest.raises(ValueError, match="hits"):
        apply_moves(state, [20, 21], [8, 9], {8: 60, 9: 55, 20: 65, 21: 40}, hits=0)


def test_a_move_that_does_not_fit_the_squad_is_refused():
    state = State(squad=tuple(range(1, 16)), bank=10, free_transfers=2)
    with pytest.raises(ValueError):
        apply_moves(state, [20], [30], {20: 65, 30: 50}, hits=0)  # 30 is not held
    with pytest.raises(ValueError):
        apply_moves(state, [9], [8], {8: 60, 9: 55}, hits=0)  # 9 is already held
    with pytest.raises(ValueError):
        apply_moves(state, [20, 21], [8], {8: 60, 20: 65, 21: 40}, hits=0)  # two for one


# --- the gate ----------------------------------------------------------------


def _rows(group, points, ft_end, gws=(2, 3, 4)):
    """Rows for one group from ``points[policy][entry]`` (a score a gameweek)
    and ``ft_end[policy][entry]`` (the FTs held after each gameweek)."""
    return [
        {
            "group": group,
            "entry": entry,
            "gw": gw,
            "policy": policy,
            "points": points[policy][entry][i],
            "ft_end": ft_end[policy][entry][i],
        }
        for policy in points
        for entry in points[policy]
        for i, gw in enumerate(gws)
    ]


def _flat(diffs_by_entry, legacy=50):
    """Points for LEGACY at a flat 50 and a candidate at 50 + the given
    differences, with FTs equal throughout so the credit changes nothing."""
    return (
        {
            "LEGACY": {e: [legacy] * 3 for e in diffs_by_entry},
            "CAND": {e: [legacy + d for d in diffs] for e, diffs in diffs_by_entry.items()},
        },
        {p: {e: [1, 1, 1] for e in diffs_by_entry} for p in ("LEGACY", "CAND")},
    )


def _renamed(rows, name):
    return [r | {"policy": name} if r["policy"] == "CAND" else r for r in rows]


def test_the_gate_sums_the_weekly_mean_differences_and_clusters_by_gameweek():
    # WINDOW − LEGACY, squad by squad:
    #   gw2: e1 52−50 = +2, e2 70−70 =  0 → mean 1.0
    #   gw3: e1 58−60 = −2, e2 35−30 = +5 → mean 1.5
    #   gw4: e1 45−40 = +5, e2 50−55 = −5 → mean 0.0
    # The FTs held after gw4 differ by e1 3−1 = +2, e2 1−2 = −1, mean +0.5, so
    # credit c adds 0.5c to gw4 alone. The ft_end on gw2 and gw3 are noise
    # that must not count.
    #   c = 0:   d̄ = (1, 1.5, 0),    d = 2.5;  deviations from 5/6 are
    #            (1/6, 2/3, −5/6), squares sum to 7/6, sample var 7/12,
    #            SE = sqrt(3 · 7/12) = sqrt(1.75)
    #   c = 1.5: d̄ = (1, 1.5, 0.75), d = 3.25; deviations from 13/12 are
    #            (−1/12, 5/12, −1/3), squares sum to 7/24, var 7/48,
    #            SE = sqrt(3 · 7/48) = sqrt(0.4375)
    #   c = 2:   d̄ = (1, 1.5, 1),    d = 3.5;  deviations from 7/6 are
    #            (−1/6, 1/3, −1/6), squares sum to 1/6, var 1/12,
    #            SE = sqrt(3 · 1/12) = 0.5
    points = {
        "LEGACY": {1: [50, 60, 40], 2: [70, 30, 55]},
        "WINDOW": {1: [52, 58, 45], 2: [70, 35, 50]},
    }
    ft_end = {
        "LEGACY": {1: [2, 5, 1], 2: [1, 1, 2]},
        "WINDOW": {1: [1, 1, 3], 2: [4, 2, 1]},
    }
    result = gate(_rows("top", points, ft_end))

    cells = result["cells"]
    assert set(cells) == {("WINDOW", "top", c) for c in (0.0, 1.5, 2.0)}
    assert cells["WINDOW", "top", 0.0]["d"] == pytest.approx(2.5)
    assert cells["WINDOW", "top", 0.0]["se"] == pytest.approx(sqrt(1.75))
    assert cells["WINDOW", "top", 1.5]["d"] == pytest.approx(3.25)
    assert cells["WINDOW", "top", 1.5]["se"] == pytest.approx(sqrt(0.4375))
    assert cells["WINDOW", "top", 2.0]["d"] == pytest.approx(3.5)
    assert cells["WINDOW", "top", 2.0]["se"] == pytest.approx(0.5)
    assert result["verdict"] == {"WINDOW": "ship"}
    assert result["pick"] == "WINDOW"


def test_the_verdicts_ship_go_to_the_owner_or_ship_nothing():
    # Per-gameweek differences, the same for both squads, FTs equal, so every
    # credit gives the same cell:
    #   (+1, +1, +1):  d = 3, SE = 0 — ship.
    #   (−3, +2,  0):  d = −1; deviations from −1/3 are (−8/3, 7/3, 1/3),
    #                  squares sum to 38/3, var 19/3, SE = sqrt(19) ≈ 4.36:
    #                  −SE ≤ d < 0 — the owner's call.
    #   (−3, −2, −4):  d = −9; deviations from −3 are (0, 1, −1), var 1,
    #                  SE = sqrt(3) ≈ 1.73: d + SE < 0 — no.
    ship = {1: [1, 1, 1], 2: [1, 1, 1]}
    owner = {1: [-3, 2, 0], 2: [-3, 2, 0]}
    worse = {1: [-3, -2, -4], 2: [-3, -2, -4]}

    # WINDOW ships in the top group and is the owner's call in the deep one;
    # ALT ships in the top group and fails in the deep one. A single failing
    # cell outweighs every passing one, and an owner's cell outweighs a ship.
    rows = (
        _renamed(_rows("top", *_flat(ship)), "WINDOW")
        + _renamed([r for r in _rows("deep", *_flat(owner)) if r["policy"] == "CAND"], "WINDOW")
        + _renamed([r for r in _rows("top", *_flat(ship)) if r["policy"] == "CAND"], "ALT")
        + _renamed(_rows("deep", *_flat(worse)), "ALT")
    )
    result = gate(rows)

    assert result["cells"]["WINDOW", "deep", 1.5]["d"] == pytest.approx(-1.0)
    assert result["cells"]["WINDOW", "deep", 1.5]["se"] == pytest.approx(sqrt(19))
    assert result["cells"]["ALT", "deep", 0.0]["d"] == pytest.approx(-9.0)
    assert result["cells"]["ALT", "deep", 0.0]["se"] == pytest.approx(sqrt(3))
    assert result["cells"]["ALT", "top", 2.0] == {"d": pytest.approx(3.0), "se": pytest.approx(0.0)}
    assert result["verdict"] == {"WINDOW": "owner", "ALT": "no"}
    assert result["pick"] is None


def test_when_both_ship_the_larger_top_group_lead_at_credit_one_and_a_half_wins():
    # Top group, two squads, LEGACY 50 a week with 1 FT held throughout.
    #   WINDOW: +1 a week, FTs equal          → d = 3 at every credit.
    #   ALT:    +2, 0, 0 a week (d = 2) and 2 FTs at the end (+1 each squad)
    #           → d = 2 at credit 0 (WINDOW leads), 2 + 1.5 = 3.5 at 1.5,
    #           4 at 2.
    # Deep group: WINDOW +10 a week (d = 30), ALT +1 once (d = 1), so a
    # tie-break read from the wrong group or the wrong credit picks WINDOW.
    gws_legacy = {e: [50, 50, 50] for e in (1, 2)}
    top_points = {
        "LEGACY": gws_legacy,
        "WINDOW": {e: [51, 51, 51] for e in (1, 2)},
        "ALT": {e: [52, 50, 50] for e in (1, 2)},
    }
    top_ft = {
        "LEGACY": {e: [1, 1, 1] for e in (1, 2)},
        "WINDOW": {e: [1, 1, 1] for e in (1, 2)},
        "ALT": {e: [1, 1, 2] for e in (1, 2)},
    }
    deep_points = {
        "LEGACY": {e: [50, 50, 50] for e in (3, 4)},
        "WINDOW": {e: [60, 60, 60] for e in (3, 4)},
        "ALT": {e: [51, 50, 50] for e in (3, 4)},
    }
    deep_ft = {p: {e: [1, 1, 1] for e in (3, 4)} for p in deep_points}
    result = gate(_rows("top", top_points, top_ft) + _rows("deep", deep_points, deep_ft))

    assert result["cells"]["ALT", "top", 0.0]["d"] == pytest.approx(2.0)
    assert result["cells"]["ALT", "top", 1.5]["d"] == pytest.approx(3.5)
    assert result["cells"]["ALT", "top", 2.0]["d"] == pytest.approx(4.0)
    assert result["cells"]["WINDOW", "top", 1.5]["d"] == pytest.approx(3.0)
    assert result["cells"]["WINDOW", "deep", 1.5]["d"] == pytest.approx(30.0)
    assert result["verdict"] == {"WINDOW": "ship", "ALT": "ship"}
    assert result["pick"] == "ALT"


def test_a_squad_missing_a_policy_is_left_out_of_the_pairing():
    # Squad 2 has no WINDOW rows (its solve was lost), so the means are over
    # squad 1 alone: +2, +2, +2 → d = 6, SE = 0. Pairing it against squad 2's
    # LEGACY rows anyway would drag every week's mean down.
    points = {
        "LEGACY": {1: [50, 50, 50], 2: [90, 90, 90]},
        "WINDOW": {1: [52, 52, 52]},
    }
    ft_end = {"LEGACY": {1: [1, 1, 1], 2: [1, 1, 1]}, "WINDOW": {1: [1, 1, 1]}}
    result = gate(_rows("top", points, ft_end))
    assert result["cells"]["WINDOW", "top", 0.0] == {"d": pytest.approx(6.0), "se": pytest.approx(0.0)}


# --- the summary ---------------------------------------------------------------


def test_the_summary_tabulates_the_cells_and_names_the_pick():
    gate_result = {
        "cells": {
            ("WINDOW", "top", 0.0): {"d": 2.5, "se": 1.3229},
            ("WINDOW", "top", 1.5): {"d": 3.25, "se": 0.6614},
            ("ALT", "top", 0.0): {"d": -1.0, "se": 4.3589},
        },
        "verdict": {"WINDOW": "ship", "ALT": "owner"},
        "pick": "WINDOW",
    }
    text = render_summary(gate_result, {"squads": 150, "skipped": 2})

    assert "| WINDOW | top | 0.0 | +2.50 | 1.32 |" in text
    assert "| WINDOW | top | 1.5 | +3.25 | 0.66 |" in text
    assert "| ALT | top | 0.0 | -1.00 | 4.36 |" in text
    assert "WINDOW: ship" in text
    assert "ALT: owner" in text
    assert "Pick: WINDOW" in text
    assert "squads: 150" in text
    assert "skipped: 2" in text
    assert "six clusters make the SE rough" in text


def test_the_summary_says_when_nothing_ships():
    text = render_summary({"cells": {}, "verdict": {"WINDOW": "no"}, "pick": None}, {})
    assert "Pick: none" in text
