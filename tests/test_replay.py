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


def test_a_defender_who_would_make_six_at_the_back_is_skipped():
    # A 1-5-3-2 eleven — 1; 3-7; 8, 9, 10; 13, 14 — with an extra defender 16
    # first on the bench, then midfielder 11, then forward 15. Midfielder 8
    # misses. Defender 16 (7 points) for 8 would make 6 DEF, past the five
    # the game allows (MID at 2 would still be legal), so 16 sits; midfielder
    # 11 (4) comes on: 5 DEF, 3 MID, 2 FWD. Ten starters at 2 = 20, plus 4,
    # plus captain 13's second 2 = 26. Letting 16 on would make 29.
    positions = _positions() | {16: DEF}
    lineup = Lineup(
        xi=[1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14], captain=13, vice=14, bench=[2, 16, 11, 15]
    )
    actual = {p: (90, 2) for p in range(1, 17)} | {8: (0, 0), 16: (90, 7), 11: (90, 4)}
    assert autosub_points(lineup, positions, actual) == 26


def test_every_starter_out_brings_on_the_whole_bench_that_fits():
    # None of the eleven plays; the bench all do: 2 scores 3, 6 scores 4,
    # 7 scores 5, 15 scores 6.
    # - Keeper 2 replaces 1.
    # - Defender 6 replaces 3, the first non-player in XI order: still 3 DEF.
    # - Defender 7 replaces 4: still 3 DEF.
    # - Forward 15 would replace 5, leaving 2 DEF: illegal; so he replaces 8,
    #   the next one, leaving 3 DEF, 4 MID, 3 FWD: legal.
    # The seven who stay on scored nothing, and neither the captain (12) nor
    # the vice (13) played, so no armband: 3 + 4 + 5 + 6 = 18.
    actual = {p: (0, 0) for p in LINEUP.xi} | {2: (90, 3), 6: (90, 4), 7: (90, 5), 15: (90, 6)}
    assert autosub_points(LINEUP, _positions(), actual) == 18


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
    with pytest.raises(ValueError, match="not held"):
        apply_moves(state, [20], [30], {20: 65, 30: 50}, hits=0)  # 30 is not held
    with pytest.raises(ValueError, match="already held"):
        apply_moves(state, [9], [8], {8: 60, 9: 55}, hits=0)  # 9 is already held
    with pytest.raises(ValueError, match="1 out"):
        apply_moves(state, [20, 21], [8], {8: 60, 20: 65, 21: 40}, hits=0)  # two for one


def test_a_move_the_bank_cannot_pay_for_is_refused():
    # Bank 0, 8 (60) out and 20 (65) in: 0 + 60 − 65 = −5. A plan solved at
    # other prices has overspent, and scoring it would hide the mispricing.
    state = State(squad=tuple(range(1, 16)), bank=0, free_transfers=1)
    with pytest.raises(ValueError, match="bank"):
        apply_moves(state, [20], [8], {8: 60, 20: 65}, hits=0)
    # At 60 apiece the bank ends at exactly 0, which is allowed.
    assert apply_moves(state, [20], [8], {8: 60, 20: 60}, hits=0).bank == 0


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
    assert result["dropped"] == {("WINDOW", "top"): 1}


def test_a_squad_missing_part_of_a_season_is_dropped_from_every_week():
    # Squad 1 is complete: WINDOW +2 a week, and 3 FTs left against LEGACY's
    # 1, so credit c adds 2c to gw4. Squad 2's WINDOW has gw2 and gw3 (−10
    # each) but no gw4, so it has no end to credit and is left out of every
    # week, not only the last. Over squad 1 alone:
    #   c = 0:   d̄ = (2, 2, 2), d = 6, SE = 0
    #   c = 1.5: d̄ = (2, 2, 5), d = 9; deviations from 3 are (−1, −1, 2),
    #            squares sum to 6, var 3, SE = sqrt(3 · 3) = 3
    #   c = 2:   d̄ = (2, 2, 6), d = 10; deviations from 10/3 are
    #            (−4/3, −4/3, 8/3), squares sum to 96/9, var 16/3,
    #            SE = sqrt(3 · 16/3) = 4
    # Pairing squad 2 in the weeks it has would give gw2 a mean of
    # (2 − 10)/2 = −4, and d at credit 0 of −4 − 4 + 2 = −6.
    rows = _rows(
        "top",
        {"LEGACY": {1: [50, 50, 50], 2: [50, 50, 50]}, "WINDOW": {1: [52, 52, 52]}},
        {"LEGACY": {1: [1, 1, 1], 2: [1, 1, 1]}, "WINDOW": {1: [1, 1, 3]}},
    ) + [
        {"group": "top", "entry": 2, "gw": gw, "policy": "WINDOW", "points": 40, "ft_end": 5}
        for gw in (2, 3)
    ]
    result = gate(rows)

    cells = result["cells"]
    assert cells["WINDOW", "top", 0.0] == {"d": pytest.approx(6.0), "se": pytest.approx(0.0)}
    assert cells["WINDOW", "top", 1.5] == {"d": pytest.approx(9.0), "se": pytest.approx(3.0)}
    assert cells["WINDOW", "top", 2.0] == {"d": pytest.approx(10.0), "se": pytest.approx(4.0)}
    assert result["dropped"] == {("WINDOW", "top"): 1}
    assert result["verdict"] == {"WINDOW": "ship"}


def test_a_squad_missing_a_middle_week_is_dropped_too():
    # A hole anywhere in the season breaks it, not only at the end. Squad 1
    # is complete, WINDOW +1 a week with FTs equal: d = 3, SE = 0. Squad 2's
    # LEGACY has no gw3 row (WINDOW is −20 a week against it), so it is
    # dropped from every week; pairing it in gw2 and gw4 would give
    # d = (1 − 20)/2 + 1 + (1 − 20)/2 = −18.
    rows = [
        r
        for r in _renamed(_rows("top", *_flat({1: [1, 1, 1], 2: [-20, -20, -20]})), "WINDOW")
        if not (r["entry"] == 2 and r["gw"] == 3 and r["policy"] == "LEGACY")
    ]
    result = gate(rows)
    assert result["cells"]["WINDOW", "top", 0.0] == {"d": pytest.approx(3.0), "se": pytest.approx(0.0)}
    assert result["dropped"] == {("WINDOW", "top"): 1}


def test_a_cell_exactly_one_se_behind_is_the_owners_call():
    # (−2, 0, 0) a week for both squads, FTs equal: d = −2; deviations from
    # −2/3 are (−4/3, 2/3, 2/3), squares sum to 24/9, var 4/3, SE =
    # sqrt(3 · 4/3) = 2. So d + SE = 0, which is not below zero: "owner", not
    # "no" — the rule for "no" is d + SE < 0.
    result = gate(_renamed(_rows("top", *_flat({1: [-2, 0, 0], 2: [-2, 0, 0]})), "WINDOW"))
    assert result["cells"]["WINDOW", "top", 0.0] == {"d": -2.0, "se": 2.0}
    assert result["verdict"] == {"WINDOW": "owner"}


def test_a_candidate_without_every_cell_never_ships():
    # Two groups are present, so each candidate needs 3 credits × 2 groups =
    # 6 cells. WINDOW is +1 a week in the top group and has no rows at all in
    # the deep one: its 3 cells would all pass, but a group it was never
    # measured in is not a group it passed. ALT has no rows anywhere, which
    # would ship on all([]) over no cells. Both are "no", for being incomplete.
    # Every squad a candidate has no whole season for is a drop: WINDOW loses
    # deep squad 3; ALT loses top squads 1 and 2 and deep squad 3.
    rows = (
        _renamed(_rows("top", *_flat({1: [1, 1, 1], 2: [1, 1, 1]})), "WINDOW")
        + [r for r in _rows("deep", *_flat({3: [0, 0, 0]})) if r["policy"] == "LEGACY"]
        + [{"group": "deep", "entry": 3, "gw": 2, "policy": "ALT", "points": 90, "ft_end": 1}]
    )
    result = gate(rows)

    assert len([k for k in result["cells"] if k[0] == "WINDOW"]) == 3
    assert not [k for k in result["cells"] if k[0] == "ALT"]
    assert result["verdict"] == {"ALT": "no", "WINDOW": "no"}
    assert result["reason"] == {"ALT": "incomplete", "WINDOW": "incomplete"}
    assert result["dropped"] == {("ALT", "deep"): 1, ("ALT", "top"): 2, ("WINDOW", "deep"): 1}
    assert result["pick"] is None


def test_no_top_group_means_no_pick():
    # Only the deep group was replayed. WINDOW is +1 a week there and ships
    # on its three cells, but the pick is read from the top group at credit
    # 1.5, and there is none: nothing is picked, and nothing raises.
    result = gate(_renamed(_rows("deep", *_flat({3: [1, 1, 1], 4: [1, 1, 1]})), "WINDOW"))
    assert result["verdict"] == {"WINDOW": "ship"}
    assert result["pick"] is None


# --- the summary ---------------------------------------------------------------


def test_the_summary_tabulates_the_cells_and_names_the_pick():
    gate_result = {
        "cells": {
            ("WINDOW", "top", 0.0): {"d": 2.5, "se": 1.3229},
            ("WINDOW", "top", 1.5): {"d": 3.25, "se": 0.6614},
            ("ALT", "top", 0.0): {"d": -1.0, "se": 4.3589},
        },
        "verdict": {"WINDOW": "ship", "ALT": "owner", "OTHER": "no"},
        "reason": {"OTHER": "incomplete"},
        "dropped": {("WINDOW", "top"): 3},
        "pick": "WINDOW",
    }
    text = render_summary(gate_result, {"squads": 150, "skipped": 2})

    assert "| WINDOW | top | 0.0 | +2.50 | 1.32 |" in text
    assert "| WINDOW | top | 1.5 | +3.25 | 0.66 |" in text
    assert "| ALT | top | 0.0 | -1.00 | 4.36 |" in text
    assert "WINDOW: ship" in text
    assert "ALT: owner" in text
    assert "OTHER: no (incomplete)" in text
    assert "WINDOW, top: 3 squads dropped as incomplete" in text
    assert "Pick: WINDOW" in text
    assert "squads: 150" in text
    assert "skipped: 2" in text
    assert "six clusters make the SE rough" in text


def test_the_summary_says_when_nothing_ships():
    text = render_summary({"cells": {}, "verdict": {"WINDOW": "no"}, "pick": None}, {})
    assert "Pick: none" in text
