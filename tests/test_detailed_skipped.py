"""The transfers-skipped list: who is offered, what he replaces, and why he lost.

Boards are built by hand as the snapshot's player rows (only the keys the pass
reads), with every expected number worked in the comment beside it.
"""

import pytest

from aigaffer.detailed.skipped import skipped_transfers
from aigaffer.solver.optimizer import HIT_POINTS, WINDOW

DECAY = 0.85


def row(pos, club, t, pr=50, xm=85.0, sp=None):
    return {"p": pos, "c": club, "t": t, "pr": pr, "sp": sp, "xm": xm}


def plan(transfers_in, ft_kept=0):
    return {"in": list(transfers_in), "ft_kept": ft_kept}


def run(players, squad, plans, chosen=0, bank=100, ft=1, window=6):
    rows = skipped_transfers(players, squad, plans, chosen, bank, ft, DECAY, window, WINDOW)
    return {r["id"]: r for r in rows}, rows


def test_the_top_three_per_position_are_offered_minus_owned_and_bought():
    players = {
        2: row("DEF", "AAA", 3.0),  # owned DEF, the lowest at his position
        20: row("DEF", "BBB", 4.0),
        1: row("MID", "AAA", 5.0),  # owned MID
        10: row("MID", "CCC", 9.0),
        11: row("MID", "DDD", 8.0),
        12: row("MID", "EEE", 7.0),
        13: row("MID", "FFF", 6.0),  # fourth best: cut
        14: row("MID", "GGG", 20.0),  # best, but the chosen plan buys him
    }
    plans = {0: plan([14]), 1: plan([])}
    found, rows = run(players, [2, 1], plans)
    # Position order GKP, DEF, MID, FWD, then total descending: 20, 10, 11, 12.
    assert [r["id"] for r in rows] == [20, 10, 11, 12]
    # DEF 20 replaces the only DEF (2): 4.0 - 3.0 = 1.0.
    assert found[20]["replaces"] == 2 and found[20]["gap"] == 1.0
    # MIDs replace 1 (5.0): 9-5 = 4.0, 8-5 = 3.0, 7-5 = 2.0.
    assert [found[i]["gap"] for i in (10, 11, 12)] == [4.0, 3.0, 2.0]
    assert all(found[i]["replaces"] == 1 for i in (10, 11, 12))
    # No sale price recorded for the man out, so his price (50) stands in:
    # 50 - 50 = 0.
    assert found[10]["price_diff"] == 0 and found[10]["pos"] == "MID"


def test_a_player_another_plan_bought_is_offered():
    players = {
        1: row("MID", "AAA", 5.0),
        10: row("MID", "BBB", 9.0),
        11: row("MID", "CCC", 8.0),
        12: row("MID", "DDD", 7.0),
        13: row("MID", "EEE", 6.5),  # fourth in the position; plan 1 buys him
        2: row("DEF", "FFF", 3.0),
    }
    plans = {0: plan([]), 1: plan([13, 10])}
    _, rows = run(players, [1, 2], plans)
    # Top three 10, 11, 12 first; 13 follows from plan 1; 10 is not repeated.
    assert [r["id"] for r in rows] == [10, 11, 12, 13]
    # 13: 6.5 - 5.0 = 1.5.
    assert rows[3]["gap"] == 1.5


def test_tags_in_order():
    # Squad MIDs: 1 (ARS, 5.0, price 100, sold at 90) and 2, 3, 4 (CHE).
    # The man out is the lowest total: 1. With bank 5 the budget is 5 + 90 = 95.
    # Nothing is bought and one free move is left, so the threshold is 0.2.
    players = {
        1: row("MID", "ARS", 5.0, pr=100, sp=90),
        2: row("MID", "CHE", 6.0),
        3: row("MID", "CHE", 7.0),
        4: row("MID", "CHE", 8.0),
        # 96 > 95, and 30 minutes is a risk too: unaffordable wins.
        10: row("MID", "LEE", 9.0, pr=96, xm=30.0),
        # Three CHE in the squad, 1 is not CHE: club limit (price 90 <= 95).
        11: row("MID", "CHE", 9.0, pr=90),
        # 50 minutes < 60: minutes risk.
        12: row("MID", "ARS", 9.0, pr=90, xm=50.0),
        # 5.1 - 5.0 = 0.1 < 0.2: not worth a transfer.
        13: row("MID", "ARS", 5.1, pr=90),
        # 9 - 5 = 4.0 >= 0.2: beaten by the chosen move.
        14: row("MID", "ARS", 9.0, pr=90),
    }
    plans = {0: plan([]), 1: plan([10, 11, 12, 13, 14])}
    found, _ = run(players, [1, 2, 3, 4], plans, bank=5, ft=1)
    assert found[10]["tag"] == "unaffordable"
    assert found[11]["tag"] == "club limit"
    assert found[12]["tag"] == "minutes risk"
    assert found[13]["tag"] == "not worth a transfer"
    assert found[14]["tag"] == "beaten by the chosen move"
    # price_diff is against the sale price: 96 - 90 = 6.
    assert found[10]["price_diff"] == 6
    # A man from the replaced man's own club is no club-limit problem: swap
    # ARS for ARS keeps the count at three CHE regardless; 12 and 13 above
    # are ARS and passed that rule.


@pytest.mark.parametrize(
    "gap_total, tag",
    [
        # Lowest squad MID has t 5.0. ft_kept 3, window 6 under WINDOW:
        # ft_values[2] = 1.6; 0.85**6 = 0.377150; 1.6 * 0.377150 = 0.603440;
        # + ft_use_penalty 0.2 = 0.80344.
        (5.7, "not worth a transfer"),  # gap 0.7 < 0.80344
        (5.9, "beaten by the chosen move"),  # gap 0.9 >= 0.80344
    ],
)
def test_not_worth_a_transfer_uses_the_lost_ft_step(gap_total, tag):
    players = {
        1: row("MID", "AAA", 5.0),
        10: row("MID", "BBB", gap_total),
        30: row("MID", "CCC", 40.0),  # the chosen plan's buy
    }
    plans = {0: plan([30], ft_kept=3), 1: plan([])}
    # A hit-sized threshold would be wrong here: ft_kept >= 2 takes the step.
    found, _ = run(players, [1], plans, ft=1, window=6)
    assert found[10]["tag"] == tag


def test_with_no_free_move_left_the_threshold_is_a_hit():
    players = {
        1: row("MID", "AAA", 5.0),
        10: row("MID", "BBB", 9.0),  # gap 4.0
        11: row("MID", "CCC", 9.5),  # gap 4.5
        30: row("MID", "DDD", 40.0),
    }
    plans = {0: plan([30]), 1: plan([])}
    # One free transfer, one bought: none left, so the threshold is
    # HIT_POINTS + 0.2 = 4.2. 4.0 < 4.2 is not worth it; 4.5 clears it.
    found, _ = run(players, [1], plans, ft=1)
    assert HIT_POINTS == 4
    assert found[10]["tag"] == "not worth a transfer"
    assert found[11]["tag"] == "beaten by the chosen move"
    # With two free transfers one is still free: threshold 0.2, both beat it.
    found, _ = run(players, [1], plans, ft=2)
    assert found[10]["tag"] == "beaten by the chosen move"
