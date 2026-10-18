"""The transfers-skipped list: who is offered, what he replaces, and why he lost.

Boards are built by hand as the snapshot's player rows (only the keys the pass
reads), with every expected number worked in the comment beside it.
"""

import pytest

from aigaffer.detailed.skipped import skipped_transfers
from aigaffer.solver.optimizer import HIT_POINTS, LEGACY, WINDOW

DECAY = 0.85


def row(pos, club, t, pr=50, xm=85.0, sp=None):
    return {"p": pos, "c": club, "t": t, "pr": pr, "sp": sp, "xm": xm}


def plan(transfers_in, ft_kept=0):
    return {"in": list(transfers_in), "ft_kept": ft_kept}


def run(players, squad, plans, chosen=0, bank=100, ft=1, window=6, settings=WINDOW, xi=()):
    # xi defaults to nobody, so the bench tag stays out of the older tests; the
    # bench tests below name their starters.
    rows = skipped_transfers(
        players, squad, plans, chosen, bank, ft, DECAY, window, settings, xi=list(xi)
    )
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
        # 9 - 5 = 4.0 >= 0.2: the plan does better without him.
        14: row("MID", "ARS", 9.0, pr=90),
    }
    plans = {0: plan([]), 1: plan([10, 11, 12, 13, 14])}
    found, _ = run(players, [1, 2, 3, 4], plans, bank=5, ft=1)
    assert found[10]["tag"] == "unaffordable"
    assert found[11]["tag"] == "club limit"
    assert found[12]["tag"] == "minutes risk"
    assert found[13]["tag"] == "not worth a transfer"
    assert found[14]["tag"] == "the plan does better without him"
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
        (5.9, "the plan does better without him"),  # gap 0.9 >= 0.80344
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
    assert found[11]["tag"] == "the plan does better without him"
    # With two free transfers one is still free: threshold 0.2, both beat it.
    found, _ = run(players, [1], plans, ft=2)
    assert found[10]["tag"] == "the plan does better without him"


def test_a_man_no_better_than_the_one_he_replaces_is_dropped_unless_a_plan_bought_him():
    # The lone squad MID 1 has t 5.0. Candidates, all in the top three:
    #   10 at 4.0: gap -1.0, nobody bought him — dropped;
    #   11 at 5.0: gap  0.0, nobody bought him — dropped (no gain is no case);
    #   12 at 4.5: gap -0.5, but plan 1 bought him — kept, since a plan the
    #              gaffer weighed thought him worth having;
    # and 13 at 6.0, gap +1.0, is kept as ever.
    players = {
        1: row("MID", "AAA", 5.0),
        10: row("MID", "BBB", 4.0),
        11: row("MID", "CCC", 5.0),
        12: row("MID", "DDD", 4.5),
        13: row("MID", "EEE", 6.0),
    }
    plans = {0: plan([]), 1: plan([12])}
    found, rows = run(players, [1], plans)
    # Top three by total: 13 (6.0), 11 (5.0), 12 (4.5); 10 is fourth and only
    # offered at all because no one bought him — he is not.
    assert [r["id"] for r in rows] == [13, 12]
    assert found[12]["gap"] == -0.5
    # The chosen plan's own buys are never offered, so "another plan" means a
    # plan other than the chosen one: chosen 1 buying 12 leaves him out.
    _, rows = run(players, [1], plans, chosen=1)
    assert [r["id"] for r in rows] == [13]


# --- tag order: a player who meets two conditions gets the earlier tag -------


def test_club_limit_beats_minutes_risk():
    # Squad MIDs: 1 (ARS, 5.0) is the man out; 2, 3, 4 are CHE (3 of them).
    # Candidate 10 is CHE with 30 minutes (< 60) and a price inside the budget
    # (50 <= 100 + 50): both club limit and minutes risk apply; club limit is
    # first in the order.
    players = {
        1: row("MID", "ARS", 5.0),
        2: row("MID", "CHE", 6.0),
        3: row("MID", "CHE", 7.0),
        4: row("MID", "CHE", 8.0),
        10: row("MID", "CHE", 9.0, xm=30.0),
    }
    found, _ = run(players, [1, 2, 3, 4], {0: plan([]), 1: plan([])})
    assert found[10]["tag"] == "club limit"


def test_minutes_risk_beats_not_worth_a_transfer():
    # Threshold 0.2 (nothing bought, one free move). Candidate 10: gap
    # 5.1 - 5.0 = 0.1 < 0.2 (not worth) and 50 minutes < 60 (minutes risk).
    # Minutes risk is first.
    players = {1: row("MID", "ARS", 5.0), 10: row("MID", "LEE", 5.1, xm=50.0)}
    found, _ = run(players, [1], {0: plan([]), 1: plan([])})
    assert found[10]["tag"] == "minutes risk"


def test_club_limit_needs_three_squad_mates_and_a_replaced_man_from_elsewhere():
    # Case A: CHE has three squad MIDs (2, 3, 4); the man out is 1 (ARS, 5.0),
    # not CHE. Candidate 10 (CHE, 9.0) would make four: club limit.
    a = {
        1: row("MID", "ARS", 5.0),
        2: row("MID", "CHE", 6.0),
        3: row("MID", "CHE", 7.0),
        4: row("MID", "CHE", 8.0),
        10: row("MID", "CHE", 9.0),
    }
    found, _ = run(a, [1, 2, 3, 4], {0: plan([]), 1: plan([])})
    assert found[10]["tag"] == "club limit"
    # Case B: same club, but the man out is a CHE man (2 at 1.0, lowest): the
    # swap keeps CHE at three, so no club limit. Gap 9.0 - 1.0 = 8.0 >= 0.2.
    b = dict(a)
    b[2] = row("MID", "CHE", 1.0)
    found, _ = run(b, [1, 2, 3, 4], {0: plan([]), 1: plan([])})
    assert found[10]["replaces"] == 2
    assert found[10]["tag"] == "the plan does better without him"
    # Case C: only two CHE in the squad (1 ARS 5.0; 2, 3 CHE): the count is
    # 2, not 3, so no club limit. Gap 9 - 5 = 4.0: beaten.
    c = {k: v for k, v in a.items() if k != 4}
    found, _ = run(c, [1, 2, 3], {0: plan([]), 1: plan([])})
    assert found[10]["tag"] == "the plan does better without him"


# --- the threshold, branch by branch ------------------------------------------


def _tag_at(gap, plans, ft, settings=WINDOW, window=6):
    """The tag for one candidate whose total is 5.0 + gap against a lone
    squad MID at 5.0 (the chosen plan's own buy, 30, is out of the pool)."""
    players = {
        1: row("MID", "AAA", 5.0),
        10: row("MID", "BBB", round(5.0 + gap, 2)),
        30: row("MID", "CCC", 99.0),
    }
    found, _ = run(players, [1], plans, ft=ft, window=window, settings=settings)
    return found[10]["tag"]


NOT, BEAT = "not worth a transfer", "the plan does better without him"


def test_two_kept_free_transfers_lose_the_second_step():
    # ft_values[1] = 2.0; 0.85**6 = 0.3771495; 2.0 * 0.3771495 = 0.754299;
    # + 0.2 = 0.954299. Gap 0.9 < it; gap 1.0 >= it.
    plans = {0: plan([30], ft_kept=2), 1: plan([])}
    assert _tag_at(0.9, plans, ft=1) == NOT
    assert _tag_at(1.0, plans, ft=1) == BEAT


def test_one_kept_with_a_free_move_left_is_just_the_penalty():
    # ft_kept 1 -> not the step branch. One bought, two free: one left, so
    # base 0; threshold 0 + 0.2. Gap 0.1 < 0.2; gap 0.3 >= 0.2.
    plans = {0: plan([30], ft_kept=1), 1: plan([])}
    assert _tag_at(0.1, plans, ft=2) == NOT
    assert _tag_at(0.3, plans, ft=2) == BEAT


def test_one_kept_with_no_free_move_left_is_a_hit():
    # One bought, one free: none left. Threshold HIT_POINTS + 0.2 = 4.2.
    plans = {0: plan([30], ft_kept=1), 1: plan([])}
    assert _tag_at(4.1, plans, ft=1) == NOT
    assert _tag_at(4.3, plans, ft=1) == BEAT


def test_legacy_has_no_step_and_no_penalty():
    # LEGACY: ft_values () and penalty 0, so even ft_kept 3 loses no step.
    # Free move left (nothing bought, one free): threshold 0. Gap -0.1 < 0
    # is not worth; gap 0.0 is not < 0, so beaten. A gap of zero or less is
    # only offered at all when another plan bought him, so plan 1 does here.
    free = {0: plan([], ft_kept=3), 1: plan([10])}
    assert _tag_at(-0.1, free, ft=1, settings=LEGACY) == NOT
    assert _tag_at(0.0, free, ft=1, settings=LEGACY) == BEAT
    # No free move left: threshold HIT_POINTS = 4 exactly. 3.9 < 4; 4.0 is not.
    spent = {0: plan([30], ft_kept=3), 1: plan([])}
    assert _tag_at(3.9, spent, ft=1, settings=LEGACY) == NOT
    assert _tag_at(4.0, spent, ft=1, settings=LEGACY) == BEAT


def test_a_kept_count_past_the_table_takes_its_last_step():
    # ft_kept 7 against five values: the last, 1.1. 1.1 * 0.3771495 =
    # 0.414864; + 0.2 = 0.614864. Gap 0.6 < it; gap 0.7 >= it.
    plans = {0: plan([30], ft_kept=7), 1: plan([])}
    assert _tag_at(0.6, plans, ft=1) == NOT
    assert _tag_at(0.7, plans, ft=1) == BEAT


def test_an_unstated_free_transfer_count_assumes_no_hit():
    # ft None, one bought, ft_kept 0: no hit assumed, threshold 0.2. Gap 0.3
    # clears it (a hit threshold, 4.2, would not have).
    plans = {0: plan([30]), 1: plan([])}
    assert _tag_at(0.3, plans, ft=None) == BEAT


# --- who is replaced, and who is not offered ----------------------------------


@pytest.mark.parametrize("pos", ["GKP", "DEF", "FWD"])
def test_the_man_replaced_is_the_lowest_total_at_the_position(pos):
    # Squad at the position, listed 7 (4.0), 3 (2.0), 9 (3.0): lowest is 3.
    players = {
        7: row(pos, "AAA", 4.0),
        3: row(pos, "BBB", 2.0),
        9: row(pos, "CCC", 3.0),
        50: row("MID", "DDD", 1.0),  # another position, lower still: ignored
        10: row(pos, "EEE", 6.0),
    }
    found, _ = run(players, [7, 3, 9, 50], {0: plan([]), 1: plan([])})
    # Gap 6.0 - 2.0 = 4.0.
    assert found[10]["replaces"] == 3 and found[10]["gap"] == 4.0


def test_equal_totals_replace_the_lower_id():
    # 8 and 4 both 2.0; the lower id, 4, is replaced.
    players = {
        8: row("DEF", "AAA", 2.0),
        4: row("DEF", "BBB", 2.0),
        10: row("DEF", "CCC", 5.0),
    }
    found, _ = run(players, [8, 4], {0: plan([]), 1: plan([])})
    assert found[10]["replaces"] == 4


def test_an_owned_player_is_never_offered_however_good():
    # Owned MID 2 has the highest total (50) and would lead the top three.
    players = {
        1: row("MID", "AAA", 5.0),
        2: row("MID", "BBB", 50.0),
        10: row("MID", "CCC", 9.0),
    }
    _, rows = run(players, [1, 2], {0: plan([]), 1: plan([])})
    assert [r["id"] for r in rows] == [10]


# --- would sit on the bench ---------------------------------------------------


def test_a_keeper_between_the_backup_and_the_starter_would_sit_on_the_bench():
    # Squad GKs: 1 (starter, 6.0, in the XI) and 2 (backup, 3.0, benched).
    # The man replaced is the lowest, 2 (3.0), so candidate 10 at 5.0 reads as
    # a gap of +2.0; but the weakest XI keeper is 1 at 6.0, and 5.0 < 6.0, so
    # he would sit behind the starter.
    players = {
        1: row("GKP", "AAA", 6.0),
        2: row("GKP", "BBB", 3.0),
        10: row("GKP", "CCC", 5.0),  # gap 2.0 >= 0.2, minutes fine: only the bench applies
    }
    found, _ = run(players, [1, 2], {0: plan([]), 1: plan([])}, xi=[1])
    assert found[10]["tag"] == "would sit on the bench"


def test_a_man_better_than_the_weakest_starter_is_not_a_bench_man():
    # Same keepers, candidate 10 at 6.5 > 6.0 (the starter): he would start.
    players = {
        1: row("GKP", "AAA", 6.0),
        2: row("GKP", "BBB", 3.0),
        10: row("GKP", "CCC", 6.5),
    }
    found, _ = run(players, [1, 2], {0: plan([]), 1: plan([])}, xi=[1])
    assert found[10]["tag"] == "the plan does better without him"
    # A man exactly level with the weakest starter is not below him.
    players[10] = row("GKP", "CCC", 6.0)
    found, _ = run(players, [1, 2], {0: plan([]), 1: plan([])}, xi=[1])
    assert found[10]["tag"] == "the plan does better without him"


def test_the_bench_comparison_is_against_starters_at_his_own_position():
    # Candidate 10 is a DEF at 4.0. The XI's weakest DEF is 3 at 5.0, the XI's
    # MID 4 is at 1.0 and must not count: 4.0 < 5.0, bench.
    players = {
        3: row("DEF", "AAA", 5.0),
        5: row("DEF", "BBB", 1.0),  # benched DEF, not in the XI
        4: row("MID", "CCC", 1.0),
        10: row("DEF", "DDD", 4.0),
    }
    found, _ = run(players, [3, 5, 4], {0: plan([]), 1: plan([])}, xi=[3, 4])
    assert found[10]["tag"] == "would sit on the bench"


def test_minutes_risk_beats_the_bench_tag():
    # Candidate 10: 50 minutes (risk) and 5.0 < the starter's 6.0 (bench).
    players = {
        1: row("GKP", "AAA", 6.0),
        2: row("GKP", "BBB", 3.0),
        10: row("GKP", "CCC", 5.0, xm=50.0),
    }
    found, _ = run(players, [1, 2], {0: plan([]), 1: plan([])}, xi=[1])
    assert found[10]["tag"] == "minutes risk"


def test_the_bench_tag_beats_not_worth_a_transfer():
    # Candidate 10: 3.1 against the replaced backup's 3.0, gap 0.1 < 0.2 (not
    # worth), and 3.1 < the starter's 6.0 (bench). Bench is first.
    players = {
        1: row("GKP", "AAA", 6.0),
        2: row("GKP", "BBB", 3.0),
        10: row("GKP", "CCC", 3.1),
    }
    found, _ = run(players, [1, 2], {0: plan([]), 1: plan([])}, xi=[1])
    assert found[10]["tag"] == "would sit on the bench"
