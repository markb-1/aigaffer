"""Tests for the transfer MILP, on a hand-built 20-player universe.

Prices and projections are picked so that every optimum below can be worked
out with a pencil, and each test spells out the arithmetic it expects.

==  ===  ====  ====  =====  ======  =============================================
id  pos  club  cost     xP  status  role
==  ===  ====  ====  =====  ======  =============================================
1   GK   1       50   30.0  a       squad, starts
2   GK   2       40   10.0  a       squad, benched
3   DEF  1       60   28.0  a       squad, starts
4   DEF  2       55   26.0  a       squad, starts
5   DEF  3       50   24.0  a       squad, starts
6   DEF  4       45   22.0  a       squad, starts
7   DEF  5       40    8.0  a       squad, benched
8   MID  1       90   40.0  a       squad, starts
9   MID  2       80   36.0  a       squad, starts
10  MID  3       70   32.0  a       squad, starts
11  MID  4       45   20.0  a       squad, starts
12  MID  6       60    6.0  i       squad, benched — injured, still ownable
13  FWD  5       95   34.0  a       squad, starts
14  FWD  6       85   30.0  a       squad, starts
15  FWD  7       45    7.0  a       squad, benched
16  MID  7       60   50.0  a       the affordable upgrade (costs what 12 costs)
17  FWD  3      130   60.0  a       a big upgrade nobody can afford at bank 0
18  DEF  1      120  100.0  a       the best player alive, and a third club-1 man
19  GK   3       45    8.0  a       strictly worse than 2, and dearer
20  DEF  7       40   90.0  u       unavailable: a steal the pool must refuse
==  ===  ====  ====  =====  ======  =============================================

The squad is players 1-15, costing 910, lining up 4-4-2 as
``1 | 3 4 5 6 | 8 9 10 11 | 13 14`` — every alternative XI swaps a starter for
a cheaper-scoring bench man. Its xP is 322 started plus 0.1 x 31 benched =
325.1.

Two boards this table cannot draw build their own and say so where they
stand: ``four_upgrades`` adds three more cost-neutral signings, and
``flat_squad`` is a fifteen priced so that only its projections matter.
"""

from collections import Counter

import pytest

from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.optimizer import MAX_HITS, Week1Lock, candidate_pool, optimize

GK, DEF, MID, FWD = 1, 2, 3, 4

# id, position, club, cost, xP, status
UNIVERSE = [
    (1, GK, 1, 50, 30.0, "a"),
    (2, GK, 2, 40, 10.0, "a"),
    (3, DEF, 1, 60, 28.0, "a"),
    (4, DEF, 2, 55, 26.0, "a"),
    (5, DEF, 3, 50, 24.0, "a"),
    (6, DEF, 4, 45, 22.0, "a"),
    (7, DEF, 5, 40, 8.0, "a"),
    (8, MID, 1, 90, 40.0, "a"),
    (9, MID, 2, 80, 36.0, "a"),
    (10, MID, 3, 70, 32.0, "a"),
    (11, MID, 4, 45, 20.0, "a"),
    (12, MID, 6, 60, 6.0, "i"),
    (13, FWD, 5, 95, 34.0, "a"),
    (14, FWD, 6, 85, 30.0, "a"),
    (15, FWD, 7, 45, 7.0, "a"),
    (16, MID, 7, 60, 50.0, "a"),
    (17, FWD, 3, 130, 60.0, "a"),
    (18, DEF, 1, 120, 100.0, "a"),
    (19, GK, 3, 45, 8.0, "a"),
    (20, DEF, 7, 40, 90.0, "u"),
]

CURRENT = list(range(1, 16))
CURRENT_XI = [1, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14]
CURRENT_XP = 325.1


def _player(pid: int, position: int, club: int, cost: int, status: str) -> Player:
    return Player(
        id=pid,
        web_name=f"P{pid}",
        team=club,
        element_type=position,
        now_cost=cost,
        status=status,
        minutes=900,
        starts=10,
        total_points=0,
        bonus=0,
        saves=0,
    )


PLAYERS = {row[0]: _player(row[0], row[1], row[2], row[3], row[5]) for row in UNIVERSE}
XP = {
    row[0]: PlayerProjection(player_id=row[0], per_gw={2: row[4]}, total=row[4])
    for row in UNIVERSE
}


def universe(*extra: int) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The current squad plus ``extra``, so a test can starve the solver."""
    ids = set(CURRENT) | set(extra)
    return {i: PLAYERS[i] for i in ids}, {i: XP[i] for i in ids}


# Three more signings, each on a club of his own and each costing exactly what
# the squad man he replaces sells for: 21 for 7, 22 for 15, 23 for 2. With 16
# in for 12 that is four upgrades worth making and nothing else to buy.
UPGRADES = [
    (21, DEF, 8, 40, 60.0, "a"),
    (22, FWD, 9, 45, 55.0, "a"),
    (23, GK, 10, 40, 40.0, "a"),
]


def four_upgrades() -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The current squad, 16, and the three ``UPGRADES``."""
    players, xp = universe(16)
    for pid, position, club, cost, points, status in UPGRADES:
        players[pid] = _player(pid, position, club, cost, status)
        xp[pid] = PlayerProjection(player_id=pid, per_gw={2: points}, total=points)
    return players, xp


FLAT_SQUAD = list(range(101, 116))


def flat_squad(
    points: list[float],
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """A fifteen of its own — ids 101-115, two keepers, five defenders, five
    midfielders, three forwards — projected ``points`` in that order.

    Everyone costs 50 and comes three to a club, so nothing but the
    projections can decide the shape of the XI. The table above does not
    apply to these ids.
    """
    positions = [GK] * 2 + [DEF] * 5 + [MID] * 5 + [FWD] * 3
    players = {
        pid: _player(pid, position, (pid - FLAT_SQUAD[0]) // 3 + 1, 50, "a")
        for pid, position in zip(FLAT_SQUAD, positions)
    }
    xp = {
        pid: PlayerProjection(player_id=pid, per_gw={2: pts}, total=pts)
        for pid, pts in zip(FLAT_SQUAD, points)
    }
    return players, xp


def assert_legal(plan, budget: int) -> None:
    """Every plan must be a squad the FPL website would let you save."""
    assert len(plan.squad) == 15
    assert Counter(PLAYERS[p].element_type for p in plan.squad) == {
        GK: 2,
        DEF: 5,
        MID: 5,
        FWD: 3,
    }
    assert max(Counter(PLAYERS[p].team for p in plan.squad).values()) <= 3
    assert sum(PLAYERS[p].now_cost for p in plan.squad) <= budget

    assert len(plan.xi) == 11
    assert set(plan.xi) <= set(plan.squad)
    xi = Counter(PLAYERS[p].element_type for p in plan.xi)
    assert xi[GK] == 1
    assert xi[DEF] >= 3
    assert xi[FWD] >= 1


def test_forced_zero_transfers_keeps_the_current_squad():
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=1, forced_transfers=0)

    assert plan.squad == CURRENT
    assert plan.transfers_in == []
    assert plan.transfers_out == []
    assert plan.hits == 0
    assert plan.xi == CURRENT_XI
    assert plan.xp_total == pytest.approx(CURRENT_XP)
    assert plan.objective == pytest.approx(CURRENT_XP)
    assert_legal(plan, budget=910)


def test_one_transfer_takes_the_only_affordable_upgrade():
    # 16 (50.0) costs exactly what 12 (6.0) sells for, so at bank 0 he is the
    # one signing available: 17 and 18 cost more than any team-mate they could
    # replace. He starts and 11 drops to the bench, for 352 + 0.1 x 45 = 356.5.
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=1, forced_transfers=1)

    assert plan.transfers_in == [16]
    assert plan.transfers_out == [12]
    assert plan.xi == [1, 3, 4, 5, 6, 8, 9, 10, 13, 14, 16]
    assert plan.hits == 0
    assert plan.xp_total == pytest.approx(356.5)
    assert plan.objective == pytest.approx(356.5)
    assert_legal(plan, budget=910)


def test_transfer_without_a_free_one_costs_a_hit():
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=0, forced_transfers=1)

    assert plan.transfers_in == [16]
    assert plan.hits == 1
    assert isinstance(plan.hits, int)
    assert plan.xp_total == pytest.approx(356.5)
    assert plan.objective == pytest.approx(352.5)


def test_a_transfer_worth_more_than_its_hit_is_taken_unprompted():
    # Left to itself the solver still signs 16: +31.4 xP for a four-point hit.
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=0)

    assert plan.transfers_in == [16]
    assert plan.hits == 1
    assert plan.objective == pytest.approx(352.5)


def test_nothing_worth_buying_means_nothing_is_bought():
    # 19 is the only signing on offer and he is worse than the keeper he would
    # replace, so the plan rolls the transfer instead.
    players, xp = universe(19)
    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=1)

    assert plan.transfers_in == []
    assert plan.transfers_out == []
    assert plan.hits == 0
    assert plan.xp_total == pytest.approx(CURRENT_XP)


def test_upgrades_beyond_the_budget_are_not_selected():
    # 18 (100.0) and 17 (60.0) are the two best players outside the squad and
    # both are out of reach at bank 0: no defender sells for 120, no forward
    # for 130.
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=1, forced_transfers=1)

    assert 17 not in plan.squad
    assert 18 not in plan.squad
    assert plan.transfers_in == [16]


def test_a_transfer_nobody_can_afford_is_infeasible():
    players, xp = universe(17, 18)
    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=1, forced_transfers=1)

    assert plan is None


def test_the_club_limit_forces_selling_the_clubmate():
    # With 100 in the bank 18 (100.0) is affordable and worth having, but club
    # 1 already fields 1, 3 and 8. Selling the weakest defender (7, 8.0) would
    # be a fourth club-1 player, so 3 (28.0) is the one who has to go:
    # 394 started + 0.1 x 31 benched = 397.1.
    plan = optimize(PLAYERS, XP, CURRENT, bank=100, free_transfers=1, forced_transfers=1)

    assert plan.transfers_in == [18]
    assert plan.transfers_out == [3]
    assert Counter(PLAYERS[p].team for p in plan.squad)[1] == 3
    assert plan.xi == [1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 18]
    assert plan.xp_total == pytest.approx(397.1)
    assert_legal(plan, budget=1010)


def test_unavailable_players_are_never_candidates():
    # 20 is 90.0 for 40 and would walk into the side in place of 7 at no cost
    # at all, but he is flagged unavailable, so the pool never offers him.
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=1, forced_transfers=1)

    assert 20 not in plan.squad
    assert plan.transfers_in == [16]


def test_more_forced_transfers_than_the_pool_allows_returns_none():
    # One player outside the squad, two transfers demanded.
    players, xp = universe(16)
    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=2, forced_transfers=2)

    assert plan is None


def test_drafts_a_whole_squad_from_scratch():
    # 15 transfers, all free. Both 12 and 20 are unavailable and nobody owns
    # them, so the pool holds 5 midfielders exactly — all of them start or sit.
    # Club 1 caps at 3, so 18 (100.0) costs the side 3 (28.0); 17 (60.0) is
    # then 85 dearer than the budget can bear. XI 424 + 0.1 x 45 = 428.5.
    plan = optimize(PLAYERS, XP, [], bank=1000, free_transfers=15, forced_transfers=15)

    assert plan.squad == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18]
    assert plan.xi == [1, 4, 5, 6, 8, 9, 10, 13, 14, 16, 18]
    assert plan.transfers_in == plan.squad
    assert plan.transfers_out == []
    assert plan.hits == 0
    assert plan.xp_total == pytest.approx(428.5)
    assert plan.objective == pytest.approx(428.5)
    assert_legal(plan, budget=1000)


def test_a_fourth_transfer_worth_making_is_still_capped_at_three():
    # Four cost-neutral upgrades are on the table and two free transfers to
    # make them with. The best three — 21 for 7, 16 for 12 and 22 for 15 —
    # leave a squad worth 428.6 for one hit, and the keeper swap that would
    # follow is worth 12 more than the hit it costs. The cap takes it off the
    # table anyway: past three moves you are wildcarding.
    players, xp = four_upgrades()

    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=2)

    assert plan.transfers_in == [16, 21, 22]
    assert plan.transfers_out == [7, 12, 15]
    assert plan.hits == 1
    assert plan.xp_total == pytest.approx(428.6)
    assert plan.objective == pytest.approx(424.6)

    # And it really is the cap doing it: forced to four, the same board comes
    # back worth more than the plan the cap allowed — 440.6 for two hits.
    forced = optimize(
        players, xp, CURRENT, bank=0, free_transfers=2, forced_transfers=4
    )
    assert forced.objective == pytest.approx(432.6)
    assert forced.objective > plan.objective


def test_a_third_hit_is_never_worth_taking():
    # Three moves with nothing banked is -12 in one gameweek, past the -8 the
    # spec allows. It is not priced and rejected, it is not on the board: the
    # count comes back infeasible and the shortlist simply has one fewer plan.
    players, xp = four_upgrades()

    assert (
        optimize(players, xp, CURRENT, bank=0, free_transfers=0, forced_transfers=3)
        is None
    )


def test_two_hits_are_the_edge_and_still_allowed():
    players, xp = four_upgrades()

    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=0, forced_transfers=2)

    assert plan.hits == MAX_HITS
    assert len(plan.transfers_in) == 2


def test_fifteen_free_transfers_lift_the_cap():
    # A wildcard or free hit is valued by asking for the best squad reachable
    # in fifteen free moves, which is more than three: all four upgrades go
    # through, none of them costs a hit, and the squad is worth 440.6.
    players, xp = four_upgrades()

    plan = optimize(players, xp, CURRENT, bank=0, free_transfers=15)

    assert plan.transfers_in == [16, 21, 22, 23]
    assert plan.transfers_out == [2, 7, 12, 15]
    assert plan.hits == 0
    assert plan.xi == [3, 4, 8, 9, 10, 13, 14, 16, 21, 22, 23]
    assert plan.xp_total == pytest.approx(440.6)


def test_the_xi_fields_one_keeper_and_three_defenders_whatever_the_points_say():
    # Unconstrained, the best eleven here are both keepers, all five
    # midfielders, all three forwards and the one defender worth anything. A
    # team sheet cannot look like that: the second keeper (99) and two of the
    # forwards make way for the next two defenders, leaving 3-5-2.
    players, xp = flat_squad(
        [100.0, 99.0]
        + [10.0, 9.0, 8.0, 7.0, 6.0]
        + [96.0, 95.0, 94.0, 93.0, 92.0]
        + [91.0, 90.0, 89.0]
    )

    plan = optimize(
        players, xp, FLAT_SQUAD, bank=0, free_transfers=1, forced_transfers=0
    )

    assert plan.xi == [101, 103, 104, 105, 108, 109, 110, 111, 112, 113, 114]


def test_the_xi_always_fields_a_forward():
    # Here the eleven best are both keepers, four defenders and every
    # midfielder, with the forwards nowhere near. Somebody has to play up
    # front, so the second keeper drops out for a 5.0 forward: 4-5-1.
    players, xp = flat_squad(
        [100.0, 99.0]
        + [91.0, 90.0, 89.0, 88.0, 87.0]
        + [96.0, 95.0, 94.0, 93.0, 92.0]
        + [5.0, 4.0, 3.0]
    )

    plan = optimize(
        players, xp, FLAT_SQUAD, bank=0, free_transfers=1, forced_transfers=0
    )

    assert plan.xi == [101, 103, 104, 105, 106, 108, 109, 110, 111, 112, 113]


def test_candidate_pool_keeps_the_squad_and_the_best_available():
    pool = candidate_pool(PLAYERS, XP, CURRENT)

    assert set(CURRENT) <= set(pool)  # 12 is injured but still ours
    assert 20 not in pool  # unavailable and not ours
    assert {16, 17, 18, 19} <= set(pool)


def test_candidate_pool_is_capped_per_position():
    # 60 midfielders, projected 1.0 to 60.0, of whom we own the worst.
    players = {pid: _player(pid, MID, pid, 50, "a") for pid in range(1, 61)}
    xp = {
        pid: PlayerProjection(player_id=pid, per_gw={2: float(pid)}, total=float(pid))
        for pid in range(1, 61)
    }

    pool = candidate_pool(players, xp, [1])

    assert pool == [1] + list(range(21, 61))


# --- what a sale actually raises -------------------------------------------
#
# The budget used to price every sale at now_cost, which overestimates what a
# riser raises: FPL pays purchase plus half the rise, rounded down. The
# optimizer takes the true figures as ``selling_prices`` — buys stay at
# now_cost, holding a player costs nothing, and a plan the real money cannot
# fund is not offered.


def test_a_sale_is_priced_at_its_selling_price_not_the_market():
    # 12 rose since we bought him: he lists at 60 and sells for 55, so at bank
    # 0 the like-for-like swap 12 -> 16 (60) no longer adds up — 60 to spend
    # against 55 raised. The best swap the true money funds is selling 10
    # (70, sells at 70): XI 322 - 32 + 50 = 340, bench 31 as before, 343.1.
    players, xp = universe(16)

    at_market = optimize(players, xp, CURRENT, 0, 1, forced_transfers=1)
    plan = optimize(
        players, xp, CURRENT, 0, 1, forced_transfers=1, selling_prices={12: 55}
    )

    assert at_market.transfers_out == [12]  # the plan the app would refuse
    assert plan.transfers_in == [16]
    assert plan.transfers_out == [10]
    assert plan.xp_total == pytest.approx(343.1)


def test_a_plan_the_true_selling_prices_cannot_fund_is_not_offered():
    # Every squad member has fallen to a selling price of 40 and 16 costs 60:
    # no single sale funds him, so the forced one-transfer board that was
    # feasible at market prices is answered honestly — with nothing.
    players, xp = universe(16)
    fallen = {pid: 40 for pid in CURRENT}

    assert optimize(players, xp, CURRENT, 0, 1, forced_transfers=1) is not None
    assert (
        optimize(
            players, xp, CURRENT, 0, 1, forced_transfers=1, selling_prices=fallen
        )
        is None
    )


def test_holding_risers_costs_nothing_even_at_a_paper_loss():
    # Every man would sell below his listed price, and the plan sells nobody.
    # Money only moves when a player does — a formulation that charged the
    # squad its own paper losses would make rolling the transfer infeasible,
    # which is the regression this pins out.
    players, xp = universe()
    haircut = {pid: PLAYERS[pid].now_cost - 5 for pid in CURRENT}

    plan = optimize(
        players, xp, CURRENT, 0, 1, forced_transfers=0, selling_prices=haircut
    )

    assert plan is not None
    assert plan.squad == CURRENT
    assert plan.xp_total == pytest.approx(CURRENT_XP)


# The week-1 lock. Unlocked, the best move on this board is 16 in for the
# injured 12 (356.5). A recorded 12 — signed this week — cannot be sold, and
# a recorded sale of 16 cannot be bought back.


def test_a_recorded_signing_stays_in_the_squad():
    plan = optimize(
        PLAYERS, XP, CURRENT, bank=0, free_transfers=1,
        lock=Week1Lock(keep=frozenset({12})),
    )

    assert 12 in plan.squad and 12 not in plan.transfers_out


def test_a_recorded_sale_is_not_bought_back():
    plan = optimize(
        PLAYERS, XP, CURRENT, bank=0, free_transfers=1,
        lock=Week1Lock(shun=frozenset({16})),
    )

    assert 16 not in plan.squad


def test_an_empty_lock_changes_nothing():
    plan = optimize(PLAYERS, XP, CURRENT, bank=0, free_transfers=1, lock=Week1Lock())

    assert plan.transfers_in == [16] and plan.transfers_out == [12]
    assert plan.objective == pytest.approx(356.5)


# A recorded wildcard week: six_arrivals(opening=20) is a board worth six moves.
# Unlocked, three moves and two hits (177.4625 on the decayed total); free, all
# six and no hit.


def test_a_recorded_wildcard_lifts_the_cap_and_the_hits_in_the_single_week_solver():
    from tests.test_multiweek import SQUAD as BOARD, six_arrivals

    players, projections = six_arrivals([5, 6], opening=20.0)

    plain = optimize(players, projections, BOARD, bank=0, free_transfers=1)
    free = optimize(
        players, projections, BOARD, bank=0, free_transfers=1,
        lock=Week1Lock(free=True),
    )

    assert len(plain.transfers_in) == 3 and plain.hits == 2
    assert sorted(free.transfers_in) == [16, 17, 18, 19, 20, 21]
    assert free.hits == 0
    assert free.objective > plain.objective
