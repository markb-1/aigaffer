"""Tests for the team sheet and the chip panel.

Projections here are next-gameweek points, not the decayed horizon the
optimizer plans against, and they are picked so every answer below can be
worked out with a pencil.

The squad is a legal fifteen, positions by id — 1-2 keepers, 3-7 defenders,
8-12 midfielders, 13-15 forwards:

==  ===  =====  ======================================================
id  pos  gw xP  role
==  ===  =====  ======================================================
1   GK     5.0  starts
2   GK     0.2  benched, and still first on the bench
3   DEF    6.0  starts
4   DEF    5.5  starts
5   DEF    5.0  starts
6   DEF    1.0  benched
7   DEF    0.5  benched
8   MID    8.0  starts, and captains
9   MID    7.5  starts, and is vice-captain
10  MID    7.0  starts
11  MID    6.5  starts
12  MID    1.5  benched
13  FWD    7.2  starts
14  FWD    6.8  starts
15  FWD    6.4  starts
==  ===  =====  ======================================================

The best eleven is 3-4-3 — 5.0 + 16.5 + 29.0 + 20.4 = 70.9 — because the
fourth defender (1.0) is worth less than the third forward (6.4).
"""

import pytest

from aigaffer.model.xp import PlayerProjection
from aigaffer.solver import lineup as lineup_module
from aigaffer.solver.lineup import (
    FORMATIONS,
    ChipEvs,
    Lineup,
    attacking_evs,
    chip_evs,
    pick_lineup,
)
from aigaffer.solver.optimizer import Plan

GK, DEF, MID, FWD = 1, 2, 3, 4

SQUAD = list(range(1, 16))
POSITIONS = dict(zip(SQUAD, [GK] * 2 + [DEF] * 5 + [MID] * 5 + [FWD] * 3))
GW_XP = dict(
    zip(
        SQUAD,
        [5.0, 0.2, 6.0, 5.5, 5.0, 1.0, 0.5, 8.0, 7.5, 7.0, 6.5, 1.5, 7.2, 6.8, 6.4],
    )
)

XI = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
XI_XP = 70.9
LINEUP = Lineup(xi=XI, captain=8, vice=9, bench=[2, 12, 6, 7])

NEXT_EVENT = 2

# Everyone the solver may pick from next gameweek: our fifteen and one more
# to sign.
BOARD_GW_XP = {**GW_XP, 16: 9.0}

# The horizon projections the plan was made against: next gameweek as above
# and the one after worth half as much.
XP = {
    pid: PlayerProjection(
        player_id=pid,
        per_gw={NEXT_EVENT: points, NEXT_EVENT + 1: points / 2},
        total=points * 1.5,
    )
    for pid, points in BOARD_GW_XP.items()
}
PLAYERS = {"players": "stand-in"}

# The plan's own eleven was chosen over six gameweeks, so it is not the eleven
# to field on Saturday: it starts 6 and 12 where the team sheet starts 11 and
# 15, and is worth 60.5 next gameweek against the team sheet's 70.9.
CURRENT_PLAN = Plan(
    squad=SQUAD,
    xi=[1, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14],
    transfers_in=[],
    transfers_out=[],
    hits=0,
    xp_total=85.0,
    objective=85.0,
)


def canned(xp_total: float) -> Plan:
    """A rebuilt squad; only its ``xp_total`` is ever read."""
    return Plan(
        squad=SQUAD,
        xi=[],
        transfers_in=[],
        transfers_out=[],
        hits=0,
        xp_total=xp_total,
        objective=xp_total,
    )


def stub_optimize(
    monkeypatch, free_hit: Plan | None, wildcard: Plan | None
) -> list[dict]:
    """Answer the two chip boards; record the calls.

    The wildcard is solved on the horizon projections it was handed, so the
    call carrying ``XP`` itself is the wildcard and the other is the free hit.
    """
    calls: list[dict] = []

    def fake_optimize(
        players, xp, current_squad, bank, free_transfers, forced_transfers=None
    ):
        calls.append(
            {
                "players": players,
                "xp": xp,
                "current_squad": current_squad,
                "bank": bank,
                "free_transfers": free_transfers,
                "forced_transfers": forced_transfers,
            }
        )
        return wildcard if xp is XP else free_hit

    monkeypatch.setattr(lineup_module, "optimize", fake_optimize)
    return calls


def evs(monkeypatch, free_hit: Plan | None, wildcard: Plan | None) -> ChipEvs:
    """``chip_evs`` over the table above, with the optimizer stubbed."""
    stub_optimize(monkeypatch, free_hit, wildcard)
    return chip_evs(
        CURRENT_PLAN, LINEUP, GW_XP, PLAYERS, XP, bank=25, next_event=NEXT_EVENT
    )


def test_the_eight_legal_formations_are_enumerated():
    # One keeper, three to five defenders, two to five midfielders, one to
    # three forwards, eleven in all.
    assert set(FORMATIONS) == {
        (1, 3, 4, 3),
        (1, 3, 5, 2),
        (1, 4, 3, 3),
        (1, 4, 4, 2),
        (1, 4, 5, 1),
        (1, 5, 2, 3),
        (1, 5, 3, 2),
        (1, 5, 4, 1),
    }
    assert len(FORMATIONS) == 8


def test_the_best_eleven_can_be_three_at_the_back():
    lineup = pick_lineup(SQUAD, POSITIONS, GW_XP)

    assert lineup.xi == XI
    assert sum(GW_XP[pid] for pid in lineup.xi) == pytest.approx(XI_XP)


def test_five_defenders_when_that_is_where_the_points_are():
    # The five best outfielders are all defenders, so the side is 5-3-2: the
    # fourth midfielder (0.9) is worth less than the second forward (2.0).
    gw_xp = dict(
        zip(
            SQUAD,
            [4.0, 0.5, 9.0, 8.5, 8.0, 7.5, 7.0, 6.0, 5.5, 1.2, 0.9, 0.8, 5.0, 2.0, 0.7],
        )
    )

    lineup = pick_lineup(SQUAD, POSITIONS, gw_xp)

    assert lineup.xi == [1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14]


def test_the_two_best_attackers_take_the_armbands():
    # Here the two best starters (8, 9) are also the two best attackers, so the
    # armbands land on them; the tests below pull the two apart.
    lineup = pick_lineup(SQUAD, POSITIONS, GW_XP)

    assert lineup.captain == 8
    assert lineup.vice == 9
    assert POSITIONS[lineup.captain] in (MID, FWD)
    assert POSITIONS[lineup.vice] in (MID, FWD)


def test_the_captain_is_an_attacker_not_the_defender_who_tops_the_sheet():
    # A nailed defender's steady floor can top the raw sheet early in a season,
    # but the armband is for ceiling — a doubled clean sheet is not a haul — so
    # it goes to the best attacker. Defender 3 tops the board at 20.0; forward
    # 13 is the best midfielder-or-forward and takes it, with midfielder 8 vice.
    gw_xp = {**GW_XP, 3: 20.0, 13: 9.5}
    lineup = pick_lineup(SQUAD, POSITIONS, gw_xp)

    assert lineup.captain == 13 and POSITIONS[13] == FWD
    assert lineup.vice == 8 and POSITIONS[8] == MID
    assert POSITIONS[lineup.captain] in (MID, FWD)
    assert POSITIONS[lineup.vice] in (MID, FWD)
    assert 3 in lineup.xi  # the defender still starts; he just does not captain


def test_tied_attackers_take_the_armbands_by_id():
    # Two attackers level at the top: the tie breaks by id so the choice never
    # wobbles between runs.
    tie = {**GW_XP, 8: 9.0, 13: 9.0}
    lineup = pick_lineup(SQUAD, POSITIONS, tie)

    assert lineup.captain == 8 and lineup.vice == 13


def test_the_armbands_do_not_depend_on_the_squad_order():
    # Same fifteen, opposite input order, same captain and vice.
    forward = pick_lineup(SQUAD, POSITIONS, GW_XP)
    backward = pick_lineup(list(reversed(SQUAD)), POSITIONS, GW_XP)

    assert (forward.captain, forward.vice) == (backward.captain, backward.vice)


# --- the armband is chosen on the ceiling, not the total -------------------


def test_the_armband_prefers_the_goal_threat_over_the_padded_total():
    # The Haaland-vs-cheap-mid case in miniature. Midfielder 8 has the highest
    # TOTAL projection (8.0), padded by appearance, a kind fixture and defensive
    # points. Forward 13 totals less (7.2) but is the real goal threat — the
    # higher goals-and-assists EV — so he takes the armband, with midfielder 9
    # (next on attacking EV) as vice. Captaining on the total would have picked 8.
    attacking = {pid: 1.0 for pid in SQUAD}
    attacking[13] = 6.5  # the forward's ceiling
    attacking[9] = 5.0  # the next-best threat
    attacking[8] = 2.0  # top total, floor-padded ceiling
    lineup = pick_lineup(SQUAD, POSITIONS, GW_XP, attacking)

    assert lineup.captain == 13 and POSITIONS[lineup.captain] == FWD
    assert lineup.vice == 9 and POSITIONS[lineup.vice] == MID
    assert POSITIONS[lineup.captain] in (MID, FWD)
    assert POSITIONS[lineup.vice] in (MID, FWD)


def test_the_attacking_armband_is_deterministic():
    # Distinct attacking figures, opposite input orders: same choice both ways.
    attacking = {pid: float(pid) for pid in SQUAD}
    a = pick_lineup(SQUAD, POSITIONS, GW_XP, attacking)
    b = pick_lineup(list(reversed(SQUAD)), POSITIONS, GW_XP, attacking)

    assert (a.captain, a.vice) == (b.captain, b.vice)
    assert POSITIONS[a.captain] in (MID, FWD) and POSITIONS[a.vice] in (MID, FWD)


def test_without_attacking_ev_the_armband_falls_back_to_the_total():
    # No ceiling figures supplied: the armband is chosen on total gw_xp, which
    # is what a set of hand-built projections with no attacking slice gets.
    lineup = pick_lineup(SQUAD, POSITIONS, GW_XP)  # attacking_ev defaults to None

    assert lineup.captain == 8 and lineup.vice == 9


def test_attacking_evs_reads_the_projection_slice_or_says_none():
    event = NEXT_EVENT
    projections = {
        1: PlayerProjection(
            player_id=1, per_gw={event: 5.0}, total=7.5, attacking_per_gw={event: 3.0}
        ),
        2: PlayerProjection(
            player_id=2, per_gw={event: 4.0}, total=6.0, attacking_per_gw={event: 1.0}
        ),
    }
    assert attacking_evs(projections, event) == {1: 3.0, 2: 1.0}

    # A hand-built projection carries no attacking slice and so makes no ceiling
    # claim: the helper says None, and pick_lineup falls back to the total.
    bare = {1: PlayerProjection(player_id=1, per_gw={event: 5.0}, total=7.5)}
    assert attacking_evs(bare, event) is None


def test_the_bench_starts_with_the_keeper_then_ranks_by_points():
    # 2 is the worst player in the squad and still first on the bench: a
    # keeper can only come on for a keeper.
    lineup = pick_lineup(SQUAD, POSITIONS, GW_XP)

    assert lineup.bench == [2, 12, 6, 7]
    assert set(lineup.bench) == set(SQUAD) - set(lineup.xi)


def test_bench_boost_is_what_the_bench_would_add(monkeypatch):
    # 0.2 + 1.5 + 1.0 + 0.5, the four who otherwise score nothing.
    chips = evs(monkeypatch, canned(90.0), canned(100.0))

    assert chips.bench_boost == pytest.approx(3.2)


def test_triple_captain_is_one_more_helping_of_the_captain(monkeypatch):
    # The armband already doubles 8, so the chip adds his 8.0 once more.
    chips = evs(monkeypatch, canned(90.0), canned(100.0))

    assert chips.triple_captain == pytest.approx(8.0)


def test_free_hit_is_the_best_one_week_squad_less_the_eleven_we_would_field(
    monkeypatch,
):
    # 90.0 for a squad picked on next gameweek alone, against the 70.9 our own
    # eleven is worth — not the 85.0 the plan scores over the horizon, nor the
    # eleven the plan named.
    chips = evs(monkeypatch, canned(90.0), canned(100.0))

    assert chips.free_hit == pytest.approx(90.0 - XI_XP)


def test_wildcard_is_the_best_horizon_squad_less_the_plan_we_have(monkeypatch):
    chips = evs(monkeypatch, canned(90.0), canned(100.0))

    assert chips.wildcard == pytest.approx(100.0 - 85.0)


def test_the_free_hit_board_is_scored_on_next_gameweek_alone(monkeypatch):
    # A free hit lasts a week, so every player is worth exactly his next
    # gameweek and the rest of the horizon is thrown away.
    calls = stub_optimize(monkeypatch, canned(90.0), canned(100.0))

    chip_evs(CURRENT_PLAN, LINEUP, GW_XP, PLAYERS, XP, bank=25, next_event=NEXT_EVENT)

    free_hit = next(call for call in calls if call["xp"] is not XP)
    assert free_hit["xp"] == {
        pid: PlayerProjection(player_id=pid, per_gw={NEXT_EVENT: points}, total=points)
        for pid, points in BOARD_GW_XP.items()
    }


def test_both_chips_rebuild_the_squad_with_fifteen_free_transfers(monkeypatch):
    calls = stub_optimize(monkeypatch, canned(90.0), canned(100.0))

    chip_evs(CURRENT_PLAN, LINEUP, GW_XP, PLAYERS, XP, bank=25, next_event=NEXT_EVENT)

    assert len(calls) == 2
    assert all(call["players"] is PLAYERS for call in calls)
    assert all(call["current_squad"] == SQUAD for call in calls)
    assert all(call["bank"] == 25 for call in calls)
    assert all(call["free_transfers"] == 15 for call in calls)
    assert all(call["forced_transfers"] is None for call in calls)


def test_a_chip_the_solver_cannot_answer_is_worth_nothing(monkeypatch):
    # No legal rebuild — an empty pool, say — is not a chip worth minus
    # seventy points; it is a chip we know nothing about.
    chips = evs(monkeypatch, None, None)

    assert chips.free_hit == 0.0
    assert chips.wildcard == 0.0
    assert chips.bench_boost == pytest.approx(3.2)
