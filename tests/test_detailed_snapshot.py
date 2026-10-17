"""The run snapshot: what one run knew and weighed, kept as a compact file.

Every board here is hand-built and small enough to check by eye. The window
is gameweeks 8, 9 and 10, and four clubs play in it:

==  ===  ========================  ==========================================
id  abv  name                      fixtures (h = at home)
==  ===  ========================  ==========================================
1   ARS  Arsenal                   8: LEE(h)  9: LEE(h), WOL(a)  10: blank
2   LEE  Leeds United              8: ARS(a)  9: ARS(a)          10: CHE(a)
3   WOL  Wolverhampton Wanderers   8: CHE(h)  9: ARS(h)          10: none
4   CHE  Chelsea                   8: WOL(a)  9: none            10: LEE(h)
==  ===  ========================  ==========================================

Two more fixtures are there to be ignored: ARS v CHE with no gameweek yet
(``event`` None, a postponement the API has not rescheduled), and ARS v CHE
in gameweek 11, outside the window.

The players:

==  =======  ====  ===  =====  =========================================
id  name     club  pos  price  part
==  =======  ====  ===  =====  =========================================
1   Saka     ARS   MID  10.1   squad; the double and the blank
2   Raya     ARS   GKP   5.6   squad
3   Palmer   CHE   MID  10.5   neither squad nor projected — never stored
4   Justin   LEE   DEF   4.6   squad
5   Cucurella CHE  DEF   6.1   projected, not owned
==  =======  ====  ===  =====  =========================================
"""

import json
from datetime import UTC, datetime

import pytest

from aigaffer.data.models import Bootstrap, Event, Fixture, Player, Team
from aigaffer.detailed.snapshot import (
    MODES_ORDER,
    build_snapshot,
    latest_snapshot,
    load_snapshots,
    snapshot_path,
    write_snapshot,
)
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan

ARS, LEE, WOL, CHE = 1, 2, 3, 4
GKP, DEF, MID, FWD = 1, 2, 3, 4


def team(tid: int, name: str, short: str) -> Team:
    return Team(
        id=tid,
        name=name,
        short_name=short,
        strength_attack_home=1100,
        strength_attack_away=1100,
        strength_defence_home=1100,
        strength_defence_away=1100,
        strength_overall_home=1100,
        strength_overall_away=1100,
    )


def player(
    pid: int, name: str, club: int, pos: int, cost: int, status: str = "a", news: str = ""
) -> Player:
    return Player(
        id=pid,
        web_name=name,
        team=club,
        element_type=pos,
        now_cost=cost,
        status=status,
        news=news,
        minutes=900,
        starts=10,
        total_points=60,
        bonus=5,
        saves=0,
    )


EVENT = Event(
    id=8,
    deadline_time=datetime(2026, 10, 24, 10, 0, tzinfo=UTC),
    is_next=True,
    is_current=False,
    finished=False,
)

BOOTSTRAP = Bootstrap(
    events=[EVENT],
    teams=[
        team(ARS, "Arsenal", "ARS"),
        team(LEE, "Leeds United", "LEE"),
        team(WOL, "Wolverhampton Wanderers", "WOL"),
        team(CHE, "Chelsea", "CHE"),
    ],
    elements=[
        player(1, "Saka", ARS, MID, 101, status="d", news="Knock - 75% chance of playing"),
        player(2, "Raya", ARS, GKP, 56),
        player(3, "Palmer", CHE, MID, 105),
        player(4, "Justin", LEE, DEF, 46),
        player(5, "Cucurella", CHE, DEF, 61),
    ],
)

FIXTURES = [
    Fixture(id=1, event=8, team_h=ARS, team_a=LEE),
    Fixture(id=2, event=8, team_h=WOL, team_a=CHE),
    Fixture(id=3, event=9, team_h=ARS, team_a=LEE),
    Fixture(id=4, event=9, team_h=WOL, team_a=ARS),
    Fixture(id=5, event=10, team_h=CHE, team_a=LEE),
    Fixture(id=6, event=None, team_h=ARS, team_a=CHE),
    Fixture(id=7, event=11, team_h=ARS, team_a=CHE),
]


def projection(pid: int, per_gw: dict[int, float], total: float) -> PlayerProjection:
    return PlayerProjection(player_id=pid, per_gw=per_gw, total=total)


# Saka's gameweek 9 is two fixtures summed (3.651 + 3.645 = 7.296); his
# gameweek 10 is the blank, which project_all scores at exactly 0.0.
PROJECTIONS = {
    1: projection(1, {8: 4.123, 9: 7.296, 10: 0.0}, 10.326),
    2: projection(2, {8: 3.5, 9: 4.25, 10: 0.0}, 7.1),
    3: projection(3, {8: 5.0, 9: 0.0, 10: 6.0}, 9.3),
    4: projection(4, {8: 2.004, 9: 1.996, 10: 2.5}, 5.502),
    5: projection(5, {8: 4.0, 9: 0.0, 10: 4.4}, 7.2),
}
XMINS = {1: 71.04, 2: 90.0, 3: 85.0, 4: 62.36, 5: 80.0}

SQUAD = [1, 2, 4]
PROJECTED = {1, 2, 5}

# Plan 0 sells Justin for Cucurella this week and plans a bench boost in
# gameweek 9; plan 1 rolls the transfer.
SWAP = Plan(
    squad=[1, 2, 5],
    xi=[1, 2],
    transfers_in=[5],
    transfers_out=[4],
    hits=0,
    xp_total=58.2468,
    objective=57.9012,
    path=PlannedPath(
        moves=[
            PlannedMove(event=9, transfers_in=[], transfers_out=[], hits=0, chip="bench_boost")
        ],
        objective=57.9012,
        weekly_xp={8: 20.0, 9: 25.0, 10: 13.0},
        ft_kept=2,
        ft_credit=1.2345,
        transfer_cost=0.2,
        bench_charge=0.1449,
    ),
    bench_charge=0.1449,
)
ROLL = Plan(
    squad=[1, 2, 4],
    xi=[1, 2],
    transfers_in=[],
    transfers_out=[],
    hits=0,
    xp_total=55.0,
    objective=56.5,
)

DECISION = {
    "free_transfers": 2,
    "captain": 1,
    "vice": 2,
    "bench": [5],
    "chip": "none",
    "rationale": "Cucurella's three fixtures beat Justin's.",
    "chip_justification": "",
    "adjustments": [{"player_id": 1, "expected_minutes": 70.0, "reason": "knock"}],
    "decision_source": "manager",
}


def snapshot(**overrides) -> dict:
    """``build_snapshot`` on the board above, with any argument replaced."""
    arguments = dict(
        event=EVENT,
        mode="early",
        ts="2026-10-21T07:00:00+00:00",
        bootstrap=BOOTSTRAP,
        fixtures=FIXTURES,
        projections=PROJECTIONS,
        xmins=XMINS,
        projected_ids=PROJECTED,
        squad_ids=SQUAD,
        selling_prices={1: 99, 2: 55, 4: 45},
        plans={0: SWAP, 1: ROLL},
        chosen_id=0,
        decision=DECISION,
        chip_evs={"bench_boost": 6.1, "triple_captain": 4.0, "free_hit": 0.0, "wildcard": 0.0},
        calendar_lines=["Bench boost: GW9, the double"],
        freehit=None,
        bank=13,
        shortlist=2,
    )
    arguments.update(overrides)
    return build_snapshot(**arguments)


# --- the player rows -------------------------------------------------------


def test_a_player_row_carries_his_fixtures_xp_and_minutes():
    row = snapshot()["players"][1]

    assert row["n"] == "Saka"
    assert row["p"] == "MID"
    assert (row["c"], row["cf"]) == ("ARS", "Arsenal")
    assert (row["pr"], row["sp"]) == (101, 99)
    assert (row["st"], row["nw"]) == ("d", "Knock - 75% chance of playing")
    # 71.04 minutes, to a tenth.
    assert row["xm"] == 71.0
    # Gameweek 8 is fixture 1 alone: Arsenal at home to Leeds.
    assert row["f"][8] == [["LEE", "h"]]
    # 4.123 to two decimals.
    assert row["x"][8] == round(4.123, 2) == 4.12
    # The decayed total, to two decimals: 10.326 -> 10.33.
    assert row["t"] == 10.33


def test_a_double_gameweek_lists_both_opponents():
    row = snapshot()["players"][1]

    # Fixture 3 (Leeds at home) then fixture 4 (at Wolves), and one xP for
    # the gameweek, the two fixtures summed: 7.296 -> 7.3.
    assert row["f"][9] == [["LEE", "h"], ["WOL", "a"]]
    assert row["x"][9] == 7.3


def test_a_blank_gameweek_has_no_opponent():
    row = snapshot()["players"][1]

    # Arsenal have no fixture in gameweek 10; the window still lists it.
    assert row["f"][10] == []
    assert row["x"][10] == 0.0
    assert sorted(row["f"]) == sorted(row["x"]) == [8, 9, 10]


def test_an_unscheduled_fixture_is_a_blank():
    # The same board with Arsenal's gameweek 9 home game unscheduled: fixture
    # 3 moves to event None. Only the trip to Wolves is left in gameweek 9,
    # and gameweek 8 is untouched. Leeds lose their gameweek 9 game entirely,
    # and the unscheduled game never turns up in any gameweek.
    postponed = [
        Fixture(id=3, event=None, team_h=ARS, team_a=LEE) if f.id == 3 else f
        for f in FIXTURES
    ]
    players = snapshot(fixtures=postponed, projected_ids={1, 2, 4})["players"]

    assert players[1]["f"] == {8: [["LEE", "h"]], 9: [["WOL", "a"]], 10: []}
    assert players[4]["f"][9] == []


def test_a_fixture_outside_the_window_is_not_listed():
    # Fixture 7 is ARS v CHE in gameweek 11; the window ends at 10.
    row = snapshot()["players"][1]

    assert 11 not in row["f"]
    assert snapshot()["events"] == [8, 9, 10]


def test_only_the_projected_set_is_stored():
    players = snapshot()["players"]

    # The squad (1, 2, 4) and the projected set (1, 2, 5); Palmer is neither.
    assert sorted(players) == [1, 2, 4, 5]
    assert 3 not in players


def test_a_selling_price_is_only_for_a_man_we_own():
    players = snapshot()["players"]

    assert players[4]["sp"] == 45
    assert players[5]["sp"] is None
    assert snapshot(selling_prices=None)["players"][1]["sp"] is None


def test_a_squad_player_without_a_projection_is_named():
    projections = {pid: p for pid, p in PROJECTIONS.items() if pid != 4}

    with pytest.raises(ValueError, match=r"\b4\b"):
        snapshot(projections=projections)


# --- the plans and the decision --------------------------------------------


def test_a_plan_carries_its_moves_its_numbers_and_its_path():
    plan = snapshot()["plans"][0]

    assert plan == {
        "in": [5],
        "out": [4],
        "hits": 0,
        "xp": 58.25,
        "net": 57.9,
        "path": [{"gw": 9, "in": [], "out": [], "chip": "bench_boost"}],
        "ft_kept": 2,
        "ft_credit": 1.23,
        "transfer_cost": 0.2,
        "bench_charge": 0.14,
        "resolved": False,
    }


def test_a_single_week_plan_has_no_path_and_no_window_figures():
    plan = snapshot()["plans"][1]

    assert plan["path"] == []
    assert plan["ft_kept"] is None
    assert (plan["ft_credit"], plan["transfer_cost"], plan["bench_charge"]) == (0.0, 0.0, 0.0)


def test_plans_from_a_resolve_are_marked():
    # The briefing's shortlist was 3 plans, ids 0-2; the re-solve went on
    # counting, so 3 and 4 are its plans and the only ones marked.
    registry = {0: SWAP, 1: ROLL, 2: ROLL, 3: SWAP, 4: ROLL}
    plans = snapshot(plans=registry, shortlist=3, chosen_id=3)["plans"]

    assert {pid: plan["resolved"] for pid, plan in plans.items()} == {
        0: False,
        1: False,
        2: False,
        3: True,
        4: True,
    }


def test_the_chosen_plan_must_be_one_of_the_plans():
    with pytest.raises(ValueError, match=r"\b7\b"):
        snapshot(chosen_id=7)


def test_the_decision_is_kept():
    shot = snapshot()

    assert shot["chosen"] == 0
    assert shot["squad"] == [1, 2, 4]
    assert (shot["captain"], shot["vice"]) == (1, 2)
    assert shot["bench"] == [5]
    # The eleven is the chosen plan's fifteen less the bench: [1, 2, 5] - [5].
    assert shot["xi"] == [1, 2]
    assert shot["decision"] == {
        "rationale": "Cucurella's three fixtures beat Justin's.",
        "adjustments": [{"player_id": 1, "expected_minutes": 70.0, "reason": "knock"}],
        "chip": "none",
        "chip_justification": "",
        "source": "manager",
    }
    assert shot["chip_evs"]["bench_boost"] == 6.1
    assert shot["calendar"] == ["Bench boost: GW9, the double"]
    assert shot["freehit"] is None
    assert shot["skipped"] == []
    assert (shot["gw"], shot["mode"], shot["ts"]) == (8, "early", "2026-10-21T07:00:00+00:00")


def test_a_run_with_no_gaffer_is_the_solvers():
    # With the manager off, the decision record carries no decision_source,
    # rationale or adjustments at all.
    bare = {k: v for k, v in DECISION.items() if k not in (
        "decision_source", "rationale", "adjustments", "chip_justification"
    )}
    decision = snapshot(decision=bare)["decision"]

    assert decision == {
        "rationale": "",
        "adjustments": [],
        "chip": "none",
        "chip_justification": "",
        "source": "solver",
    }


def test_a_free_hit_week_fields_the_temporary_eleven():
    # The chip fields Raya, Cucurella and Justin, with Justin on the bench.
    # The standing squad less the bench would be [1, 2, 5] - [4] = [1, 2, 5],
    # which is not the team that plays: the eleven is the free hit's own.
    shot = snapshot(freehit=([2, 5, 4], [2, 5]), decision={**DECISION, "bench": [4]})

    assert shot["freehit"] == {"squad": [2, 5, 4], "xi": [2, 5]}
    assert shot["xi"] == [2, 5]
    assert shot["bench"] == [4]


def test_bank_and_free_transfers_are_kept():
    shot = snapshot(bank=13, decision={**DECISION, "free_transfers": 3})

    # Tenths, as the API serves them: 13 is £1.3m.
    assert shot["bank"] == 13
    assert shot["ft"] == 3


# --- the files ---------------------------------------------------------------


def test_snapshots_load_in_mode_order_and_latest_is_newest(tmp_path):
    write_snapshot(tmp_path, 8, "deadline", snapshot(mode="deadline"))
    write_snapshot(tmp_path, 8, "early", snapshot(mode="early"))
    gw9 = EVENT.model_copy(update={"id": 9})
    write_snapshot(tmp_path, 9, "early", snapshot(mode="early", event=gw9))

    assert [s["mode"] for s in load_snapshots(tmp_path, 8)] == ["early", "deadline"]
    latest = latest_snapshot(tmp_path)
    assert (latest["gw"], latest["mode"]) == (9, "early")


def test_within_a_gameweek_the_latest_mode_wins(tmp_path):
    for mode in ("scout", "early"):
        write_snapshot(tmp_path, 8, mode, snapshot(mode=mode))

    assert latest_snapshot(tmp_path)["mode"] == "scout"


def test_no_snapshot_at_all_is_none(tmp_path):
    assert latest_snapshot(tmp_path) is None
    assert load_snapshots(tmp_path, 8) == []


def test_a_snapshot_reads_back_as_it_was_built(tmp_path):
    # JSON turns every int key into a string; the reader turns them back, so
    # a renderer handed a loaded snapshot reads players[1]["f"][8] exactly as
    # one handed the snapshot in memory does.
    built = snapshot()
    write_snapshot(tmp_path, 8, "early", built)

    assert load_snapshots(tmp_path, 8) == [built]


def test_the_write_is_atomic(tmp_path):
    first = write_snapshot(tmp_path, 8, "early", snapshot(ts="first"))
    second = write_snapshot(tmp_path, 8, "early", snapshot(ts="second"))

    assert first == second == snapshot_path(tmp_path, 8, "early")
    assert first == tmp_path / "detailed" / "gw8-early.json"
    # Nothing left behind but the one file, and the rerun's is the one kept.
    assert [p.name for p in first.parent.iterdir()] == ["gw8-early.json"]
    assert json.loads(first.read_text(encoding="utf-8"))["ts"] == "second"


def test_an_unknown_mode_is_refused(tmp_path):
    # A file named for a mode the reader does not know would be written and
    # then never read, which is worse than not writing it.
    assert "weekly" not in MODES_ORDER
    with pytest.raises(ValueError, match="weekly"):
        write_snapshot(tmp_path, 8, "weekly", snapshot())


# --- the skipped transfers -------------------------------------------------


def test_the_snapshot_lists_the_transfers_the_chosen_plan_skipped():
    """Chosen = the roll (plan 0), so Cucurella (plan 1's buy) is skipped.

    He is the only unowned DEF, so he is both a top-three DEF and a plan-1 buy:
    one row. He replaces Justin, the only squad DEF: totals are 7.2 and 5.5
    (5.502 to two decimals), so the gap is 1.7. Price 61 against bank 13 + Justin's sale 45 = 58: 61 > 58,
    unaffordable, and 61 - 45 = 16 tenths over the sale.
    """
    result = snapshot(plans={0: ROLL, 1: SWAP}, chosen_id=0)
    assert result["skipped"] == [
        {
            "id": 5,
            "pos": "DEF",
            "replaces": 4,
            "gap": 1.7,
            "price_diff": 16,
            "tag": "unaffordable",
        }
    ]
