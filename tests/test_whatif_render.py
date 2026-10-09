"""Message 1: the numbers, in the order the expert asked for, exact to the
character — this is the text on the owner's phone."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aigaffer.chips import BENCH_BOOST, FREE_HIT, WILDCARD
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.orchestrator import PreparedWeek
from aigaffer.report import whatif as render
from aigaffer.report.whatif import render_numbers, render_unsure_line
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.lineup import Lineup
from aigaffer.solver.optimizer import Plan
from aigaffer.whatif import HOLD, MARGINAL, PLAY, FreeHitTerms, HoldWeek, PathSide, WhatIf

WINDOW = [6, 7, 8, 9, 10, 11]
# Sat 10 Oct 2026 10:00 UTC is 11:00 in London (BST).
DEADLINE = datetime(2026, 10, 10, 10, 0, tzinfo=UTC)
NOW = datetime(2026, 10, 8, 19, 0, tzinfo=UTC)  # 1 day 15 hours before

# A fifteen held by the owner: two keepers, five defenders, five midfielders,
# three forwards, ids 1–15, every one on his own club and priced 50. Projected
# xP for GW6 descends with the id, so the eleven, the bench and the armbands
# are obvious: 1 in goal (2 benched), 3–7 at the back... see LINEUP below.
POSITIONS = {1: 1, 2: 1, **{p: 2 for p in range(3, 8)}, **{p: 3 for p in range(8, 13)}, **{p: 4 for p in range(13, 16)}}
NAMES = {
    1: "Verbruggen", 2: "Kinsky", 3: "Diop", 4: "Muharemovic", 5: "Guehi", 6: "Calafiori", 7: "Vuskovic",
    8: "Bruno", 9: "Mbeumo", 10: "Gross", 11: "LewisPotter", 12: "Sangare",
    13: "Haaland", 14: "CalvertLewin", 15: "JoaoPedro",
    # Wildcard signings / free-hit team members.
    21: "Pickford", 22: "Gvardiol", 23: "Mitchell", 24: "Tavernier", 25: "Schade",
}
CHANCE: dict[int, int] = {}
STATUS: dict[int, str] = {}


def _player(pid: int) -> Player:
    position = POSITIONS.get(pid, {21: 1, 22: 2, 23: 2, 24: 3, 25: 3}.get(pid, 3))
    return Player(
        id=pid, web_name=NAMES[pid], team=pid, element_type=position, now_cost=50,
        status=STATUS.get(pid, "a"), chance_of_playing_next_round=CHANCE.get(pid),
        minutes=900, starts=10, total_points=0, bonus=0, saves=0,
    )


def _week(*, xmins=None, executed=None, deadlines=None, free_transfers=2) -> PreparedWeek:
    players = {pid: _player(pid) for pid in NAMES}
    deadlines = deadlines or {6: DEADLINE, 7: DEADLINE + timedelta(days=7)}
    events = [SimpleNamespace(id=gw, deadline_time=when) for gw, when in sorted(deadlines.items())]
    bootstrap = SimpleNamespace(events=events, chips=[])
    # GW6 points: the squad descends 30, 29, ... by id; signings score high.
    xp = {pid: float(31 - pid) for pid in range(1, 16)}
    xp.update({21: 20.0, 22: 20.0, 23: 19.0, 24: 25.0, 25: 24.0})
    projections = {
        pid: PlayerProjection(player_id=pid, per_gw={gw: xp[pid] for gw in WINDOW}, total=6 * xp[pid])
        for pid in xp
    }
    effective = SimpleNamespace(
        players=players,
        squad=SimpleNamespace(player_ids=list(range(1, 16)), bank=10),
        free_transfers=free_transfers,
        event=SimpleNamespace(id=6, deadline_time=DEADLINE),
        executed=executed,
        bootstrap=bootstrap,
        chips_used=[],
    )
    return PreparedWeek(
        inputs=effective, ledger=None, effective=effective, executed=executed,
        prices={3: 40}, strengths=None, xmins=xmins or {pid: 90.0 for pid in xp},
        projections=projections, calendar=None,
    )


def _side(squad, *, moves=(), week1="none", fh_squad=None, fh_xi=None, ins=(), outs=(), hits=0, proven=True):
    path = PlannedPath(
        moves=list(moves), objective=0.0, weekly_xp={}, week1_chip=week1,
        week1_freehit_squad=fh_squad, week1_freehit_xi=fh_xi, proven=proven,
    )
    plan = Plan(squad=list(squad), xi=list(squad)[:11], transfers_in=list(ins),
                transfers_out=list(outs), hits=hits, xp_total=0.0, objective=0.0, path=path)
    chips = {6: week1} if week1 != "none" else {}
    chips.update({m.event: m.chip for m in moves if m.chip != "none"})
    # The eleven: the free-hit team's on a free-hit side, else the squad's
    # first eleven, armbands on its first two — fixed here so the renderer's
    # exact strings do not depend on the lineup picker.
    fielded = list(fh_squad or squad)
    lineup = Lineup(xi=list(fh_xi or fielded[:11]), bench=fielded[11:15], captain=fielded[0], vice=fielded[1])
    return PathSide(plan=plan, path=path, chips=chips, lineup=lineup, seconds=1.0)


# The wildcard: out Verbruggen, Diop, Muharemovic, Sangare; in Pickford,
# Gvardiol, Mitchell, Tavernier. The new fifteen's GW6 eleven by xP, captain
# the top scorer (Calafiori is not; see the numbers above).
WC_SQUAD = sorted(set(range(1, 16)) - {1, 3, 4, 12} | {21, 22, 23, 24})


def _wildcard(*, band=MARGINAL, lean=HOLD, net=2.0, gain=21.0, bars=19.0, hold_week=None,
              off_moves=(), on_moves=(), unsure=False, off_ins=(), off_outs=()) -> WhatIf:
    on = _side(WC_SQUAD, week1=WILDCARD, moves=on_moves, ins=[21, 22, 23, 24], outs=[1, 3, 4, 12])
    off = _side(range(1, 16), moves=off_moves, ins=off_ins, outs=off_outs, proven=not unsure)
    return WhatIf(
        kind=WILDCARD, event=6, window=WINDOW, on=on, off=off, net=net, gain=gain, bars_diff=bars,
        weekly_gain={6: 3.0, 7: 6.0, 8: 5.0, 9: 4.0, 10: 2.0, 11: 1.0},
        band=band, lean=lean, unsure=unsure, margin=8.0, hold_week=hold_week, fh_terms=None,
        refunded_hits=0, minutes_source="Wednesday's scout",
    )


VERDICT = {"event": 6, "transfers_in": [], "transfers_out": [],
           "adjustments": [{"player_id": 15, "expected_minutes": 10.0}]}


def test_a_marginal_wildcard_reads_as_the_spec_lays_it_out():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=NOW, numbers_only=False)

    assert text.splitlines()[:3] == [
        "🃏 Wildcard GW6 — MARGINAL, lean hold (+2 net; noise ±8)",
        "Deadline Sat 10 Oct 11:00 UK (in 1d 15h)",
        "",
    ]
    assert "Now buys +21 over GW6–11 vs your best path without it; keeping it is worth ~19." in text
    assert "(off: roll, 3 FTs into GW7 · wildcard: not in the window)" in text
    assert "Chips — on: WC GW6 · off: none" in text
    assert "Gain by week: +3 +6 +5 +4 +2 +1" in text
    assert "Minutes: Wednesday's scout (JoaoPedro 10)." in text
    # Marginal: the squad block is there, and so is the hold sentence.
    assert "If you play it — 4 moves:" in text
    assert "GK  Verbruggen → Pickford" in text
    assert "DEF Diop → Gvardiol, Muharemovic → Mitchell" in text
    assert "MID Sangare → Tavernier" in text
    assert text.endswith(render.FOOTER + " " + render.GAFFER_FOLLOWS)


def test_the_bank_armbands_and_bench_are_the_new_fifteens():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=NOW, numbers_only=False)

    # Bank 10 + sales (Diop at his ledger price 40, the rest at 50: 190) −
    # buys (4 × 50 = 200) = 0. Captain the best GW6 score in the new eleven
    # The armbands and bench come from the side's fixed lineup (see _side).
    bank_line = next(line for line in text.splitlines() if line.startswith("Bank "))
    assert bank_line.startswith("Bank £0.0m · C ")
    assert " · Bench: " in bank_line


def test_numbers_only_says_so_instead_of_promising_the_gaffer():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=NOW, numbers_only=True)

    assert text.endswith(render.FOOTER + " " + render.NUMBERS_ONLY)


def test_a_hold_gives_one_line_for_the_squad_and_names_the_week():
    hold = HoldWeek(event=9, value=10.0, source="off_path", provisional=False)
    later = PlannedMove(event=9, transfers_in=[], transfers_out=[], hits=0, chip=WILDCARD)
    text = render_numbers(
        _wildcard(band=HOLD, lean=None, net=-10.0, gain=20.0, bars=30.0, hold_week=hold, off_moves=[later]),
        _week(), verdict=VERDICT, now=NOW, numbers_only=False,
    )

    assert "🃏 Wildcard GW6 — HOLD (−10 net; noise ±8)" in text
    assert "If you play it —" not in text
    assert "(Playing it anyway would mean 4 moves.)" in text
    assert "Hold → GW9: the path without it plays it then, worth 10 more than now." in text
    assert "(off: roll, 3 FTs into GW7 · wildcard: GW9)" in text
    # The off path plays the wildcard later, so the bars line is chip timing.
    assert "chip timing −30" in text
    assert "Ask again from GW7." in text


def test_a_provisional_week_says_so():
    hold = HoldWeek(event=11, value=9.0, source="off_path", provisional=True)
    text = render_numbers(_wildcard(band=HOLD, lean=None, net=-9.0, hold_week=hold), _week(),
                          verdict=None, now=NOW, numbers_only=False)

    assert "Hold → GW11 (provisional): the path without it plays it then, worth 9 more than now." in text


def test_a_wildcard_held_past_the_window_invents_no_week(monkeypatch):
    hold = HoldWeek(event=None, value=35.0, source="past_window", provisional=False)
    stop_deadline = datetime(2027, 1, 2, 13, 30, tzinfo=UTC)
    week = _week(deadlines={6: DEADLINE, 7: DEADLINE + timedelta(days=7), 19: stop_deadline})
    monkeypatch.setattr(
        render, "held_in_week",
        lambda bootstrap, chips_used, event_id, executed=None: (render.HeldChip(WILDCARD, 2, 19),),
    )
    text = render_numbers(_wildcard(band=HOLD, lean=None, net=-12.0, hold_week=hold), week,
                          verdict=None, now=NOW, numbers_only=False)

    assert (
        "Hold: nothing in GW6–11 beats keeping it. It's yours until the GW19 deadline"
        " (Sat 02 Jan 13:30 UK). Keeping it is priced at ~35 now on a generic curve,"
        " not a specific week. It becomes a Play when 4+ starters need changing, a"
        " fixture swing lines up, or a bench boost is 1–3 weeks ahead."
    ) in text


def test_an_international_break_moves_the_ask_again():
    hold = HoldWeek(event=9, value=10.0, source="off_path", provisional=False)
    week = _week(deadlines={6: DEADLINE, 7: DEADLINE + timedelta(days=14)})
    text = render_numbers(_wildcard(band=HOLD, lean=None, net=-10.0, hold_week=hold), week,
                          verdict=None, now=NOW, numbers_only=False)

    assert "Ask again after the international break." in text


def test_no_verdict_means_the_models_minutes():
    text = render_numbers(_wildcard(), _week(), verdict=None, now=NOW, numbers_only=False)

    assert "Minutes: the model's own (no report yet this gameweek)." in text


def test_a_verdict_with_no_calls_says_no_changes():
    text = render_numbers(_wildcard(), _week(), verdict={**VERDICT, "adjustments": []}, now=NOW,
                          numbers_only=False)

    assert "Minutes: Wednesday's scout (no changes)." in text


def test_the_off_path_differing_from_the_verdict_is_said():
    whatif = _wildcard(off_ins=[22], off_outs=[3])
    text = render_numbers(whatif, _week(), verdict={**VERDICT, "transfers_in": [], "transfers_out": []},
                          now=NOW, numbers_only=False)

    assert "This differs from your latest report's plan (roll) — a quicker solve." in text
    assert "(off: 1 move (Diop → Gvardiol), 2 FTs into GW7 · wildcard: not in the window)" in text


def test_the_off_path_agreeing_with_the_verdict_says_nothing():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=NOW, numbers_only=False)

    assert "differs from your latest report" not in text


def test_risk_flags_name_the_doubts_in_the_new_players(monkeypatch):
    monkeypatch.setitem(CHANCE, 24, 75)
    monkeypatch.setitem(STATUS, 22, "d")
    week = _week(xmins={**{pid: 90.0 for pid in NAMES}, 23: 62.0})
    text = render_numbers(_wildcard(), week, verdict=VERDICT, now=NOW, numbers_only=False)

    # Signings in order: Pickford (fine), Gvardiol (doubtful), Mitchell (62
    # minutes, but benched: at 19 he is the new fifteen's lowest outfield
    # scorer bar the forwards the shape needs, so the eleven is 4-5-1 without
    # him and his minutes are no doubt worth a flag), Tavernier (75%).
    flags = next(line for line in text.splitlines() if line.startswith("⚠️"))
    assert flags == "⚠️ Gvardiol doubtful · Tavernier 75%"


def test_no_doubts_means_no_flag_line():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=NOW, numbers_only=False)

    assert "⚠️" not in text


def test_the_unsure_line_names_the_side_that_stopped():
    assert render_unsure_line(_wildcard()) is None
    assert render_unsure_line(_wildcard(unsure=True)) == (
        "Solver unsure: the solve without it stopped on its time limit — treat the numbers as rough."
    )


def test_the_free_hit_shows_its_sum_and_its_fifteen():
    fh = sorted(set(range(1, 16)) - {1, 3, 12} | {21, 22, 25})
    on = _side(range(1, 16), week1=FREE_HIT, fh_squad=fh, fh_xi=fh[:11])
    off = _side(range(1, 16))
    whatif = WhatIf(
        kind=FREE_HIT, event=6, window=WINDOW, on=on, off=off, net=4.6, gain=19.6, bars_diff=15.0,
        weekly_gain={6: 13.1, 7: -1.0, 8: -0.5, 9: 0.0, 10: 0.0, 11: 0.0},
        band=PLAY, lean=None, unsure=False, margin=4.0, hold_week=None,
        fh_terms=FreeHitTerms(one_week=13.1, knock_on=6.5, keeping=15.0), refunded_hits=0,
        minutes_source=None,
    )
    text = render_numbers(whatif, _week(), verdict=None, now=NOW, numbers_only=False)

    assert text.splitlines()[0] == "🃏 Free hit GW6 — PLAY (+5 net; noise ±4)"
    assert "FH team vs your GW6 XI +13.1 · knock-on GW7–11 +6.5 · keeping it −15.0 = net +4.6" in text
    assert render.KNOCK_ON_UP.format(knock=6.5) in text
    assert "If you play it — your free-hit team:" in text
    assert any(line.startswith("Bench: ") for line in text.splitlines())
    assert any(line.startswith("C ") and " · V " in line for line in text.splitlines())


def test_a_free_hit_knock_on_that_costs_is_said_in_words():
    on = _side(range(1, 16), week1=FREE_HIT, fh_squad=list(range(1, 16)), fh_xi=list(range(1, 12)))
    whatif = WhatIf(
        kind=FREE_HIT, event=6, window=WINDOW, on=on, off=_side(range(1, 16)), net=-3.4, gain=11.6,
        bars_diff=15.0, weekly_gain={gw: 0.0 for gw in WINDOW}, band=MARGINAL, lean=HOLD, unsure=False,
        margin=4.0, hold_week=HoldWeek(event=12, value=14.0, source="calendar", provisional=True),
        fh_terms=FreeHitTerms(one_week=13.1, knock_on=-1.5 - 3.0, keeping=15.0), refunded_hits=0,
        minutes_source=None,
    )
    text = render_numbers(whatif, _week(), verdict=None, now=NOW, numbers_only=False)

    assert render.KNOCK_ON_DOWN.format(knock=4.5) in text
    assert "Hold → GW12 (provisional): the calendar saves it for then, worth ~14." in text


def test_a_deadline_already_passed_says_so():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=DEADLINE + timedelta(minutes=1),
                          numbers_only=False)

    assert text.splitlines()[1] == "Deadline Sat 10 Oct 11:00 UK (passed)"


def test_under_a_day_the_countdown_is_hours_and_minutes():
    text = render_numbers(_wildcard(), _week(), verdict=VERDICT, now=DEADLINE - timedelta(hours=3, minutes=5),
                          numbers_only=False)

    assert text.splitlines()[1] == "Deadline Sat 10 Oct 11:00 UK (in 3h 05m)"


def test_other_chips_played_differently_are_chip_timing_not_keeping():
    bb = PlannedMove(event=7, transfers_in=[], transfers_out=[], hits=0, chip=BENCH_BOOST)
    bb_later = PlannedMove(event=9, transfers_in=[], transfers_out=[], hits=0, chip=BENCH_BOOST)
    text = render_numbers(_wildcard(on_moves=[bb], off_moves=[bb_later]), _week(), verdict=VERDICT, now=NOW,
                          numbers_only=False)

    assert "Chips — on: WC GW6, BB GW7 · off: BB GW9" in text
    assert "; chip timing −19." in text
