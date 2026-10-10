"""Message 1: the call, the points and the team, exact to the character —
this is the text on the owner's phone."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.orchestrator import PreparedWeek
from aigaffer.report import whatif as render
from aigaffer.report.whatif import render_numbers, render_unsure_line
from aigaffer.solver.multiweek import PlannedPath
from aigaffer.solver.lineup import Lineup
from aigaffer.solver.optimizer import Plan
from aigaffer.whatif import HOLD, MARGINAL, PLAY, FreeHitTerms, HoldWeek, PathSide, WhatIf

WINDOW = [6, 7, 8, 9, 10, 11]
# Sat 10 Oct 2026 10:00 UTC is 11:00 in Ireland (IST, the same offset as BST).
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


VERDICT = {"event": 6, "transfers_in": [], "transfers_out": [], "adjustments": []}

# The wildcard's lineup is _side's fixed one: the first eleven of the sorted
# new fifteen (2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15), armbands on its first two
# (Kinsky 2, Guehi 5), the last four as the bench in order. Rows, by hand:
# GK 2; DEF 5, 6, 7; MID 8, 9, 10, 11; FWD 13, 14, 15. Bank: 10 + sales (Diop
# at his ledger price 40, the other three at 50: 190) - buys (4 x 50 = 200) = 0.
WC_TEAM = """If you played it — C Kinsky · V Guehi · bank £0.0m:
GK  Kinsky (C)
DEF Guehi (V), Calafiori, Vuskovic
MID Bruno, Mbeumo, Gross, LewisPotter
FWD Haaland, CalvertLewin, JoaoPedro
Bench: 1 Pickford · 2 Gvardiol · 3 Mitchell · 4 Tavernier"""
FOOTER = 'Not recorded — if you play it, don\'t send "Transfers made". Gaffer\'s view next.'


def _text(whatif, week=None, *, now=NOW, numbers_only=False) -> str:
    return render_numbers(whatif, week or _week(), verdict=VERDICT, now=now, numbers_only=numbers_only)


def test_a_wildcard_play_is_three_lines_the_team_and_the_footer():
    # Gain +21 over GW6-11, net +12: whole points, "xP" after the first only.
    text = _text(_wildcard(band=PLAY, lean=None, net=12.0, gain=21.0))

    assert text == (
        "🃏 Wildcard GW6 — PLAY\n"
        "+21 xP over GW6–11 · +12 net of keeping it\n"
        "Deadline Sat 10 Oct 11:00 Irish (in 1d 15h)\n"
        "\n" + WC_TEAM + "\n"
        "\n" + FOOTER
    )


def test_a_hold_off_path_names_the_better_week_and_is_provisional_when_far_out():
    hold = HoldWeek(event=11, value=9.0, source="off_path", provisional=True)
    text = _text(_wildcard(band=HOLD, lean=None, net=-9.0, gain=21.0, hold_week=hold))

    # A Hold still shows the team he would field if he played it anyway.
    assert text == (
        "🃏 Wildcard GW6 — HOLD (GW11 looks better, provisional)\n"
        "+21 xP over GW6–11 · −9 net of keeping it\n"
        "Deadline Sat 10 Oct 11:00 Irish (in 1d 15h)\n"
        "\n" + WC_TEAM + "\n"
        "\n" + FOOTER
    )


def test_a_calendar_hold_week_reads_the_same_way():
    hold = HoldWeek(event=9, value=10.0, source="calendar", provisional=False)
    text = _text(_wildcard(band=HOLD, lean=None, net=-10.0, hold_week=hold))

    assert text.splitlines()[0] == "🃏 Wildcard GW6 — HOLD (GW9 looks better)"
    assert text.splitlines()[1] == "+21 xP over GW6–11 · −10 net of keeping it"


def test_a_wildcard_held_past_the_window_invents_no_week():
    hold = HoldWeek(event=None, value=35.0, source="past_window", provisional=False)
    text = _text(_wildcard(band=HOLD, lean=None, net=-12.0, hold_week=hold))

    assert text.splitlines()[0] == "🃏 Wildcard GW6 — HOLD (nothing in GW6–11 beats keeping it)"


def test_a_plain_hold_with_no_week_has_no_parenthesis():
    text = _text(_wildcard(band=HOLD, lean=None, net=-12.0, hold_week=None))

    assert text.splitlines()[0] == "🃏 Wildcard GW6 — HOLD"


def test_a_maybe_leaning_hold_says_so_and_names_the_week():
    assert _text(_wildcard()).splitlines()[0] == "🃏 Wildcard GW6 — MAYBE (leans hold)"
    hold = HoldWeek(event=9, value=10.0, source="off_path", provisional=False)
    text = _text(_wildcard(hold_week=hold))

    assert text.splitlines()[0] == "🃏 Wildcard GW6 — MAYBE (leans hold, GW9 looks better)"


def test_a_maybe_leaning_play_has_no_week():
    hold = HoldWeek(event=9, value=10.0, source="off_path", provisional=False)
    text = _text(_wildcard(lean=PLAY, net=6.0, hold_week=hold))

    assert text.splitlines()[0] == "🃏 Wildcard GW6 — MAYBE (leans play)"
    assert text.splitlines()[1] == "+21 xP over GW6–11 · +6 net of keeping it"


def _free_hit(**changes) -> WhatIf:
    fh = sorted(set(range(1, 16)) - {1, 3, 12} | {21, 22, 25})
    on = _side(range(1, 16), week1=FREE_HIT, fh_squad=fh, fh_xi=fh[:11])
    fields = dict(
        kind=FREE_HIT, event=6, window=WINDOW, on=on, off=_side(range(1, 16)), net=4.6, gain=19.6,
        bars_diff=15.0, weekly_gain={6: 13.1, 7: -1.0, 8: -0.5, 9: 0.0, 10: 0.0, 11: 0.0},
        band=PLAY, lean=None, unsure=False, margin=4.0, hold_week=None,
        fh_terms=FreeHitTerms(one_week=13.1, knock_on=6.5, keeping=15.0), refunded_hits=0,
        minutes_source=None,
    )
    fields.update(changes)
    return WhatIf(**fields)


def test_a_free_hit_leads_with_its_one_week_gain_and_fields_the_free_hit_team():
    # The free-hit fifteen sorted: 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 21,
    # 22, 25; the eleven its first eleven (2 ... 14), armbands on 2 and 4, the
    # bench 15, 21, 22, 25. Rows: GK 2; DEF 4, 5, 6, 7; MID 8, 9, 10, 11;
    # FWD 13, 14 -- a 4-4-2. One week +13.1 -> +13; net +4.6 -> +5.
    text = _text(_free_hit())

    assert text == (
        "🃏 Free hit GW6 — PLAY\n"
        "+13 xP this week vs your XI · +5 net over GW6–11\n"
        "Deadline Sat 10 Oct 11:00 Irish (in 1d 15h)\n"
        "\n"
        "If you played it — your free-hit team · C Kinsky · V Muharemovic:\n"
        "GK  Kinsky (C)\n"
        "DEF Muharemovic (V), Guehi, Calafiori, Vuskovic\n"
        "MID Bruno, Mbeumo, Gross, LewisPotter\n"
        "FWD Haaland, CalvertLewin\n"
        "Bench: 1 JoaoPedro · 2 Pickford · 3 Gvardiol · 4 Schade\n"
        "\n" + FOOTER
    )


def test_the_unsure_line_sits_directly_under_the_deadline():
    text = _text(_wildcard(unsure=True))

    lines = text.splitlines()
    assert lines[2].startswith("Deadline ")
    assert lines[3] == (
        "Solver unsure: the solve without it stopped on its time limit — treat the numbers as rough."
    )
    assert lines[4] == ""
    assert render_unsure_line(_wildcard()) is None
    assert "Solver unsure" not in _text(_wildcard())


def test_the_unsure_line_names_the_side_that_stopped():
    assert render_unsure_line(_wildcard(unsure=True)) == (
        "Solver unsure: the solve without it stopped on its time limit — treat the numbers as rough."
    )


def test_a_doubtful_newcomer_is_noted_after_his_name_not_in_a_warning_line(monkeypatch):
    # Bruno (8) captains and Mbeumo (9) is vice. Gvardiol (22, doubtful) and
    # Tavernier (24, 75%) both start; Pickford (21) is on the bench; Mitchell
    # (23, 62 minutes) is on the bench too, where minutes are no doubt.
    monkeypatch.setitem(CHANCE, 24, 75)
    monkeypatch.setitem(STATUS, 22, "d")
    whatif = _wildcard()
    lineup = Lineup(xi=[2, 5, 6, 22, 8, 9, 10, 24, 13, 14, 15], bench=[21, 7, 11, 23], captain=8, vice=9)
    whatif = replace(whatif, on=replace(whatif.on, lineup=lineup))
    week = _week(xmins={**{pid: 90.0 for pid in NAMES}, 23: 62.0})

    text = _text(whatif, week)

    assert text.split("\n\n")[1] == (
        "If you played it — C Bruno · V Mbeumo · bank £0.0m:\n"
        "GK  Kinsky\n"
        "DEF Guehi, Calafiori, Gvardiol (doubtful)\n"
        "MID Bruno (C), Mbeumo (V), Gross, Tavernier (75%)\n"
        "FWD Haaland, CalvertLewin, JoaoPedro\n"
        "Bench: 1 Pickford · 2 Vuskovic · 3 LewisPotter · 4 Mitchell"
    )
    assert "⚠️" not in text


def test_a_newcomer_in_the_eleven_on_thin_minutes_is_noted_in_mins():
    whatif = _wildcard()
    lineup = Lineup(xi=[21, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15], bench=[2, 22, 23, 24], captain=8, vice=9)
    whatif = replace(whatif, on=replace(whatif.on, lineup=lineup))
    week = _week(xmins={**{pid: 90.0 for pid in NAMES}, 21: 62.0})

    assert "GK  Pickford (62 mins)" in _text(whatif, week).splitlines()


def test_an_old_hand_with_thin_minutes_is_not_a_newcomer_and_gets_no_note():
    week = _week(xmins={**{pid: 90.0 for pid in NAMES}, 8: 40.0})

    assert "Bruno (40 mins)" not in _text(_wildcard(), week)


def test_numbers_only_says_so_instead_of_promising_the_gaffer():
    text = _text(_wildcard(), numbers_only=True)

    assert text.endswith('Not recorded — if you play it, don\'t send "Transfers made". '
                         "Numbers only — the gaffer is switched off.")


def test_a_deadline_already_passed_says_so():
    text = _text(_wildcard(), now=DEADLINE + timedelta(minutes=1))

    assert text.splitlines()[2] == "Deadline Sat 10 Oct 11:00 Irish (passed)"


def test_under_a_day_the_countdown_is_hours_and_minutes():
    text = _text(_wildcard(), now=DEADLINE - timedelta(hours=3, minutes=5))

    assert text.splitlines()[2] == "Deadline Sat 10 Oct 11:00 Irish (in 3h 05m)"


def test_the_deadline_is_read_in_irish_time_in_winter_too():
    # 13:30 UTC on 2 Jan is 13:30 in Dublin (GMT); the offsets match London's.
    winter = datetime(2027, 1, 2, 13, 30, tzinfo=UTC)
    week = _week(deadlines={6: winter})
    week.effective.event.deadline_time = winter

    text = _text(_wildcard(), week, now=winter - timedelta(days=2))

    assert text.splitlines()[2] == "Deadline Sat 02 Jan 13:30 Irish (in 2d 0h)"
