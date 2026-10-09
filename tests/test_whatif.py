"""The what-if core: two window solves pinned on and off, and what they mean.

The solver is stubbed in all but the last test. What is under test here is the
arithmetic and the wiring around the two solves — which pin each gets, which
free transfers, one set of free-hit prices shared between them, the net, the
gain and the bars, the band and the week to hold for — and with a stub every
number is chosen by hand. Pinning itself is the solver's business and is tested
with the solver (tests/test_multiweek.py).
"""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from aigaffer import whatif
from aigaffer.chips import BENCH_BOOST, FREE_HIT, WILDCARD, HeldChip
from aigaffer.model.xp import PlayerProjection
from aigaffer.orchestrator import PreparedWeek, prepare_week
from aigaffer.solver.lineup import Lineup
from aigaffer.solver.calendar import CalendarEntry, ChipCalendar
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan
from aigaffer.store import DB_NAME, Store
from aigaffer.whatif import (
    HOLD,
    MARGINAL,
    PLAY,
    WHATIF_MARGIN,
    band_for,
    simulate,
    verdict_minutes,
)
from tests.fixtures import config, make_client, make_executed, pipeline_routes

WINDOW = [6, 7, 8, 9, 10, 11]
WC1 = HeldChip(WILDCARD, 2, 19)
FH1 = HeldChip(FREE_HIT, 2, 19)
BB1 = HeldChip(BENCH_BOOST, 1, 19)
SQUAD = list(range(1, 16))


class _AnyPlayer(dict):
    """A player table that has everyone, all goalkeepers: simulate only asks
    the board for positions so week1_lineup can hand them to the lineup picker,
    and the picker is stubbed (see ``stub_lineup``), so which position is
    immaterial. A hand-built squad of fifteen legal positions would add nothing
    to what these tests check."""

    def __missing__(self, pid):
        return SimpleNamespace(element_type=1)


@pytest.fixture
def stub_lineup(monkeypatch):
    """The lineup picker is the lineup module's to test; here it answers a
    fixed eleven so the arithmetic tests are about the arithmetic. Not autouse:
    the end-to-end test must run the real picker on a real board, so the stubbed
    tests ask for it through ``held``, which every one of them uses."""
    monkeypatch.setattr(
        whatif, "pick_lineup",
        lambda squad, *rest: Lineup(xi=list(squad[:11]), captain=squad[0], vice=squad[1], bench=list(squad[11:])),
    )


def _projections(extra: dict[int, dict[int, float]] | None = None) -> dict[int, PlayerProjection]:
    """Every squad player worth 1.0 a week over the window, unless ``extra``
    says otherwise for a player — enough for projected_events and the bench
    strip, which are the only things simulate reads off the projections."""
    table = {pid: {gw: 1.0 for gw in WINDOW} for pid in SQUAD}
    for pid, weeks in (extra or {}).items():
        table[pid] = {gw: weeks.get(gw, 0.0) for gw in WINDOW}
    return {
        pid: PlayerProjection(player_id=pid, per_gw=weeks, total=sum(weeks.values()))
        for pid, weeks in table.items()
    }


def _week(*, executed=None, free_transfers=1, calendar=None, projections=None) -> PreparedWeek:
    effective = SimpleNamespace(
        players=_AnyPlayer(),
        squad=SimpleNamespace(player_ids=SQUAD, bank=10),
        free_transfers=free_transfers,
        event=SimpleNamespace(id=WINDOW[0]),
        executed=executed,
        bootstrap=None,
    )
    return PreparedWeek(
        inputs=effective,
        ledger=None,
        effective=effective,
        executed=executed,
        prices={1: 50},
        strengths=None,
        xmins={},
        projections=projections or _projections(),
        calendar=calendar,
    )


def _side(objective, *, bars=0.0, week1="none", moves=(), weekly=None, proven=True,
          fh_squad=None, fh_xi=None):
    """One stubbed solve's answer: a plan and its path with the numbers given."""
    path = PlannedPath(
        moves=list(moves),
        objective=objective,
        weekly_xp=weekly or {gw: 50.0 for gw in WINDOW},
        week1_chip=week1,
        week1_freehit_squad=fh_squad,
        week1_freehit_xi=fh_xi,
        proven=proven,
        bars_paid=bars,
    )
    plan = Plan(
        squad=SQUAD, xi=SQUAD[:11], transfers_in=[], transfers_out=[], hits=0,
        xp_total=objective, objective=objective, path=path,
    )
    return plan, path


class FakeSolver:
    """optimize_path, answering by pin: ``on``/``off`` are the two answers,
    and every call's keyword arguments are kept for the test to read."""

    def __init__(self, on, off):
        self.answers = {True: on, False: off}
        self.calls: list[dict] = []

    def __call__(self, players, projections, squad, bank, free_transfers, events, decay, **options):
        self.calls.append({"free_transfers": free_transfers, "events": events, **options})
        return self.answers[options["pin_chip"][1]]


@pytest.fixture
def held(monkeypatch, stub_lineup):
    """Patch the held chips simulate sees; returns the setter."""

    def set_held(*chips: HeldChip) -> None:
        monkeypatch.setattr(whatif, "_held", lambda cfg, inputs: tuple(chips))

    set_held(WC1)
    return set_held


@pytest.fixture
def no_fh_prices(monkeypatch):
    calls = []

    def fake(*args, **kwargs):
        calls.append((args, kwargs))
        return {1: (60.0, SQUAD, SQUAD[:11])}

    monkeypatch.setattr(whatif, "_free_hit_prices", fake)
    return calls


# ----------------------------------------------------------------- band_for


@pytest.mark.parametrize(
    ("kind", "net", "unsure", "band", "lean"),
    [
        (WILDCARD, 8.0, False, PLAY, None),        # the edge is a Play
        (WILDCARD, 7.99, False, MARGINAL, PLAY),
        (WILDCARD, 0.0, False, MARGINAL, HOLD),     # no gain at all leans hold
        (WILDCARD, -7.99, False, MARGINAL, HOLD),
        (WILDCARD, -8.0, False, HOLD, None),
        (FREE_HIT, 4.0, False, PLAY, None),
        (FREE_HIT, 3.9, False, MARGINAL, PLAY),
        (FREE_HIT, -4.0, False, HOLD, None),
        # A solve stopped on time doubles the margin before it will call it.
        (WILDCARD, 15.99, True, MARGINAL, PLAY),
        (WILDCARD, 16.0, True, PLAY, None),
        (FREE_HIT, -7.9, True, MARGINAL, HOLD),
        (FREE_HIT, -8.0, True, HOLD, None),
    ],
)
def test_the_band_reads_the_net_against_the_chip_margin(kind, net, unsure, band, lean):
    assert band_for(kind, net, unsure) == (band, lean)


def test_the_margins_are_the_expert_bands():
    assert WHATIF_MARGIN == {WILDCARD: 8.0, FREE_HIT: 4.0}


# ----------------------------------------------------------------- simulate


def test_the_net_is_the_difference_of_the_two_objectives(monkeypatch, held):
    # On: objective 120 after paying a 30-point wildcard bar. Off: 118, no bar.
    # net 2, bars_diff 30, gain 32 — and +2 on an 8-point margin is Marginal,
    # leaning play.
    weekly_on = {6: 58.0, 7: 56.0, 8: 55.0, 9: 54.0, 10: 52.0, 11: 51.0}
    weekly_off = {6: 55.0, 7: 50.0, 8: 50.0, 9: 50.0, 10: 50.0, 11: 50.0}
    solver = FakeSolver(
        _side(120.0, bars=30.0, week1=WILDCARD, weekly=weekly_on),
        _side(118.0, weekly=weekly_off),
    )
    monkeypatch.setattr(whatif, "optimize_path", solver)

    result = simulate(_week(), config(chips=True), WILDCARD, minutes_source="Wednesday's scout")

    assert result.net == pytest.approx(2.0)
    assert result.bars_diff == pytest.approx(30.0)
    assert result.gain == pytest.approx(32.0)
    assert (result.band, result.lean, result.unsure) == (MARGINAL, PLAY, False)
    assert result.margin == 8.0
    assert result.weekly_gain == pytest.approx({6: 3.0, 7: 6.0, 8: 5.0, 9: 4.0, 10: 2.0, 11: 1.0})
    assert result.on.chips == {6: WILDCARD}
    assert result.off.chips == {}
    assert result.window == WINDOW and result.event == 6 and result.kind == WILDCARD
    assert result.fh_terms is None and result.hold_week is None
    assert result.minutes_source == "Wednesday's scout"


def test_both_solves_are_pinned_and_see_the_same_board(monkeypatch, held, no_fh_prices):
    held(WC1, FH1)
    solver = FakeSolver(_side(100.0, week1=WILDCARD), _side(100.0))
    monkeypatch.setattr(whatif, "optimize_path", solver)

    simulate(_week(), config(chips=True), WILDCARD, time_limit=45)

    on, off = solver.calls
    assert on["pin_chip"] == (WILDCARD, True) and off["pin_chip"] == (WILDCARD, False)
    # One set of free-hit prices, priced once, handed to both: the same object.
    assert len(no_fh_prices) == 1
    assert on["freehit_prices"] is off["freehit_prices"] is not None
    # Never a pinned opening count: the uncapped wildcard week lives only in
    # the unpinned branch of optimize_path.
    assert on["forced_first_transfers"] is None and off["forced_first_transfers"] is None
    assert on["time_limit"] == off["time_limit"] == 45
    assert on["selling_prices"] == off["selling_prices"] == {1: 50}
    assert on["held_chips"] == off["held_chips"] == (WC1, FH1)
    assert on["events"] == off["events"] == WINDOW


def test_no_free_hit_in_play_prices_nothing(monkeypatch, held, no_fh_prices):
    solver = FakeSolver(_side(100.0, week1=WILDCARD), _side(100.0))
    monkeypatch.setattr(whatif, "optimize_path", solver)

    simulate(_week(), config(chips=True), WILDCARD)

    assert no_fh_prices == []
    assert solver.calls[0]["freehit_prices"] is None


def test_a_board_with_no_free_hit_squad_is_no_what_if(monkeypatch, held):
    held(FH1)
    monkeypatch.setattr(whatif, "_free_hit_prices", lambda *a, **k: None)
    monkeypatch.setattr(whatif, "optimize_path", FakeSolver(None, None))

    assert simulate(_week(), config(chips=True), FREE_HIT) is None


@pytest.mark.parametrize("missing", ["on", "off"])
def test_a_solve_without_an_answer_is_no_what_if(monkeypatch, held, missing):
    on = None if missing == "on" else _side(100.0, week1=WILDCARD)
    off = None if missing == "off" else _side(100.0)
    monkeypatch.setattr(whatif, "optimize_path", FakeSolver(on, off))

    assert simulate(_week(), config(chips=True), WILDCARD) is None


def test_a_chip_not_held_this_week_is_no_what_if(monkeypatch, held):
    held(BB1)
    solver = FakeSolver(_side(1.0), _side(1.0))
    monkeypatch.setattr(whatif, "optimize_path", solver)

    assert simulate(_week(), config(chips=True), WILDCARD) is None
    assert solver.calls == []


@pytest.mark.parametrize(("on_proven", "off_proven"), [(False, True), (True, False)])
def test_either_solve_stopped_on_time_makes_it_unsure(monkeypatch, held, on_proven, off_proven):
    # net +10: a Play on the plain 8-point margin, but under the 16 an unsure
    # answer needs — so Marginal, leaning play.
    monkeypatch.setattr(
        whatif,
        "optimize_path",
        FakeSolver(_side(110.0, week1=WILDCARD, proven=on_proven), _side(100.0, proven=off_proven)),
    )

    result = simulate(_week(), config(chips=True), WILDCARD)

    assert result.unsure is True
    assert (result.band, result.lean) == (MARGINAL, PLAY)


def test_a_wildcard_after_entered_moves_refunds_their_hits(monkeypatch, held):
    # He entered two moves on one free transfer: one 4-point hit. A wildcard
    # now folds both moves in and refunds the hit, and the week's free
    # transfers go back to the pre-move count — the on-solve starts from
    # ft_before, the off-solve from what the moves left.
    executed = make_executed(gw=6, transfers_in=[17, 18], transfers_out=[6, 7], ft_before=1, ft_after=0)
    solver = FakeSolver(_side(100.0, week1=WILDCARD), _side(100.0))
    monkeypatch.setattr(whatif, "optimize_path", solver)

    result = simulate(_week(executed=executed, free_transfers=0), config(chips=True), WILDCARD)

    on, off = solver.calls
    assert on["free_transfers"] == 1 and off["free_transfers"] == 0
    assert result.refunded_hits == 1
    assert result.net == pytest.approx(4.0)       # 0 between the objectives, +4 refunded
    assert result.gain == pytest.approx(4.0)
    # The recorded moves still bind both solves (conservative, accepted).
    assert on["lock"] is not None and on["lock"].keep == frozenset({17, 18})
    assert off["lock"] == on["lock"]


def test_a_wildcard_with_no_entered_moves_refunds_nothing(monkeypatch, held):
    solver = FakeSolver(_side(100.0, week1=WILDCARD), _side(100.0))
    monkeypatch.setattr(whatif, "optimize_path", solver)

    result = simulate(_week(free_transfers=2), config(chips=True), WILDCARD)

    assert [call["free_transfers"] for call in solver.calls] == [2, 2]
    assert result.refunded_hits == 0


def test_the_free_hit_week_drops_its_bench_share_and_sums_to_the_net(monkeypatch, held, no_fh_prices):
    # The free-hit team: players 21–35. Its eleven (21–31) earn 66 with the
    # armband in; its bench (32–35) is projected 10 each, 40 in all, which the
    # path's week-1 figure carries at BENCH_WEIGHT 0.1: 66 + 4 = 70. Stripped,
    # the free-hit week is 66 against the off-path's 60: one_week +6.
    held(FH1)
    fh_squad = list(range(21, 36))
    fh_xi = list(range(21, 32))
    projections = _projections({pid: {6: 10.0} for pid in range(32, 36)})
    on = _side(
        95.0, bars=12.0, week1=FREE_HIT,
        weekly={6: 70.0, 7: 50.0, 8: 50.0, 9: 50.0, 10: 50.0, 11: 50.0},
        fh_squad=fh_squad, fh_xi=fh_xi,
    )
    off = _side(97.0, weekly={6: 60.0, 7: 51.0, 8: 51.0, 9: 50.0, 10: 50.0, 11: 50.0})
    monkeypatch.setattr(whatif, "optimize_path", FakeSolver(on, off))

    result = simulate(_week(projections=projections), config(chips=True), FREE_HIT)

    assert result.weekly_gain[6] == pytest.approx(6.0)
    assert result.net == pytest.approx(-2.0)
    assert result.gain == pytest.approx(10.0)        # −2 + 12 − 0
    terms = result.fh_terms
    assert terms.one_week == pytest.approx(6.0)
    assert terms.knock_on == pytest.approx(4.0)      # gain − one_week
    assert terms.keeping == pytest.approx(12.0)      # bars_diff
    assert terms.one_week + terms.knock_on - terms.keeping == pytest.approx(result.net)
    assert (result.band, result.lean) == (MARGINAL, HOLD)


def test_hold_names_the_week_the_off_path_plays_it(monkeypatch, held):
    # The path without it wildcards in GW9 and comes out 10 ahead: Hold, GW9,
    # worth 10 more than now. Three weeks out is not provisional.
    later = PlannedMove(event=9, transfers_in=[], transfers_out=[], hits=0, chip=WILDCARD)
    monkeypatch.setattr(
        whatif, "optimize_path",
        FakeSolver(_side(100.0, bars=30.0, week1=WILDCARD), _side(110.0, bars=30.0, moves=[later])),
    )

    result = simulate(_week(), config(chips=True), WILDCARD)

    assert result.band == HOLD
    assert result.off.chips == {9: WILDCARD}
    assert result.hold_week == whatif.HoldWeek(event=9, value=10.0, source="off_path", provisional=False)


def test_a_hold_week_more_than_four_weeks_out_is_provisional(monkeypatch, held):
    later = PlannedMove(event=11, transfers_in=[], transfers_out=[], hits=0, chip=WILDCARD)
    monkeypatch.setattr(
        whatif, "optimize_path",
        FakeSolver(_side(100.0, week1=WILDCARD), _side(109.0, moves=[later])),
    )

    result = simulate(_week(), config(chips=True), WILDCARD)

    assert result.hold_week.event == 11 and result.hold_week.provisional is True


def test_a_free_hit_held_past_the_window_names_the_calendar_week(monkeypatch, held, no_fh_prices):
    held(FH1)
    calendar = ChipCalendar(
        entries=(CalendarEntry(held=FH1, saved_for=12, value=14.0, bars={6: 24.0}),),
        discount=0.97, proxy_scale={}, horizon_end=19,
    )
    monkeypatch.setattr(
        whatif, "optimize_path",
        FakeSolver(_side(90.0, bars=24.0, week1=FREE_HIT, fh_squad=SQUAD, fh_xi=SQUAD[:11]), _side(100.0)),
    )

    result = simulate(_week(calendar=calendar), config(chips=True), FREE_HIT)

    assert result.band == HOLD
    assert result.hold_week == whatif.HoldWeek(event=12, value=14.0, source="calendar", provisional=True)


def test_a_wildcard_held_past_the_window_invents_no_week(monkeypatch, held):
    calendar = ChipCalendar(
        entries=(CalendarEntry(held=WC1, saved_for=None, value=None, bars={gw: 35.0 for gw in WINDOW}),),
        discount=0.97, proxy_scale={}, horizon_end=19,
    )
    monkeypatch.setattr(
        whatif, "optimize_path",
        FakeSolver(_side(80.0, bars=35.0, week1=WILDCARD), _side(100.0)),
    )

    result = simulate(_week(calendar=calendar), config(chips=True), WILDCARD)

    assert result.hold_week == whatif.HoldWeek(event=None, value=35.0, source="past_window", provisional=False)


def test_a_marginal_lean_hold_also_names_its_week(monkeypatch, held):
    later = PlannedMove(event=8, transfers_in=[], transfers_out=[], hits=0, chip=WILDCARD)
    monkeypatch.setattr(
        whatif, "optimize_path",
        FakeSolver(_side(100.0, week1=WILDCARD), _side(103.0, moves=[later])),
    )

    result = simulate(_week(), config(chips=True), WILDCARD)

    assert (result.band, result.lean) == (MARGINAL, HOLD)
    assert result.hold_week.event == 8


def test_a_play_names_no_hold_week(monkeypatch, held):
    monkeypatch.setattr(
        whatif, "optimize_path", FakeSolver(_side(120.0, week1=WILDCARD), _side(100.0)),
    )

    assert simulate(_week(), config(chips=True), WILDCARD).hold_week is None


# ----------------------------------------------------------- verdict_minutes


def test_the_minutes_come_from_the_newest_verdict_and_only_its_applied_calls(tmp_path: Path):
    store = Store(tmp_path / DB_NAME)
    store.save_run(6, "early", "r", {"adjustments": [{"player_id": 9, "expected_minutes": 0.0}]})
    store.save_run(
        6, "scout", "r",
        {
            "adjustments": [{"player_id": 3, "expected_minutes": 10.0, "reason": "knee"}],
            "unapplied": [{"player_id": 4, "expected_minutes": 0.0, "reason": "noted"}],
        },
    )

    overrides, label, record = verdict_minutes(store, 6)

    ts = store.latest_verdict(6, ("early", "scout", "deadline")).ts
    day = datetime.fromisoformat(ts).astimezone(UTC).strftime("%A")
    assert overrides == {3: 10.0}
    assert label == f"{day}'s scout"
    assert record["adjustments"][0]["player_id"] == 3


def test_an_early_scout_is_named_in_full(tmp_path: Path):
    store = Store(tmp_path / DB_NAME)
    store.save_run(6, "early", "r", {"adjustments": []})

    overrides, label, _ = verdict_minutes(store, 6)

    assert overrides == {}
    assert label.endswith("'s early scout")


def test_no_verdict_means_base_minutes(tmp_path: Path):
    assert verdict_minutes(Store(tmp_path / DB_NAME), 6) == ({}, None, None)


# ----------------------------------------------------------------- end to end


def test_the_real_solver_end_to_end_keeps_the_invariants(tmp_path: Path):
    """The pipeline universe, the real preamble and the real solver: whatever
    the board, the on side plays the free hit this week and the off side does
    not, net is the difference of the objectives, and the parts add up."""
    cfg = config(chips=True, state_dir=tmp_path)
    store = Store(tmp_path / DB_NAME)
    week = prepare_week(cfg, make_client(pipeline_routes()), store)

    result = simulate(week, cfg, FREE_HIT)

    assert result is not None
    event = week.effective.event.id
    assert result.on.chips.get(event) == FREE_HIT
    assert result.off.chips.get(event) != FREE_HIT
    assert result.net == pytest.approx(result.on.path.objective - result.off.path.objective)
    assert result.gain == pytest.approx(result.net + result.bars_diff)
    terms = result.fh_terms
    assert terms.one_week + terms.knock_on - terms.keeping == pytest.approx(result.net)
    assert result.band in (PLAY, MARGINAL, HOLD)
    # Each side's lineup is picked from its own squad: the free-hit fifteen on
    # the on side, the standing plan's squad on the off side.
    assert set(result.on.lineup.xi) <= set(result.on.path.week1_freehit_squad)
    assert set(result.off.lineup.xi) <= set(result.off.plan.squad)


def test_next_break_finds_the_first_long_gap():
    start = datetime(2026, 10, 3, 10, tzinfo=UTC)
    gaps = {5: 0, 6: 7, 7: 14, 8: 21, 9: 35, 10: 42}  # GW8 -> GW9 is 14 days
    events = [SimpleNamespace(id=gw, deadline_time=start + timedelta(days=d)) for gw, d in gaps.items()]
    assert whatif.next_break(events, 6) == (8, 9)
    assert whatif.next_break(events, 9) is None


def test_week1_lineup_fields_the_free_hit_team_on_a_free_hit_week(monkeypatch):
    seen = {}
    monkeypatch.setattr(whatif, "pick_lineup", lambda squad, *rest: seen.setdefault("squad", squad))
    fh = list(range(100, 115))
    path = PlannedPath(moves=[], objective=0.0, weekly_xp={}, week1_chip=FREE_HIT, week1_freehit_squad=fh)
    plan = Plan(squad=SQUAD, xi=SQUAD[:11], transfers_in=[], transfers_out=[], hits=0,
                xp_total=0.0, objective=0.0, path=path)
    week = SimpleNamespace(effective=SimpleNamespace(players={pid: SimpleNamespace(element_type=1) for pid in fh + SQUAD}),
                           projections={})
    whatif.week1_lineup(plan, path, week, 6)
    assert seen["squad"] == fh
