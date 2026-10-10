"""Tests for the orchestrator: the clock that starts a run, and the run.

``run_pipeline`` is the only place the whole of the bot is assembled, so the
tests here are integration tests: a real ``FplClient`` over a fake transport,
the real minutes model, the real solver and the real renderer, over the
``PIPELINE_*`` universe in :mod:`tests.fixtures`. What they assert is that the
pieces are wired to each other — the injured man we own is never fielded, the
shortlist has more than one plan on it, the chip panel is priced off a squad
that exists — rather than the numbers those pieces produce, which are pinned
by the unit tests of the modules that produce them.

The manager is the one piece stubbed rather than run. His loop is a
conversation with an API and is tested against a scripted one in
:mod:`tests.test_manager_agent`; what matters here is the wiring around it,
which is why these runs hand a canned decision back and then ask what the
report, the store and the phone did with it.
"""

import argparse
import copy
import json
import sys
from types import SimpleNamespace
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import NamedTuple

import anthropic
import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.chips import TRIPLE_CAPTAIN, held_for, whole_season
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import GwHistory, Player
from aigaffer.ledger import Observation
from aigaffer.manager import agent
from aigaffer.manager.agent import ManagerDecision
from aigaffer.manager.briefing import CHIP_ALREADY_PLAYED
from aigaffer.news_ledger import ledger_path
from aigaffer.model.xp import PlayerProjection, projected_events
from aigaffer.orchestrator import (
    CALENDAR_FAILED,
    CHIP_SPENT,
    NO_CHIPS,
    PipelineError,
    PipelineInputs,
    PreparedWeek,
    SolveResult,
    _calendar,
    _fielded_lineup,
    _held,
    _entered_lineup,
    _recorded_lineup,
    _selling_prices,
    _week1_lock,
    _stored_actions,
    build_projections,
    decide_mode,
    diff_actions,
    fetch_inputs,
    history_pool,
    prepare_week,
    run_pipeline,
    solve,
)
from aigaffer.report import render
from aigaffer.recording import record_transfers_made
from aigaffer.report.render import render_report, working_from
from aigaffer.report.telegram import KEYBOARD, send_report
from aigaffer.solver.calendar import CHIP_DISCOUNT
from aigaffer.solver.lineup import Lineup, attacking_evs, pick_lineup
from aigaffer.solver.multiweek import FALLBACK_BARS, PlannedPath
from aigaffer.solver.optimizer import FORWARD, MIDFIELDER, Plan
from aigaffer.store import Store
from tests.fixtures import (
    HISTORY_PATH,
    PICKS_15_IDS,
    PICKS_PATH,
    TEAM_ID,
    TRANSFERS_PATH,
    CountingTransport,
    config,
    make_client,
    pipeline_routes,
    ELEMENT_SUMMARY_JSON,
    HISTORY_JSON,
    PICKS_15_JSON,
    PIPELINE_BOOTSTRAP_JSON,
    PIPELINE_ELEMENTS_JSON,
    PIPELINE_FIXTURES_JSON,
    TRANSFERS_JSON,
    fake_fpl_transport,
    make_executed,
)

TOKEN = "1234:super-secret-bot-token"
# The fixture's GW2 deadline itself: zero hours left, past every floor the
# withholding rule measures against (tests/test_manager_retry.py). The runs
# below whose manager fails assert the labelled, saved report, which is the
# branch this clock puts them on — said here rather than left to the wall
# clock, which only agrees because 2025 is behind us.
PAST_THE_FLOOR = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)


DEADLINE = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def chips_off_from_the_environment(monkeypatch):
    """The same default for the runs that build their configuration from the
    environment — the command line's — which is :func:`config`'s rule by
    the switch's own name."""
    monkeypatch.setenv("AIGAFFER_CHIPS", "off")


# --- the mode clock --------------------------------------------------------


@pytest.mark.parametrize(
    ("hours_to_go", "mode"),
    [
        # The windows are contiguous: each stays open until the next report's
        # territory, so a tick that lands late — GitHub dropped nine hours of
        # cron once — still calls for the report instead of standing down.
        # has_run in __main__ is what keeps a late-open window from sending
        # the same report twice.
        (0.5, "reminder"),
        (1.5, "reminder"),
        (2, "reminder"),
        (3, "reminder"),
        (3.5, "deadline"),
        (6, "deadline"),
        (17.6, "deadline"),
        # The deadline report's anchor is T-18h, after Friday's press
        # conferences: 18 is still its window, and 20 (where it used to be
        # called) is the scout's.
        (18, "deadline"),
        (18.1, "scout"),
        (20, "scout"),
        (22.5, "scout"),
        (23.9, "scout"),
        (24, "scout"),
        (25, "scout"),
        (36, "scout"),
        (36.5, "scout"),
        (48, "scout"),
        (60, "scout"),
        (60.5, None),
        (100, None),
    ],
)
def test_the_mode_follows_the_clock(hours_to_go, mode):
    assert decide_mode(DEADLINE - timedelta(hours=hours_to_go), DEADLINE) == mode


def test_a_deadline_that_has_gone_is_no_window():
    # The gameweek is under way; the next one is what the next run is about.
    assert decide_mode(DEADLINE, DEADLINE) is None
    assert decide_mode(DEADLINE + timedelta(minutes=1), DEADLINE) is None


def test_overlapping_windows_prefer_the_report_nearest_the_deadline(monkeypatch):
    # The three windows do not overlap as configured, but the constants are
    # constants and someone will widen one. The precedence is pinned here: the
    # report nearest the deadline wins, because it is the one whose moment
    # cannot be made up later.
    monkeypatch.setattr(orchestrator, "REMINDER_WINDOW", (0, 30))
    monkeypatch.setattr(orchestrator, "DEADLINE_WINDOW", (0, 60))
    monkeypatch.setattr(orchestrator, "SCOUT_WINDOW", (0, 100))

    assert decide_mode(DEADLINE - timedelta(hours=20), DEADLINE) == "reminder"
    assert decide_mode(DEADLINE - timedelta(hours=50), DEADLINE) == "deadline"
    assert decide_mode(DEADLINE - timedelta(hours=90), DEADLINE) == "scout"


# --- the missed full report, caught up at the reminder ----------------------


def _auto_args():
    return argparse.Namespace(command="auto", force=False)


def _client_hours_out(hours, gw=2):
    event = SimpleNamespace(
        id=gw, deadline_time=datetime.now(UTC) + timedelta(hours=hours)
    )
    bootstrap = SimpleNamespace(next_event=lambda: event)
    return SimpleNamespace(bootstrap=lambda: bootstrap)


def test_a_missing_full_report_outranks_the_reminder(tmp_path):
    # Every tick before this one was dropped, so there is no full report to
    # remind anyone of. The reminder's hour runs the report it would have
    # been checking instead; the reminder gets a later tick.
    store = Store(tmp_path / "state.db")
    assert cli._mode(_auto_args(), _client_hours_out(2), store) == ("deadline", 2)


def test_the_reminder_runs_once_the_full_report_exists(tmp_path):
    store = Store(tmp_path / "state.db")
    store.save_run(2, "deadline", "md", {})
    assert cli._mode(_auto_args(), _client_hours_out(2), store) == ("reminder", 2)


def test_a_week_fully_reported_stands_down(tmp_path):
    store = Store(tmp_path / "state.db")
    store.save_run(2, "deadline", "md", {})
    store.save_run(2, "reminder", "md", {})
    assert cli._mode(_auto_args(), _client_hours_out(2), store) == (None, 2)


def test_auto_force_overrules_the_store_not_the_clock(tmp_path):
    # --force skips the dedupe; the promotion still consults the store, so
    # with the full report on record the reminder hour stays the reminder's.
    args = argparse.Namespace(command="auto", force=True)
    store = Store(tmp_path / "state.db")
    store.save_run(2, "deadline", "md", {})
    store.save_run(2, "reminder", "md", {})
    assert cli._mode(args, _client_hours_out(2), store) == ("reminder", 2)


def test_a_reminder_asked_for_by_name_is_not_promoted(tmp_path):
    # Promotion is auto's business: a person naming the reminder on the
    # command line gets the reminder, missing full report or not.
    args = argparse.Namespace(command="reminder", force=False)
    store = Store(tmp_path / "state.db")
    assert cli._mode(args, _client_hours_out(2), store) == ("reminder", 2)


# --- the pipeline ----------------------------------------------------------


def preseason_bootstrap() -> dict:
    """The universe before a ball is kicked: GW1 next, nothing current."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    payload["events"][0].update(is_current=False, is_next=True, finished=False)
    payload["events"][1].update(is_next=False)
    return payload


def bullets(report: str, heading: str) -> list[str]:
    """The list items under ``## heading``."""
    lines = report.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"## {heading}"))
    rest = lines[start + 1 :]
    end = next((i for i, line in enumerate(rest) if line.startswith("## ")), len(rest))
    return [line for line in rest[:end] if line.startswith("- ")]


class Run(NamedTuple):
    report: str
    cfg: Config
    store: Store


@pytest.fixture(scope="module")
def scout_run(tmp_path_factory) -> Run:
    """One scout run, shared: the pipeline is deterministic and its half-dozen
    solves are not worth repeating for every assertion about the same report."""
    state = tmp_path_factory.mktemp("scout_run") / "state"
    cfg = config(state_dir=state)
    store = Store(state / "aigaffer.db")
    report = run_pipeline(cfg, make_client(pipeline_routes()), store, "scout")
    return Run(report=report, cfg=cfg, store=store)


def test_the_report_is_the_whole_report(scout_run):
    headings = [line for line in scout_run.report.splitlines() if line.startswith("#")]

    assert headings[0] == "# AI Gaffer — GW2 scout"
    assert headings[1] == "## Do this"
    assert headings[3].startswith("## Starting XI (")
    assert [headings[2], *headings[4:]] == [
        "## Recommendation",
        "## Candidate plans",
        # The window is the default engine, so an ordinary run has somewhere
        # it is going as well as something it is doing.
        "## The road ahead",
        "## Chip EV",
        "## Watchlist",
    ]
    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in scout_run.report


def test_the_window_the_projections_cover_is_the_window_the_planner_plans(
    seam, monkeypatch
):
    # The planner and the projections have to agree about which gameweeks
    # exist: a window with a gameweek nobody projected in it is a window
    # planned on zeros, and the arithmetic that says how long it is lives in
    # one place.
    asked: list[dict] = []
    real = orchestrator.generate_plans

    def spy(*args, **kwargs):
        asked.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "generate_plans", spy)
    cfg = replace(seam.cfg, chips=True)
    _, projections = build_projections(seam.inputs, cfg)

    solve(seam.inputs, projections, cfg)

    covered = sorted({gw for p in projections.values() for gw in p.per_gw})
    assert len(covered) == cfg.horizon and covered[0] == seam.inputs.event.id
    # Both sets the live rules hand out are in hand — the history's GW1
    # wildcard falls outside the 2-19 window and spends nothing — derived in
    # one place and threaded through the sweep. A solve asked without a ledger
    # reading, or without a calendar, passes those absences through too.
    held = asked[0].pop("held_chips")
    assert [chip.id for chip in held] == [
        "bench_boost@19", "triple_captain@19", "wildcard@19", "free_hit@19",
        "bench_boost@38", "triple_captain@38", "wildcard@38", "free_hit@38",
    ]
    assert asked == [
        {
            "projections_events": covered,
            "decay": cfg.decay,
            "planner": "multi",
            # No calendar handed in, so no bars: the window's fallback.
            "bars": None,
            "selling_prices": None,
            # Nothing recorded this week, so nothing locked.
            "lock": None,
        }
    ]


def test_the_window_answers_and_the_recommendation_carries_its_path(scout_run):
    # The engine is on by default, so an ordinary run comes back with a plan
    # that knows what its opening move is for.
    decision = scout_run.store.last_runs(1)[0]["decision"]

    assert decision["engine"] == "multi"
    assert decision["path"], "a window with nothing after this week is not one"
    for move in decision["path"]:
        assert move["event"] > 2
        assert len(move["in"]) == len(move["out"]) >= 1
        assert all(isinstance(pid, int) for pid in move["in"] + move["out"])
        assert isinstance(move["hits"], int)
        assert isinstance(move["chip"], str), "the forecast reads the path's chip weeks"


def test_the_report_says_where_the_recommendation_is_going(scout_run):
    ahead = bullets(scout_run.report, "The road ahead")
    decision = scout_run.store.last_runs(1)[0]["decision"]

    assert len(ahead) == len(decision["path"])
    assert ahead[0].startswith(f"- GW{decision['path'][0]['event']}: out ")
    assert (
        "Advisory — re-planned every run; only this week's moves are ever made."
        in scout_run.report
    )
    assert "Single-week engine" not in scout_run.report


def test_the_single_week_planner_is_recorded_as_the_engine_it_is(tmp_path):
    # AIGAFFER_PLANNER=single: the other engine, on purpose. There is no path
    # to print and nothing to apologise for.
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state", planner="single")

    report = run_pipeline(cfg, make_client(pipeline_routes()), store, "scout")
    decision = store.last_runs(1)[0]["decision"]

    assert decision["engine"] == "single"
    assert "path" not in decision
    assert "## The road ahead" not in report
    assert "Single-week engine" not in report


def test_a_window_that_answers_nothing_says_so_under_the_shortlist(
    monkeypatch, tmp_path
):
    # The multi-week engine was asked for and came back with nothing, so the
    # single-week solver drew this shortlist up. The report says which.
    # Every window solve comes back with nothing, which is what an infeasible
    # board or a sweep that ran out of time looks like from here.
    monkeypatch.setattr("aigaffer.solver.plans.optimize_path", lambda *a, **k: None)
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        store,
        "scout",
    )

    assert "Single-week engine (multi-week solve unavailable this run)." in report
    assert store.last_runs(1)[0]["decision"]["engine"] == "single"


def test_a_draft_never_claims_the_window_was_unavailable(tmp_path):
    # Fifteen players from nothing is not a question the window is asked —
    # a whole squad will not fit under a gameweek's transfer ceiling — so the
    # single-week solver answering it is not a degradation to report.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(routes),
        store,
        "scout",
    )

    assert "Single-week engine" not in report
    assert store.last_runs(1)[0]["decision"]["engine"] == "single"


def test_the_shortlist_offers_more_than_one_plan(scout_run):
    # generate_plans ran, and the report ranked what it came back with.
    plans = bullets(scout_run.report, "Candidate plans")

    assert len(plans) > 1
    assert sum("recommended" in row for row in plans) == 1


def test_the_injured_player_we_own_is_not_fielded(scout_run):
    # Ito is in the picks and out for the gameweek: minutes, to xP, to lineup.
    eleven = bullets(scout_run.report, "Starting XI")[:4]

    assert not any("Ito" in row for row in eleven)


def test_the_run_is_recorded_with_its_report(scout_run):
    assert scout_run.store.has_run(2, "scout") is True
    assert scout_run.store.has_run(2, "deadline") is False

    written = scout_run.cfg.state_dir / "reports" / "gw2-scout.md"
    assert written.read_text(encoding="utf-8") == scout_run.report


def test_the_root_gw_file_carries_the_report(tmp_path):
    # Beside README.md, where the GitHub homepage's file listing shows it:
    # GW{n}.md is the copy a visitor reads without digging into state/. Its
    # root is the directory holding state_dir, which on a CI checkout is the
    # checkout and here is a tmp_path the test owns.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")

    report = run_pipeline(cfg, make_client(pipeline_routes()), store, "scout")

    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == report


def test_the_readme_points_at_the_latest_verdict(tmp_path):
    # GitHub's tab row cannot host GW{n}.md, so the README carries one
    # marked line the run rewrites: a link to the verdict the homepage
    # would otherwise bury. Only the marked line moves; without a marker —
    # or without a README at all — the run touches nothing and says nothing.
    cfg = config(state_dir=tmp_path / "state")
    readme = tmp_path / "README.md"
    readme.write_text(
        "# AI Gaffer\n\nplaceholder <!-- latest-verdict -->\n\nThe pitch.\n",
        encoding="utf-8",
    )

    run_pipeline(
        cfg, make_client(pipeline_routes()), Store(cfg.state_dir / "a.db"), "scout"
    )

    text = readme.read_text(encoding="utf-8")
    assert "📋 **Latest verdict: [GW2](GW2.md)** <!-- latest-verdict -->" in text
    assert "placeholder" not in text
    assert text.startswith("# AI Gaffer\n"), "the rest of the file is untouched"
    assert text.endswith("The pitch.\n")


def test_a_readme_without_the_marker_is_left_alone(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    readme = tmp_path / "README.md"
    readme.write_text("# Fork without the line\n", encoding="utf-8")

    run_pipeline(
        cfg, make_client(pipeline_routes()), Store(cfg.state_dir / "a.db"), "scout"
    )

    assert readme.read_text(encoding="utf-8") == "# Fork without the line\n"


def test_the_deadline_run_overwrites_the_scouts_root_file(tmp_path):
    # Latest wins at the root — the midweek scout report stands until the
    # deadline run replaces it with the operative plan — while the per-mode
    # history in state/reports keeps both.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    scout = run_pipeline(cfg, client, store, "scout")
    deadline = run_pipeline(cfg, client, store, "deadline")

    assert scout != deadline
    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == deadline
    reports = cfg.state_dir / "reports"
    assert (reports / "gw2-scout.md").read_text(encoding="utf-8") == scout
    assert (reports / "gw2-deadline.md").read_text(encoding="utf-8") == deadline


def test_the_decision_says_what_was_decided(scout_run):
    decision = scout_run.store.last_runs(1)[0]["decision"]

    assert decision["mode"] == "scout"
    assert decision["event"] == 2
    assert decision["free_transfers"] == 1
    assert decision["captain"] in {element["id"] for element in PIPELINE_ELEMENTS_JSON}
    # A transfer is a swap: the fifteen stays fifteen.
    assert len(decision["transfers_in"]) == len(decision["transfers_out"])
    assert set(decision["chip_evs"]) == {
        "bench_boost",
        "triple_captain",
        "free_hit",
        "wildcard",
    }
    assert isinstance(decision["objective"], float)


def test_the_chip_switch_off_is_phase_2_5_to_the_byte(tmp_path):
    # The whole feature is gated. With the switch off the window is handed no
    # chips and builds the chip-blind model: no chip played, no calendar built,
    # and the report is, character for character, the one a run writes with
    # the switch on and no chip left in hand — every window the rules hand out
    # closed before this gameweek. Nothing held is the pre-chip model, however
    # it came about.
    #
    # Near-tautological at this level, and kept as the end-to-end smoke for
    # the switch: both runs hand the window an empty ``held_chips``, so of
    # course they agree. The guarantee that empty chips build the pre-chip
    # MILP to the byte is pinned where the model is built, in
    # tests/test_multiweek.py: test_no_chips_available_is_byte_for_byte_the_old_solve,
    # test_a_chip_whose_window_misses_the_window_gets_no_variables and
    # test_generate_plans_with_no_held_chips_is_the_pre_calendar_shortlist.
    off = Store(tmp_path / "off.db")
    closed = [{**rule, "start_event": 1, "stop_event": 1} for rule in HALVES[:4]]

    report = run_pipeline(
        config(state_dir=tmp_path / "off", chips=False),
        make_client(pipeline_routes()),
        off,
        "scout",
        send=False,
    )
    nothing_held = run_pipeline(
        config(state_dir=tmp_path / "on", chips=True),
        make_client(halves_routes(closed)),
        Store(tmp_path / "on.db"),
        "scout",
        send=False,
    )

    assert report == nothing_held
    assert "PLAY " not in report
    decision = off.last_runs(1)[0]["decision"]
    assert decision["chip"] == "none" and decision["chip_calendar"] is None


def test_the_decision_records_the_chip_this_week_plays(scout_run):
    # No manager and the chip switch off (this suite's default), so none is
    # played and the record says so. The field is there either way — the
    # diary reads the same as the phone about what chip, if any, went in.
    assert scout_run.store.last_runs(1)[0]["decision"]["chip"] == "none"


def test_a_free_hit_the_solver_plans_reaches_the_report_and_the_record(
    monkeypatch, tmp_path
):
    # Ruling 4 and the T3 carry end to end, with the solver's answer stubbed so
    # a free hit is actually planned: the report labels the free-hit eleven and
    # the record keeps the chip. No manager, so the chip is the solver's own.
    inputs = fetch_inputs(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
    )
    _, projections = build_projections(inputs, config())
    standing = inputs.squad.player_ids
    positions = {pid: p.element_type for pid, p in inputs.players.items()}
    gw_xp = {pid: pr.per_gw.get(inputs.event.id, 0.0) for pid, pr in projections.items()}
    fh_squad = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
    choice = Plan(
        squad=standing, xi=[], transfers_in=[], transfers_out=[], hits=0,
        xp_total=0.0, objective=0.0,
        path=PlannedPath(
            moves=[], objective=0.0, weekly_xp={}, week1_chip="free_hit",
            week1_freehit_squad=fh_squad, week1_freehit_xi=fh_squad[:11],
        ),
    )
    solved = SolveResult(
        plans=[choice],
        choice=choice,
        lineup=pick_lineup(standing, positions, gw_xp),
        chips=NO_CHIPS,
        draft_mode=False,
    )
    monkeypatch.setattr(orchestrator, "solve", lambda *a, **k: solved)
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
    )

    assert "Free Hit XI (this week only)" in report
    assert "PLAY Free Hit" in report
    assert store.last_runs(1)[0]["decision"]["chip"] == "free_hit"


def test_the_record_keeps_the_squad_it_was_solved_from_and_the_bench(scout_run):
    # "Transfers made" composes a later verdict onto the recorded position only
    # when the verdict was solved from it, so every record says what it started
    # from; and the T-3h reminder diffs the bench of what was entered.
    decision = scout_run.store.last_runs(1)[0]["decision"]

    assert decision["squad_before"] == sorted(PICKS_15_IDS)
    assert len(decision["bench"]) == 4
    assert decision["freehit_squad"] is None and decision["freehit_xi"] is None


def test_a_free_hit_week_records_its_fifteen_and_eleven_at_the_top_level(
    monkeypatch, tmp_path
):
    # The solver's answer stubbed to a planned free hit, as the test above
    # does. The temporary team goes on the record beside the standing squad's
    # moves — and never into solver_actions, whose diff would otherwise start
    # comparing fifteen ids at T-3h.
    inputs = fetch_inputs(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes())
    )
    _, projections = build_projections(inputs, config())
    standing = inputs.squad.player_ids
    positions = {pid: p.element_type for pid, p in inputs.players.items()}
    gw_xp = {pid: pr.per_gw.get(inputs.event.id, 0.0) for pid, pr in projections.items()}
    fh_squad = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
    fh_xi = [1, 3, 4, 10, 12, 5, 6, 11, 14, 17, 7]  # 4-5-1, one keeper
    choice = Plan(
        squad=standing, xi=[], transfers_in=[], transfers_out=[], hits=0,
        xp_total=0.0, objective=0.0,
        path=PlannedPath(
            moves=[], objective=0.0, weekly_xp={}, week1_chip="free_hit",
            week1_freehit_squad=fh_squad, week1_freehit_xi=fh_xi,
        ),
    )
    solved = SolveResult(
        plans=[choice],
        choice=choice,
        lineup=pick_lineup(standing, positions, gw_xp),
        chips=NO_CHIPS,
        draft_mode=False,
    )
    monkeypatch.setattr(orchestrator, "solve", lambda *a, **k: solved)
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
    )
    record = store.decision(2, "scout")

    assert record["freehit_squad"] == fh_squad
    assert record["freehit_xi"] == fh_xi
    assert set(record["solver_actions"]) == {
        "transfers", "captain", "vice", "chip", "formation",
    }


def test_the_chips_are_priced_off_the_squad_we_hold(scout_run):
    decision = scout_run.store.last_runs(1)[0]["decision"]

    # The bench has four fit-enough players on it, so a bench boost is worth
    # something — the number was computed, not defaulted.
    assert decision["chip_evs"]["bench_boost"] > 0
    assert decision["chip_baseline"] == "roll"


def test_a_squad_the_api_will_not_show_is_drafted_instead(tmp_path):
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(routes),
        store,
        "scout",
    )

    signings = bullets(report, "Recommendation")
    assert report.startswith("# AI Gaffer — GW2 scout — initial squad draft")
    assert signings == [signings[0]]  # nobody to sell, so no sales bullet
    assert signings[0].count("£") == 15  # a whole squad bought
    assert store.last_runs(1)[0]["decision"]["free_transfers"] is None


def test_a_preseason_run_drafts_a_squad(tmp_path):
    # No gameweek is current, and the entry endpoints do not answer yet.
    routes = pipeline_routes(bootstrap=preseason_bootstrap())
    for path in (PICKS_PATH, TRANSFERS_PATH, HISTORY_PATH):
        del routes[path]
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(routes),
        store,
        "scout",
    )

    assert report.startswith("# AI Gaffer — GW1 scout — initial squad draft")
    assert len(bullets(report, "Starting XI")) == 5
    assert bullets(report, "Chip EV") == [
        "- Bench boost: +0.0",
        "- Triple captain: +0.0",
        "- Free hit: +0.0",
        "- Wildcard: +0.0 xP over 6 GWs (horizon)",
    ]
    assert store.has_run(1, "scout") is True


def test_one_history_the_api_will_not_serve_does_not_lose_the_report(tmp_path):
    # Ferrer's element-summary rate-limits every try. Two hundred requests a
    # run and one of them failing is a Saturday, not an outage: he falls back
    # on the starts-based minutes guess and the week's report still comes out.
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes(), statuses={"/api/element-summary/5/": 429}),
        store,
        "scout",
    )

    assert report.startswith("# AI Gaffer — GW2 scout")
    assert "Ferrer" in report  # projected from his starts, not written off
    assert store.has_run(2, "scout") is True


def test_a_season_with_no_gameweek_ahead_is_an_error(tmp_path):
    over = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for event in over["events"]:
        event.update(is_next=False, is_current=False)

    with pytest.raises(PipelineError):
        run_pipeline(
            config(state_dir=tmp_path / "state"),
            make_client(pipeline_routes(bootstrap=over)),
            Store(tmp_path / "aigaffer.db"),
            "scout",
        )


def test_a_dry_run_leaves_nothing_behind(tmp_path):
    store = Store(tmp_path / "state" / "aigaffer.db")

    run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        store,
        "deadline",
        send=False,
        save=False,
    )

    assert store.last_runs() == []
    assert not (tmp_path / "state" / "reports").exists()
    assert list(tmp_path.glob("GW*.md")) == []
    # The ledger too: a dry run prices its sales in memory and writes nothing.
    assert store.purchases() == {}
    assert store.squad_record(1) is None


# --- the purchase ledger in the run ----------------------------------------
#
# The ledger's own rules are tested in tests/test_ledger.py; what is pinned
# here is the wiring — every run with a squad maintains it, the selling prices
# it computes are the ones the solver is handed, and the reconciliation line
# reaches the report a person actually reads.


def test_the_first_run_seeds_the_ledger_and_hands_the_solver_true_prices(
    tmp_path, monkeypatch
):
    # Ferrer has risen £0.6m since the season opened, so the seed says he was
    # bought at 119 and the selling rule pays 122 of his listed 125 — and that
    # 122, not the 125, is what reaches the solver's sale side. Everyone else
    # never moved and sells at par.
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    next(e for e in payload["elements"] if e["id"] == FERRER)["cost_change_start"] = 6
    asked: list[dict] = []
    real = orchestrator.generate_plans

    def spy(*args, **kwargs):
        asked.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "generate_plans", spy)
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")

    run_pipeline(
        cfg, make_client(pipeline_routes(bootstrap=payload)), store, "scout",
        send=False,
    )

    held = set(PICKS_15_JSON["picks"][index]["element"] for index in range(15))
    assert store.purchases()[FERRER] == 119
    assert store.purchases()[1] == 55  # unmoved, seeded at his listed price
    assert set(store.purchases()) == held
    assert store.squad_record(1) == {
        "gw": 1, "bank": 28, "player_ids": [p["element"] for p in PICKS_15_JSON["picks"]],
    }
    prices = asked[0]["selling_prices"]
    assert prices[FERRER] == 122
    assert prices[1] == 55
    assert set(prices) == held


def test_the_reminder_maintains_the_ledger_too(tmp_path):
    # The reminder fetched the same picks the full report would, so the ledger
    # is kept current on every mode — a transfer made between the deadline run
    # and the reminder is sighted three hours out, not a week later.
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")

    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    held = {p["element"] for p in PICKS_15_JSON["picks"]}
    assert set(store.purchases()) == held


def test_a_draft_week_has_no_ledger_to_keep(tmp_path):
    # No squad yet: nothing to seed and nothing to snapshot. The ledger's
    # first run is the first run that actually holds fifteen players.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")

    run_pipeline(cfg, make_client(routes), store, "scout", send=False)

    assert store.purchases() == {}
    assert store.squad_record(1) is None


def test_a_bank_the_ledger_did_not_predict_earns_one_line_in_the_report(tmp_path):
    # The picks have rolled to GW2 and the diff against the remembered GW1 is
    # one swap: Reyes (17) out, Quill (16) in. The ledger bought Reyes at 90
    # and he lists at 95, so his sale should have raised 92; Quill cost 50.
    # From a previous bank of 0 that predicts 42, and the game published 28 —
    # £1.4m adrift, which is exactly what the report has to say, once.
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")
    held = [p["element"] for p in PICKS_15_JSON["picks"]]
    previous = [pid if pid != 16 else 17 for pid in held]
    for pid in previous:
        store.record_purchase(pid, buy_price=90 if pid == 17 else 50, gw_seen=1)
    store.record_squad(1, bank=0, player_ids=previous)
    routes = unplayed_routes(midweek_bootstrap(), (1, 90))

    report = run_pipeline(cfg, make_client(routes), store, "scout", send=False)

    assert "the bank is £1.4m below what the purchase ledger predicted" in report
    assert report.count("purchase ledger") == 1


def test_a_bank_the_ledger_predicted_exactly_earns_no_line(tmp_path):
    # Same rolled gameweek, no transfers made, previous bank equal to the
    # published one: the audit passes and the report says nothing about it.
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")
    held = [p["element"] for p in PICKS_15_JSON["picks"]]
    for pid in held:
        store.record_purchase(pid, buy_price=50, gw_seen=1)
    store.record_squad(1, bank=28, player_ids=held)
    routes = unplayed_routes(midweek_bootstrap(), (1, 90))

    report = run_pipeline(cfg, make_client(routes), store, "scout", send=False)

    assert "purchase ledger" not in report


def deep_season_bootstrap() -> dict:
    """Five finished gameweeks behind GW2's deadline — a season old enough
    that a board of six hundred unmoved prices cannot be a market."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    payload["events"][0].update(finished=True)
    for gw in range(3, 7):
        payload["events"].append(
            {
                "id": gw,
                "name": f"Gameweek {gw}",
                "deadline_time": "2026-01-30T17:30:00Z",
                "finished": True,
                "is_previous": False,
                "is_current": False,
                "is_next": False,
                "average_entry_score": 0,
            }
        )
    return payload


def test_a_deep_season_where_no_price_ever_moved_reads_as_a_missing_field(
    tmp_path, capsys
):
    # The model defaults cost_change_start to 0 so a hand-built fixture need
    # not mention it — which means the live API dropping or renaming the field
    # would parse cleanly and quietly seed the ledger at now_cost. Weeks into
    # a season, every element claiming an unmoved price is that regression's
    # signature, and the run says so, once.
    cfg = config(state_dir=tmp_path / "state")

    fetch_inputs(cfg, make_client(pipeline_routes(bootstrap=deep_season_bootstrap())))
    printed = capsys.readouterr().out

    assert orchestrator.COST_CHANGES_MISSING in printed
    assert printed.count("cost_change_start") == 1


def test_a_chip_missing_from_the_rules_is_said_once(tmp_path, capsys):
    # A rename on FPL's side would drop the chip from the plan without a word:
    # the run names what is missing and what it did not recognise, once.
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for rule in payload["chips"]:
        if rule["name"] == "bboost":
            rule["name"] = "benchboost"
    cfg = config(state_dir=tmp_path / "state")

    fetch_inputs(cfg, make_client(pipeline_routes(bootstrap=payload)))
    printed = capsys.readouterr().out

    assert printed.count(orchestrator.CHIP_RULES_GAP) == 1
    assert "chip rules: bench_boost not offered (unknown chips: benchboost)" in printed


def test_every_chip_renamed_says_the_fallback_is_in_force(tmp_path, capsys):
    # Nothing maps, so the planner falls back to whole-season windows and the
    # chips stay planned: the line must not leave it at "not offered".
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for rule in payload["chips"]:
        rule["name"] = "new_" + rule["name"]
    cfg = config(state_dir=tmp_path / "state")

    fetch_inputs(cfg, make_client(pipeline_routes(bootstrap=payload)))
    printed = capsys.readouterr().out

    assert printed.count(orchestrator.CHIP_RULES_GAP) == 1
    assert "whole-season fallback in force" in printed


def test_the_live_chip_rules_say_nothing(tmp_path, capsys):
    fetch_inputs(config(state_dir=tmp_path / "state"), make_client(pipeline_routes()))

    assert orchestrator.CHIP_RULES_GAP not in capsys.readouterr().out


def test_a_single_price_that_moved_proves_the_field_alive(tmp_path, capsys):
    payload = deep_season_bootstrap()
    payload["elements"][0]["cost_change_start"] = 2
    cfg = config(state_dir=tmp_path / "state")

    fetch_inputs(cfg, make_client(pipeline_routes(bootstrap=payload)))

    assert "cost_change_start" not in capsys.readouterr().out


def test_an_opening_month_of_unmoved_prices_is_not_an_anomaly(tmp_path, capsys):
    # One finished gameweek and every price where it started is simply
    # August: the seed of now_cost minus 0 is exact, and a warning here would
    # cry wolf on every fresh season.
    cfg = config(state_dir=tmp_path / "state")

    fetch_inputs(cfg, make_client(pipeline_routes()))

    assert "cost_change_start" not in capsys.readouterr().out


# --- the seam: fetch, project, solve ---------------------------------------
#
# The three stages ``run_pipeline`` is made of, called on their own. What the
# manager agent will do with them is re-project on minutes it has read the
# team news for and solve again, so the tests here are about the two things
# that makes possible: the fetch is a value the later stages can be re-run
# over without touching the API, and an override reaches the projection.

FERRER = 5  # the premium midfielder in the pipeline universe, and its captain
REYES = 17  # the next-best attacker, whose total tops Ferrer's once his bonus goes


class Seam(NamedTuple):
    cfg: Config
    inputs: PipelineInputs


@pytest.fixture(scope="module")
def seam(tmp_path_factory) -> Seam:
    """One fetch, shared: every stage test below re-runs from these inputs,
    which is the point of them being a value."""
    cfg = config(state_dir=tmp_path_factory.mktemp("seam") / "state")
    return Seam(cfg=cfg, inputs=fetch_inputs(cfg, make_client(pipeline_routes())))


def test_the_chips_already_played_are_fetched_once_and_kept(tmp_path):
    # The free-transfer sum reads them and so does the manager's chip panel;
    # a season's chip history fetched twice is one request nobody needed.
    transport = CountingTransport(pipeline_routes())
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)

    inputs = fetch_inputs(config(state_dir=tmp_path / "state"), client)

    assert inputs.chips_used == HISTORY_JSON["chips"]
    assert transport.counts[HISTORY_PATH] == 1


def test_a_manager_with_no_squad_has_played_no_chips(tmp_path):
    # Nothing has been played by somebody who has not played, and the entry
    # endpoints do not answer for him either.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    cfg = config(state_dir=tmp_path / "state")

    inputs = fetch_inputs(cfg, make_client(routes))

    assert inputs.squad is None
    assert inputs.chips_used == []


def test_the_fetch_gathers_everything_the_later_stages_need(seam):
    inputs = seam.inputs

    assert inputs.event.id == 2
    assert inputs.squad is not None
    assert inputs.free_transfers == 1
    assert inputs.players.keys() == {e["id"] for e in PIPELINE_ELEMENTS_JSON}
    assert len(inputs.fixtures) == len(PIPELINE_FIXTURES_JSON)
    # A history each for the squad and the shortlist, and nobody else.
    assert list(inputs.histories) == history_pool(
        inputs.players, inputs.squad.player_ids
    )


def test_the_stages_compose_to_the_report_the_pipeline_wrote(seam, scout_run):
    # The seam is a refactor, not a rewrite: fetch, project and solve in that
    # order still produce the report to the character.
    _, projections = build_projections(seam.inputs, seam.cfg)
    solved = solve(seam.inputs, projections, seam.cfg)

    report = render_report(
        "scout",
        seam.inputs.event,
        solved.plans,
        solved.choice,
        solved.lineup,
        solved.chips,
        seam.inputs.bootstrap,
        projections,
    )

    assert report == scout_run.report
    assert solved.draft_mode is False


def test_an_override_of_no_minutes_writes_a_player_out_of_the_coming_week(seam):
    event = seam.inputs.event.id
    _, before = build_projections(seam.inputs, seam.cfg)

    xmins, after = build_projections(seam.inputs, seam.cfg, {FERRER: 0.0})

    assert before[FERRER].per_gw[event] > 0  # there was something to take away
    assert xmins[FERRER] == 0.0, "the minutes handed back are the coming week's"
    assert after[FERRER].per_gw[event] == 0.0
    assert after[FERRER].attacking_per_gw[event] == 0.0
    # Out this week is not out for the window: the week after, he is back on
    # the minutes the model gave him, so his total is that week's and no more.
    later = [gw for gw in after[FERRER].per_gw if gw > event]
    assert later and all(
        after[FERRER].per_gw[gw] == before[FERRER].per_gw[gw] for gw in later
    )
    assert 0 < after[FERRER].total < before[FERRER].total
    # Nobody else moved: an override is about one player's minutes.
    assert {pid: p for pid, p in after.items() if pid != FERRER} == {
        pid: p for pid, p in before.items() if pid != FERRER
    }


def test_an_adjustment_leaves_the_road_ahead_as_the_model_saw_it(seam):
    # The window the multi-week solver plans over is every gameweek in the
    # projections, so an override that leaked past the coming week would
    # re-plan weeks the team news said nothing about — a player ruled out on
    # Friday sold for the month, his bench boost and his armband in the
    # weeks after priced at nothing. Several players adjusted at once, up and
    # down, and from the coming week on nothing the solver is fed has moved.
    event = seam.inputs.event.id
    dodd = 4  # a 75% doubt, so the model had him on 67.5 minutes
    overrides = {FERRER: 0.0, REYES: 20.0, dodd: 90.0}
    _, before = build_projections(seam.inputs, seam.cfg)

    _, after = build_projections(seam.inputs, seam.cfg, overrides)

    assert after.keys() == before.keys()
    for pid in after:
        for gw, points in before[pid].per_gw.items():
            if gw > event:
                assert after[pid].per_gw[gw] == points, (pid, gw)
                assert (
                    after[pid].attacking_per_gw[gw]
                    == before[pid].attacking_per_gw[gw]
                ), (pid, gw)
    assert all(after[pid].per_gw[event] != before[pid].per_gw[event] for pid in overrides)


def test_a_player_the_api_rules_out_stays_out_beyond_the_coming_week(seam):
    # The other half of the trade. Ito is injured on the API's own flag, which
    # the minutes model reads for every gameweek in the window, so his zero
    # beyond the coming week never came from an override and an override does
    # not lift it: told he is fit for this one, he plays this one.
    ito = 8
    event = seam.inputs.event.id

    xmins, after = build_projections(seam.inputs, seam.cfg, {ito: 90.0})

    assert xmins[ito] == 90.0
    assert after[ito].per_gw[event] > 0
    assert all(points == 0.0 for gw, points in after[ito].per_gw.items() if gw > event)


@pytest.mark.parametrize(
    ("override", "expected"), [(120.0, 90.0), (-30.0, 0.0), (30.0, 30.0)]
)
def test_an_override_is_clamped_to_a_match(seam, override, expected):
    # The overrides come from an agent reading team news; a match is ninety
    # minutes long whatever it thinks it read.
    xmins, _ = build_projections(seam.inputs, seam.cfg, {FERRER: override})

    assert xmins[FERRER] == expected


def test_a_player_given_no_minutes_is_not_fielded(seam, scout_run):
    # Minutes to projection to lineup, which is the whole point of the seam:
    # Ferrer captains the report the pipeline wrote, and told he is not
    # playing the same solve leaves him out of the eleven altogether.
    assert scout_run.store.last_runs(1)[0]["decision"]["captain"] == FERRER

    _, projections = build_projections(seam.inputs, seam.cfg, {FERRER: 0.0})

    solved = solve(seam.inputs, projections, seam.cfg)

    assert FERRER not in solved.lineup.xi


def test_the_pipeline_captains_the_goal_threat_not_the_padded_total(tmp_path):
    # The armband plumbing, end to end: solve hands pick_lineup the attacking
    # slice project_all computed, and on the stock universe nothing would
    # notice if it stopped — Ferrer tops the total and the ceiling both, so
    # the total-ranked fallback crowns the same man. So the universe is bent
    # until the two columns disagree: with his bonus struck out Ferrer's floor
    # sinks below Reyes' total while his goals and assists stay the best on
    # the board. Only a captaincy ranked on the attacking slice finds him now.
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    next(e for e in payload["elements"] if e["id"] == FERRER)["bonus"] = 0
    cfg = config(state_dir=tmp_path / "state")
    inputs = fetch_inputs(cfg, make_client(pipeline_routes(bootstrap=payload)))

    _, projections = build_projections(inputs, cfg)
    solved = solve(inputs, projections, cfg)

    # The scenario holds: among the eleven's attackers, the biggest total and
    # the biggest ceiling are different players — otherwise this proves
    # nothing about which of the two the armband was ranked on.
    gw = inputs.event.id
    positions = {pid: p.element_type for pid, p in inputs.players.items()}
    attacking = attacking_evs(projections, gw)
    candidates = [
        pid for pid in solved.lineup.xi if positions[pid] in (MIDFIELDER, FORWARD)
    ]
    by_total = max(candidates, key=lambda pid: projections[pid].per_gw[gw])
    by_ceiling = max(candidates, key=lambda pid: attacking[pid])
    assert by_total == REYES and by_ceiling == FERRER

    # Ferrer captains on the ceiling; ranked on the total he would not even be
    # vice, which is how this test fails if the plumbing is ever dropped.
    assert solved.lineup.captain == FERRER
    assert solved.lineup.vice == REYES


def test_a_solve_with_no_squad_drafts_a_fifteen(tmp_path):
    # The draft path lives inside solve now, and says so.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    cfg = config(state_dir=tmp_path / "state")
    inputs = fetch_inputs(cfg, make_client(routes))

    _, projections = build_projections(inputs, cfg)
    solved = solve(inputs, projections, cfg)

    assert inputs.squad is None
    assert inputs.free_transfers is None
    assert solved.draft_mode is True
    assert solved.plans == [solved.choice]
    assert len(solved.choice.transfers_in) == 15
    assert solved.chips == NO_CHIPS  # no squad to play a chip against


# --- which chips the window may plan ---------------------------------------
#
# The chips in hand at this gameweek, each with the window it must be played
# in, when the switch is on and there is a squad to play them against. Off, or
# drafting, the set is empty and the window builds Phase 2.5's chip-blind model.

# The rules as the bootstrap serves them: every chip twice, one set to GW19 and
# a fresh one from GW20.
HALVES = [
    {"name": api, "start_event": start, "stop_event": stop}
    for start, stop in ((1, 19), (20, 38))
    for api in ("bboost", "3xc", "wildcard", "freehit")
]
FIRST_SET_ONLY = [rule for rule in HALVES if rule["stop_event"] == 19]


def halves_routes(rules: list[dict] = HALVES, chips_used: list[dict] | None = None) -> dict:
    """The pipeline universe with chip rules on its bootstrap and, given one, a
    chip history of its own (the fixture's is the wildcard in GW1)."""
    routes = pipeline_routes(bootstrap={**PIPELINE_BOOTSTRAP_JSON, "chips": rules})
    if chips_used is not None:
        routes[HISTORY_PATH] = {**HISTORY_JSON, "chips": chips_used}
    return routes


def test_with_no_rules_every_chip_not_played_is_held_all_season(seam):
    # A board with no chip rules is the model before halves: one window per
    # chip over the whole season, the played ones gone.
    inputs = replace(
        seam.inputs,
        bootstrap=seam.inputs.bootstrap.model_copy(update={"chips": []}),
        chips_used=[],
    )

    assert _held(config(chips=True), inputs) == whole_season(
        "bench_boost", "triple_captain", "wildcard", "free_hit"
    )


def test_the_held_chips_are_both_sets_less_the_one_spent(tmp_path):
    # The wildcard went in GW1, which spends the first set's and leaves the
    # second's: held from the start of the season, playable from GW20.
    inputs = fetch_inputs(
        config(state_dir=tmp_path), make_client(halves_routes())
    )

    held = _held(config(chips=True), inputs)

    assert [chip.id for chip in held] == [
        "bench_boost@19", "triple_captain@19", "free_hit@19",
        "bench_boost@38", "triple_captain@38", "wildcard@38", "free_hit@38",
    ]
    assert held_for(held, "wildcard", inputs.event.id) is None


def test_the_chip_switch_off_holds_nothing(seam):
    inputs = replace(seam.inputs, chips_used=[])

    assert _held(config(chips=False), inputs) == ()


def test_a_draft_holds_no_chips_to_plan(seam):
    # No squad, so nothing to play a chip against — the empty set the window
    # reads as "advisory only", which is the pre-chip model.
    inputs = replace(seam.inputs, squad=None)

    assert _held(config(chips=True), inputs) == ()
    assert _calendar(inputs, config(chips=True), {}, None) is None


def test_the_single_week_planner_builds_no_calendar(monkeypatch, tmp_path):
    # Chips on and seven held, but the single-week solver reads no bars: a
    # calendar for it would be projections to GW19 and a dozen weeks of
    # sub-solves nobody consults.
    # None — the same answer as holding nothing — and build_calendar is never
    # asked.
    calls = calendar_spy(monkeypatch)
    inputs = fetch_inputs(
        config(state_dir=tmp_path), make_client(halves_routes())
    )
    _, projections = build_projections(inputs, config(chips=True))
    single = config(chips=True, planner="single")

    assert _held(single, inputs) != ()
    assert _calendar(inputs, single, projections, None) is None
    assert calls == []


def test_the_fielded_lineup_is_the_free_hit_team_on_a_free_hit_week(seam):
    # The critical T3 carry at the orchestrator seam: on a free-hit week the
    # eleven fielded is the temporary team the solver priced, not the standing
    # squad. Any other chip, or none, leaves the decided eleven standing.
    _, projections = build_projections(seam.inputs, seam.cfg)
    positions = {pid: p.element_type for pid, p in seam.inputs.players.items()}
    gw_xp = {
        pid: pr.per_gw.get(seam.inputs.event.id, 0.0)
        for pid, pr in projections.items()
    }
    standing_squad = seam.inputs.squad.player_ids
    standing = pick_lineup(standing_squad, positions, gw_xp)

    # A legal 2/5/5/3 fifteen from the pipeline pool, deliberately not the
    # standing one — Jarvis, Meier, Reyes and Sarr are not in the squad we hold.
    fh_squad = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
    choice = Plan(
        squad=standing_squad, xi=[], transfers_in=[], transfers_out=[], hits=0,
        xp_total=0.0, objective=0.0,
        path=PlannedPath(
            moves=[], objective=0.0, weekly_xp={}, week1_chip="free_hit",
            week1_freehit_squad=fh_squad, week1_freehit_xi=fh_squad[:11],
        ),
    )

    fielded = _fielded_lineup(
        "free_hit", choice, standing, seam.inputs.players, projections,
        seam.inputs.event.id,
    )
    assert set(fielded.xi) <= set(fh_squad)
    assert fielded.xi != standing.xi

    # No chip: the decided eleven is handed straight back, unchanged.
    assert _fielded_lineup(
        "none", choice, standing, seam.inputs.players, projections,
        seam.inputs.event.id,
    ) is standing


# --- rounds nobody has played yet ------------------------------------------
#
# The moment a deadline goes, element-summary grows a history row for that
# gameweek with 0 minutes in it, for every player whose fixture has not
# kicked off yet. Checked against the live endpoint on 2026-08-21, five hours
# after GW1's deadline and two days before his match: Haaland's history was
# one row — round 1, 0 minutes, 0 points, kickoff 2026-08-23T13:00Z — while
# Saka, whose match had been played that evening, had a row of 67 minutes for
# the same round.
#
# A row for a match nobody has played is not a gameweek the player sat out,
# and the fetch takes them out so that nothing downstream has to know the
# difference. The rule is the row's own fixture: a history row names the match
# it belongs to, and /fixtures/ says whether that match has been played. A row
# that carries no fixture id at all falls back on the event's ``finished``
# flag, which is the coarser thing the filter used to be written on.


def midweek_bootstrap() -> dict:
    """GW1 played, GW2's deadline gone and none of its matches finished, GW3
    next. Between a deadline and a kickoff is the whole of the bug."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    payload["events"][0].update(is_current=False, is_next=False, finished=True)
    payload["events"][1].update(is_current=True, is_next=False, finished=False)
    payload["events"].append(
        {
            "id": 3,
            "name": "Gameweek 3",
            "deadline_time": "2025-08-29T17:30:00Z",
            "finished": False,
            "is_previous": False,
            "is_current": False,
            "is_next": True,
            "average_entry_score": 0,
        }
    )
    return payload


def opening_weekend_bootstrap() -> dict:
    """GW1's deadline has gone and not a ball has been kicked; GW2 is next.
    The live universe of 2026-08-21, and the run this fix came out of."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    payload["events"][0].update(finished=False)
    return payload


# The pipeline universe has three fixtures a gameweek; this is one of each, and
# it is the fixture a history row for that round carries unless a test says
# otherwise. Ferrer's club (team 1) plays in every one of them.
ROUND_FIXTURE = {1: 1, 2: 4, 3: 7}


def fixtures_played(*played: int, provisional: tuple[int, ...] = ()) -> list[dict]:
    """The pipeline universe's fixtures, with exactly ``played`` finished.

    ``provisional`` are the ones at full time and not yet data-checked — the
    live shape of a Saturday evening, where the minutes are in the payload and
    ``finished`` is still false.
    """
    payload = copy.deepcopy(PIPELINE_FIXTURES_JSON)
    for fixture in payload:
        fixture["finished"] = fixture["id"] in played
        fixture["finished_provisional"] = (
            fixture["finished"] or fixture["id"] in provisional
        )
    return payload


def summary(*rounds: tuple[int, ...]) -> dict:
    """An element-summary from ``(round, minutes)`` pairs.

    A row names the fixture it belongs to and the fetch's filter reads it, so a
    pair takes that round's fixture from :data:`ROUND_FIXTURE`. A
    ``(round, minutes, fixture)`` triple says which one instead — 0 for a
    payload that carried no fixture id at all.
    """
    return {
        "fixtures": [{"id": 7, "event": 3, "is_home": True}],
        "history": [
            {
                "element": 5,
                "fixture": row[2] if len(row) > 2 else ROUND_FIXTURE[row[0]],
                "round": row[0],
                "minutes": row[1],
                "total_points": 0,
                "bonus": 0,
            }
            for row in rounds
        ],
        "history_past": [],
    }


def unplayed_routes(
    bootstrap: dict,
    *rounds: tuple[int, ...],
    fixtures: list[dict] = PIPELINE_FIXTURES_JSON,
) -> dict:
    """The universe of ``bootstrap``, with ``rounds`` as everybody's history."""
    routes = pipeline_routes(bootstrap, fixtures)
    # The current gameweek moves, and the picks are asked for by its id.
    routes[f"/api/entry/{TEAM_ID}/event/2/picks/"] = PICKS_15_JSON
    routes.update(
        {
            f"/api/element-summary/{element['id']}/": summary(*rounds)
            for element in PIPELINE_ELEMENTS_JSON
        }
    )
    return routes


def test_a_round_nobody_has_played_is_not_history(tmp_path):
    # GW1 was played and GW2 has only been entered, so a history of both is a
    # history of one.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(midweek_bootstrap(), (1, 90), (2, 0))

    inputs = fetch_inputs(cfg, make_client(routes))

    assert inputs.event.id == 3
    assert inputs.histories
    for history in inputs.histories.values():
        assert [entry.round for entry in history] == [1]


def test_an_unplayed_round_does_not_drag_the_minutes_down(tmp_path):
    # Ninety minutes in the gameweek that happened, and a phantom nothing in
    # the one that has not. The mean of what happened is ninety; averaging the
    # phantom in halves him, and the starts floor then hides the damage at 75
    # — a fit ninety-minute player marked down for a match nobody has played.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(midweek_bootstrap(), (1, 90), (2, 0))
    inputs = fetch_inputs(cfg, make_client(routes))

    xmins, _ = build_projections(inputs, cfg)

    assert xmins[FERRER] == 90.0


def test_a_played_fixture_inside_an_unfinished_round_is_kept(tmp_path):
    # The half the event-level cut threw away. GW2 runs Saturday to Monday:
    # Ferrer's match was played on the Saturday and his real eighty-five
    # minutes are already in the payload, while the Monday match has a row of
    # nothing. The event is unfinished either way, so a rule written on the
    # event drops the eighty-five along with the phantom.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        midweek_bootstrap(),
        (1, 90),
        (2, 85, 4),  # played on the Saturday
        (2, 0, 5),  # kicks off on the Monday
        fixtures=fixtures_played(1, 2, 3, 4),
    )

    inputs = fetch_inputs(cfg, make_client(routes))

    assert inputs.histories
    for history in inputs.histories.values():
        assert [(entry.round, entry.minutes) for entry in history] == [(1, 90), (2, 85)]


def test_a_fixture_at_full_time_counts_before_it_is_data_checked(tmp_path):
    # ``finished`` goes up when the round's data is checked, which is hours or
    # a day after the match. ``finished_provisional`` goes up at full time,
    # which is when the minutes appear. Checked live on 2026-08-22: a match
    # kicked off at 19:00 the previous evening still read ``finished: false``
    # with 90 minutes on the clock and its history rows served.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        midweek_bootstrap(),
        (1, 90),
        (2, 85, 4),
        fixtures=fixtures_played(1, 2, 3, provisional=(4,)),
    )

    inputs = fetch_inputs(cfg, make_client(routes))

    for history in inputs.histories.values():
        assert [entry.round for entry in history] == [1, 2]


def test_a_row_with_no_fixture_id_falls_back_on_its_round(tmp_path):
    # Nothing is finished in this fixtures payload, so a row judged on its
    # fixture would be dropped whatever round it is in. These rows carry no
    # fixture id, so the event rule decides: GW1 is finished and GW2 is not.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        midweek_bootstrap(),
        (1, 90, 0),
        (2, 0, 0),
        fixtures=fixtures_played(),
    )

    inputs = fetch_inputs(cfg, make_client(routes))

    assert inputs.histories
    for history in inputs.histories.values():
        assert [entry.round for entry in history] == [1]


def test_a_row_naming_a_fixture_nobody_has_heard_of_is_dropped(tmp_path):
    # Never silently kept: an id the fixtures payload does not carry is not
    # evidence that a match was played, and its round has not finished either.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        midweek_bootstrap(), (1, 90), (2, 0, 4242), fixtures=fixtures_played(1, 2, 3)
    )

    inputs = fetch_inputs(cfg, make_client(routes))

    for history in inputs.histories.values():
        assert [entry.round for entry in history] == [1]


def test_a_history_of_played_rounds_is_kept_whole(seam):
    # The other half of the claim: the filter takes out what has not happened
    # and nothing else. The ordinary universe has one finished gameweek behind
    # it and the history the API served for it arrives intact.
    assert seam.inputs.histories[FERRER] == [
        GwHistory.model_validate(entry) for entry in ELEMENT_SUMMARY_JSON["history"]
    ]


def test_the_opening_weekend_has_no_history_rather_than_a_history_of_zeroes(tmp_path):
    # Friday's live universe: one row per player, 0 minutes, for matches that
    # kick off tomorrow. None of it is evidence of anything, so none of it is
    # kept, and the minutes model is left with the empty history it knows how
    # to fall back from.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        opening_weekend_bootstrap(), (1, 0), fixtures=fixtures_played()
    )

    inputs = fetch_inputs(cfg, make_client(routes))

    assert inputs.histories
    assert all(history == [] for history in inputs.histories.values())


# --- last season's minutes -------------------------------------------------
#
# The fetch carries a prior per player: what he averaged a gameweek in his last
# Premier League season. It is what the minutes model stands on when this
# season is too thin to read, and it is the difference between a returning
# premium projected at a substitute's twenty minutes in GW1 and one projected
# at the eighty-five he actually plays.


def past_season(minutes: int) -> list[dict]:
    return [{"season_name": "2024/25", "minutes": minutes, "total_points": 180}]


def rested_opener_routes(minutes: int | None = 3230) -> dict:
    """GW1 played, nobody in this squad any of it, GW2 next.

    Every season-to-date counter is zero, because after one gameweek a player
    who did not feature has started nothing — which is the state that leaves
    the starts guess with nothing to say. ``minutes`` is what last season had
    to say instead; None is a player with no Premier League behind him.
    """
    bootstrap = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for element in bootstrap["elements"]:
        element.update(minutes=0, starts=0, total_points=0)

    summary_payload = summary((1, 0))
    if minutes is not None:
        summary_payload["history_past"] = past_season(minutes)
    routes = pipeline_routes(bootstrap)
    routes.update(
        {
            f"/api/element-summary/{element['id']}/": summary_payload
            for element in PIPELINE_ELEMENTS_JSON
        }
    )
    return routes


def test_the_fetch_carries_last_seasons_minutes_a_gameweek(tmp_path):
    # 3230 minutes over 38 gameweeks is 85.0.
    cfg = config(state_dir=tmp_path / "state")

    inputs = fetch_inputs(cfg, make_client(pipeline_routes()))

    assert inputs.prior_minutes[FERRER] == pytest.approx(85.0)


def test_a_player_with_no_premier_league_past_has_no_prior(tmp_path):
    # Absent rather than zero: nothing to read is not the same as a season of
    # not playing, and the minutes model has to be able to tell them apart.
    cfg = config(state_dir=tmp_path / "state")
    routes = pipeline_routes()
    routes[f"/api/element-summary/{FERRER}/"] = summary((1, 90))

    inputs = fetch_inputs(cfg, make_client(routes))

    assert FERRER not in inputs.prior_minutes
    assert inputs.prior_minutes  # everybody else still has one


def test_last_season_lifts_a_returning_premium_the_opener_rested(tmp_path):
    # The week the gaffer spent overriding seven players by hand, with the
    # evidence he was overriding it with now in the payload. One played
    # gameweek, none of it his, and no starts to fall back on: the old floor
    # was a substitute's twenty. 3230 minutes last season is 85.0 a gameweek.
    cfg = config(state_dir=tmp_path / "state")
    inputs = fetch_inputs(cfg, make_client(rested_opener_routes()))

    xmins, _ = build_projections(inputs, cfg)

    assert xmins[FERRER] == pytest.approx(85.0)


def test_a_promoted_clubs_player_still_falls_back_on_his_starts(tmp_path):
    # Same week, same rested opener, and no season behind him: the old
    # behaviour, unchanged, because there is nothing better to have.
    cfg = config(state_dir=tmp_path / "state")
    inputs = fetch_inputs(cfg, make_client(rested_opener_routes(minutes=None)))

    xmins, _ = build_projections(inputs, cfg)

    assert inputs.prior_minutes == {}
    assert xmins[FERRER] == 20.0


def test_a_prior_does_not_overrule_a_season_that_has_been_played(seam):
    # The ordinary universe: one played gameweek of ninety minutes, and a
    # prior of 85 underneath it. The floor is a floor.
    xmins, _ = build_projections(seam.inputs, seam.cfg)

    assert seam.inputs.prior_minutes[FERRER] == pytest.approx(85.0)
    assert xmins[FERRER] == 90.0


# --- when the filter takes everything -------------------------------------
#
# The cut above is the right one and it is still a cut, made against a payload
# somebody else serves. If a flag is renamed, a fixture id stops matching or an
# opening weekend simply has nothing behind it, every history in the run comes
# back empty and every player is projected off his priors — a whole run's
# recommendation resting on a fallback, with nothing on the log to say so. One
# line says it, once, and only when rows actually went in and nothing came out.


def test_a_filter_that_empties_every_history_says_so(tmp_path, capsys):
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        opening_weekend_bootstrap(), (1, 0), fixtures=fixtures_played()
    )

    inputs = fetch_inputs(cfg, make_client(routes))
    printed = capsys.readouterr().out

    assert all(history == [] for history in inputs.histories.values())
    assert "history filter removed every played round" in printed
    assert f"({len(inputs.histories)} players had rows)" in printed
    # One line for the run, not one per player.
    assert printed.count("history filter removed") == 1


def test_a_fetch_with_no_rows_to_remove_is_not_an_anomaly(tmp_path, capsys):
    # Pre-season, and the case the warning must stay quiet for: nobody had a
    # row, so nothing was taken. An empty history here is the season not having
    # started, not the filter having gone wrong.
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(opening_weekend_bootstrap(), fixtures=fixtures_played())

    inputs = fetch_inputs(cfg, make_client(routes))

    assert all(history == [] for history in inputs.histories.values())
    assert "history filter" not in capsys.readouterr().out


def test_a_fetch_that_keeps_a_played_match_says_nothing(tmp_path, capsys):
    cfg = config(state_dir=tmp_path / "state")

    inputs = fetch_inputs(cfg, make_client(pipeline_routes()))

    assert any(inputs.histories.values())
    assert "history filter" not in capsys.readouterr().out


def test_a_deadline_that_has_gone_does_not_bench_a_fit_starter(tmp_path):
    # Friday's live run, end to end and offline. Every player's whole season
    # is a row for a match that has not started, so nobody has a mean to be
    # read off and everybody falls back on his starts — which for the fit
    # premium is a starter's minutes and a place in the eleven, not the zero
    # that sent the gaffer overriding seven players by hand.
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(state_dir=tmp_path / "state")
    routes = unplayed_routes(
        opening_weekend_bootstrap(), (1, 0), fixtures=fixtures_played()
    )

    report = run_pipeline(cfg, make_client(routes), store, "scout", send=False)
    decision = store.last_runs(1)[0]["decision"]

    assert decision["event"] == 2
    assert decision["captain"] == FERRER
    assert any("Ferrer" in line for line in bullets(report, "Starting XI"))
    assert decision["xp_total"] > 0


# --- the gaffer ------------------------------------------------------------
#
# The manager's own loop is tested against a scripted API in
# ``tests/test_manager_agent``; what is tested here is the wiring around it.
# ``run_manager`` is stubbed at the module the pipeline reaches into, so these
# runs are the real fetch, the real solver and the real renderer with one
# canned decision dropped in the middle — which is exactly the seam this task
# builds. Nothing here can reach Anthropic: the stub is called instead of the
# loop, and the client it is handed is never used.

GAFFER_RATIONALE = (
    "Grant is suspended and the solver did not know it. I have rolled the"
    " transfer rather than pay for a replacement I do not want."
)
GRANT = 6  # in the fifteen, and the player the solver's own plan sells

GOOD_CHIP = (
    "The bench boost is the chip this week: the panel prices it above anything"
    " else on the board, all four of the bench have home fixtures against the"
    " bottom three, and what we give up is the double gameweek in GW34 —"
    " eight months of injuries away, against points on offer on Saturday."
)


class Consult(NamedTuple):
    """One call of ``run_manager``, as the pipeline made it."""

    client: object
    cfg: Config
    inputs: PipelineInputs
    solve0: SolveResult
    projections: dict
    briefing: str
    resolver: object


class Gaffer:
    """The manager, stubbed: what he was asked, and what he answered."""

    def __init__(self, decide) -> None:
        self.decide = decide
        self.consults: list[Consult] = []
        self.decisions: list[ManagerDecision] = []
        self.kwargs: list[dict] = []

    def __call__(self, client, cfg, inputs, solve0, projections, briefing, resolver, **kwargs):
        self.kwargs.append(kwargs)
        consult = Consult(client, cfg, inputs, solve0, projections, briefing, resolver)
        self.consults.append(consult)
        decision = self.decide(consult)
        self.decisions.append(decision)
        return decision


def lineup_for(plan, inputs: PipelineInputs, projections: dict) -> Lineup:
    """The eleven ``plan`` fields, picked the way the manager's loop picks it."""
    positions = {pid: player.element_type for pid, player in inputs.players.items()}
    gw_xp = {
        pid: projection.per_gw.get(inputs.event.id, 0.0)
        for pid, projection in projections.items()
    }
    return pick_lineup(plan.squad, positions, gw_xp)


def decided(
    consult: Consult,
    plan=None,
    chip: str = "none",
    justification: str = "",
    searches: int = 2,
) -> ManagerDecision:
    """A decision of his own: the plan that rolls, and his own armbands.

    The captain is the highest id in the eleven rather than the best player in
    it, so that a report or a decision record carrying the solver's captain
    instead of his is a test failure and not a coincidence.
    """
    chosen = plan or next(p for p in consult.solve0.plans if not p.transfers_in)
    lineup = lineup_for(chosen, consult.inputs, consult.projections)
    return ManagerDecision(
        plan=chosen,
        lineup=replace(lineup, captain=max(lineup.xi), vice=min(lineup.xi)),
        captain=max(lineup.xi),
        vice=min(lineup.xi),
        chip=chip,
        chip_justification=justification,
        rationale=GAFFER_RATIONALE,
        # He changed his mind about Grant, which is what a conversation does.
        adjustments=[
            {"player_id": GRANT, "expected_minutes": 20.0, "reason": "a doubt (paper)"},
            {"player_id": GRANT, "expected_minutes": 0.0, "reason": "suspended (club)"},
        ],
        searches=searches,
        source="manager",
    )


def unavailable(consult: Consult, reason: str = "RateLimitError") -> ManagerDecision:
    """The week the manager's own fallback hands back when he cannot be asked."""
    lineup = consult.solve0.lineup
    return ManagerDecision(
        plan=consult.solve0.choice,
        lineup=lineup,
        captain=lineup.captain,
        vice=lineup.vice,
        chip="none",
        chip_justification="",
        rationale=agent.NO_VIEW,
        adjustments=[],
        searches=1,
        source=f"solver-fallback: {reason}",
    )


def stub_gaffer(monkeypatch, decide=decided) -> Gaffer:
    """Answer the pipeline's manager with ``decide``; record what it was asked."""
    gaffer = Gaffer(decide)
    monkeypatch.setattr(agent, "run_manager", gaffer)
    return gaffer


def gaffer_cfg(tmp_path, **kwargs) -> Config:
    return config(state_dir=tmp_path / "state", anthropic_api_key="sk-test", **kwargs)


def gaffer_run(
    monkeypatch, tmp_path, decide=decided, mode="scout", routes=None, chips=False,
    **kwargs,
):
    """One run with a manager in it; the report, the store and the manager."""
    gaffer = stub_gaffer(monkeypatch, decide)
    store = Store(tmp_path / "aigaffer.db")
    report = run_pipeline(
        gaffer_cfg(tmp_path, chips=chips),
        make_client(routes or pipeline_routes()),
        store,
        mode,
        **kwargs,
        now=PAST_THE_FLOOR,
    )
    return report, store, gaffer


def recorded(consult, searches=2):
    """``decided`` with two calls for Grant: a doubt, then a ban."""
    decision = decided(consult, searches=searches)
    decision.record = [
        {"player_id": GRANT, "expected_minutes": 20.0, "reason": "a doubt (paper)", "category": "doubt",
         "tier": 2, "quote_date": "2026-08-20", "source": "paper", "return_gw": None},
        {"player_id": GRANT, "expected_minutes": 0.0, "reason": "suspended (club)", "category": "suspended",
         "tier": 1, "quote_date": "2026-08-20", "source": "club", "return_gw": None},
    ]
    return decision


def test_a_decided_run_writes_his_calls_into_the_ledger_last_call_winning(monkeypatch, tmp_path):
    # gaffer_run runs at PAST_THE_FLOOR; two calls for Grant, the ban is later.
    gaffer_run(monkeypatch, tmp_path, decide=recorded)
    written = json.loads(ledger_path(tmp_path / "state").read_text(encoding="utf-8"))
    grant = written[str(GRANT)]
    assert grant["category"] == "suspended" and grant["expected_minutes"] == 0.0
    assert grant["run"] == "scout" and grant["checked_at"] == PAST_THE_FLOOR.isoformat()


def test_a_run_that_saves_nothing_or_has_nothing_to_say_writes_no_ledger(monkeypatch, tmp_path):
    for decide, extra in ((recorded, {"save": False, "send": False}), (unavailable, {}), (decided, {})):
        run_dir = tmp_path / decide.__name__ / str(bool(extra))
        run_dir.mkdir(parents=True)
        gaffer_run(monkeypatch, run_dir, decide=decide, **extra)
        assert not ledger_path(run_dir / "state").exists()


def test_the_decision_record_carries_no_ledger_record(monkeypatch, tmp_path):
    _, store, _ = gaffer_run(monkeypatch, tmp_path, decide=recorded)
    decision = store.decision(2, "scout")  # the pipeline universe's coming gameweek is GW2
    assert "record" not in decision
    assert all(set(a) == {"player_id", "expected_minutes", "reason"} for a in decision["adjustments"])


def test_an_existing_ledger_reaches_the_briefing_and_the_budget_is_stated(monkeypatch, tmp_path):
    state = tmp_path / "state"
    path = ledger_path(state)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({str(GRANT): {
        "player_id": GRANT, "gw": 1, "category": "doubt", "expected_minutes": 30.0, "tier": 1,
        "quote_date": "2025-08-15", "source": "presser", "note": "knock", "return_gw": None,
        "checked_at": "2025-08-15T10:00:00+00:00", "run": "deadline",
        "fpl": {"status": "a", "chance_of_playing_next_round": None, "news": "", "news_added": None},
    }}), encoding="utf-8")
    _, _, gaffer = gaffer_run(monkeypatch, tmp_path)
    # The fixtures carry no kickoff_time, so rules 2 and 3 are inert; Grant
    # (status "a", chance None) is a GW1 doubt read in GW2, so rule 4 fires.
    text = gaffer.consults[0].briefing
    assert "## What we already know" in text and "re-check — a new gameweek" in text
    assert "Search budget this run: 6 searches" in text
    assert gaffer.kwargs[0]["today"] == PAST_THE_FLOOR.date()


@pytest.mark.parametrize(("mode", "budget"), [("scout", 6), ("deadline", 10)])
def test_a_run_past_its_budget_says_so_on_the_log(monkeypatch, tmp_path, capsys, mode, budget):
    spent = budget + 1
    gaffer_run(monkeypatch, tmp_path, decide=lambda c: decided(c, searches=spent), mode=mode)
    assert f"the gaffer decided: {spent} searches — search budget exceeded ({spent}/{budget})" in capsys.readouterr().out


def test_a_ledger_that_cannot_be_written_never_costs_the_report(monkeypatch, tmp_path, capsys):
    saved_first = []

    def boom(*args, **kwargs):
        # The write comes after store.save_run (spec section 6): the run is already recorded.
        saved_first.append(Store(tmp_path / "aigaffer.db").has_run(2, "scout"))
        raise OSError("disk full at /secret/path")
    monkeypatch.setattr(orchestrator, "write_ledger", boom)
    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=recorded)
    out = capsys.readouterr().out
    assert "aigaffer news ledger: not written (OSError)" in out and "/secret/path" not in out
    assert saved_first == [True]
    assert store.has_run(2, "scout")


def test_a_ledger_write_that_fails_in_any_way_never_costs_the_report(monkeypatch, tmp_path, capsys):
    def boom(*args, **kwargs):
        raise ValueError("secret words")
    monkeypatch.setattr(orchestrator, "write_ledger", boom)
    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=recorded)
    out = capsys.readouterr().out
    assert "not written (ValueError)" in out and "secret words" not in out
    assert report and store.has_run(2, "scout")


def test_the_gaffer_decides_the_week(monkeypatch, tmp_path):
    # The solver wants a transfer; the manager rolls. Everything downstream
    # has to be his week and not the solver's — the recommendation, the
    # armbands, the eleven and the record kept of all three.
    report, store, gaffer = gaffer_run(monkeypatch, tmp_path, send=False)
    decision = store.last_runs(1)[0]["decision"]
    his = gaffer.decisions[0]

    assert gaffer.consults[0].solve0.choice.transfers_in, "the solver would have moved"
    assert "Roll the transfer." in report and "- Out:" not in report
    assert GAFFER_RATIONALE in report
    assert decision["transfers_in"] == [] and decision["transfers_out"] == []
    assert decision["captain"] == his.captain != FERRER, "his captain, not the solver's"
    assert decision["vice"] == his.vice
    # And the armband is on the team sheet, against the player he gave it to.
    named = next(e["web_name"] for e in PIPELINE_ELEMENTS_JSON if e["id"] == his.captain)
    assert f"{named} (C)" in report


def test_the_record_keeps_the_words_as_well_as_the_numbers(monkeypatch, tmp_path):
    _, store, _ = gaffer_run(monkeypatch, tmp_path, send=False)
    decision = store.last_runs(1)[0]["decision"]

    assert decision["decision_source"] == "manager"
    assert decision["rationale"] == GAFFER_RATIONALE
    # Whole, in the order he made them, the one he thought better of included:
    # the record is what happened, and the report is where it reads tidily.
    assert decision["adjustments"] == [
        {"player_id": GRANT, "expected_minutes": 20.0, "reason": "a doubt (paper)"},
        {"player_id": GRANT, "expected_minutes": 0.0, "reason": "suspended (club)"},
    ]
    assert decision["chip"] == "none"
    assert decision["chip_justification"] == ""
    assert decision["searches"] == 2


def test_the_report_prints_the_minutes_he_settled_on(monkeypatch, tmp_path):
    report, _, _ = gaffer_run(monkeypatch, tmp_path, send=False)

    assert bullets(report, "The Gaffer's view") == [
        "- Set Grant to 0 mins — suspended (club)"
    ]


def test_what_he_only_wrote_down_is_kept_and_marked_as_such(monkeypatch, tmp_path):
    # Minutes he set after the last re-solve changed nothing, so the report
    # says so and the record keeps them anyway: it is what happened.
    noted = [{"player_id": FERRER, "expected_minutes": 90.0, "reason": "fit, he says"}]

    def late(consult: Consult) -> ManagerDecision:
        return replace(decided(consult), unapplied=noted)

    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=late, send=False)

    assert store.last_runs(1)[0]["decision"]["unapplied"] == noted
    assert "Noted but not applied (no re-solve followed):" in report
    assert "- Ferrer at 90 mins — fit, he says" in report


def test_the_kill_switch_leaves_the_solver_to_it(monkeypatch, tmp_path, scout_run):
    # A key in the environment and AIGAFFER_MANAGER=0: the one way to turn the
    # manager off on a machine that could perfectly well reach him.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked with the switch off")

    stub_gaffer(monkeypatch, never)
    cfg = config(
        state_dir=tmp_path / "state",
        anthropic_api_key="sk-test",
        manager_enabled=False,
    )

    report = run_pipeline(
        cfg,
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert report == scout_run.report


def test_the_gaffer_is_briefed_on_the_week_the_solver_solved(monkeypatch, tmp_path):
    _, _, gaffer = gaffer_run(monkeypatch, tmp_path, send=False)
    his = gaffer.consults[0]

    assert len(gaffer.consults) == 1
    assert his.briefing.startswith("# AI Gaffer — manager briefing: GW2")
    assert "Free transfers: 1" in his.briefing
    # The expected minutes the projection was built on: the one number he is
    # allowed to overwrite, and he cannot sensibly overwrite what he is not shown.
    assert "xMins" in his.briefing
    assert his.inputs.event.id == 2 and his.solve0.draft_mode is False


def test_the_client_is_built_against_the_clock(monkeypatch, tmp_path):
    # The SDK's own defaults are ten minutes and two retries, which is half an
    # hour of one request — longer than the whole job is allowed to take. The
    # loop's time budget bounds the conversation; this bounds the turn that is
    # in flight when the budget runs out.
    #
    # Five minutes and not two. Two bounded a turn that hangs, and also every
    # turn that works: this model at high effort runs its web searches before it
    # answers at all, and against the live API it never once came back inside
    # two — every run fell back to the solver with no turns and no searches on
    # the clock. A ceiling low enough to catch the good case is not a ceiling.
    built: list[dict] = []

    class Recorder:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    _, _, gaffer = gaffer_run(monkeypatch, tmp_path, send=False)

    assert built == [{"api_key": "sk-test", "timeout": 300.0, "max_retries": 1}]
    assert isinstance(gaffer.consults[0].client, Recorder), "and it is what he is given"


def test_the_resolver_reprojects_and_resolves_on_his_minutes(monkeypatch, tmp_path):
    seen = {}

    def re_solve(consult: Consult) -> ManagerDecision:
        solved, projections = consult.resolver({FERRER: 0.0})
        seen["xi"] = solved.lineup.xi
        seen["ferrer"] = projections[FERRER].per_gw
        seen["before"] = consult.projections[FERRER].per_gw
        return decided(consult, plan=solved.choice)

    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=re_solve, send=False)

    # Out for the coming week, and only the coming week: the weeks after it
    # are what the solver was fed before he said a word.
    assert seen["before"][2] > 0 and seen["ferrer"][2] == 0.0
    assert seen["ferrer"][3] == seen["before"][3] > 0
    assert FERRER not in seen["xi"], "told he is not playing, the solver drops him"
    assert store.has_run(2, "scout") is True
    assert "## The Gaffer's view" in report


def test_a_re_solve_plans_the_road_ahead_again_on_his_minutes(monkeypatch, tmp_path):
    # The resolver is this module's own solve, so a re-solve re-plans the
    # window as well as the week — and the plan he finalizes off it carries
    # the path that came back, into the report and into the record.
    seen = {}

    def re_solve(consult: Consult) -> ManagerDecision:
        solved, _ = consult.resolver({FERRER: 0.0})
        seen["paths"] = [plan.path for plan in solved.plans]
        seen["chosen"] = solved.choice.path
        seen["before"] = consult.solve0.choice.path
        return decided(consult, plan=solved.choice)

    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=re_solve, send=False)
    decision = store.last_runs(1)[0]["decision"]

    assert seen["before"] is not None, "the first solve planned a window too"
    assert seen["paths"] and all(path is not None for path in seen["paths"])
    assert decision["engine"] == "multi"
    assert decision["path"] == [
        {
            "event": move.event,
            "in": move.transfers_in,
            "out": move.transfers_out,
            "hits": move.hits,
            "chip": move.chip,
        }
        for move in seen["chosen"].moves
    ]
    assert ("## The road ahead" in report) == bool(seen["chosen"].moves)


def test_the_report_is_costed_on_the_projections_he_decided_on(monkeypatch, tmp_path):
    # A manager who re-solves decides on projections the pipeline never saw.
    # Printing his eleven beside the solver's numbers would be a report whose
    # team sheet and whose columns disagree about what week it is.
    REYES = 17

    def re_costed(consult: Consult) -> ManagerDecision:
        his = dict(consult.projections)
        his[REYES] = PlayerProjection(player_id=REYES, per_gw={2: 9.9}, total=99.9)
        return replace(decided(consult), projections=his)

    report, _, _ = gaffer_run(monkeypatch, tmp_path, decide=re_costed, send=False)

    assert bullets(report, "Watchlist")[0] == "- Reyes (MID, EAS, £9.5m) — 99.9 xP"


def test_a_decision_with_no_projections_is_costed_by_the_solvers(
    monkeypatch, tmp_path, scout_run
):
    # The fallback is the solver's own week, and the solver's own numbers are
    # the ones already in hand.
    report, _, _ = gaffer_run(
        monkeypatch, tmp_path, decide=unavailable, send=False
    )

    assert bullets(report, "Watchlist") == bullets(scout_run.report, "Watchlist")


def test_a_gaffer_who_could_not_be_reached_leaves_the_solvers_week(
    monkeypatch, tmp_path
):
    report, store, gaffer = gaffer_run(
        monkeypatch, tmp_path, decide=unavailable, send=False
    )
    decision = store.last_runs(1)[0]["decision"]

    assert "The gaffer was unavailable (RateLimitError)" in report
    assert decision["decision_source"] == "solver-fallback: RateLimitError"
    assert decision["transfers_in"] == gaffer.consults[0].solve0.choice.transfers_in
    assert decision["captain"] == FERRER, "the solver's own eleven, and his captain"


def test_a_manager_that_falls_over_does_not_take_the_report_with_it(
    monkeypatch, tmp_path
):
    # run_manager is not supposed to raise. If it ever does, the week's report
    # is not the thing to lose over it.
    def explode(consult: Consult) -> ManagerDecision:
        raise ValueError("the loop has a bug")

    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=explode, send=False)

    assert "The gaffer was unavailable (unexpected ValueError)" in report
    assert "the loop has a bug" not in report
    assert store.has_run(2, "scout") is True


class PoisonedModule:
    """A module whose every attribute blows up on the way out.

    What a half-installed dependency does to an import that is not an
    ImportError: a C extension that will not load, a module whose top level
    raises. ``from x import y`` on this raises RuntimeError, which is what a
    guard written for ImportError alone would let past.
    """

    def __getattr__(self, name: str):
        raise RuntimeError(f"{name} is not coming out of here")


def test_an_import_that_blows_up_is_a_fallback_and_not_a_lost_report(
    monkeypatch, tmp_path
):
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked with his briefing broken")

    stub_gaffer(monkeypatch, never)
    monkeypatch.setitem(sys.modules, "aigaffer.manager.briefing", PoisonedModule())
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert "The gaffer was unavailable (unexpected RuntimeError)" in report
    assert "not coming out of here" not in report
    assert store.has_run(2, "scout") is True


def test_a_manager_who_never_loaded_at_all_says_so_in_the_report(
    monkeypatch, tmp_path
):
    # A dependency that will not import leaves no decision to hang a section
    # on, so the report is Phase 1's — and a fork whose install broke could
    # otherwise read a season of solver-only reports without ever being told.
    # One line, in the document that actually goes to the phone.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked with his own module broken")

    stub_gaffer(monkeypatch, never)
    monkeypatch.setitem(sys.modules, "aigaffer.manager.agent", PoisonedModule())
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert "## The Gaffer's view" not in report
    assert report.endswith(f"\n{orchestrator.MANAGER_UNAVAILABLE}\n")
    written = (tmp_path / "state" / "reports" / "gw2-scout.md").read_text(
        encoding="utf-8"
    )
    assert written == report, "the diary and the phone read the same report"


def test_the_manager_unavailable_notice_rides_the_digest(monkeypatch, tmp_path):
    # The digest is the document that actually goes to the phone now, so the
    # accident has to be said there too — a fork whose install broke must not
    # read a season of confident solver digests without being told.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked with his own module broken")

    stub_gaffer(monkeypatch, never)
    monkeypatch.setitem(sys.modules, "aigaffer.manager.agent", PoisonedModule())
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
        anthropic_api_key="sk-test",
    )

    run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "scout",
        now=PAST_THE_FLOOR,
    )

    [(_, _, message)] = sent
    assert orchestrator.MANAGER_UNAVAILABLE in message


# The fifteen-man picks fixture's season so far, as the phone's first line
# says it: 61 points, a rank in the millions, £100.0m all in, £2.8m of it
# banked. The free-transfer count follows on the same line.
STANDING_LINE = "61 pts · rank 2,345,678 · value £100.0m · bank £2.8m · "


def test_the_digest_opens_with_the_standing(monkeypatch, tmp_path):
    # The phone's first screen says how the season is going before it says
    # what to do, and the figures come off the picks the run already fetched.
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )

    run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    [(_, _, message)] = sent
    lines = message.splitlines()
    [standing] = [line for line in lines if line.startswith(STANDING_LINE)]
    assert standing == STANDING_LINE + "1 free transfer"
    assert lines[lines.index(standing) - 2].startswith("Deadline: ")


def test_a_delivered_digest_carries_the_keyboard(monkeypatch, tmp_path):
    # The scheduled send is what puts the four buttons under his message box
    # even if he never types anything, so _deliver passes the keyboard on.
    calls = []
    monkeypatch.setattr(
        orchestrator, "send_report", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    cfg = config(
        telegram_token=TOKEN, telegram_chat_id="42", state_dir=tmp_path / "state"
    )

    run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    [(args, kwargs)] = calls
    assert args[:2] == (TOKEN, "42")
    assert kwargs == {"reply_markup": KEYBOARD}


def test_the_reminder_opens_with_the_standing_too(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    run_pipeline(cfg, client, store, "deadline", send=False)
    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    lines = alert.splitlines()
    [standing] = [line for line in lines if line.startswith(STANDING_LINE)]
    assert lines[lines.index(standing) - 2].startswith("Deadline: ")


def test_the_audit_line_rides_the_digest_as_well_as_the_report(
    monkeypatch, tmp_path
):
    # The ledger's discrepancy note fires at most once per gameweek and the
    # phone is where it has to be seen — a note only the diary carries is a
    # note nobody reads on deadline day.
    monkeypatch.setattr(orchestrator, "_audit_line", lambda ledger: "\nAUDIT-MARK\n")
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )

    report = run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    [(_, _, message)] = sent
    assert "AUDIT-MARK" in report
    assert "AUDIT-MARK" in message


def test_a_manager_who_was_reached_never_gets_the_note(monkeypatch, tmp_path):
    # The Gaffer's view says whose pick it is, in his own section. A second
    # line saying the same thing would be the report explaining itself twice.
    report, _, _ = gaffer_run(monkeypatch, tmp_path, decide=unavailable, send=False)

    assert orchestrator.MANAGER_UNAVAILABLE not in report


def test_a_run_with_no_manager_asked_for_carries_no_note(
    monkeypatch, tmp_path, scout_run
):
    # No key is not a degradation, it is a configuration, and the invariant is
    # that it renders Phase 1's report byte for byte.
    assert orchestrator.MANAGER_UNAVAILABLE not in scout_run.report


def test_a_draft_carries_no_note_either(monkeypatch, tmp_path):
    # Fifteen players from nothing is not a week the manager is asked about,
    # and the report says it is a draft in its own heading.
    stub_gaffer(monkeypatch)
    routes = pipeline_routes()
    del routes[PICKS_PATH]

    report = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(routes),
        Store(tmp_path / "aigaffer.db"),
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert orchestrator.MANAGER_UNAVAILABLE not in report


def test_a_chip_he_still_holds_is_played(monkeypatch, tmp_path):
    report, store, _ = gaffer_run(
        monkeypatch,
        tmp_path,
        decide=partial(decided, chip="bench_boost", justification=GOOD_CHIP),
        send=False,
        chips=True,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert "Playing the bench boost." in report
    assert decision["chip"] == "bench_boost"
    assert decision["chip_justification"] == GOOD_CHIP


def test_with_chips_off_a_chip_he_holds_is_still_his_to_play(monkeypatch, tmp_path):
    # The switch makes chips advisory — priced in the panel, planned by
    # nobody — and not unplayable. The solver planned no chip and built no
    # calendar, and the bench boost he finalizes, which the rules say we hold,
    # is the week that goes in.
    report, store, gaffer = gaffer_run(
        monkeypatch,
        tmp_path,
        decide=partial(decided, chip="bench_boost", justification=GOOD_CHIP),
        send=False,
        chips=False,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert gaffer.consults[0].cfg.chips is False
    assert gaffer.consults[0].solve0.calendar is None
    assert decision["decision_source"] == "manager"
    assert decision["chip"] == "bench_boost"
    assert decision["chip_calendar"] is None
    assert "Playing the bench boost." in report


def test_a_chip_he_has_already_played_is_refused_at_the_door(monkeypatch, tmp_path):
    # The season's history says the wildcard went in GW1, and on these rules
    # (both sets, the first from GW1) that spends the first set's. The briefing
    # tells him so; this is the belt under that brace, and it costs him the
    # decision rather than the chip, because a week built on a chip we cannot
    # play is not a week anybody can enter.
    report, store, gaffer = gaffer_run(
        monkeypatch,
        tmp_path,
        decide=partial(decided, chip="wildcard", justification=GOOD_CHIP),
        send=False,
        routes=halves_routes(),
        chips=True,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert "The gaffer was unavailable (chip not held for this gameweek)" in report
    assert decision["decision_source"] == f"solver-fallback: {CHIP_SPENT}"
    assert decision["chip"] == "none"
    assert decision["transfers_in"] == gaffer.consults[0].solve0.choice.transfers_in
    assert decision["searches"] == 2, "the searches were still paid for"
    assert GAFFER_RATIONALE not in report


def test_a_chip_not_held_this_gameweek_sends_the_week_back_to_the_solver(
    monkeypatch, tmp_path
):
    # The rules hand out one set only and the bench boost went in GW1: there
    # is no bench boost in hand at all, so a week built on one is refused
    # whole, exactly as a spent chip always was.
    routes = halves_routes(FIRST_SET_ONLY, chips_used=[{"name": "bboost", "event": 1}])
    gaffer = stub_gaffer(
        monkeypatch, partial(decided, chip="bench_boost", justification=GOOD_CHIP)
    )
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        gaffer_cfg(tmp_path, chips=True), make_client(routes), store, "scout",
        send=False, now=PAST_THE_FLOOR,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert decision["decision_source"] == f"solver-fallback: {CHIP_SPENT}"
    assert decision["chip"] == "none"
    assert decision["transfers_in"] == gaffer.consults[0].solve0.choice.transfers_in
    assert GAFFER_RATIONALE not in report


def test_a_second_half_chip_before_gw20_is_refused(monkeypatch, tmp_path):
    # Both sets on the board, the first bench boost spent in GW1. The second
    # is in hand — held from the start of the season — but not playable until
    # GW20, so at GW2 it is no more a chip to play than the spent one.
    routes = halves_routes(chips_used=[{"name": "bboost", "event": 1}])
    gaffer = stub_gaffer(
        monkeypatch, partial(decided, chip="bench_boost", justification=GOOD_CHIP)
    )
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(
        gaffer_cfg(tmp_path, chips=True), make_client(routes), store, "scout",
        send=False, now=PAST_THE_FLOOR,
    )
    decision = store.last_runs(1)[0]["decision"]
    inputs = gaffer.consults[0].inputs

    assert "bench_boost@38" in {
        chip.id for chip in _held(gaffer_cfg(tmp_path, chips=True), inputs)
    }
    assert decision["decision_source"] == f"solver-fallback: {CHIP_SPENT}"
    assert decision["chip"] == "none"


def test_without_a_key_there_is_no_gaffer_and_no_difference(
    monkeypatch, tmp_path, scout_run
):
    # The Phase 1 report, byte for byte, and a manager who was never asked.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked without a key")

    stub_gaffer(monkeypatch, never)
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert report == scout_run.report
    assert "decision_source" not in store.last_runs(1)[0]["decision"]


def test_the_gaffer_is_never_asked_to_draft_a_squad(monkeypatch, tmp_path):
    # Fifteen players from nothing is not a week to read the news about: there
    # is no team, no chip to play and no transfer to talk him out of.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked to draft")

    gaffer = stub_gaffer(monkeypatch, never)
    routes = pipeline_routes()
    del routes[PICKS_PATH]

    report = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(routes),
        Store(tmp_path / "aigaffer.db"),
        "scout",
        send=False,
        now=PAST_THE_FLOOR,
    )

    assert gaffer.consults == []
    assert "## The Gaffer's view" not in report


@pytest.mark.parametrize("mode", ["scout", "deadline"])
def test_the_gaffer_reads_the_news_for_both_reports(monkeypatch, tmp_path, mode):
    # The scout report is two days out and the deadline report is on the day;
    # the news is worth reading for both, and it is the deadline run that has
    # the team news in it.
    report, store, gaffer = gaffer_run(monkeypatch, tmp_path, mode=mode, send=False)

    assert len(gaffer.consults) == 1
    assert store.has_run(2, mode) is True
    assert "## The Gaffer's view" in report


def test_a_dry_run_still_asks_the_gaffer(monkeypatch, tmp_path):
    # Reading the news is analysis and costs nothing but tokens; a dry run
    # that skipped it would print a report nobody could check.
    report, store, gaffer = gaffer_run(monkeypatch, tmp_path, send=False, save=False)

    assert len(gaffer.consults) == 1
    assert "## The Gaffer's view" in report
    assert store.last_runs() == []


def test_the_run_says_whether_the_gaffer_decided(monkeypatch, capsys, tmp_path):
    gaffer_run(monkeypatch, tmp_path, send=False)

    assert capsys.readouterr().out.strip() == "the gaffer decided: 2 searches"


def test_the_run_says_when_the_gaffer_stood_down(monkeypatch, capsys, tmp_path):
    # The class name and nothing else: an exception's own words can carry a
    # key, and this line goes into a log anybody can read.
    gaffer_run(monkeypatch, tmp_path, decide=unavailable, send=False)

    assert capsys.readouterr().out.strip() == "the gaffer stood down: RateLimitError"


def test_the_report_the_gaffer_wrote_is_the_one_that_is_sent(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    gaffer = stub_gaffer(monkeypatch)
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
        anthropic_api_key="sk-test",
    )

    report = run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline",
        now=PAST_THE_FLOOR,
    )

    assert len(gaffer.consults) == 1
    assert "## The Gaffer's view" in report
    # The digest that went to the phone is built from the same decision: the
    # gaffer's view rides it, opening paragraph and source line included.
    [(_, _, message)] = sent
    assert "## The Gaffer's view" in message
    assert "Decided by the gaffer." in message


# --- the chip calendar ------------------------------------------------------
#
# Built once a run, outside the solve, from projections of its own: base
# minutes and a horizon running to the furthest expiry the window can reach.
# The solve and every re-solve the gaffer asks for read the same one, and the
# decision record keeps it.


def calendar_spy(monkeypatch) -> list[SimpleNamespace]:
    """Record each ``build_calendar`` call — what it was handed — and build it."""
    calls: list[SimpleNamespace] = []
    real = orchestrator.build_calendar

    def spy(held, window, players, projections, xmins, *rest):
        calls.append(
            SimpleNamespace(
                held=held,
                window=window,
                xmins=dict(xmins),
                events=projected_events(projections),
            )
        )
        return real(held, window, players, projections, xmins, *rest)

    monkeypatch.setattr(orchestrator, "build_calendar", spy)
    return calls


def test_the_calendar_is_built_once_from_base_minutes_and_a_resolve_does_not_move_it(
    monkeypatch, tmp_path, seam
):
    calls = calendar_spy(monkeypatch)
    seen = {}

    def re_solve(consult: Consult) -> ManagerDecision:
        # Ferrer ruled out for the coming week: his minutes, not the model's.
        solved, _ = consult.resolver({FERRER: 0.0})
        seen["first"], seen["again"] = consult.solve0.calendar, solved.calendar
        return decided(consult, plan=solved.choice)

    stub_gaffer(monkeypatch, re_solve)
    store = Store(tmp_path / "aigaffer.db")
    run_pipeline(
        gaffer_cfg(tmp_path, chips=True), make_client(halves_routes()), store,
        "scout", send=False, now=PAST_THE_FLOOR,
    )
    base, _ = build_projections(seam.inputs, seam.cfg)

    assert len(calls) == 1, "once a run, and never again for a re-solve"
    assert calls[0].xmins[FERRER] == base[FERRER] > 0, "the model's minutes, not his"
    # The window it judges is the solve's; the weeks it projects run on to the
    # first set's expiry, which is where its chips are saved for.
    assert calls[0].window == list(range(2, 2 + seam.cfg.horizon))
    assert calls[0].events == list(range(2, 20))
    assert seen["first"] is not None and seen["again"] is seen["first"]


def test_a_failed_calendar_says_so_and_the_report_still_goes_out(
    monkeypatch, capsys, tmp_path
):
    def broken(*args, **kwargs):
        raise ValueError("a payload nobody has seen before")

    monkeypatch.setattr(orchestrator, "build_calendar", broken)
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
        chips=True,
    )
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(cfg, make_client(halves_routes()), store, "deadline")

    printed = capsys.readouterr().out.splitlines()
    assert f"{CALENDAR_FAILED}: ValueError" in printed
    assert not any("nobody has seen" in line for line in printed), "the type, never the words"
    assert len(sent) == 1 and report.startswith("# AI Gaffer — GW2")
    record = store.decision(2, "deadline")["chip_calendar"]
    assert record["fell_back"] is True
    # The old flat bars, for every first-set chip still in hand: each has
    # weeks left beyond the window, so none is forced to play inside it.
    bars = {entry["chip"]: entry["bars"]["2"] for entry in record["entries"]}
    assert bars == {
        "bench_boost@19": FALLBACK_BARS["bench_boost"],
        "triple_captain@19": FALLBACK_BARS["triple_captain"],
        "free_hit@19": FALLBACK_BARS["free_hit"],
    }


def test_the_record_keeps_the_calendar(tmp_path):
    cfg = config(state_dir=tmp_path / "state", chips=True)
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(cfg, make_client(halves_routes()), store, "scout", send=False)
    record = store.last_runs(1)[0]["decision"]["chip_calendar"]

    assert record["fell_back"] is False
    assert record["discount"] == CHIP_DISCOUNT
    assert record["horizon_end"] == 19
    # The first set the window can reach, soonest to expire and in chip order;
    # the second set starts beyond the window and is not on it yet, and the
    # wildcard went in GW1.
    assert [entry["chip"] for entry in record["entries"]] == [
        "bench_boost@19", "triple_captain@19", "free_hit@19",
    ]
    assert record["entries"][0]["chip"] == "bench_boost@19"
    assert set(record["entries"][0]["bars"]) == {str(gw) for gw in range(2, 8)}


def test_chips_off_builds_no_calendar_and_records_none(monkeypatch, tmp_path):
    calls = calendar_spy(monkeypatch)
    cfg = config(state_dir=tmp_path / "state", chips=False)
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(cfg, make_client(halves_routes()), store, "scout", send=False)

    assert calls == []
    assert store.last_runs(1)[0]["decision"]["chip_calendar"] is None


@pytest.mark.parametrize("mode", ["scout", "reminder"])
def test_team_strengths_are_fitted_once_a_run(monkeypatch, tmp_path, mode):
    # The run's projections, the calendar's and every re-solve's read one fit.
    fits = []
    real = orchestrator.build_team_strengths

    def counting(*args, **kwargs):
        fits.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "build_team_strengths", counting)
    calls = calendar_spy(monkeypatch)

    def re_solve(consult: Consult) -> ManagerDecision:
        solved, _ = consult.resolver({FERRER: 0.0})
        return decided(consult, plan=solved.choice)

    stub_gaffer(monkeypatch, re_solve)
    run_pipeline(
        gaffer_cfg(tmp_path, chips=True), make_client(halves_routes()),
        Store(tmp_path / "aigaffer.db"), mode, send=False, now=PAST_THE_FLOOR,
    )

    assert len(calls) == 1, "a calendar was built, so its projections were asked for"
    assert len(fits) == 1


# --- the reminder ----------------------------------------------------------
#
# Three hours out the schedule runs the solver again — never the manager — and
# sends a short alert: the operative plan, and whether the news has moved
# under it. Whether to shout is diffed like against like — the solver's own
# pre-manager plan, which the deadline record keeps under ``solver_actions``,
# against the fresh solver-only solve — because diffing the gaffer's verdict
# against a fresh solver would shout on every week he overrode the solver,
# which is settled at T-18h and is not news. What the alert shows is still
# the gaffer's verdict. The diff is machine-readable and decided in one
# place, so the record and the message cannot disagree about whether the
# news moved.

NAMES = {element["id"]: element["web_name"] for element in PIPELINE_ELEMENTS_JSON}


def plan_shape(**overrides) -> dict:
    """One plan as the actions dict the reminder diffs: a swap, the armbands,
    no chip, and the formation."""
    shape = {
        "transfers": [[7, 18]],
        "captain": 5,
        "vice": 13,
        "chip": "none",
        "formation": "3-4-3",
    }
    shape.update(overrides)
    return shape


def test_identical_actions_have_no_diff():
    assert diff_actions(plan_shape(), plan_shape()) == {}


def test_a_changed_buy_is_a_buy_change_and_only_a_buy_change():
    # The sale is the same player either way; only the signing moved. Read as
    # positional pairs this was two dropped and two added moves, each naming
    # a sale that never changed.
    diff = diff_actions(plan_shape(), plan_shape(transfers=[[7, 19]]))

    assert diff == {"buys_added": [19], "buys_dropped": [18]}


def test_a_replaced_sale_never_implicates_the_unchanged_moves():
    # The reviewer's example: two sorted lists zipped into pairs re-pair
    # everything after the change, so one replaced sale (9 for 4) used to
    # read as two swaps dropped and two added. As sets it is exactly what
    # happened: one sale in, one sale out, the buys untouched.
    stored = plan_shape(transfers=[[5, 18], [9, 20]])
    fresh = plan_shape(transfers=[[4, 18], [5, 20]])

    assert diff_actions(stored, fresh) == {"sells_added": [4], "sells_dropped": [9]}


def test_a_moved_armband_and_a_changed_chip_are_each_named():
    diff = diff_actions(
        plan_shape(),
        plan_shape(captain=13, vice=5, chip="free_hit", formation="3-5-2"),
    )

    assert diff == {
        "captain": [5, 13],
        "vice": [13, 5],
        "chip": ["none", "free_hit"],
        "formation": ["3-4-3", "3-5-2"],
    }


def test_a_record_from_before_formations_were_kept_is_not_a_change():
    # A full report written before this field existed reads back with no
    # formation. That is a record with less in it, not a plan that moved, and
    # a reminder that shouted about it would be crying wolf on week one.
    stored = plan_shape(formation=None)

    assert diff_actions(stored, plan_shape()) == {}


def test_a_stored_decision_becomes_the_actions_the_diff_reads():
    record = {
        "mode": "deadline",
        "transfers_in": [18],
        "transfers_out": [7],
        "hits": 0,
        "captain": 5,
        "vice": 13,
        "chip": "none",
        "formation": "3-4-3",
    }

    assert _stored_actions(record) == plan_shape()
    assert _stored_actions(None) is None


def test_the_full_report_records_the_actions_for_the_reminder(tmp_path):
    # The persistence half of the round trip: the deadline run's decision
    # record, already in the store, carries everything the reminder reads —
    # the verdict's swaps, armbands, chip and formation, and the solver's own
    # pre-manager plan whole under ``solver_actions``, which is the side the
    # reminder actually diffs. On a week with no manager the two describe the
    # same plan, and that is the invariant pinned here.
    cfg = config(state_dir=tmp_path)
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(cfg, make_client(pipeline_routes()), store, "deadline", send=False)
    record = store.decision(2, "deadline")
    stored = _stored_actions(record)

    assert record["formation"].count("-") == 2
    assert stored["transfers"] == [
        [out, bought]
        for out, bought in zip(record["transfers_out"], record["transfers_in"])
    ]
    assert stored["captain"] == record["captain"] == FERRER
    assert stored["chip"] == "none"
    assert stored["formation"] == record["formation"]
    assert record["solver_actions"] == stored, "no manager: his verdict is the solver's"


def test_a_plan_that_held_reads_as_a_calm_reminder(tmp_path):
    # The pipeline is deterministic, so a reminder straight after the full
    # report re-solves to the same week and says so quietly.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    run_pipeline(cfg, client, store, "deadline", send=False)
    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    assert alert.startswith("# AI Gaffer — GW2 reminder")
    assert render.REMINDER_UNCHANGED in alert
    assert "⚠️" not in alert
    assert "## Candidate plans" not in alert and "## Watchlist" not in alert


def test_the_gaffer_deviating_from_the_solver_is_not_news(monkeypatch, tmp_path):
    # At T-18h the manager overrode the solver: he rolled the transfer the
    # solver wanted and moved the armband. The T-3h solve re-derives roughly
    # the solver's own answer, so a diff of verdict-against-fresh-solver
    # would shout every week he ever deviates — crying wolf about a
    # disagreement that was settled a day ago. The news has not moved, so the
    # reminder is calm, and the plan it shows is his verdict, not the
    # solver's rediscovered one.
    _, store, gaffer = gaffer_run(monkeypatch, tmp_path, mode="deadline", send=False)
    his = gaffer.decisions[0]
    assert gaffer.consults[0].solve0.choice.transfers_in, "the solver would have moved"

    alert = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(pipeline_routes()),
        store,
        "reminder",
        send=False,
    )

    assert render.REMINDER_UNCHANGED in alert
    assert "⚠️" not in alert
    assert "No transfers — roll." in alert, "his roll, not the solver's swap"
    assert f"CAPTAIN {NAMES[his.captain]}" in alert
    assert store.decision(2, "reminder")["changes"] == {}


def test_the_news_moving_the_solver_is_shouted_about(tmp_path):
    # The record's ``solver_actions`` — the solver's own day-old plan — is
    # doctored so the fresh solve disagrees with it on every axis the diff
    # reads: the warning leads and each change is named. The gaffer's fields
    # are left alone, so the verdict block shows his stored plan, labelled as
    # the operative one, beside the fresh solve — shown, never replaced.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())
    run_pipeline(cfg, client, store, "deadline", send=False)
    record = store.decision(2, "deadline")
    doctored = dict(
        record,
        solver_actions=dict(
            record["solver_actions"],
            transfers=[],
            captain=record["vice"],
            vice=record["captain"],
            chip="bench_boost",
        ),
    )
    store.save_run(2, "deadline", "the doctored yardstick", doctored)

    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    assert render.NEWS_MOVED in alert
    assert alert.index("⚠️") < alert.index("⏰"), "the warning leads"
    out, bought = record["transfers_out"][0], record["transfers_in"][0]
    assert f"- Now selling: {NAMES[out]}" in alert
    assert f"- Now buying: {NAMES[bought]}" in alert
    assert (
        f"- Captain moved from {NAMES[record['vice']]}"
        f" to {NAMES[record['captain']]}" in alert
    )
    assert "- Chip changed from bench boost to none" in alert
    assert render.GAFFER_VERDICT in alert and render.FRESH_SOLVE in alert
    assert alert.index(render.GAFFER_VERDICT) < alert.index(render.FRESH_SOLVE)
    # The verdict block still carries the gaffer's own stored moves.
    verdict = alert[alert.index(render.GAFFER_VERDICT):alert.index(render.FRESH_SOLVE)]
    assert f"SELL {NAMES[out]}" in verdict
    assert render.HUMAN_JUDGES in alert
    # And the record of the reminder keeps the same diff, machine-readably.
    changes = store.decision(2, "reminder")["changes"]
    assert changes["sells_added"] == [out] and changes["buys_added"] == [bought]
    assert changes["captain"] == [record["vice"], record["captain"]]
    assert changes["chip"] == ["bench_boost", "none"]


def test_the_reminder_resolves_with_a_fresh_calendar_and_still_diffs_chips(
    monkeypatch, tmp_path
):
    # Three hours out the solve is run again, and so is the calendar it is
    # judged against: built afresh for this run, handed to the solve, kept in
    # the reminder's record. The chip is still a kind on both sides of the
    # diff, so a solver-then of "none" against a solver-now that plays the
    # triple captain is a chip change like any other.
    cfg = config(state_dir=tmp_path / "state", chips=True)
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(halves_routes())
    run_pipeline(cfg, client, store, "deadline", send=False)
    record = store.decision(2, "deadline")
    store.save_run(
        2, "deadline", "the yardstick",
        dict(record, solver_actions=dict(record["solver_actions"], chip="none")),
    )

    calls = calendar_spy(monkeypatch)
    given = []
    real = orchestrator.solve

    def triple_captain_now(inputs, projections, cfg, selling_prices=None, calendar=None):
        given.append(calendar)
        solved = real(inputs, projections, cfg, selling_prices, calendar)
        path = replace(solved.choice.path, week1_chip=TRIPLE_CAPTAIN)
        return replace(solved, choice=replace(solved.choice, path=path))

    monkeypatch.setattr(orchestrator, "solve", triple_captain_now)

    alert = run_pipeline(cfg, client, store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert len(calls) == 1 and len(given) == 1
    assert given[0] is not None and given[0].fell_back is False
    assert reminder["chip_calendar"] == given[0].record()
    assert reminder["changes"]["chip"] == ["none", TRIPLE_CAPTAIN]
    assert "- Chip changed from none to triple captain" in alert


def test_a_record_from_before_solver_actions_were_kept_stays_calm(tmp_path):
    # A deadline record written before this branch has no ``solver_actions``:
    # there is no solver-then to hold the fresh solve against, and unknowable
    # is not changed — the same rule the diff applies to a record from before
    # formations were kept. The verdict here is doctored to disagree with the
    # fresh solve on every axis, which is exactly the shape an old record of
    # a manager-overridden week has, and it must not shout.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())
    run_pipeline(cfg, client, store, "deadline", send=False)
    record = store.decision(2, "deadline")
    legacy = dict(
        record,
        transfers_in=[],
        transfers_out=[],
        captain=record["vice"],
        vice=record["captain"],
    )
    del legacy["solver_actions"]
    store.save_run(2, "deadline", "a verdict from before this branch", legacy)

    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    assert render.REMINDER_UNCHANGED in alert
    assert "⚠️" not in alert
    assert store.decision(2, "reminder")["full_report_solver_plan"] is None


def test_a_reminder_with_no_full_report_behind_it_says_so(tmp_path):
    # The T-18h tick can be dropped wholesale. The reminder still goes, with
    # the fresh block and one honest line about what it could not compare.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")

    alert = run_pipeline(
        cfg, make_client(pipeline_routes()), store, "reminder", send=False
    )

    assert render.NO_FULL_REPORT in alert
    assert "⚠️" not in alert
    assert "⏰" in alert
    assert store.decision(2, "reminder")["full_report_plan"] is None


def test_the_reminder_never_touches_the_root_verdict(tmp_path):
    # GW{n}.md at the root is the polished verdict the homepage shows. The
    # reminder is history and a phone buzz, so it goes to state/reports and
    # the store and nowhere else.
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    full = run_pipeline(cfg, client, store, "deadline", send=False)
    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == full
    reports = cfg.state_dir / "reports"
    assert (reports / "gw2-reminder.md").read_text(encoding="utf-8") == alert
    assert store.has_run(2, "reminder") is True


def test_the_manager_is_never_asked_for_the_reminder(monkeypatch, tmp_path):
    # Three hours out is too late for a conversation that can take twenty
    # minutes, and the verdict was his yesterday: the reminder is the solver
    # checking the weather, with a key in the environment or without one.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked in reminder mode")

    gaffer = stub_gaffer(monkeypatch, never)

    alert = run_pipeline(
        gaffer_cfg(tmp_path),
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        "reminder",
        send=False,
    )

    assert gaffer.consults == []
    assert "The Gaffer's view" not in alert


def test_the_reminder_goes_to_telegram_and_only_telegram(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )

    alert = run_pipeline(
        cfg,
        make_client(pipeline_routes()),
        Store(cfg.state_dir / "aigaffer.db"),
        "reminder",
    )

    assert sent == [(TOKEN, "42", alert)]
    assert list(tmp_path.glob("GW*.md")) == []


def test_a_reminder_that_never_buzzed_is_not_marked_done(
    monkeypatch, capsys, tmp_path
):
    # The reminder's entire value is the buzz, so it delivers before it saves
    # — the reverse of the full report, whose diary copy is worth keeping on
    # its own. A send that fails leaves nothing behind, not the store row and
    # not the history file, so the next tick inside the window tries again;
    # the risk taken in exchange is one duplicate buzz if a send lands and
    # the save then dies, which is the cheaper failure.
    def explode(*args, **kwargs):
        raise httpx.ConnectError(f"connecting to /bot{TOKEN}/sendMessage failed")

    monkeypatch.setattr(orchestrator, "send_report", explode)
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    run_pipeline(cfg, client, store, "reminder")

    printed = capsys.readouterr().out
    assert printed.strip() == "telegram send failed: ConnectError"
    assert TOKEN not in printed
    assert store.has_run(2, "reminder") is False, "the next tick retries"
    assert not (cfg.state_dir / "reports" / "gw2-reminder.md").exists()

    # The next tick: the phone is reachable again, and the retry completes
    # the reminder exactly once.
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    alert = run_pipeline(cfg, client, store, "reminder")

    assert sent == [(TOKEN, "42", alert)]
    assert store.has_run(2, "reminder") is True
    assert (cfg.state_dir / "reports" / "gw2-reminder.md").read_text(
        encoding="utf-8"
    ) == alert


def test_a_reminders_failed_buzz_does_not_swallow_the_reconciliation_note(
    monkeypatch, tmp_path
):
    # The £1.4m audit board from the ledger section, met by a reminder whose
    # phone is down. The ledger's observation joins the deliver-first dance:
    # persisted before the send, the failed buzz would have left the GW2
    # snapshot written, the retrying tick would have found the gameweek
    # already reconciled and predicted nothing, and the one line saying the
    # bank was adrift would only ever have been in the alert nobody got.
    def explode(*args, **kwargs):
        raise httpx.ConnectError("the phone is down")

    monkeypatch.setattr(orchestrator, "send_report", explode)
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )
    store = Store(cfg.state_dir / "aigaffer.db")
    held = [p["element"] for p in PICKS_15_JSON["picks"]]
    previous = [pid if pid != 16 else 17 for pid in held]
    for pid in previous:
        store.record_purchase(pid, buy_price=90 if pid == 17 else 50, gw_seen=1)
    store.record_squad(1, bank=0, player_ids=previous)
    client = make_client(unplayed_routes(midweek_bootstrap(), (1, 90)))

    report = run_pipeline(cfg, client, store, "reminder")

    assert "the bank is £1.4m below what the purchase ledger predicted" in report
    assert store.squad_record(2) is None, "nothing persisted before the buzz"
    assert 16 not in store.purchases() and store.purchases()[17] == 90

    # The next tick: the phone answers, the retried alert carries the same
    # note, and only then does the observation land — the snapshot, Quill's
    # sighting and Reyes's departure.
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    alert = run_pipeline(cfg, client, store, "reminder")

    assert sent == [(TOKEN, "42", alert)]
    assert "the bank is £1.4m below what the purchase ledger predicted" in alert
    assert store.squad_record(2) is not None
    assert 16 in store.purchases() and 17 not in store.purchases()


# --- whose history to fetch ------------------------------------------------


def forward(pid: int, points: int, cost: int) -> Player:
    """An available forward, ranked only by what the pool ranks on."""
    return Player(
        id=pid,
        web_name=f"P{pid}",
        team=1,
        element_type=4,
        now_cost=cost,
        status="a",
        minutes=900,
        starts=10,
        total_points=points,
        bonus=0,
        saves=0,
    )


def test_the_history_pool_ranks_on_points_first():
    # The highest scorers are the cheapest here, so a pool that led on price
    # would come back with the other forty.
    players = {pid: forward(pid, points=pid, cost=100 - pid) for pid in range(1, 51)}

    assert history_pool(players, []) == list(range(11, 51))


def test_a_pool_with_no_points_to_rank_on_falls_back_to_price():
    # Between seasons every total is zero. Id order would spend the run's two
    # hundred requests on whoever the API numbers first, which is nobody in
    # particular; price is the market's own ranking and the only one left.
    players = {pid: forward(pid, points=0, cost=40 + pid) for pid in range(1, 51)}

    assert history_pool(players, []) == list(range(11, 51))


# --- delivery --------------------------------------------------------------


def test_the_report_is_sent_to_telegram(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )

    report = run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    # The phone gets the digest — the checklist and the reasoning's opening,
    # not the whole document; the full report is the file's and the store's.
    assert len(sent) == 1
    token, chat, message = sent[0]
    assert (token, chat) == (TOKEN, "42")
    assert message != report
    assert "## Do this" in message
    assert "Full report: GW2.md in the repo." in message
    assert "Candidate plans" not in message
    assert "Watchlist" not in message


def test_nothing_is_sent_without_somewhere_to_send_it(monkeypatch, capsys, tmp_path):
    # Half-configured is the easy mistake — a token in the secrets and no
    # chat id — and it looks exactly like a working bot until the phone stays
    # quiet, so the run that could not deliver says so.
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))

    run_pipeline(
        config(telegram_token=TOKEN, state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        "deadline",
    )

    printed = capsys.readouterr().out
    assert sent == []
    assert printed.strip() == orchestrator.NOT_CONFIGURED
    assert TOKEN not in printed


def test_a_failed_send_keeps_the_run_and_never_prints_the_token(
    monkeypatch, capsys, tmp_path
):
    def explode(*args, **kwargs):
        # httpx puts the request URL — and so the token — in its messages.
        raise httpx.ConnectError(f"connecting to /bot{TOKEN}/sendMessage failed")

    monkeypatch.setattr(orchestrator, "send_report", explode)
    store = Store(tmp_path / "aigaffer.db")
    cfg = config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
    )

    report = run_pipeline(cfg, make_client(pipeline_routes()), store, "deadline")

    printed = capsys.readouterr().out
    assert printed.strip() == "telegram send failed: ConnectError"
    assert TOKEN not in printed
    assert report.startswith("# AI Gaffer")
    assert store.has_run(2, "deadline") is True


# --- the command line ------------------------------------------------------


def bootstrap_due_in(hours: float) -> dict:
    """The universe with GW2's deadline that many hours from now."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    deadline = datetime.now(UTC) + timedelta(hours=hours)
    payload["events"][1]["deadline_time"] = deadline.isoformat()
    return payload


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A configured environment, and the store the CLI will keep runs in."""
    monkeypatch.setenv("FPL_TEAM_ID", str(TEAM_ID))
    monkeypatch.setenv("AIGAFFER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    return Store(tmp_path / "state" / "aigaffer.db")


def serve(monkeypatch, routes: dict, statuses: dict[str, int] | None = None) -> None:
    monkeypatch.setattr(cli, "FplClient", lambda: make_client(routes, statuses))


def telegram(monkeypatch) -> list[dict]:
    """Point the CLI's alert at a fake Telegram; return the messages posted.

    The real :func:`send_report` runs — chunking, URL and all — over a mock
    transport, so the token never leaves the process but everything that
    handles it is exercised.
    """
    posted: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert TOKEN in str(request.url)  # the token travels in the URL
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.setattr(cli, "send_report", partial(send_report, http=http))
    return posted


def test_auto_runs_the_scout_two_days_out(monkeypatch, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(48)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "scout") is True


def test_auto_runs_the_full_report_the_evening_before(monkeypatch, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(17)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "deadline") is True


def test_auto_runs_the_reminder_in_the_final_hours(monkeypatch, store):
    # The full report went out the day before, as it should have; the final
    # hours then belong to the reminder. Without that first run the same tick
    # would catch the full report up instead — pinned at the _mode tests.
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(17)))
    assert cli.main(["auto"]) == 0

    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(2)))
    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "reminder") is True


def test_a_deadline_tick_that_died_is_retried_by_the_next(monkeypatch, capsys, store):
    # The insurance the late-open window is designed around: a run that dies
    # must leave has_run false — nothing half-saved — for the next tick to
    # try again, and the retry must not then produce a duplicate of anything.
    serve(
        monkeypatch,
        pipeline_routes(bootstrap=bootstrap_due_in(17)),
        statuses={"/api/fixtures/": 500},
    )

    assert cli.main(["auto"]) == 1
    assert store.has_run(2, "deadline") is False
    assert "the deadline run failed" in capsys.readouterr().out

    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(17)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "deadline") is True
    assert len(store.last_runs()) == 1, "one record: the tick that succeeded"


def test_a_reminder_already_sent_is_not_sent_twice(monkeypatch, capsys, store):
    # The windows stay open until the deadline, so it is the store and only
    # the store that keeps every later tick in the reminder's hours quiet.
    store.save_run(2, "deadline", "the full report from the day before", {})
    store.save_run(2, "reminder", "the reminder from an hour ago", {})
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(2)))

    assert cli.main(["auto"]) == 0
    assert len(store.last_runs()) == 2, "the two seeded runs and nothing new"
    assert capsys.readouterr().out == ""


def test_force_runs_the_reminder_again(monkeypatch, store):
    store.save_run(2, "reminder", "the reminder from half an hour ago", {})
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["reminder", "--force"]) == 0
    assert len(store.last_runs()) == 2


def test_a_dry_run_reminder_prints_and_leaves_nothing(monkeypatch, capsys, store):
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["reminder", "--dry-run", "--force"]) == 0

    assert capsys.readouterr().out.startswith("# AI Gaffer — GW2 reminder")
    assert store.last_runs() == []


def test_auto_stands_down_between_the_windows(monkeypatch, capsys, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(100)))

    assert cli.main(["auto"]) == 0
    assert store.last_runs() == []
    assert capsys.readouterr().out == ""


def test_auto_stands_down_when_the_gameweek_has_already_run(monkeypatch, capsys, store):
    store.save_run(2, "scout", "the report from four hours ago", {})
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(48)))

    assert cli.main(["auto"]) == 0
    assert len(store.last_runs()) == 1
    assert capsys.readouterr().out == ""


def test_a_mode_asked_for_by_name_says_why_it_did_not_run(monkeypatch, capsys, store):
    store.save_run(2, "scout", "the report from four hours ago", {})
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["scout"]) == 0
    assert len(store.last_runs()) == 1
    assert "--force" in capsys.readouterr().out


def test_force_runs_auto_again_inside_a_window(monkeypatch, store):
    store.save_run(2, "scout", "the report from four hours ago", {})
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(48)))

    assert cli.main(["auto", "--force"]) == 0
    assert len(store.last_runs()) == 2


def test_force_runs_a_mode_the_gameweek_has_already_had(monkeypatch, store):
    store.save_run(2, "scout", "the report from four hours ago", {})
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["scout", "--force"]) == 0
    assert len(store.last_runs()) == 2


def test_a_dry_run_prints_the_report_and_saves_nothing(monkeypatch, capsys, store):
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["deadline", "--dry-run", "--force"]) == 0

    assert capsys.readouterr().out.startswith("# AI Gaffer — GW2 deadline")
    assert store.last_runs() == []


def test_a_pipeline_that_cannot_run_exits_non_zero(monkeypatch, capsys, store):
    over = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for event in over["events"]:
        event.update(is_next=False, is_current=False)
    serve(monkeypatch, pipeline_routes(bootstrap=over))

    assert cli.main(["deadline", "--force"]) == 1
    assert "gameweek" in capsys.readouterr().out


def test_an_api_that_will_not_answer_exits_non_zero_without_quoting_itself(
    monkeypatch, capsys, store
):
    # httpx names the URL it was calling in every message it raises, and for
    # the Telegram leg of a run that URL has the bot token in it. So the CLI
    # prints the class of the failure and the stage it happened at, and never
    # the exception's own words.
    serve(monkeypatch, pipeline_routes(), statuses={"/api/fixtures/": 500})

    assert cli.main(["deadline", "--force"]) == 1

    printed = capsys.readouterr().out
    assert printed.strip() == "aigaffer: the deadline run failed: HTTPStatusError"
    assert "premierleague.com" not in printed
    assert store.last_runs() == []


def test_a_schedule_the_api_will_not_serve_is_reported_the_same_way(
    monkeypatch, capsys, store
):
    # Nothing has been decided yet, so there is no mode to name — only the
    # stage that failed.
    serve(monkeypatch, {}, statuses={"/api/bootstrap-static/": 503})

    assert cli.main(["auto"]) == 1
    assert (
        capsys.readouterr().out.strip()
        == "aigaffer: reading the schedule failed: HTTPStatusError"
    )


def test_a_failed_run_tells_the_phone_that_expects_the_report(
    monkeypatch, capsys, store
):
    # A run that dies in silence looks exactly like a quiet week, and the
    # point of the schedule is that nobody has to check.
    posted = telegram(monkeypatch)
    serve(
        monkeypatch,
        pipeline_routes(bootstrap=bootstrap_due_in(17)),
        statuses={"/api/fixtures/": 500},
    )

    assert cli.main(["auto"]) == 1

    assert posted == [
        {
            "chat_id": "42",
            "text": "aigaffer run failed: HTTPStatusError (gw 2)",
            "parse_mode": "HTML",
        }
    ]
    assert TOKEN not in capsys.readouterr().out


def test_a_failure_with_no_gameweek_named_still_gets_an_alert(monkeypatch, store):
    # --force skips the schedule read, so nothing has asked which gameweek
    # this is. The alert says what it knows and nothing more.
    posted = telegram(monkeypatch)
    over = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for event in over["events"]:
        event.update(is_next=False, is_current=False)
    serve(monkeypatch, pipeline_routes(bootstrap=over))

    assert cli.main(["deadline", "--force"]) == 1
    assert posted == [
        {
            "chat_id": "42",
            "text": "aigaffer run failed: PipelineError",
            "parse_mode": "HTML",
        }
    ]


def test_an_alert_that_fails_too_is_not_a_second_failure(monkeypatch, capsys, store):
    # The exit code is already 1 and the reason is already printed; the alert
    # is the last thing anybody wants an exception from.
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def explode(*args, **kwargs):
        raise httpx.ConnectError(f"connecting to /bot{TOKEN}/sendMessage failed")

    monkeypatch.setattr(cli, "send_report", explode)
    serve(monkeypatch, pipeline_routes(), statuses={"/api/fixtures/": 500})

    assert cli.main(["deadline", "--force"]) == 1
    assert TOKEN not in capsys.readouterr().out


def test_a_dry_run_that_fails_keeps_it_off_the_phone(monkeypatch, store):
    # --dry-run means nothing leaves the machine, and there is a person at
    # the keyboard reading the failure already.
    posted = telegram(monkeypatch)
    serve(monkeypatch, pipeline_routes(), statuses={"/api/fixtures/": 500})

    assert cli.main(["deadline", "--force", "--dry-run"]) == 1
    assert posted == []


def test_a_manager_asked_for_without_a_key_says_so(monkeypatch, capsys, store):
    # AIGAFFER_MANAGER is an opt-out, so asking for the manager and getting
    # the solver alone is a silent disappointment unless the run says why.
    # A missing repository secret arrives looking exactly like this.
    monkeypatch.setenv("AIGAFFER_MANAGER", "1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["scout", "--force", "--dry-run"]) == 0

    printed = capsys.readouterr().out
    assert cli.NO_MANAGER in printed
    assert "## The Gaffer's view" not in printed


def test_a_manager_turned_off_on_purpose_says_nothing(monkeypatch, capsys, store):
    monkeypatch.setenv("AIGAFFER_MANAGER", "0")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    serve(monkeypatch, pipeline_routes())

    assert cli.main(["scout", "--force", "--dry-run"]) == 0
    assert cli.NO_MANAGER not in capsys.readouterr().out


def test_an_unknown_command_is_refused():
    with pytest.raises(SystemExit) as refusal:
        cli.main(["wildcard"])

    assert refusal.value.code == 2


def test_the_strength_switch_is_honored(monkeypatch, tmp_path):
    # The kill switch is the feature's whole safety story: off must mean
    # the fit is never even asked for, and on must mean it is.
    asked = []
    monkeypatch.setattr(
        orchestrator, "build_team_strengths", lambda *args, **kw: asked.append(1)
    )
    serve(monkeypatch, pipeline_routes())

    def run_with(enabled: bool) -> int:
        cfg = config(
            state_dir=tmp_path / f"state-{enabled}",
            strength_enabled=enabled,
        )
        client = make_client(pipeline_routes())
        store = Store(tmp_path / f"db-{enabled}.db")
        run_pipeline(cfg, client, store, "scout", send=False, save=False)
        return len(asked)

    assert run_with(False) == 0, "off: the fit is never asked for"
    assert run_with(True) >= 1, "on: it is"


def test_a_chips_run_carries_the_calendar_in_the_report_and_one_line_on_the_phone(
    monkeypatch, tmp_path
):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    cfg = config(
        telegram_token=TOKEN, telegram_chat_id="42", state_dir=tmp_path / "state", chips=True
    )

    report = run_pipeline(
        cfg, make_client(halves_routes()), Store(tmp_path / "aigaffer.db"), "scout"
    )

    [(_, _, message)] = sent
    assert "## Chip calendar" in report
    assert "## Chip calendar" not in message
    [line] = [text for text in message.splitlines() if text.startswith("Chips: ")]
    assert line.endswith("expire GW19")


def test_a_recorded_signing_outside_the_cut_still_gets_a_history(monkeypatch, tmp_path):
    # With the cut at one a position, Sarr (FWD, 31 points) is not in the
    # history pool: Haas tops the forwards. Recorded as signed this week, he
    # must be — or he projects at zero and the solve sells him at once.
    monkeypatch.setattr(orchestrator, "CANDIDATES_PER_POSITION", 1)
    cfg = config(state_dir=tmp_path / "state")
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed(transfers_in=[18], transfers_out=[8]))

    without = fetch_inputs(cfg, make_client(pipeline_routes()))
    with_row = fetch_inputs(cfg, make_client(pipeline_routes()), store=store)

    assert 18 not in without.histories
    assert 18 in with_row.histories
    assert with_row.executed is None, "the fetch is still the API's truth"


def test_week1_lock_reads_the_recorded_week_off_the_inputs(seam):
    from aigaffer.solver.optimizer import Week1Lock
    from tests.fixtures import make_executed

    def lock(**fields):
        return _week1_lock(replace(seam.inputs, executed=make_executed(**fields)))

    assert _week1_lock(replace(seam.inputs, executed=None)) is None
    assert lock(transfers_in=[17, 18], transfers_out=[6]) == Week1Lock(
        keep=frozenset({17, 18}), shun=frozenset({6}), hold=False, free=False
    )
    assert lock(chip="free_hit") == Week1Lock(
        keep=frozenset({17}), shun=frozenset({6}), hold=True, free=False
    )
    assert lock(chip="wildcard") == Week1Lock(
        keep=frozenset({17}), shun=frozenset({6}), hold=False, free=True
    )
    assert lock(chip="bench_boost") == Week1Lock(
        keep=frozenset({17}), shun=frozenset({6}), hold=False, free=False
    )


# --- after "Transfers made" -----------------------------------------------
#
# The owner texted "Transfers made" for the newest verdict on record; the run
# after it must solve from the squad he entered, say so, never sell what he
# signed, and leave the ledger on the API's truth.

FH_SQUAD = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
FH_XI = [1, 3, 4, 10, 12, 5, 6, 11, 14, 17, 7]  # 4-5-1
PLAYERS = {e["id"]: Player.model_validate(e) for e in PIPELINE_ELEMENTS_JSON}


def enter_latest(store: Store):
    """Text "Transfers made" for the newest verdict on record, as the inbox
    would (no git, no Telegram); return the row it wrote."""
    client = make_client(pipeline_routes())
    now = datetime.now(UTC)
    outcome = record_transfers_made(
        store, client.bootstrap(), client.picks(TEAM_ID, 1),
        HISTORY_JSON["chips"], now + timedelta(seconds=1), now,
    )
    assert outcome.changed, outcome.reply
    return store.executed(2)


class Entered(NamedTuple):
    report: str
    store: Store
    row: object


@pytest.fixture(scope="module")
def entered_run(tmp_path_factory) -> Entered:
    """Scout, then "Transfers made" on it, then the deadline report."""
    state = tmp_path_factory.mktemp("entered_run") / "state"
    cfg = config(state_dir=state)
    store = Store(state / "aigaffer.db")
    client = make_client(pipeline_routes())
    run_pipeline(cfg, client, store, "scout", send=False)
    row = enter_latest(store)
    report = run_pipeline(cfg, client, store, "deadline", send=False)
    return Entered(report=report, store=store, row=row)


def test_a_run_after_transfers_made_says_what_it_works_from(entered_run):
    line = working_from(entered_run.row, PLAYERS)

    assert entered_run.row.transfers_in, "the scout made a move to enter"
    assert line in entered_run.report
    assert entered_run.report.index(line) < entered_run.report.index("## Do this")


def test_it_solves_from_the_entered_squad_and_never_undoes_it(entered_run):
    row = entered_run.row
    decision = entered_run.store.decision(2, "deadline")

    assert decision["squad_before"] == row.squad_after
    assert decision["free_transfers"] == row.ft_after
    assert decision["executed_from"] == row.verdicts
    assert not set(row.transfers_in) & set(decision["transfers_out"]), "no signing sold"
    assert not set(row.transfers_out) & set(decision["transfers_in"]), "no sale bought back"


def test_the_ledger_never_sees_the_entered_squad(entered_run):
    store = entered_run.store

    assert set(store.purchases()) == set(PICKS_15_IDS)
    assert store.squad_record(1) == {"gw": 1, "bank": 28, "player_ids": PICKS_15_IDS}


def test_both_ledger_calls_read_the_api_squad(monkeypatch, tmp_path):
    # The read before the solve prices the sales and the write after the
    # report keeps the snapshot; both are the ledger's, and the ledger only
    # ever sees the picks the game has. Grant-for-Reyes is recorded, so the
    # effective squad holds Reyes and not Grant — neither call may.
    seen = []
    real = orchestrator.observe

    def spy(store, squad, *args, **kwargs):
        seen.append((squad.player_ids, kwargs.get("persist", True)))
        return real(store, squad, *args, **kwargs)

    monkeypatch.setattr(orchestrator, "observe", spy)
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())

    run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "deadline", send=False,
    )

    assert seen == [(PICKS_15_IDS, False), (PICKS_15_IDS, True)]
    assert store.decision(2, "deadline")["squad_before"] == sorted(
        set(PICKS_15_IDS) - {6} | {17}
    ), "while the run itself solved from the entered squad"


def test_a_row_for_another_gameweek_changes_nothing(tmp_path):
    # Last gameweek's row is history: the API shows its moves now, and a run
    # that applied it again would count them twice. Byte for byte the run
    # with nothing recorded.
    # Separate directories: each store keeps its recordings beside itself.
    stale = Store(tmp_path / "stale" / "aigaffer.db")
    stale.save_executed(make_executed(gw=1))

    plain = run_pipeline(
        config(state_dir=tmp_path / "plain_state"), make_client(pipeline_routes()),
        Store(tmp_path / "plain" / "aigaffer.db"), "deadline", send=False,
    )
    with_stale = run_pipeline(
        config(state_dir=tmp_path / "stale_state"), make_client(pipeline_routes()),
        stale, "deadline", send=False,
    )

    assert with_stale == plain
    assert stale.decision(2, "deadline")["executed_from"] is None


@pytest.mark.parametrize("mode", ["deadline", "reminder"])
def test_an_unreadable_recorded_week_runs_from_the_api_squad(tmp_path, capsys, mode):
    # A gw2.json cut off mid-write (or written by another schema) must cost
    # the owner his "Transfers made", never the run: it goes ahead from the
    # API's squad — byte for byte the run with nothing recorded — and says
    # in one line why it ignored the row.
    broken = Store(tmp_path / "broken" / "aigaffer.db")
    broken.executed_dir.mkdir(parents=True)
    (broken.executed_dir / "gw2.json").write_text('{"gw": 2, "mode": "dead')

    plain = run_pipeline(
        config(state_dir=tmp_path / "plain_state"), make_client(pipeline_routes()),
        Store(tmp_path / "plain" / "aigaffer.db"), mode, send=False,
    )
    capsys.readouterr()
    with_broken = run_pipeline(
        config(state_dir=tmp_path / "broken_state"), make_client(pipeline_routes()),
        broken, mode, send=False,
    )

    assert with_broken == plain
    assert broken.decision(2, mode).get("executed_from") is None
    assert (
        "recorded week gw2 unreadable: JSONDecodeError — running from the API's squad"
        in capsys.readouterr().out
    )


def test_a_recorded_free_hit_fields_the_entered_team_and_holds(tmp_path):
    # With one free transfer the solver would sign Reyes for Grant; on a free
    # hit he entered, the standing squad makes no move and the team on the
    # sheet is his, with his armbands.
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(
        make_executed(
            chip="free_hit", transfers_in=[], transfers_out=[],
            squad_after=sorted(PICKS_15_IDS), bank_after=28, ft_after=1,
            buy_prices={}, sell_prices={}, freehit_squad=FH_SQUAD, freehit_xi=FH_XI,
            arrival_status={9: "a", 10: "a", 12: "a", 17: "a", 18: "a"},
        )
    )

    report = run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "deadline", send=False,
    )
    decision = store.decision(2, "deadline")

    assert render.CHIP_ENTERED.format(chip="Free Hit") in report
    assert "Free Hit XI (this week only)" in report
    assert decision["chip"] == "free_hit"
    assert decision["solver_actions"]["chip"] == "free_hit", "the overlay reaches plan_actions"
    assert decision["freehit_squad"] == FH_SQUAD and decision["freehit_xi"] == FH_XI
    assert decision["transfers_in"] == [] and decision["transfers_out"] == []
    assert decision["captain"] == 5 and decision["vice"] == 17
    # The sheet prints the eleven he entered, in his order, and his armbands —
    # not an eleven re-picked over his fifteen, and not the standing squad's.
    # Ferrer is captain and Reyes, whom only the free hit brings in, vice.
    eleven = ", ".join(PLAYERS[pid].web_name for pid in FH_XI)
    assert f"{render.FREE_HIT_XI}: {eleven}" in report
    assert "CAPTAIN Ferrer · VICE Reyes" in report
    assert "Ferrer (C)" in report and "Reyes (V)" in report
    assert decision["formation"] == "4-5-1"
    # The four left over, keeper first: Jarvis, then the three outfielders.
    assert decision["bench"][0] == 9
    assert set(decision["bench"]) == {9, 13, 15, 18}


def test_a_recorded_free_hit_with_no_team_on_record_fields_the_standing_eleven(
    tmp_path,
):
    # The gaffer finalized a free hit the solver had built no team for, and
    # the owner entered it: the row has the chip and no fifteen. There is no
    # other team to field, so the sheet is the standing squad's best eleven —
    # never a crash — and the checklist says the chip is played already
    # rather than telling him to play it.
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(
        make_executed(
            chip="free_hit", transfers_in=[], transfers_out=[],
            squad_after=sorted(PICKS_15_IDS), bank_after=28, ft_after=1,
            buy_prices={}, sell_prices={}, arrival_status={},
        )
    )

    report = run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "deadline", send=False,
    )
    decision = store.decision(2, "deadline")

    assert render.CHIP_ENTERED.format(chip="Free Hit") in report
    assert "PLAY Free Hit" not in report
    assert render.FREE_HIT_XI not in report, "no temporary team to name"
    assert decision["chip"] == "free_hit"
    assert decision["freehit_squad"] is None and decision["freehit_xi"] is None
    assert decision["transfers_in"] == [] and decision["transfers_out"] == []
    # The standing fifteen's eleven: Ferrer, our own premium, wears the
    # armband, and Ito — injured — sits on the bench.
    assert decision["captain"] == FERRER
    assert set(decision["bench"]) < set(PICKS_15_IDS)
    assert 8 in decision["bench"]


def test_an_entered_free_hit_is_fielded_as_entered(seam):
    # His eleven in his order and his armbands, whatever the projections
    # would pick over his fifteen; the bench is the four left over, the
    # keeper first and the outfielders by this week's projection — the order
    # the app auto-subs them in.
    _, projections = build_projections(seam.inputs, seam.cfg)
    event_id = seam.inputs.event.id
    positions = {pid: p.element_type for pid, p in seam.inputs.players.items()}
    gw_xp = {pid: pr.per_gw.get(event_id, 0.0) for pid, pr in projections.items()}
    standing_squad = seam.inputs.squad.player_ids
    standing = pick_lineup(standing_squad, positions, gw_xp)
    # A forced-hold solve: no path, so no free-hit squad of the solver's own.
    choice = Plan(
        squad=standing_squad, xi=[], transfers_in=[], transfers_out=[], hits=0,
        xp_total=0.0, objective=0.0,
    )
    executed = make_executed(
        chip="free_hit", transfers_in=[], transfers_out=[],
        squad_after=sorted(PICKS_15_IDS), freehit_squad=FH_SQUAD, freehit_xi=FH_XI,
        captain=FERRER, vice=REYES,
    )

    fielded = _fielded_lineup(
        "free_hit", choice, standing, seam.inputs.players, projections, event_id,
        executed,
    )

    assert fielded.xi == FH_XI
    assert (fielded.captain, fielded.vice) == (FERRER, REYES)
    outfield = sorted([13, 15, 18], key=lambda pid: -gw_xp.get(pid, 0.0))
    assert fielded.bench == [9, *outfield]
    # Without a team on the row there is nothing else to field: the decided
    # eleven comes straight back.
    bare = replace(executed, freehit_squad=None, freehit_xi=None)
    assert _fielded_lineup(
        "free_hit", choice, standing, seam.inputs.players, projections, event_id,
        bare,
    ) is standing


def recorded_wildcard_store(tmp_path) -> Store:
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed(chip="wildcard", ft_after=1))
    return store


def test_the_gaffer_cannot_play_a_second_chip_in_a_recorded_chip_week(
    monkeypatch, tmp_path
):
    store = recorded_wildcard_store(tmp_path)
    stub_gaffer(monkeypatch, partial(decided, chip="bench_boost"))

    run_pipeline(
        gaffer_cfg(tmp_path), make_client(pipeline_routes()), store, "scout",
        send=False, now=PAST_THE_FLOOR,
    )
    decision = store.decision(2, "scout")

    assert decision["decision_source"] == f"solver-fallback: {CHIP_SPENT}"
    assert decision["chip"] == "wildcard", "the recorded chip stands"


def test_the_gaffers_none_in_a_recorded_chip_week_stands(monkeypatch, tmp_path):
    store = recorded_wildcard_store(tmp_path)
    gaffer = stub_gaffer(monkeypatch)

    run_pipeline(
        gaffer_cfg(tmp_path), make_client(pipeline_routes()), store, "scout",
        send=False, now=PAST_THE_FLOOR,
    )
    decision = store.decision(2, "scout")

    assert decision["decision_source"] == "manager"
    assert decision["chip"] == "wildcard"
    assert CHIP_ALREADY_PLAYED.format(chip="Wildcard") in gaffer.consults[0].briefing


# --- the T-3h reminder after "Transfers made" --------------------------------
#
# The yardstick is what he entered: a hold with his armbands and the shape and
# bench of the verdict he entered, against a fresh solve on the squad he
# entered. A calm week still buzzes, with the calm line.


def test_the_bench_is_diffed_only_when_both_sides_keep_it():
    base = {"transfers": [], "captain": 1, "vice": 2, "chip": "none", "formation": "4-4-2"}

    assert diff_actions({**base, "bench": [3, 4]}, {**base, "bench": [4, 3]}) == {
        "bench": [[3, 4], [4, 3]]
    }
    assert diff_actions(base, {**base, "bench": [4, 3]}) == {}


def doubtful(pid: int) -> dict:
    """The pipeline board with ``pid`` flagged a doubt since yesterday."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for element in payload["elements"]:
        if element["id"] == pid:
            element.update(status="d", chance_of_playing_next_round=50)
    return payload


def test_an_entered_week_with_nothing_new_gets_the_calm_alert(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())
    run_pipeline(cfg, client, store, "deadline", send=False)
    row = enter_latest(store)

    alert = run_pipeline(cfg, client, store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert render.RECORDED_CALM in alert
    assert render.ENTERED_ALREADY in alert
    assert working_from(row, PLAYERS) in alert
    assert "⚠️" not in alert
    assert reminder["changes"] == {}
    assert reminder["executed_from"] == row.verdicts
    # And the eleven he entered: the squad after his moves less the bench of
    # the verdict he entered, with his armbands.
    deadline = store.decision(2, "deadline")
    entered = _entered_lineup(row, deadline, PLAYERS)
    assert entered is not None
    assert render.phone_lineup(entered, PLAYERS) in alert
    assert alert.index("Bench:") < alert.index(render.RECORDED_CALM)
    assert reminder["full_report_plan"]["transfers"] == []
    assert reminder["full_report_plan"]["captain"] == row.captain
    assert set(store.purchases()) == set(PICKS_15_IDS), "the ledger kept the API's truth"


def test_a_signing_flagged_since_he_was_entered_is_news_at_t_minus_3(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    run_pipeline(cfg, make_client(pipeline_routes()), store, "deadline", send=False)
    row = enter_latest(store)
    assert row.transfers_in, "the deadline verdict signs someone"
    signing = row.transfers_in[0]

    alert = run_pipeline(
        cfg, make_client(pipeline_routes(bootstrap=doubtful(signing))), store,
        "reminder", send=False,
    )
    reminder = store.decision(2, "reminder")

    assert [signing, "a", "d"] in reminder["changes"]["arrivals"]
    assert render.NEWS_MOVED_ENTERED in alert
    assert "(entered this week) is now doubtful — available when you entered him" in alert


# --- what a recorded signing sells for, before the game has seen him ---------
#
# The ledger reads the real picks, so before the deadline a signing he
# entered has no ledger price and would be sold at his listed one. FPL pays
# half the rise: Reyes bought at £9.3m and listed at £9.5m sells for £9.4m.


def test_a_recorded_signing_sells_for_half_his_rise():
    ledger = Observation(selling_prices={1: 55, 6: 75})
    effective = SimpleNamespace(players=PLAYERS)

    prices = _selling_prices(ledger, make_executed(buy_prices={17: 93}), effective)

    assert prices == {1: 55, 6: 75, 17: 94}, "93 + (95 - 93) // 2, not his listed 95"
    assert ledger.selling_prices == {1: 55, 6: 75}, "the ledger itself is untouched"
    assert _selling_prices(ledger, None, effective) is ledger.selling_prices


@pytest.mark.parametrize("mode", ["deadline", "reminder"])
def test_the_solve_sells_a_recorded_signing_at_half_his_rise(monkeypatch, tmp_path, mode):
    asked = []
    real = orchestrator.solve

    def spy(inputs, projections, cfg, selling_prices=None, calendar=None):
        asked.append(selling_prices)
        return real(inputs, projections, cfg, selling_prices, calendar)

    monkeypatch.setattr(orchestrator, "solve", spy)
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed(buy_prices={17: 93}))

    run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, mode, send=False,
    )

    [prices] = asked
    assert prices[17] == 94, "credited +0.1 on a 0.2 rise, not his listed 95"
    assert set(prices) == set(PICKS_15_IDS) | {17}


# --- the phone after "Transfers made" -----------------------------------------


def test_the_digest_on_the_phone_says_what_it_works_from(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())
    cfg = config(
        telegram_token=TOKEN, telegram_chat_id="42", state_dir=tmp_path / "state"
    )

    run_pipeline(cfg, make_client(pipeline_routes()), store, "deadline")

    [(_, _, message)] = sent
    assert working_from(make_executed(), PLAYERS) in message


def test_the_withheld_alert_on_the_phone_says_it_too(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args, **kwargs: sent.append(args))
    stub_gaffer(monkeypatch, unavailable)
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())
    cfg = gaffer_cfg(tmp_path, telegram_token=TOKEN, telegram_chat_id="42")

    run_pipeline(
        cfg, make_client(pipeline_routes()), store, "deadline",
        now=PAST_THE_FLOOR - timedelta(hours=20),
    )

    [(_, _, alert)] = sent
    assert "The gaffer did not decide" in alert, "withheld, not the digest"
    assert working_from(make_executed(), PLAYERS) in alert


# --- like against like, after "Transfers made" --------------------------------
#
# The armbands, shape and bench are diffed solver-then against solver-now,
# as the ordinary reminder diffs them: a week where the gaffer overruled the
# solver's armband was settled at T-18h, and is not news at T-3h. What he
# entered is still the plan the alert shows as operative.


def test_a_gaffer_overruled_armband_is_not_news_at_t_minus_3(monkeypatch, tmp_path):
    # The solver's own plan, with the gaffer's armbands (the highest id in
    # the eleven, never the solver's captain).
    stub_gaffer(monkeypatch, lambda consult: decided(consult, plan=consult.solve0.choice))
    cfg = gaffer_cfg(tmp_path)
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())
    run_pipeline(cfg, client, store, "deadline", send=False, now=PAST_THE_FLOOR)
    record = store.decision(2, "deadline")
    assert record["captain"] != record["solver_actions"]["captain"], "he overruled it"
    row = enter_latest(store)

    alert = run_pipeline(cfg, client, store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert reminder["changes"] == {}
    assert render.RECORDED_CALM in alert
    assert f"CAPTAIN {PLAYERS[row.captain].web_name}" in alert, "his armband, operative"


def test_the_solvers_own_bench_is_kept_beside_the_operative_one(tmp_path):
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "deadline", send=False,
    )
    record = store.decision(2, "deadline")

    # No gaffer: the solver's eleven is the operative one, bench and all.
    assert len(record["solver_bench"]) == 4
    assert record["solver_bench"] == record["bench"]
    assert "bench" not in record["solver_actions"], "the five-field shape stays"


def entered_week(tmp_path):
    """A deadline run, then "Transfers made" on it: the config, store, row."""
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    run_pipeline(cfg, make_client(pipeline_routes()), store, "deadline", send=False)
    return cfg, store, enter_latest(store)


def read_back_as(monkeypatch, store: Store, edit) -> None:
    """Have the reminder read the recorded verdict back with ``edit`` applied."""
    real = store.decision_at

    def read(gw, mode, ts):
        record = copy.deepcopy(real(gw, mode, ts))
        edit(record)
        return record

    monkeypatch.setattr(store, "decision_at", read)


def test_a_moved_solver_captain_is_news_under_his_operative_armbands(
    monkeypatch, tmp_path
):
    cfg, store, row = entered_week(tmp_path)
    # Yesterday the solver wanted his vice as captain; today it wants his
    # captain. That is the solver moving, whatever he entered.
    read_back_as(
        monkeypatch, store,
        lambda record: record["solver_actions"].update(captain=row.vice),
    )

    alert = run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert reminder["changes"] == {"captain": [row.vice, row.captain]}
    assert render.NEWS_MOVED_ENTERED in alert
    entered_block = alert[alert.index(render.ENTERED_VERDICT):alert.index(render.FRESH_SOLVE)]
    captain, vice = PLAYERS[row.captain].web_name, PLAYERS[row.vice].web_name
    assert f"CAPTAIN {captain} · VICE {vice}" in entered_block
    assert reminder["full_report_plan"]["captain"] == row.captain, "shown: his"
    assert reminder["full_report_solver_plan"]["captain"] == row.vice, "diffed: the solver's"


def test_the_bench_is_diffed_against_the_solvers_own(monkeypatch, tmp_path):
    cfg, store, row = entered_week(tmp_path)
    bench = store.decision(2, "deadline")["solver_bench"]
    read_back_as(
        monkeypatch, store, lambda record: record.update(solver_bench=bench[::-1])
    )

    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    assert store.decision(2, "reminder")["changes"] == {"bench": [bench[::-1], bench]}


def test_a_record_from_before_the_solvers_bench_skips_the_bench(monkeypatch, tmp_path):
    cfg, store, _ = entered_week(tmp_path)
    read_back_as(monkeypatch, store, lambda record: record.pop("solver_bench"))

    alert = run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    assert store.decision(2, "reminder")["changes"] == {}
    assert render.RECORDED_CALM in alert


def test_a_row_with_no_verdicts_still_buzzes(tmp_path):
    cfg, store, row = entered_week(tmp_path)
    store.save_executed(replace(row, verdicts=[]))

    alert = run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert render.ENTERED_ALREADY in alert
    assert reminder["full_report_plan"]["formation"] is None, "nothing to read it from"
    assert reminder["full_report_plan"]["bench"] is None


def wanting_a_move(monkeypatch) -> None:
    """Have every solve want Sarr for Ito, whatever it was locked to."""
    real = orchestrator.solve

    def solve(*args, **kwargs):
        solved = real(*args, **kwargs)
        choice = replace(solved.choice, transfers_in=[18], transfers_out=[8])
        return replace(solved, choice=choice)

    monkeypatch.setattr(orchestrator, "solve", solve)


def wanting_a_chip(monkeypatch, chip: str) -> None:
    """Have every solve want to play ``chip`` this week, on its own path."""
    real = orchestrator.solve

    def solve(*args, **kwargs):
        solved = real(*args, **kwargs)
        assert solved.choice.path is not None, "the chip rides the plan's path"
        path = replace(solved.choice.path, week1_chip=chip)
        return replace(solved, choice=replace(solved.choice, path=path))

    monkeypatch.setattr(orchestrator, "solve", solve)


def test_a_solver_chip_the_gaffer_overruled_is_not_news_at_t_minus_3(
    monkeypatch, tmp_path
):
    # Yesterday the solver wanted to bench boost and the gaffer said no, so
    # he entered no chip. Today the solver, alone, still wants to: solver
    # against solver, nothing moved. Measured against the "none" he entered
    # it would cry "Chip changed" over a call settled at T-18h.
    cfg, store, row = entered_week(tmp_path)
    assert row.chip == "none"
    read_back_as(
        monkeypatch, store,
        lambda record: record["solver_actions"].update(chip="bench_boost"),
    )
    wanting_a_chip(monkeypatch, "bench_boost")

    alert = run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert reminder["changes"] == {}
    assert render.RECORDED_CALM in alert
    assert reminder["full_report_plan"]["chip"] == "none", "shown: his"
    assert reminder["full_report_solver_plan"]["chip"] == "bench_boost", "diffed: the solver's"


def test_a_chip_the_solver_moved_to_is_news_at_t_minus_3(monkeypatch, tmp_path):
    # The other way about: yesterday's solver played nothing, today's wants
    # to bench boost — the solver moved, and that is news.
    cfg, store, _ = entered_week(tmp_path)
    assert store.decision(2, "deadline")["solver_actions"]["chip"] == "none"
    wanting_a_chip(monkeypatch, "bench_boost")

    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    assert store.decision(2, "reminder")["changes"] == {"chip": ["none", "bench_boost"]}


def test_a_chip_he_entered_over_the_solvers_none_is_not_news(monkeypatch, tmp_path):
    # The gaffer played a wildcard the solver did not want, and he entered
    # it. An entered chip is played: the fresh side reports it whatever the
    # solver now wants, so it can never move, and diffing it against the
    # solver's "none" would cry "Chip changed" every reminder that week.
    cfg, store, row = entered_week(tmp_path)
    store.save_executed(replace(row, chip="wildcard"))
    assert store.decision(2, "deadline")["solver_actions"]["chip"] == "none"

    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)
    reminder = store.decision(2, "reminder")

    assert "chip" not in reminder["changes"]
    assert reminder["actions"]["chip"] == "wildcard"


def test_a_record_with_no_solver_side_skips_the_chip(monkeypatch, tmp_path):
    # A record from before solver_actions were kept cannot say what the
    # solver played, so the chip is not diffed — as with the bench.
    cfg, store, _ = entered_week(tmp_path)
    read_back_as(monkeypatch, store, lambda record: record.pop("solver_actions"))
    wanting_a_chip(monkeypatch, "bench_boost")

    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    assert "chip" not in store.decision(2, "reminder")["changes"]


def test_a_recorded_free_hit_week_has_no_transfer_news(monkeypatch, tmp_path):
    # The fresh solve plans the standing squad, which he cannot touch in a
    # free-hit week: a move it wants is not news he can act on.
    wanting_a_move(monkeypatch)
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(
        make_executed(
            chip="free_hit", transfers_in=[], transfers_out=[],
            squad_after=sorted(PICKS_15_IDS), bank_after=28, ft_after=1,
            buy_prices={}, sell_prices={}, freehit_squad=FH_SQUAD, freehit_xi=FH_XI,
            arrival_status={9: "a", 10: "a", 12: "a", 17: "a", 18: "a"},
        )
    )

    alert = run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "reminder", send=False,
    )
    reminder = store.decision(2, "reminder")

    assert not {"sells_added", "buys_added"} & set(reminder["changes"])
    assert reminder["actions"]["transfers"] == []
    assert "Now buying" not in alert and "Now selling" not in alert


def test_an_ordinary_entered_week_keeps_its_transfer_news(monkeypatch, tmp_path):
    wanting_a_move(monkeypatch)
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())

    alert = run_pipeline(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()),
        store, "reminder", send=False,
    )
    reminder = store.decision(2, "reminder")

    assert reminder["changes"]["buys_added"] == [18]
    assert reminder["changes"]["sells_added"] == [8]
    assert "Now buying: Sarr" in alert


def test_prepare_week_reads_the_recorded_week_once_and_lays_it_over(tmp_path):
    # One read of the recorded week a run: fetch_inputs needs it for the
    # history pool, apply_executed for the effective squad, and both must see
    # the same row — a what-if hands in a row it read under the state lock.
    calls: list[int] = []
    row = make_executed()

    def reader(gw: int):
        calls.append(gw)
        return row

    week = prepare_week(
        config(state_dir=tmp_path / "state"),
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        executed_for=reader,
    )

    assert isinstance(week, PreparedWeek)
    assert calls == [2], "once, for the coming gameweek"
    assert week.executed == row
    assert week.effective.executed == row and week.inputs.executed is None
    assert sorted(week.effective.squad.player_ids) == row.squad_after
    assert sorted(week.inputs.squad.player_ids) == sorted(PICKS_15_IDS), "the API's truth kept"
    assert 17 in week.prices, "the recorded signing is priced to sell"


def test_prepare_week_reads_the_store_when_no_reader_is_given(tmp_path):
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed())

    week = prepare_week(
        config(state_dir=tmp_path / "state"), make_client(pipeline_routes()), store
    )

    assert week.executed == make_executed()


def test_a_reader_overrules_the_store(monkeypatch, tmp_path):
    # The what-if reads the row once, from the state it snapshotted, and the
    # store it then hands over is the snapshot's: the reader is the one
    # source of truth for the row, in the fetch and in the overlay alike.
    monkeypatch.setattr(orchestrator, "CANDIDATES_PER_POSITION", 1)
    cfg = config(state_dir=tmp_path / "state")
    store = Store(tmp_path / "aigaffer.db")
    store.save_executed(make_executed(transfers_in=[18], transfers_out=[8]))

    inputs = fetch_inputs(
        cfg, make_client(pipeline_routes()), store=store, executed_for=lambda gw: None
    )
    week = prepare_week(cfg, make_client(pipeline_routes()), store, executed_for=lambda gw: None)

    assert 18 not in inputs.histories, "Sarr is outside the cut and nobody recorded him"
    assert week.executed is None and 18 not in week.inputs.histories


def test_minute_overrides_move_the_week_and_never_the_calendar(monkeypatch, tmp_path):
    # The gaffer's minutes are for the coming gameweek (build_projections'
    # docstring); the calendar prices weeks months out on the model's own.
    # A what-if replays his minutes from the latest verdict, so they must
    # reach the week's projections and stop there.
    seen: list[tuple[dict | None, int | None]] = []
    real = orchestrator.build_projections

    def spy(inputs, cfg, minute_overrides=None, *, horizon=None, **kwargs):
        seen.append((minute_overrides, horizon))
        return real(inputs, cfg, minute_overrides, horizon=horizon, **kwargs)

    monkeypatch.setattr(orchestrator, "build_projections", spy)

    week = prepare_week(
        config(state_dir=tmp_path / "state", chips=True),
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        minute_overrides={FERRER: 0.0},
    )

    week_call, *calendar_calls = seen
    assert week_call == ({FERRER: 0.0}, None)
    assert calendar_calls and all(not overrides for overrides, _ in calendar_calls)
    assert all(horizon is not None for _, horizon in calendar_calls), "the calendar's own horizon"
    assert week.xmins[FERRER] == 0.0
    assert week.projections[FERRER].per_gw[2] == pytest.approx(0.0, abs=1e-9)
    assert week.calendar is not None


def test_every_run_opens_with_the_shared_preamble(monkeypatch, tmp_path):
    # The full report and the reminder both take their week from
    # prepare_week, so a what-if built on it can never drift from either.
    opened: list[str] = []
    real = orchestrator.prepare_week

    def spy(*args, **kwargs):
        opened.append("week")
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator, "prepare_week", spy)
    cfg = config(state_dir=tmp_path / "state")
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(cfg, make_client(pipeline_routes()), store, "scout", send=False)
    run_pipeline(cfg, make_client(pipeline_routes()), store, "reminder", send=False)

    assert opened == ["week", "week"]


# --- the lineup the reminders carry ---------------------------------------------

# A hand-built record over ids 1..21. squad_before is 1..15; the verdict sells
# 7 and buys 18, so the squad after is {1..15} - {7} | {18}. The bench of four
# [2, 12, 6, 14] comes off that fifteen, leaving the eleven
# {1, 3, 4, 5, 8, 9, 10, 11, 13, 15, 18}, sorted by id.
PLAN_RECORD = {
    "squad_before": list(range(1, 16)),
    "transfers_out": [7],
    "transfers_in": [18],
    "captain": 8,
    "vice": 13,
    "bench": [2, 12, 6, 14],
    "freehit_squad": None,
    "freehit_xi": None,
}


def test_the_planned_lineup_is_rebuilt_from_the_deadline_record():
    lineup = _recorded_lineup(PLAN_RECORD, PLAYERS)

    assert lineup.xi == [1, 3, 4, 5, 8, 9, 10, 11, 13, 15, 18]
    assert lineup.bench == [2, 12, 6, 14]
    assert (lineup.captain, lineup.vice) == (8, 13)


def test_a_free_hit_weeks_planned_lineup_is_the_temporary_team():
    # The free-hit fifteen is 1..14 plus 18 (21 bought for the week instead of
    # 15), its eleven is the record's freehit_xi in the record's order, and the
    # bench is the four of the fifteen outside it, from the record.
    xi = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 21]
    record = {
        **PLAN_RECORD,
        "freehit_squad": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 20, 21],
        "freehit_xi": xi,
        "bench": [20, 2, 12, 6],
    }

    lineup = _recorded_lineup(record, PLAYERS)

    assert lineup.xi == xi
    assert lineup.bench == [20, 2, 12, 6]


def test_a_legacy_record_has_no_rebuildable_lineup():
    legacy = {key: value for key, value in PLAN_RECORD.items() if key != "squad_before"}

    assert _recorded_lineup(legacy, PLAYERS) is None
    assert _recorded_lineup(None, PLAYERS) is None
    assert _recorded_lineup({**PLAN_RECORD, "squad_before": None}, PLAYERS) is None


def test_a_free_hit_record_with_no_free_hit_team_has_no_rebuildable_lineup():
    # The chip is free hit but the record names no temporary team: the standing
    # squad is not what he fields, so no block rather than a mislabelled one.
    record = {**PLAN_RECORD, "chip": "free_hit"}

    assert _recorded_lineup(record, PLAYERS) is None
    assert _recorded_lineup({**record, "freehit_xi": [], "freehit_squad": []}, PLAYERS) is None


def test_a_free_hit_lineup_is_built_from_the_records_bench_not_its_xi():
    # The record's freehit_xi (the MILP's) has 3 where the picker's bench
    # [20, 2, 12, 3] has put him, so it disagrees with the fifteen less the
    # bench by one player. The block must still be there, built from the
    # bench: 1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 21.
    squad = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 20, 21]
    record = {
        **PLAN_RECORD, "chip": "free_hit", "freehit_squad": squad,
        "freehit_xi": [1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 21],
        "bench": [20, 2, 12, 3],
    }

    lineup = _recorded_lineup(record, PLAYERS)

    assert lineup.xi == [1, 4, 5, 6, 8, 9, 10, 11, 13, 14, 21]
    assert lineup.bench == [20, 2, 12, 3]
    disagreeing = {**record, "freehit_xi": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 21]}
    assert _recorded_lineup(disagreeing, PLAYERS).xi == lineup.xi, "same eleven"


def test_a_bench_of_the_wrong_length_has_no_rebuildable_lineup():
    # Three on the bench leaves a twelve, which is no eleven to field.
    assert _recorded_lineup({**PLAN_RECORD, "bench": [2, 12, 6]}, PLAYERS) is None
    # A bench man outside the squad leaves the eleven the wrong size too.
    assert _recorded_lineup({**PLAN_RECORD, "bench": [2, 12, 6, 7]}, PLAYERS) is None


def test_a_planned_player_the_board_dropped_has_no_rebuildable_lineup():
    # An armband on someone outside the eleven is as inconsistent as a player
    # the board has never heard of: neither is a lineup to field.
    assert _recorded_lineup({**PLAN_RECORD, "captain": 99}, PLAYERS) is None
    assert _recorded_lineup({**PLAN_RECORD, "transfers_in": [99]}, PLAYERS) is None


def test_the_entered_lineup_is_the_squad_he_entered_minus_the_recorded_bench():
    row = make_executed(
        squad_after=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18],
        captain=8, vice=13,
    )

    lineup = _entered_lineup(row, {"bench": [2, 12, 6, 14]}, PLAYERS)

    assert lineup.xi == [1, 3, 4, 5, 8, 9, 10, 11, 13, 15, 18]
    assert lineup.bench == [2, 12, 6, 14]
    assert (lineup.captain, lineup.vice) == (8, 13)


def test_an_entered_free_hit_lineup_is_the_temporary_team():
    xi = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 21]
    squad = [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 20, 21]
    row = make_executed(
        chip="free_hit", freehit_squad=squad, freehit_xi=xi, captain=8, vice=13
    )

    lineup = _entered_lineup(row, {"bench": [20, 2, 12, 6]}, PLAYERS)

    assert lineup.xi == xi and lineup.bench == [20, 2, 12, 6]


def test_an_entered_free_hit_with_no_team_on_record_is_left_out():
    row = make_executed(
        chip="free_hit", freehit_squad=None, freehit_xi=None,
        squad_after=[1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 18],
        captain=8, vice=13,
    )

    assert _entered_lineup(row, {"bench": [2, 12, 6, 14]}, PLAYERS) is None


def test_an_entered_row_with_no_squad_is_left_out():
    row = make_executed(squad_after=None, captain=8, vice=13)

    assert _entered_lineup(row, {"bench": [2, 12, 6, 14]}, PLAYERS) is None


def test_an_entered_lineup_with_no_recorded_bench_is_left_out():
    assert _entered_lineup(make_executed(), None, PLAYERS) is None
    assert _entered_lineup(make_executed(), {"bench": None}, PLAYERS) is None


def test_a_calm_reminder_carries_the_deadline_verdicts_lineup(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")
    client = make_client(pipeline_routes())

    run_pipeline(cfg, client, store, "deadline", send=False)
    alert = run_pipeline(cfg, client, store, "reminder", send=False)

    planned = _recorded_lineup(store.decision(2, "deadline"), PLAYERS)
    assert render.phone_lineup(planned, PLAYERS) in alert
    assert alert.index("## Do this") < alert.index("Bench:") < alert.index(
        render.REMINDER_UNCHANGED
    )


def test_a_reminder_with_no_full_report_carries_the_fresh_lineup(tmp_path):
    cfg = config(state_dir=tmp_path / "state")
    store = Store(cfg.state_dir / "aigaffer.db")

    alert = run_pipeline(
        cfg, make_client(pipeline_routes()), store, "reminder", send=False
    )

    assert "Starting XI (" in alert and "Bench: 1 " in alert
    assert alert.index("Bench:") < alert.index(render.NO_FULL_REPORT)


def test_a_calm_reminder_after_a_free_hit_deadline_carries_the_free_hit_team(
    monkeypatch, tmp_path
):
    # A stubbed planned free hit (as the record test above builds it), the
    # deadline run, then a calm reminder. The block is headed as the free hit
    # team and its eleven is the temporary fifteen less the record's bench.
    cfg = config(state_dir=tmp_path / "state")
    client = make_client(pipeline_routes())
    inputs = fetch_inputs(cfg, client)
    _, projections = build_projections(inputs, config())
    standing = inputs.squad.player_ids
    positions = {pid: p.element_type for pid, p in inputs.players.items()}
    gw_xp = {pid: pr.per_gw.get(inputs.event.id, 0.0) for pid, pr in projections.items()}
    choice = Plan(
        squad=standing, xi=[], transfers_in=[], transfers_out=[], hits=0,
        xp_total=0.0, objective=0.0,
        path=PlannedPath(
            moves=[], objective=0.0, weekly_xp={}, week1_chip="free_hit",
            week1_freehit_squad=FH_SQUAD, week1_freehit_xi=FH_XI,
        ),
    )
    solved = SolveResult(
        plans=[choice], choice=choice,
        lineup=pick_lineup(standing, positions, gw_xp),
        chips=NO_CHIPS, draft_mode=False,
    )
    monkeypatch.setattr(orchestrator, "solve", lambda *a, **k: solved)
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(cfg, client, store, "deadline", send=False)
    alert = run_pipeline(cfg, client, store, "reminder", send=False)
    record = store.decision(2, "deadline")

    eleven = sorted(set(FH_SQUAD) - set(record["bench"]))
    assert len(eleven) == 11
    assert render.REMINDER_UNCHANGED in alert
    block = alert[alert.index("Starting XI \u2014 free hit team"):]
    shown = block[: block.index("Bench:")]
    for pid in eleven:
        assert PLAYERS[pid].web_name in shown
    for pid in record["bench"]:
        assert PLAYERS[pid].web_name not in shown
