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

import copy
import json
import sys
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import NamedTuple

import anthropic
import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Player
from aigaffer.manager import agent
from aigaffer.manager.agent import ManagerDecision
from aigaffer.model.xp import PlayerProjection
from aigaffer.orchestrator import (
    NO_CHIPS,
    PipelineError,
    PipelineInputs,
    SolveResult,
    build_projections,
    decide_mode,
    fetch_inputs,
    history_pool,
    run_pipeline,
    solve,
)
from aigaffer.report.render import render_report
from aigaffer.report.telegram import send_report
from aigaffer.solver.lineup import Lineup, pick_lineup
from aigaffer.store import Store
from tests.fixtures import (
    ELEMENT_SUMMARY_JSON,
    HISTORY_JSON,
    PICKS_15_JSON,
    PIPELINE_BOOTSTRAP_JSON,
    PIPELINE_ELEMENTS_JSON,
    PIPELINE_FIXTURES_JSON,
    TRANSFERS_JSON,
    fake_fpl_transport,
)

TEAM_ID = 99
TOKEN = "1234:super-secret-bot-token"

PICKS_PATH = f"/api/entry/{TEAM_ID}/event/1/picks/"
TRANSFERS_PATH = f"/api/entry/{TEAM_ID}/transfers/"
HISTORY_PATH = f"/api/entry/{TEAM_ID}/history/"

DEADLINE = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)


# --- the mode clock --------------------------------------------------------


@pytest.mark.parametrize(
    ("hours_to_go", "mode"),
    [
        (0.5, "deadline"),
        (2, "deadline"),
        (3, "deadline"),
        (3.5, None),
        (6, None),
        (20, None),
        (36, None),
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


# --- the pipeline ----------------------------------------------------------


def pipeline_routes(bootstrap: dict = PIPELINE_BOOTSTRAP_JSON) -> dict:
    """Every endpoint a run touches, for the fifteen-man universe."""
    return {
        "/api/bootstrap-static/": bootstrap,
        "/api/fixtures/": PIPELINE_FIXTURES_JSON,
        PICKS_PATH: PICKS_15_JSON,
        TRANSFERS_PATH: TRANSFERS_JSON,
        HISTORY_PATH: HISTORY_JSON,
        **{
            f"/api/element-summary/{element['id']}/": ELEMENT_SUMMARY_JSON
            for element in PIPELINE_ELEMENTS_JSON
        },
    }


def make_client(routes: dict, statuses: dict[str, int] | None = None) -> FplClient:
    """A client over the fake transport that never waits between retries."""
    return FplClient(
        http=httpx.Client(transport=fake_fpl_transport(routes, statuses)),
        sleep=lambda _: None,
    )


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
    state = tmp_path_factory.mktemp("state")
    cfg = Config(team_id=TEAM_ID, state_dir=state)
    store = Store(state / "aigaffer.db")
    report = run_pipeline(cfg, make_client(pipeline_routes()), store, "scout")
    return Run(report=report, cfg=cfg, store=store)


def test_the_report_is_the_whole_report(scout_run):
    headings = [line for line in scout_run.report.splitlines() if line.startswith("#")]

    assert headings[0] == "# AI Gaffer — GW2 scout"
    assert headings[2].startswith("## Starting XI (")
    assert [headings[1], *headings[3:]] == [
        "## Recommendation",
        "## Candidate plans",
        "## Chip EV",
        "## Watchlist",
    ]
    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in scout_run.report


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
        Config(team_id=TEAM_ID, state_dir=tmp_path), make_client(routes), store, "scout"
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
        Config(team_id=TEAM_ID, state_dir=tmp_path), make_client(routes), store, "scout"
    )

    assert report.startswith("# AI Gaffer — GW1 scout — initial squad draft")
    assert len(bullets(report, "Starting XI")) == 5
    assert bullets(report, "Chip EV") == [
        "- Bench boost: +0.0",
        "- Triple captain: +0.0",
        "- Free hit: +0.0",
        "- Wildcard: +0.0",
    ]
    assert store.has_run(1, "scout") is True


def test_one_history_the_api_will_not_serve_does_not_lose_the_report(tmp_path):
    # Ferrer's element-summary rate-limits every try. Two hundred requests a
    # run and one of them failing is a Saturday, not an outage: he falls back
    # on the starts-based minutes guess and the week's report still comes out.
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        Config(team_id=TEAM_ID, state_dir=tmp_path),
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
            Config(team_id=TEAM_ID, state_dir=tmp_path),
            make_client(pipeline_routes(bootstrap=over)),
            Store(tmp_path / "aigaffer.db"),
            "scout",
        )


def test_a_dry_run_leaves_nothing_behind(tmp_path):
    store = Store(tmp_path / "aigaffer.db")

    run_pipeline(
        Config(team_id=TEAM_ID, state_dir=tmp_path),
        make_client(pipeline_routes()),
        store,
        "deadline",
        send=False,
        save=False,
    )

    assert store.last_runs() == []
    assert not (tmp_path / "reports").exists()


# --- the seam: fetch, project, solve ---------------------------------------
#
# The three stages ``run_pipeline`` is made of, called on their own. What the
# manager agent will do with them is re-project on minutes it has read the
# team news for and solve again, so the tests here are about the two things
# that makes possible: the fetch is a value the later stages can be re-run
# over without touching the API, and an override reaches the projection.

FERRER = 5  # the premium midfielder in the pipeline universe, and its captain


class Seam(NamedTuple):
    cfg: Config
    inputs: PipelineInputs


@pytest.fixture(scope="module")
def seam(tmp_path_factory) -> Seam:
    """One fetch, shared: every stage test below re-runs from these inputs,
    which is the point of them being a value."""
    cfg = Config(team_id=TEAM_ID, state_dir=tmp_path_factory.mktemp("seam"))
    return Seam(cfg=cfg, inputs=fetch_inputs(cfg, make_client(pipeline_routes())))


class CountingTransport(httpx.BaseTransport):
    """The fake transport, with a tally of what was asked of it."""

    def __init__(self, routes: dict) -> None:
        self.inner = fake_fpl_transport(routes)
        self.counts: Counter = Counter()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.counts[request.url.path] += 1
        return self.inner.handle_request(request)


def test_the_chips_already_played_are_fetched_once_and_kept(tmp_path):
    # The free-transfer sum reads them and so does the manager's chip panel;
    # a season's chip history fetched twice is one request nobody needed.
    transport = CountingTransport(pipeline_routes())
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)

    inputs = fetch_inputs(Config(team_id=TEAM_ID, state_dir=tmp_path), client)

    assert inputs.chips_used == HISTORY_JSON["chips"]
    assert transport.counts[HISTORY_PATH] == 1


def test_a_manager_with_no_squad_has_played_no_chips(tmp_path):
    # Nothing has been played by somebody who has not played, and the entry
    # endpoints do not answer for him either.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    cfg = Config(team_id=TEAM_ID, state_dir=tmp_path)

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


def test_an_override_of_no_minutes_writes_a_player_out(seam):
    _, before = build_projections(seam.inputs, seam.cfg)

    xmins, after = build_projections(seam.inputs, seam.cfg, {FERRER: 0.0})

    assert before[FERRER].total > 0  # there was something to take away
    assert xmins[FERRER] == 0.0
    assert after[FERRER].total == 0.0
    # Nobody else moved: an override is about one player's minutes.
    assert {pid: p.total for pid, p in after.items() if pid != FERRER} == {
        pid: p.total for pid, p in before.items() if pid != FERRER
    }


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


def test_a_solve_with_no_squad_drafts_a_fifteen(tmp_path):
    # The draft path lives inside solve now, and says so.
    routes = pipeline_routes()
    del routes[PICKS_PATH]
    cfg = Config(team_id=TEAM_ID, state_dir=tmp_path)
    inputs = fetch_inputs(cfg, make_client(routes))

    _, projections = build_projections(inputs, cfg)
    solved = solve(inputs, projections, cfg)

    assert inputs.squad is None
    assert inputs.free_transfers is None
    assert solved.draft_mode is True
    assert solved.plans == [solved.choice]
    assert len(solved.choice.transfers_in) == 15
    assert solved.chips == NO_CHIPS  # no squad to play a chip against


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

    def __call__(self, client, cfg, inputs, solve0, projections, briefing, resolver):
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


def gaffer_cfg(tmp_path) -> Config:
    return Config(team_id=TEAM_ID, state_dir=tmp_path, anthropic_api_key="sk-test")


def gaffer_run(monkeypatch, tmp_path, decide=decided, mode="scout", **kwargs):
    """One run with a manager in it; the report, the store and the manager."""
    gaffer = stub_gaffer(monkeypatch, decide)
    store = Store(tmp_path / "aigaffer.db")
    report = run_pipeline(
        gaffer_cfg(tmp_path), make_client(pipeline_routes()), store, mode, **kwargs
    )
    return report, store, gaffer


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
    cfg = Config(
        team_id=TEAM_ID,
        state_dir=tmp_path,
        anthropic_api_key="sk-test",
        manager_enabled=False,
    )

    report = run_pipeline(
        cfg,
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        "scout",
        send=False,
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
    built: list[dict] = []

    class Recorder:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    _, _, gaffer = gaffer_run(monkeypatch, tmp_path, send=False)

    assert built == [{"api_key": "sk-test", "timeout": 120.0, "max_retries": 1}]
    assert isinstance(gaffer.consults[0].client, Recorder), "and it is what he is given"


def test_the_resolver_reprojects_and_resolves_on_his_minutes(monkeypatch, tmp_path):
    seen = {}

    def re_solve(consult: Consult) -> ManagerDecision:
        solved, projections = consult.resolver({FERRER: 0.0})
        seen["xi"] = solved.lineup.xi
        seen["ferrer"] = projections[FERRER].total
        seen["before"] = consult.projections[FERRER].total
        return decided(consult, plan=solved.choice)

    report, store, _ = gaffer_run(monkeypatch, tmp_path, decide=re_solve, send=False)

    assert seen["before"] > 0 and seen["ferrer"] == 0.0
    assert FERRER not in seen["xi"], "told he is not playing, the solver drops him"
    assert store.has_run(2, "scout") is True
    assert "## The Gaffer's view" in report


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
    )

    assert "The gaffer was unavailable (unexpected RuntimeError)" in report
    assert "not coming out of here" not in report
    assert store.has_run(2, "scout") is True


def test_a_chip_he_still_holds_is_played(monkeypatch, tmp_path):
    report, store, _ = gaffer_run(
        monkeypatch,
        tmp_path,
        decide=partial(decided, chip="bench_boost", justification=GOOD_CHIP),
        send=False,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert "Playing the bench boost." in report
    assert decision["chip"] == "bench_boost"
    assert decision["chip_justification"] == GOOD_CHIP


def test_a_chip_he_has_already_played_is_refused_at_the_door(monkeypatch, tmp_path):
    # The season's history says the wildcard went in GW1. The briefing tells
    # him so; this is the belt under that brace, and it costs him the decision
    # rather than the chip, because a week built on a chip we cannot play is
    # not a week anybody can enter.
    report, store, gaffer = gaffer_run(
        monkeypatch,
        tmp_path,
        decide=partial(decided, chip="wildcard", justification=GOOD_CHIP),
        send=False,
    )
    decision = store.last_runs(1)[0]["decision"]

    assert "The gaffer was unavailable (chip already played)" in report
    assert decision["decision_source"] == "solver-fallback: chip already played"
    assert decision["chip"] == "none"
    assert decision["transfers_in"] == gaffer.consults[0].solve0.choice.transfers_in
    assert decision["searches"] == 2, "the searches were still paid for"
    assert GAFFER_RATIONALE not in report


def test_without_a_key_there_is_no_gaffer_and_no_difference(
    monkeypatch, tmp_path, scout_run
):
    # The Phase 1 report, byte for byte, and a manager who was never asked.
    def never(consult: Consult) -> ManagerDecision:
        raise AssertionError("the manager was asked without a key")

    stub_gaffer(monkeypatch, never)
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        Config(team_id=TEAM_ID, state_dir=tmp_path),
        make_client(pipeline_routes()),
        store,
        "scout",
        send=False,
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
    monkeypatch.setattr(orchestrator, "send_report", lambda *args: sent.append(args))
    gaffer = stub_gaffer(monkeypatch)
    cfg = Config(
        team_id=TEAM_ID,
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path,
        anthropic_api_key="sk-test",
    )

    report = run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    assert len(gaffer.consults) == 1
    assert sent == [(TOKEN, "42", report)]
    assert "## The Gaffer's view" in report


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
    monkeypatch.setattr(orchestrator, "send_report", lambda *args: sent.append(args))
    cfg = Config(
        team_id=TEAM_ID, telegram_token=TOKEN, telegram_chat_id="42", state_dir=tmp_path
    )

    report = run_pipeline(
        cfg, make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "deadline"
    )

    assert sent == [(TOKEN, "42", report)]


def test_nothing_is_sent_without_somewhere_to_send_it(monkeypatch, capsys, tmp_path):
    # Half-configured is the easy mistake — a token in the secrets and no
    # chat id — and it looks exactly like a working bot until the phone stays
    # quiet, so the run that could not deliver says so.
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args: sent.append(args))

    run_pipeline(
        Config(team_id=TEAM_ID, telegram_token=TOKEN, state_dir=tmp_path),
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
    def explode(*args):
        # httpx puts the request URL — and so the token — in its messages.
        raise httpx.ConnectError(f"connecting to /bot{TOKEN}/sendMessage failed")

    monkeypatch.setattr(orchestrator, "send_report", explode)
    store = Store(tmp_path / "aigaffer.db")
    cfg = Config(
        team_id=TEAM_ID, telegram_token=TOKEN, telegram_chat_id="42", state_dir=tmp_path
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
    monkeypatch.setenv("AIGAFFER_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    return Store(tmp_path / "aigaffer.db")


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


def test_auto_runs_the_deadline_check_on_the_day(monkeypatch, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(2)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "deadline") is True


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
        pipeline_routes(bootstrap=bootstrap_due_in(2)),
        statuses={"/api/fixtures/": 500},
    )

    assert cli.main(["auto"]) == 1

    assert posted == [
        {"chat_id": "42", "text": "aigaffer run failed: HTTPStatusError (gw 2)"}
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
    assert posted == [{"chat_id": "42", "text": "aigaffer run failed: PipelineError"}]


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
