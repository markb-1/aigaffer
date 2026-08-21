"""Tests for the orchestrator: the clock that starts a run, and the run.

``run_pipeline`` is the only place the whole of Phase 1 is assembled, so the
tests here are integration tests: a real ``FplClient`` over a fake transport,
the real minutes model, the real solver and the real renderer, over the
``PIPELINE_*`` universe in :mod:`tests.fixtures`. What they assert is that the
pieces are wired to each other — the injured man we own is never fielded, the
shortlist has more than one plan on it, the chip panel is priced off a squad
that exists — rather than the numbers those pieces produce, which are pinned
by the unit tests of the modules that produce them.
"""

import copy
from datetime import UTC, datetime, timedelta
from typing import NamedTuple

import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.orchestrator import PipelineError, decide_mode, run_pipeline
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
        (5, "deadline"),
        (6, "deadline"),
        (6.5, None),
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


def make_client(routes: dict) -> FplClient:
    return FplClient(http=httpx.Client(transport=fake_fpl_transport(routes)))


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
    assert written.read_text() == scout_run.report


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
    assert signings[0] == "- Out: "  # nobody to sell
    assert signings[1].count("£") == 15  # a whole squad bought
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


def test_nothing_is_sent_without_somewhere_to_send_it(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(orchestrator, "send_report", lambda *args: sent.append(args))

    run_pipeline(
        Config(team_id=TEAM_ID, telegram_token=TOKEN, state_dir=tmp_path),
        make_client(pipeline_routes()),
        Store(tmp_path / "aigaffer.db"),
        "deadline",
    )

    assert sent == []


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


def serve(monkeypatch, routes: dict) -> None:
    monkeypatch.setattr(cli, "FplClient", lambda: make_client(routes))


def test_auto_runs_the_scout_two_days_out(monkeypatch, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(48)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "scout") is True


def test_auto_runs_the_deadline_check_on_the_day(monkeypatch, store):
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(5)))

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


def test_an_unknown_command_is_refused():
    with pytest.raises(SystemExit) as refusal:
        cli.main(["wildcard"])

    assert refusal.value.code == 2
