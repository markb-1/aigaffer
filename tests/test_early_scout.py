"""Tests for the early scout: the scout that runs the evening after a gameweek.

The three reports hang off the next deadline. This one hangs off the last
kickoff of the gameweek just played: the evening of the day after it, when
the results are in and the week ahead is at its emptiest, a first look at
the transfer plans. It is the same report as the Thursday scout under its
own name, sent once, and superseded rather than made up if it never lands —
the Thursday scout is the one whose window closes on a deadline.
"""

import argparse
import copy
from datetime import UTC, datetime, timedelta

import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.config import EARLY_SCOUT_HOUR_UTC, MANAGER_RETRY_LIMIT, SCOUT_HORIZON_HOURS
from aigaffer.data.models import Fixture
from aigaffer.orchestrator import decide_mode, last_kickoff, run_pipeline
from aigaffer.store import Store
from tests.fixtures import PIPELINE_FIXTURES_JSON
from tests.test_manager_retry import AUTH_FAILED, Phone, hours_out, phone_cfg
from tests.test_orchestrator import (
    TEAM_ID,
    bootstrap_due_in,
    decided,
    gaffer_cfg,
    make_client,
    pipeline_routes,
    serve,
    store,  # noqa: F401 — the CLI's configured environment, a fixture
    stub_gaffer,
)

# The fixture's GW2 deadline, and a GW1 that ended on the Sunday before it.
DEADLINE = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)
LAST_KICKOFF = datetime(2025, 8, 17, 15, 30, tzinfo=UTC)
ANCHOR = datetime(2025, 8, 18, EARLY_SCOUT_HOUR_UTC, 0, tzinfo=UTC)
SCOUT_OPENS = DEADLINE - timedelta(hours=SCOUT_HORIZON_HOURS)


# --- the clock ---------------------------------------------------------------


@pytest.mark.parametrize(
    "now, mode",
    [
        (ANCHOR - timedelta(minutes=1), None),
        (ANCHOR, "early"),
        (ANCHOR + timedelta(hours=18), "early"),
        (SCOUT_OPENS - timedelta(minutes=1), "early"),
        (SCOUT_OPENS, "scout"),
    ],
)
def test_the_early_scout_opens_the_evening_after_and_closes_on_the_scout(now, mode):
    assert decide_mode(now, DEADLINE, last_kickoff=LAST_KICKOFF) == mode


def test_without_a_last_kickoff_there_is_no_early_scout():
    assert decide_mode(ANCHOR, DEADLINE) is None
    assert decide_mode(ANCHOR, DEADLINE, last_kickoff=None) is None


def test_a_midweek_round_leaves_the_thursday_scout_to_cover_it():
    # The round ended Wednesday night; the evening after is Thursday, which
    # is already inside the scout window. One scout, not two.
    midweek = datetime(2025, 8, 19, 19, 0, tzinfo=UTC)
    evening_after = datetime(2025, 8, 20, EARLY_SCOUT_HOUR_UTC, 0, tzinfo=UTC)
    assert evening_after > SCOUT_OPENS
    assert decide_mode(evening_after, DEADLINE, last_kickoff=midweek) == "scout"


def test_the_reports_nearest_the_deadline_still_win():
    assert decide_mode(DEADLINE - timedelta(hours=5), DEADLINE, last_kickoff=LAST_KICKOFF) == "deadline"
    assert decide_mode(DEADLINE - timedelta(hours=1), DEADLINE, last_kickoff=LAST_KICKOFF) == "reminder"


def test_a_naive_last_kickoff_is_read_as_utc():
    assert decide_mode(ANCHOR, DEADLINE, last_kickoff=LAST_KICKOFF.replace(tzinfo=None)) == "early"


# --- the last kickoff --------------------------------------------------------


_ids = iter(range(1, 10_000))


def _fixture(event, kickoff):
    """A fixture as the API serves it: the kickoff an ISO string, or absent."""
    return Fixture(
        id=next(_ids),
        event=event,
        team_h=1,
        team_a=2,
        kickoff_time=None if kickoff is None else kickoff.isoformat().replace("+00:00", "Z"),
    )


def test_the_last_kickoff_is_the_current_rounds_latest_before_the_deadline():
    fixtures = [
        _fixture(1, datetime(2025, 8, 15, 19, 0, tzinfo=UTC)),
        _fixture(1, LAST_KICKOFF),
        _fixture(1, datetime(2025, 8, 16, 14, 0, tzinfo=UTC)),
        _fixture(2, datetime(2025, 8, 23, 14, 0, tzinfo=UTC)),
    ]
    assert last_kickoff(fixtures, current=1, before=DEADLINE) == LAST_KICKOFF


def test_a_fixture_rescheduled_past_the_next_deadline_is_not_the_rounds_end():
    # A postponed match keeps its round number and moves months away; the
    # round it belonged to still ended on the Sunday.
    fixtures = [
        _fixture(1, LAST_KICKOFF),
        _fixture(1, datetime(2025, 11, 5, 20, 0, tzinfo=UTC)),
    ]
    assert last_kickoff(fixtures, current=1, before=DEADLINE) == LAST_KICKOFF


def test_a_round_with_no_kickoffs_has_no_end():
    assert last_kickoff([], current=1, before=DEADLINE) is None
    assert last_kickoff([_fixture(1, None)], current=1, before=DEADLINE) is None
    assert last_kickoff([_fixture(2, LAST_KICKOFF)], current=1, before=DEADLINE) is None


# --- the command line --------------------------------------------------------


def fixtures_ended(hours_ago: float) -> list[dict]:
    """GW1's fixtures with kickoffs that many hours ago, GW2's a week on."""
    payload = copy.deepcopy(PIPELINE_FIXTURES_JSON)
    ended = datetime.now(UTC) - timedelta(hours=hours_ago)
    for fixture in payload:
        if fixture["event"] == 1:
            fixture["kickoff_time"] = (ended - timedelta(hours=2)).isoformat()
        elif fixture["event"] == 2:
            fixture["kickoff_time"] = (ended + timedelta(days=7)).isoformat()
    payload[1]["kickoff_time"] = ended.isoformat()
    return payload


def _auto():
    return argparse.Namespace(command="auto", force=False)


def evening_after(now: datetime) -> datetime:
    """``now`` plus one day, at the early scout's hour."""
    return (now + timedelta(days=1)).replace(
        hour=EARLY_SCOUT_HOUR_UTC, minute=0, second=0, microsecond=0
    )


def hours_since_a_round_that_ended_before_yesterdays_anchor() -> float:
    """A last kickoff far enough back that its evening-after has passed."""
    now = datetime.now(UTC)
    # Two days ago at the anchor hour: the day after it is yesterday's evening.
    ended = (now - timedelta(days=2)).replace(hour=EARLY_SCOUT_HOUR_UTC, minute=0)
    return (now - ended).total_seconds() / 3600


def test_auto_runs_the_early_scout_the_evening_after_the_round(tmp_path):
    ended = hours_since_a_round_that_ended_before_yesterdays_anchor()
    client = make_client(
        pipeline_routes(bootstrap=bootstrap_due_in(100), fixtures=fixtures_ended(ended))
    )
    store = Store(tmp_path / "state.db")

    assert cli._mode(_auto(), client, store) == ("early", 2)

    store.save_run(2, "early", "md", {})
    assert cli._mode(_auto(), client, store) == (None, 2)


def test_auto_waits_for_the_evening_after(tmp_path):
    # The round ended an hour ago: the evening after is tomorrow's.
    client = make_client(
        pipeline_routes(bootstrap=bootstrap_due_in(100), fixtures=fixtures_ended(1))
    )
    assert cli._mode(_auto(), client, Store(tmp_path / "state.db")) == (None, 2)


def test_the_fixtures_are_only_fetched_when_the_answer_could_be_early(tmp_path):
    # Inside sixty hours the clock already has its answer; a tick there must
    # not spend a request on the fixtures. A client with no fixtures at all
    # proves it by not blowing up.
    from types import SimpleNamespace

    event = SimpleNamespace(id=2, deadline_time=datetime.now(UTC) + timedelta(hours=30))
    bootstrap = SimpleNamespace(next_event=lambda: event, current_event=lambda: None)
    client = SimpleNamespace(bootstrap=lambda: bootstrap)

    assert cli._mode(_auto(), client, Store(tmp_path / "state.db")) == ("scout", 2)


def test_the_early_scout_can_be_asked_for_by_name(tmp_path):
    args = argparse.Namespace(command="early", force=False)
    client = make_client(pipeline_routes(bootstrap=bootstrap_due_in(100)))

    assert cli._mode(args, client, Store(tmp_path / "state.db")) == ("early", 2)
    assert cli._parse_args(["early"]).command == "early"


def test_auto_end_to_end_sends_the_early_scout_once(monkeypatch, tmp_path, store):
    ended = hours_since_a_round_that_ended_before_yesterdays_anchor()
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(100), fixtures=fixtures_ended(ended)))

    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "early") is True
    assert cli.main(["auto"]) == 0
    assert len(store.last_runs()) == 1


# --- the report --------------------------------------------------------------


def test_the_early_scout_is_the_scout_under_its_own_name(monkeypatch, tmp_path):
    stub_gaffer(monkeypatch, decided)
    store = Store(tmp_path / "aigaffer.db")

    report = run_pipeline(
        gaffer_cfg(tmp_path), make_client(pipeline_routes()), store, "early", send=False
    )

    assert report.startswith("# AI Gaffer — GW2 early scout\n")
    assert "## The Gaffer's view" in report
    assert (tmp_path / "state" / "reports" / "gw2-early.md").read_text(encoding="utf-8") == report
    assert (tmp_path / "GW2.md").read_text(encoding="utf-8") == report
    assert store.has_run(2, "early") is True
    assert store.has_run(2, "scout") is False


def test_the_early_scouts_digest_carries_its_name(monkeypatch, tmp_path):
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    stub_gaffer(monkeypatch, decided)

    run_pipeline(phone_cfg(tmp_path), make_client(pipeline_routes()), Store(tmp_path / "aigaffer.db"), "early")

    [digest] = phone.messages
    assert digest.startswith("# AI Gaffer — GW2 early scout\n")


@pytest.mark.parametrize("hours", [100.0, SCOUT_HORIZON_HOURS + 0.5])
def test_an_early_scout_the_gaffer_cannot_decide_is_never_sent_labelled(
    monkeypatch, tmp_path, hours
):
    # Withheld all the way to the scout window, then superseded: the
    # Thursday scout is hours away, and a labelled early scout would be a
    # second text saying less.
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    stub_gaffer(monkeypatch, AUTH_FAILED)
    store = Store(tmp_path / "state" / "aigaffer.db")

    run_pipeline(
        phone_cfg(tmp_path), make_client(pipeline_routes()), store, "early", now=hours_out(hours)
    )

    assert store.has_run(2, "early") is False
    [alert] = phone.messages
    assert "Full report: GW2.md" not in alert


def test_an_early_scout_past_its_retry_budget_is_shelved_quietly(monkeypatch, tmp_path):
    # The retry limit bounds the bill for every report; for this one the
    # labelled solver report it would otherwise release is a second text
    # saying less than Thursday's will. So the week is written down — the
    # diary, the store, so the ticks stand down — and the phone hears nothing
    # more than the alert it already had.
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    stub_gaffer(monkeypatch, AUTH_FAILED)
    store = Store(tmp_path / "state" / "aigaffer.db")
    for _ in range(MANAGER_RETRY_LIMIT):
        store.record_withheld(2, "early", "AuthenticationError")

    report = run_pipeline(
        phone_cfg(tmp_path), make_client(pipeline_routes()), store, "early", now=hours_out(100)
    )

    assert store.has_run(2, "early") is True
    assert phone.messages == []
    assert (tmp_path / "state" / "reports" / "gw2-early.md").read_text(encoding="utf-8") == report


def test_a_forced_early_scout_is_sent_whatever_the_gaffer_did(monkeypatch, tmp_path):
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    stub_gaffer(monkeypatch, AUTH_FAILED)
    store = Store(tmp_path / "state" / "aigaffer.db")

    run_pipeline(
        phone_cfg(tmp_path), make_client(pipeline_routes()), store, "early",
        now=hours_out(100), force=True,
    )

    assert store.has_run(2, "early") is True
    [digest] = phone.messages
    assert digest.startswith("# AI Gaffer — GW2 early scout\n")
