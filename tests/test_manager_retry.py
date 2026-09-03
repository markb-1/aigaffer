"""Tests for the manager's second chance: a week he did not decide is withheld.

A gaffer who is configured and does not decide — a dead key, a rate limit,
an install that will not import, a chip he cannot play — used to send the
solver's report wearing his fallback label and mark the week done, which
left the backup scheduler nothing to redo and the owner one buried line to
notice. Now the run withholds the report while there is time for the next
tick to ask him again, tells the phone once why and what the solver would
do meanwhile, and only past the floor — the last hours of the window — lets
the solver's own report through and marks it done, so a week is never left
without one.

The pipeline is driven the way :mod:`tests.test_orchestrator` drives it — a
real client over the fake transport, the manager stubbed — with the clock
handed in, because the floor is a question about how long the window has
left and the fixture's deadline is a fixed date.
"""

import sys
from datetime import UTC, datetime, timedelta
from functools import partial

import pytest

from aigaffer import orchestrator
from aigaffer.config import MANAGER_RETRY_FLOOR_HOURS, Config
from aigaffer.orchestrator import run_pipeline
from aigaffer.store import Store
from tests.test_orchestrator import (
    GOOD_CHIP,
    TEAM_ID,
    TOKEN,
    PoisonedModule,
    decided,
    make_client,
    pipeline_routes,
    stub_gaffer,
    unavailable,
)

# The fixture's GW2 deadline; every clock below is read back from it.
DEADLINE = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)
# The floors as the design states them: the last three hours of each window.
DEADLINE_FLOOR = 3.0 + MANAGER_RETRY_FLOOR_HOURS
SCOUT_FLOOR = 24.0 + MANAGER_RETRY_FLOOR_HOURS

AUTH_FAILED = partial(unavailable, reason="AuthenticationError")
RATE_LIMITED = partial(unavailable, reason="RateLimitError")
FALLBACK_LINE = "The gaffer was unavailable (AuthenticationError); this is the solver's pick."


def hours_out(hours: float) -> datetime:
    return DEADLINE - timedelta(hours=hours)


def phone_cfg(tmp_path, key: str | None = "sk-test") -> Config:
    return Config(
        team_id=TEAM_ID,
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
        anthropic_api_key=key,
    )


class Phone:
    """Every message the run sent, in order."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, token: str, chat_id: str, text: str) -> None:
        self.messages.append(text)


@pytest.fixture
def phone(monkeypatch) -> Phone:
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    return phone


def tick(
    monkeypatch,
    tmp_path,
    decide,
    hours: float,
    mode: str = "deadline",
    store: Store | None = None,
    cfg: Config | None = None,
    **kwargs,
):
    """One tick of ``mode`` with ``hours`` left on the clock and the manager
    answering with ``decide``; the report and the store."""
    stub_gaffer(monkeypatch, decide)
    store = store or Store(tmp_path / "state" / "aigaffer.db")
    report = run_pipeline(
        cfg or phone_cfg(tmp_path),
        make_client(pipeline_routes()),
        store,
        mode,
        now=hours_out(hours),
        **kwargs,
    )
    return report, store


def readme_with_marker(tmp_path) -> str:
    text = f"# A README\n\nverdict {orchestrator.LATEST_MARKER}\n"
    (tmp_path / "README.md").write_text(text, encoding="utf-8")
    return text


def test_a_gaffer_who_fails_with_time_to_spare_has_the_report_withheld(
    monkeypatch, tmp_path, phone, capsys
):
    readme = readme_with_marker(tmp_path)

    report, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)

    assert store.has_run(2, "deadline") is False
    assert not (tmp_path / "state" / "reports").exists()
    assert not (tmp_path / "GW2.md").exists()
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == readme
    # The run still hands back the report it would have sent: a dry run
    # prints it, and a person at the keyboard can read what was withheld.
    assert FALLBACK_LINE in report
    assert orchestrator.WITHHELD in capsys.readouterr().out


def test_the_phone_is_told_why_and_what_the_solver_would_do(
    monkeypatch, tmp_path, phone
):
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)

    [alert] = phone.messages
    assert "AuthenticationError" in alert
    assert "## Do this" in alert
    assert "⏰ Make these by Fri 22 Aug 2025 17:30 UTC — GW2" in alert
    # It says the next tick will try again, and until when.
    assert "next tick" in alert
    assert f"Fri 22 Aug 2025 {17 - int(DEADLINE_FLOOR):02d}:30 UTC" in alert
    # It is an alert, not the digest: no view to quote, no report to point at.
    assert "## The Gaffer's view" not in alert
    assert "Full report: GW2.md" not in alert


def test_the_same_failure_is_reported_once(monkeypatch, tmp_path, phone):
    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=19, store=store)
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=18, store=store)

    assert len(phone.messages) == 1
    assert store.has_run(2, "deadline") is False


def test_a_new_reason_is_a_new_alert(monkeypatch, tmp_path, phone):
    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)
    tick(monkeypatch, tmp_path, RATE_LIMITED, hours=19, store=store)

    assert len(phone.messages) == 2
    assert "RateLimitError" in phone.messages[1]


def test_a_gaffer_who_recovers_reports_the_week_as_normal(
    monkeypatch, tmp_path, phone
):
    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)
    tick(monkeypatch, tmp_path, decided, hours=19, store=store)

    assert store.has_run(2, "deadline") is True
    assert store.decision(2, "deadline")["decision_source"] == "manager"
    assert (tmp_path / "GW2.md").exists()
    alert, digest = phone.messages
    assert "## The Gaffer's view" in digest
    assert "Full report: GW2.md" in digest


@pytest.mark.parametrize(
    "mode, hours",
    [
        ("deadline", DEADLINE_FLOOR),
        ("deadline", DEADLINE_FLOOR - 1),
        ("scout", SCOUT_FLOOR),
        ("scout", SCOUT_FLOOR - 2),
    ],
)
def test_at_the_floor_the_solvers_report_goes_out_and_counts(
    monkeypatch, tmp_path, phone, mode, hours
):
    # The window is closing and he has not answered: the week gets the
    # solver's report, labelled as today, and the store marks it done so the
    # reminder — and the promotion rule in ``__main__`` — read a week that ran.
    report, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=hours, mode=mode)

    assert store.has_run(2, mode) is True
    assert FALLBACK_LINE in report
    [digest] = phone.messages
    assert "## Do this" in digest
    assert "Full report: GW2.md" in digest


@pytest.mark.parametrize(
    "mode, hours", [("deadline", DEADLINE_FLOOR + 0.5), ("scout", SCOUT_FLOOR + 0.5)]
)
def test_just_above_the_floor_the_report_is_still_withheld(
    monkeypatch, tmp_path, phone, mode, hours
):
    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=hours, mode=mode)

    assert store.has_run(2, mode) is False
    [alert] = phone.messages
    assert "Full report: GW2.md" not in alert


def test_a_dry_run_with_a_failing_gaffer_leaves_nothing_behind(
    monkeypatch, tmp_path, phone
):
    report, store = tick(
        monkeypatch, tmp_path, AUTH_FAILED, hours=20, send=False, save=False
    )

    assert phone.messages == []
    assert store.withheld_before(2, "deadline", "AuthenticationError") is False
    assert store.last_runs() == []
    assert not (tmp_path / "GW2.md").exists()
    assert FALLBACK_LINE in report


def test_a_gaffer_who_never_loaded_is_withheld_too(monkeypatch, tmp_path, phone):
    # The other way he can fail to decide: his own module will not import,
    # so there is no decision to label — and no reason the week is done.
    def never(consult):
        raise AssertionError("the manager was asked with his own module broken")

    monkeypatch.setitem(sys.modules, "aigaffer.manager.agent", PoisonedModule())

    report, store = tick(monkeypatch, tmp_path, never, hours=20)

    assert store.has_run(2, "deadline") is False
    [alert] = phone.messages
    assert orchestrator.NOT_LOADED in alert
    assert "## Do this" in alert


def test_a_chip_he_cannot_play_is_not_a_decision_either(monkeypatch, tmp_path, phone):
    # Refused at the door is the solver's week under his name, and a week the
    # solver decided is exactly what the next tick exists to improve on.
    plays_spent_chip = partial(decided, chip="wildcard", justification=GOOD_CHIP)

    _, store = tick(monkeypatch, tmp_path, plays_spent_chip, hours=20)

    assert store.has_run(2, "deadline") is False
    [alert] = phone.messages
    assert "chip already played" in alert


def test_a_gaffer_nobody_asked_for_is_never_withheld(monkeypatch, tmp_path, phone):
    # No key means no manager was asked, and the solver's week is the week —
    # Phase 1's run, byte for byte, with nothing to wait for.
    _, store = tick(
        monkeypatch, tmp_path, AUTH_FAILED, hours=20, cfg=phone_cfg(tmp_path, key=None)
    )

    assert store.has_run(2, "deadline") is True
    [digest] = phone.messages
    assert "Full report: GW2.md" in digest
    assert "withheld" not in digest


def test_the_clock_defaults_to_now(monkeypatch, tmp_path, phone):
    # Without a clock handed in the run reads the wall clock — and the
    # fixture's deadline is long gone, which is past every floor: the report
    # goes out and counts, exactly as every other test in the suite expects.
    stub_gaffer(monkeypatch, AUTH_FAILED)
    store = Store(tmp_path / "state" / "aigaffer.db")

    run_pipeline(phone_cfg(tmp_path), make_client(pipeline_routes()), store, "deadline")

    assert store.has_run(2, "deadline") is True
