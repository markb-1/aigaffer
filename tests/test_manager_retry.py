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

import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import orchestrator
from aigaffer.config import DEADLINE_ANCHOR_HOURS, MANAGER_RETRY_FLOOR_HOURS, MANAGER_RETRY_LIMIT, Config
from aigaffer.orchestrator import run_pipeline
from aigaffer.store import Store
from tests.fixtures import PICKS_15_JSON
from tests.test_orchestrator import (
    GOOD_CHIP,
    TOKEN,
    PoisonedModule,
    bootstrap_due_in,
    chips_off_from_the_environment,  # noqa: F401 — autouse: chips off, as there
    config,
    decided,
    halves_routes,
    make_client,
    midweek_bootstrap,
    pipeline_routes,
    serve,
    store,  # noqa: F401 — the CLI's configured environment, a fixture
    stub_gaffer,
    unavailable,
    unplayed_routes,
)

# The fixture's GW2 deadline; every clock below is read back from it.
DEADLINE = datetime(2025, 8, 22, 17, 30, tzinfo=UTC)
# The floors as the design states them: the last three hours of each window.
DEADLINE_FLOOR = 3.0 + MANAGER_RETRY_FLOOR_HOURS
SCOUT_FLOOR = DEADLINE_ANCHOR_HOURS + MANAGER_RETRY_FLOOR_HOURS

AUTH_FAILED = partial(unavailable, reason="AuthenticationError")
RATE_LIMITED = partial(unavailable, reason="RateLimitError")
FALLBACK_LINE = "The gaffer was unavailable (AuthenticationError); this is the solver's pick."


def hours_out(hours: float) -> datetime:
    return DEADLINE - timedelta(hours=hours)


def phone_cfg(tmp_path, key: str | None = "sk-test") -> Config:
    return config(
        telegram_token=TOKEN,
        telegram_chat_id="42",
        state_dir=tmp_path / "state",
        anthropic_api_key=key,
    )


class Phone:
    """Every message the run sent, in order."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def __call__(self, token: str, chat_id: str, text: str, **kwargs) -> None:
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
    routes: dict | None = None,
    **kwargs,
):
    """One tick of ``mode`` with ``hours`` left on the clock and the manager
    answering with ``decide``; the report and the store."""
    stub_gaffer(monkeypatch, decide)
    store = store or Store(tmp_path / "state" / "aigaffer.db")
    report = run_pipeline(
        cfg or phone_cfg(tmp_path),
        make_client(routes or pipeline_routes()),
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


def test_the_withheld_alert_opens_with_the_standing(monkeypatch, tmp_path, phone):
    # A withheld week is still a text, and it opens the way the digest does.
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)

    [alert] = phone.messages
    lines = alert.splitlines()
    [standing] = [
        line for line in lines
        if line.startswith("61 pts · rank 2,345,678 · value £100.0m · bank £2.8m · ")
    ]
    assert lines[lines.index(standing) - 2].startswith("Deadline: ")


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
    # Both sets on the board and the first wildcard spent in GW1.
    plays_spent_chip = partial(decided, chip="wildcard", justification=GOOD_CHIP)

    _, store = tick(
        monkeypatch, tmp_path, plays_spent_chip, hours=20, routes=halves_routes()
    )

    assert store.has_run(2, "deadline") is False
    [alert] = phone.messages
    assert orchestrator.CHIP_SPENT in alert


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


# --- what the review asked for ---------------------------------------------


def test_the_ledgers_note_survives_a_withheld_tick(monkeypatch, tmp_path, phone):
    # The ledger reconciles the bank once per gameweek, on the first run that
    # sees it; a withheld tick that wrote the snapshot would spend that one
    # line on a report nobody saved or sent. Seeded as the ledger's own test
    # seeds it: a purchase the published bank contradicts.
    store = Store(tmp_path / "state" / "aigaffer.db")
    held = [pick["element"] for pick in PICKS_15_JSON["picks"]]
    previous = [pid if pid != 16 else 17 for pid in held]
    for pid in previous:
        store.record_purchase(pid, buy_price=90 if pid == 17 else 50, gw_seen=1)
    store.record_squad(1, bank=0, player_ids=previous)
    routes = unplayed_routes(midweek_bootstrap(), (1, 90))
    gw3_deadline = datetime(2025, 8, 29, 17, 30, tzinfo=UTC)

    stub_gaffer(monkeypatch, AUTH_FAILED)
    withheld = run_pipeline(
        phone_cfg(tmp_path), make_client(routes), store, "deadline",
        now=gw3_deadline - timedelta(hours=20),
    )
    stub_gaffer(monkeypatch, decided)
    final = run_pipeline(
        phone_cfg(tmp_path), make_client(routes), store, "deadline",
        now=gw3_deadline - timedelta(hours=19),
    )

    assert "purchase ledger" in withheld
    assert "purchase ledger" in final
    assert "purchase ledger" in phone.messages[-1]


def test_an_alert_that_failed_to_send_is_tried_again_next_tick(monkeypatch, tmp_path):
    calls: list[str] = []

    def flaky(token: str, chat_id: str, text: str, **kwargs) -> None:
        calls.append(text)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")

    monkeypatch.setattr(orchestrator, "send_report", flaky)

    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20)
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=19, store=store)
    tick(monkeypatch, tmp_path, AUTH_FAILED, hours=18, store=store)

    # The one that failed, the one that landed, and then silence.
    assert len(calls) == 2


def test_the_retries_stop_at_the_limit(monkeypatch, tmp_path, phone):
    # A manager who fails slowly — a refusal, a loop that runs out of time —
    # costs a full run per tick, from two schedulers. The floor bounds the
    # hours; this bounds the bill.
    _, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=23)
    for n in range(1, MANAGER_RETRY_LIMIT):
        tick(monkeypatch, tmp_path, AUTH_FAILED, hours=23 - n * 0.5, store=store)
    assert store.has_run(2, "deadline") is False
    assert len(phone.messages) == 1

    report, _ = tick(
        monkeypatch, tmp_path, AUTH_FAILED,
        hours=23 - MANAGER_RETRY_LIMIT * 0.5, store=store,
    )

    assert store.has_run(2, "deadline") is True
    assert FALLBACK_LINE in report
    assert "Full report: GW2.md" in phone.messages[-1]


def test_a_forced_run_takes_whatever_answer_it_gets(monkeypatch, tmp_path, phone):
    # ``--force`` is a person overruling the store, and the same person is
    # the authority the floor stands in for: they asked for a report now,
    # they can see the stood-down line, and they can force again.
    report, store = tick(monkeypatch, tmp_path, AUTH_FAILED, hours=20, force=True)

    assert store.has_run(2, "deadline") is True
    assert FALLBACK_LINE in report
    [digest] = phone.messages
    assert "Full report: GW2.md" in digest


def test_a_naive_clock_is_read_as_utc(monkeypatch, tmp_path, phone):
    stub_gaffer(monkeypatch, AUTH_FAILED)
    store = Store(tmp_path / "state" / "aigaffer.db")

    run_pipeline(
        phone_cfg(tmp_path), make_client(pipeline_routes()), store, "deadline",
        now=hours_out(20).replace(tzinfo=None),
    )

    assert store.has_run(2, "deadline") is False


def test_an_unconfigured_phone_is_not_told_the_report_was_kept(
    monkeypatch, tmp_path, capsys
):
    stub_gaffer(monkeypatch, AUTH_FAILED)
    quiet = config(state_dir=tmp_path / "state", anthropic_api_key="sk-test")

    run_pipeline(
        quiet, make_client(pipeline_routes()), Store(tmp_path / "state" / "aigaffer.db"),
        "deadline", now=hours_out(20),
    )

    out = capsys.readouterr().out
    assert orchestrator.NOT_CONFIGURED not in out
    assert orchestrator.ALERT_NOT_CONFIGURED in out


# --- through the command line, where the incident happened -----------------


def test_auto_withholds_then_reports_then_stands_down(
    monkeypatch, tmp_path, capsys, store
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(12)))

    stub_gaffer(monkeypatch, AUTH_FAILED)
    assert cli.main(["auto"]) == 0
    assert store.has_run(2, "deadline") is False
    assert orchestrator.WITHHELD in capsys.readouterr().out

    stub_gaffer(monkeypatch, decided)
    assert cli.main(["auto"]) == 0
    assert store.decision(2, "deadline")["decision_source"] == "manager"

    assert cli.main(["auto"]) == 0
    assert len(store.last_runs()) == 1


def test_a_forced_deadline_at_the_keyboard_is_not_withheld(monkeypatch, tmp_path, store):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    serve(monkeypatch, pipeline_routes(bootstrap=bootstrap_due_in(40)))
    stub_gaffer(monkeypatch, AUTH_FAILED)

    assert cli.main(["deadline", "--force"]) == 0

    assert store.has_run(2, "deadline") is True
