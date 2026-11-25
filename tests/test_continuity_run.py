"""The earlier call through a whole run: the briefing, the report, the phone,
the record, and everything that can be wrong with what was saved.

The pipeline is the orchestrator's own test board, with the gaffer stubbed
there as it is in tests/test_orchestrator.py: he rolls, with armbands of his own.
"""

import json
from dataclasses import replace

import pytest

from aigaffer import orchestrator
from aigaffer.continuity import Continuity
from aigaffer.executed import Executed
from aigaffer.orchestrator import run_pipeline
from aigaffer.store import Store
from tests.test_manager_retry import Phone, phone_cfg
from tests.test_orchestrator import (
    GRANT,
    PAST_THE_FLOOR,
    chips_off_from_the_environment,  # noqa: F401 — autouse: chips off, as there
    decided,
    halves_routes,
    make_client,
    pipeline_routes,
    stub_gaffer,
    unavailable,
)

SECTION = "## Your earlier call this gameweek"


@pytest.fixture
def phone(monkeypatch) -> Phone:
    phone = Phone()
    monkeypatch.setattr(orchestrator, "send_report", phone)
    return phone


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "state" / "aigaffer.db")


def run(tmp_path, store, mode, routes=None, **cfg):
    return run_pipeline(
        replace(phone_cfg(tmp_path), **cfg), make_client(routes or pipeline_routes()), store, mode, now=PAST_THE_FLOOR
    )


def moved(consult):
    """The first plan on the board that makes a move, for a stated reason."""
    plan = next(p for p in consult.solve0.plans if p.transfers_in)
    return replace(decided(consult, plan=plan), change_reason="the winger is out")


def opening(text: str) -> str:
    return text.split("## Do this\n\n", 1)[1].splitlines()[0]


def corrupt(store, **values):
    for column, value in values.items():
        with store._connect() as conn:
            conn.execute(f"UPDATE runs SET {column} = ?", (value,))


def edit(store, **changes):
    """The saved decision, changed but still JSON."""
    record = store.decision(2, "early")
    corrupt(store, decision_json=json.dumps({**record, **changes}))


# --- said -------------------------------------------------------------------


def test_the_first_run_of_a_gameweek_says_nothing(monkeypatch, tmp_path, store, phone):
    gaffer = stub_gaffer(monkeypatch)
    report = run(tmp_path, store, "early")
    assert SECTION not in gaffer.consults[0].briefing
    assert gaffer.kwargs[0]["continuity"] is None
    assert opening(report).startswith("⏰") and opening(phone.messages[0]).startswith("⏰")
    assert "earlier" not in store.decision(2, "early")


def test_the_same_call_is_said_in_the_report_the_phone_and_the_long_read(monkeypatch, tmp_path, store, phone):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    report = run(tmp_path, store, "scout")

    said = opening(report)
    assert said.startswith("Same call as ") and said.endswith("early look.")
    assert opening(phone.messages[1]) == said
    snapshot = json.loads((tmp_path / "state" / "detailed" / "gw2-scout.json").read_text())
    assert opening(snapshot["do_this"]) == said
    briefing = gaffer.consults[1].briefing
    assert briefing.index(SECTION) < briefing.index("## Candidate plans")
    assert "- Transfers: roll" in briefing
    assert isinstance(gaffer.kwargs[1]["continuity"], Continuity)
    assert store.decision(2, "scout")["earlier"][0] == "early"


def test_a_changed_call_is_said_with_his_reason_and_kept(monkeypatch, tmp_path, store, phone):
    stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    stub_gaffer(monkeypatch, moved)
    report = run(tmp_path, store, "scout")

    said = opening(report)
    assert said.startswith("Changed from ") and said.endswith(", because the winger is out.")
    assert opening(phone.messages[1]) == said
    assert report.splitlines().index(said) < next(i for i, x in enumerate(report.splitlines()) if x.startswith("⏰"))
    assert store.decision(2, "scout")["change_reason"] == "the winger is out"
    snapshot = json.loads((tmp_path / "state" / "detailed" / "gw2-scout.json").read_text())
    assert snapshot["decision"]["change_reason"] == "the winger is out"


def test_a_week_the_gaffer_could_not_decide_says_whose_pick_it_is(monkeypatch, tmp_path, store, phone):
    stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    stub_gaffer(monkeypatch, unavailable)
    report = run(tmp_path, store, "scout")
    assert opening(report).startswith("Changed from ")
    assert opening(report).endswith("because this is the solver's pick.")


def test_with_no_manager_configured_there_is_nobody_to_name(monkeypatch, tmp_path, store, phone):
    stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    report = run(tmp_path, store, "scout", manager_enabled=False)
    assert opening(report).startswith("Changed from ") and "because" not in opening(report)


# --- never the report's price ---------------------------------------------------


def no_earlier_call(gaffer, report, consult=1):
    assert SECTION not in gaffer.consults[consult].briefing
    assert gaffer.kwargs[consult]["continuity"] is None
    assert opening(report).startswith("⏰")
    assert "Decided by the gaffer." in report


@pytest.mark.parametrize("values", [{"decision_json": "{half"}, {"decision_json": "null"}, {"ts": "soon"}])
def test_a_corrupt_record_is_a_run_with_no_earlier_call(monkeypatch, tmp_path, store, phone, values):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    corrupt(store, **values)
    no_earlier_call(gaffer, run(tmp_path, store, "scout"))


@pytest.mark.parametrize("changes", [
    {"squad_before": [None]}, {"transfers_in": None}, {"captain": "x"}, {"adjustments": [None]},
    {"chip": {}}, {"squad_before": None},
])
def test_a_record_of_another_shape_is_a_run_with_no_earlier_call(monkeypatch, tmp_path, store, phone, changes):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    edit(store, **changes)
    no_earlier_call(gaffer, run(tmp_path, store, "scout"))


def test_a_rationale_that_is_not_words_costs_only_the_reason(monkeypatch, tmp_path, store, phone):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    edit(store, rationale=["x"])
    report = run(tmp_path, store, "scout")
    assert SECTION in gaffer.consults[1].briefing and "- Why:" not in gaffer.consults[1].briefing
    assert opening(report).startswith("Same call")


def test_a_section_that_will_not_render_is_a_run_with_no_earlier_call(monkeypatch, tmp_path, store, phone, capsys):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")

    def broken(*args, **kwargs):
        raise TypeError("secret")

    monkeypatch.setattr(orchestrator, "earlier_section", broken)
    no_earlier_call(gaffer, run(tmp_path, store, "scout"))
    out = capsys.readouterr().out
    assert "could not be read (TypeError)" in out and "secret" not in out


@pytest.mark.parametrize("how", ["missing", "corrupt", "keys", "squad", "fixtures", "player_rows"])
def test_a_bad_snapshot_costs_the_facts_and_not_the_call(monkeypatch, tmp_path, store, phone, how, capsys):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    path = tmp_path / "state" / "detailed" / "gw2-early.json"
    saved = json.loads(path.read_text())
    if how == "missing":
        path.unlink()
    elif how == "corrupt":
        path.write_text("{half")
    elif how == "keys":
        path.write_text(json.dumps({**saved, "players": {pid: {} for pid in saved["players"]}}))
    elif how == "squad":
        path.write_text(json.dumps({**saved, "squad": [1]}))
    elif how == "fixtures":
        path.write_text(json.dumps({**saved, "players": {p: {**r, "f": None} for p, r in saved["players"].items()}}))
    else:
        real, calls = orchestrator.player_rows, []

        def once(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise KeyError("club")
            return real(**kwargs)

        monkeypatch.setattr(orchestrator, "player_rows", once)

    report = run(tmp_path, store, "scout")
    context = gaffer.kwargs[1]["continuity"]
    assert context.since is None and SECTION in context.section and "Since then" not in context.section
    assert opening(report).startswith("Same call")
    assert "what has changed since is unknown (" in capsys.readouterr().out


def test_a_line_that_will_not_render_is_a_report_without_one(monkeypatch, tmp_path, store, phone, capsys):
    stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")

    def broken(*args, **kwargs):
        raise TypeError("secret")

    monkeypatch.setattr(orchestrator, "report_line", broken)
    assert opening(run(tmp_path, store, "scout")).startswith("⏰")
    assert "line was skipped (TypeError)" in capsys.readouterr().out


# --- what the rule is handed ----------------------------------------------------


def handed(gaffer) -> Continuity:
    return gaffer.kwargs[-1]["continuity"]


def test_the_minutes_used_are_the_earlier_runs_own(monkeypatch, tmp_path, store, phone):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    edit(store, adjustments=[{"player_id": GRANT, "expected_minutes": 33.0, "reason": "x"}])
    run(tmp_path, store, "scout")
    used = handed(gaffer).used
    # His own number for the man he adjusted; the snapshot's for everybody else.
    assert used[GRANT] == 33.0 and len(used) > 15


@pytest.mark.parametrize("spent, kept", [([], True), ([{"name": "bboost", "event": 1}], False)])
def test_a_chip_already_spent_cannot_be_kept(monkeypatch, tmp_path, store, phone, spent, kept):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    edit(store, chip="bench_boost")
    run(tmp_path, store, "scout", routes=halves_routes(chips_used=spent))
    assert handed(gaffer).chip_kept is kept


@pytest.mark.parametrize("shift, moved", [(0.0, False), (1.4, False), (1.6, True), (-1.6, True), (None, True)])
def test_the_panel_says_whether_the_chip_moved(monkeypatch, tmp_path, store, phone, shift, moved):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    panel = store.decision(2, "early")["chip_evs"]
    then = {} if shift is None else {**panel, "bench_boost": panel["bench_boost"] + shift}
    edit(store, chip="bench_boost", chip_evs=then)
    run(tmp_path, store, "scout")
    assert handed(gaffer).chip_moved is moved


@pytest.mark.parametrize("of_this_call", [True, False])
def test_the_briefing_says_when_he_has_entered_the_call(monkeypatch, tmp_path, store, phone, of_this_call):
    gaffer = stub_gaffer(monkeypatch)
    run(tmp_path, store, "early")
    verdict = store.latest_verdict(2, ("early",))
    stamp = verdict.ts if of_this_call else "2025-08-19T00:00:00+00:00"
    record = verdict.decision
    store.save_executed(Executed(
        gw=2, mode="early", recorded_at=stamp, transfers_in=[], transfers_out=[],
        squad_after=record["squad_before"], chip="none", captain=record["captain"], vice=record["vice"],
        buy_prices={}, sell_prices={}, bank_after=0, ft_after=1, ft_before=1, verdicts=[["early", stamp]],
    ))
    report = run(tmp_path, store, "scout")
    assert ("He has entered it" in handed(gaffer).section) is of_this_call
    # Both lines, the squad it works from and the call it is, in one text.
    assert "Working from" in report and opening(report).startswith("Same call as ")
