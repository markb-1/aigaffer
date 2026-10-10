"""Tests for the chip what-if handler: the order it does things in, the
gates, the snapshot, the log, and what a killed process leaves behind.

The heavy stages — the week's preamble, the two solves, the renderers and
the gaffer — are the modules' own business and have their own tests; here
they are stubbed in the handler's namespace, so what is pinned is the
handler's contract: one message at a time, in the spec's order (§4), the cap
spent only by a what-if that produced numbers, and a kill that never runs
anything twice.
"""

import copy
import fcntl
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aigaffer import whatif_handler
from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.data.fpl_api import FplClient
from aigaffer.inbox import mark_failed
from aigaffer.news_ledger import ledger_path
from aigaffer.statesync import REPORT_RUNNING, state_lock
from aigaffer.store import DB_NAME, Store
from aigaffer.whatif_handler import (
    ACK,
    CAPPED,
    CHIPS_OFF,
    DIDNT_FINISH,
    FH_AFTER_MOVES,
    FH_BACK_TO_BACK,
    GAFFER_DOWN,
    NO_ANSWER,
    NO_GAMEWEEK,
    NO_SQUAD,
    NOT_HELD,
    ONE_CHIP_A_WEEK,
    PICKS_FAILED,
    WHATIF_DAILY_CAP,
    WHATIF_NICE,
    chip_whatif,
)
from aigaffer.whatif_log import LOG_FILE, WhatIfLog
from tests.fixtures import (
    HISTORY_PATH,
    PICKS_15_JSON,
    PICKS_PATH,
    PIPELINE_BOOTSTRAP_JSON,
    TEAM_ID,
    CountingTransport,
    config,
    fake_fpl_transport,
    make_client,
    make_executed,
    pipeline_routes,
)

NOW = datetime(2025, 8, 21, 18, 0, tzinfo=UTC)
SENT = NOW - timedelta(seconds=30)
UID = 41


class Stages:
    """Every heavy stage the handler calls, faked and recorded in order."""

    def __init__(self, monkeypatch, *, whatif="the what-if", band="marginal") -> None:
        self.order: list[str] = []
        self.whatif = whatif
        self.band = band
        self.seen: dict = {}
        fakes = {
            "verdict_minutes": self.verdict_minutes,
            "prepare_week": self.prepare_week,
            "simulate": self.simulate,
            "render_numbers": self.render_numbers,
            "numbers_entry": self.numbers_entry,
            "evaluate_ledger": self.evaluate_ledger,
            "build_chip_briefing": self.build_chip_briefing,
            "run_chip_opinion": self.run_chip_opinion,
            "render_opinion": self.render_opinion,
            "opinion_entry": self.opinion_entry,
        }
        for name, fake in fakes.items():
            monkeypatch.setattr(whatif_handler, name, fake)

    def verdict_minutes(self, store, gw):
        self.order.append("verdict_minutes")
        self.seen["verdict_store"] = store
        self.seen["verdict_gw"] = gw
        return {7: 10.0}, "Wednesday's scout", {"mode": "scout"}

    def prepare_week(self, cfg, client, store, *, executed_for, minute_overrides):
        self.order.append("prepare_week")
        self.seen.update(
            store=store, db_path=store.db_path, executed_for=executed_for,
            minute_overrides=minute_overrides,
        )
        # The handler reads the effective inputs to judge the ledger.
        effective = SimpleNamespace(bootstrap="bootstrap", fixtures=[], event="event")
        return SimpleNamespace(name="the week", effective=effective)

    def simulate(self, week, cfg, kind, *, minutes_source=None):
        self.order.append("simulate")
        self.seen.update(kind=kind, minutes_source=minutes_source)
        if self.whatif is None:
            return None
        return SimpleNamespace(band=self.band, event=2, kind=kind)

    def render_numbers(self, whatif, week, *, verdict, now, numbers_only):
        self.order.append("render_numbers")
        self.seen.update(verdict=verdict, numbers_only=numbers_only)
        return "message 1"

    def numbers_entry(self, whatif, *, ts, total_seconds):
        self.order.append("numbers_entry")
        return {"ts": ts, "event": whatif.event, "chip": whatif.kind, "band": whatif.band}

    def evaluate_ledger(self, raw, bootstrap, fixtures, event, now):
        self.order.append("evaluate_ledger")
        self.seen["ledger_raw"] = raw
        return "the view"

    def build_chip_briefing(self, week, whatif, verdict, **kwargs):
        self.order.append("build_chip_briefing")
        self.seen["ledger"] = kwargs.get("ledger")
        self.seen["today"] = kwargs.get("today")
        return "the briefing"

    def run_chip_opinion(self, client, cfg, briefing, band):
        self.order.append("run_chip_opinion")
        self.seen.update(briefing=briefing, band=band, anthropic=client)
        return SimpleNamespace(verdict="hold")

    def render_opinion(self, opinion, band):
        self.order.append("render_opinion")
        return "message 2"

    def opinion_entry(self, opinion, *, ts, model):
        self.order.append("opinion_entry")
        return {"ts": ts, "verdict": opinion.verdict, "model": model}


class Phone:
    """The reply closure, recorded; ``fail_on`` makes the n-th send raise."""

    def __init__(self, fail_on: int | None = None) -> None:
        self.sent: list[str] = []
        self.fail_on = fail_on

    def __call__(self, text: str) -> None:
        if self.fail_on is not None and len(self.sent) + 1 == self.fail_on:
            raise httpx.ConnectError("telegram down")
        self.sent.append(text)


def setup(tmp_path, *, chips=True, key=True, **cfg_overrides):
    cfg = config(
        state_dir=tmp_path / "state",
        chips=chips,
        anthropic_api_key="test-key" if key else None,
        **cfg_overrides,
    )
    Store(cfg.state_dir / DB_NAME)  # the live database exists, as on the box
    return cfg, tmp_path / "inbox"


def ask(cfg, inbox, client=None, kind=WILDCARD, phone=None, renice=None, uid=UID):
    phone = phone if phone is not None else Phone()
    niced: list[int] = []
    chip_whatif(
        cfg, client or make_client(pipeline_routes()), inbox, inbox / "state.lock",
        kind, uid, SENT, phone,
        now=lambda: NOW,
        anthropic_factory=lambda cfg: "anthropic client",
        renice=renice or niced.append,
    )
    return phone, niced


def ack(kind=WILDCARD) -> str:
    return ACK.format(chip="wildcard" if kind == WILDCARD else "free hit")


def log_of(inbox) -> WhatIfLog:
    return WhatIfLog(inbox / LOG_FILE)


# --- the order of things -------------------------------------------------------


def test_a_what_if_runs_in_the_specs_order(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    phone, niced = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1", "message 2"]
    assert stages.order == [
        "verdict_minutes", "prepare_week", "simulate", "render_numbers",
        "evaluate_ledger", "build_chip_briefing", "numbers_entry", "run_chip_opinion",
        "render_opinion", "opinion_entry",
    ]
    assert niced == [WHATIF_NICE]
    assert stages.seen["kind"] == WILDCARD
    assert stages.seen["minute_overrides"] == {7: 10.0}
    assert stages.seen["minutes_source"] == "Wednesday's scout"
    assert stages.seen["verdict"] == {"mode": "scout"}
    assert stages.seen["numbers_only"] is False
    assert stages.seen["anthropic"] == "anthropic client"
    assert stages.seen["band"] == "marginal"
    log = log_of(inbox)
    assert log.numbers(UID)["band"] == "marginal"
    assert log.numbers(UID)["briefing"] == "the briefing"
    assert log.has_opinion(UID)


def test_the_what_if_hands_the_gaffer_the_ledger_and_writes_none(tmp_path, monkeypatch):
    # The what-if reads the news ledger for the briefing and never writes it:
    # the file's bytes are the same before and after, and the gaffer is handed
    # whatever evaluate_ledger made of the raw entries (here the fake's "view").
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    path = ledger_path(cfg.state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"player_id": 7, "category": "doubt"}
    path.write_text(json.dumps({"7": entry}), encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    ask(cfg, inbox)

    assert stages.seen["ledger_raw"] == {"7": entry}
    assert stages.seen["ledger"] == "the view"
    assert path.read_text(encoding="utf-8") == before


def test_the_what_if_ages_the_news_on_the_handlers_clock(tmp_path, monkeypatch):
    # The ledger is judged at the handler's now(); "N days ago" must count
    # from the same instant's date, not from the machine's local today.
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    ask(cfg, inbox)

    assert stages.seen["today"] == NOW.date()


def test_with_the_manager_off_the_ledger_is_not_even_judged(tmp_path, monkeypatch):
    # No gaffer, no briefing: judging the ledger could only fail into a
    # "couldn't be reached" for a gaffer that was switched off.
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path, key=False)

    def broken(raw, bootstrap, fixtures, event, now):
        raise KeyError("shape nobody foresaw")

    monkeypatch.setattr(whatif_handler, "evaluate_ledger", broken)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1"]
    assert "build_chip_briefing" not in stages.order


def test_a_ledger_that_cannot_be_judged_is_the_gaffer_being_down(tmp_path, monkeypatch):
    # evaluate_ledger sits inside the try that wraps the briefing: a failure
    # there leaves message 1 sent, the numbers entry logged with no briefing,
    # and message 2 saying the gaffer couldn't be reached.
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    def broken(raw, bootstrap, fixtures, event, now):
        raise KeyError("shape nobody foresaw")

    monkeypatch.setattr(whatif_handler, "evaluate_ledger", broken)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1", GAFFER_DOWN.format(reason="KeyError")]
    assert "run_chip_opinion" not in stages.order
    assert log_of(inbox).numbers(UID)["briefing"] is None


def test_the_marker_is_set_before_anything_else(tmp_path, monkeypatch):
    # Spec §8: the handler's first act, so an exception anywhere after it is
    # never mistaken for a kill. The first reply is the earliest observable
    # moment after the marker; it must already be there.
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    seen = []

    def phone(text: str) -> None:
        seen.append((inbox / "failed").read_text())

    ask(cfg, inbox, phone=phone)

    assert seen[0] == str(UID)


def test_the_ack_goes_before_any_request_or_lock(tmp_path, monkeypatch):
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    transport = CountingTransport(pipeline_routes())
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)
    at_ack = {}

    def phone(text: str) -> None:
        if not at_ack:
            at_ack["requests"] = sum(transport.counts.values())

    ask(cfg, inbox, client=client, phone=phone)

    assert at_ack["requests"] == 0


def test_a_running_report_is_said_and_waited_for(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    holding, release = threading.Event(), threading.Event()

    def tick() -> None:
        with state_lock(inbox / "state.lock"):
            holding.set()
            release.wait(5)

    holder = threading.Thread(target=tick)
    holder.start()
    holding.wait()
    phone = Phone()

    def on_send(text: str) -> None:
        phone(text)
        if text == REPORT_RUNNING:
            release.set()  # the tick finishes once he has been told

    ask(cfg, inbox, phone=on_send)
    holder.join()

    assert phone.sent == [ack(), REPORT_RUNNING, "message 1", "message 2"]
    assert stages.order[0] == "verdict_minutes", "nothing heavy ran before the lock"


# --- the snapshot --------------------------------------------------------------


def test_everything_after_the_lock_reads_a_copy_that_is_then_gone(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    live = cfg.state_dir / DB_NAME
    Store(live).save_run(2, "scout", "# report", {"mode": "scout", "event": 2})

    ask(cfg, inbox)

    snapshot = stages.seen["db_path"]
    assert snapshot != live
    assert stages.seen["verdict_store"].db_path == snapshot
    assert not Path(snapshot).exists(), "the temp copy is deleted at the end"


def test_the_copy_holds_what_the_live_store_held(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    Store(cfg.state_dir / DB_NAME).save_run(2, "scout", "# report", {"mode": "scout"})
    copied = {}

    def verdict_minutes(store, gw):
        copied["decision"] = store.decision(2, "scout")
        return {}, None, None

    monkeypatch.setattr(whatif_handler, "verdict_minutes", verdict_minutes)

    ask(cfg, inbox)

    assert copied["decision"] == {"mode": "scout"}


def test_the_copy_is_gone_when_a_stage_raises(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    def exploding(week, cfg, kind, *, minutes_source=None):
        raise RuntimeError("solver fell over")

    monkeypatch.setattr(whatif_handler, "simulate", exploding)

    with pytest.raises(RuntimeError):
        ask(cfg, inbox)

    assert not Path(stages.seen["db_path"]).exists()


def test_the_week_is_given_the_gates_recorded_row(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    row = make_executed()  # GW2, Grant for Reyes, no chip
    Store(cfg.state_dir / DB_NAME).save_executed(row)

    ask(cfg, inbox)

    executed_for = stages.seen["executed_for"]
    assert executed_for(2) == row
    assert executed_for(3) is None


def test_a_relative_state_directory_is_snapshotted(tmp_path, monkeypatch):
    # On the box the state directory is the relative "state" (the tick runs
    # from the clone's root); a file URI cannot be built from a relative path.
    Stages(monkeypatch)
    monkeypatch.chdir(tmp_path)
    cfg = config(state_dir=Path("state"), chips=True, anthropic_api_key="test-key")
    Store(cfg.state_dir / DB_NAME)

    phone, _ = ask(cfg, tmp_path / "inbox")

    assert phone.sent == [ack(), "message 1", "message 2"]
    assert log_of(tmp_path / "inbox").numbers(UID) is not None


# --- the cap and the log -------------------------------------------------------


def test_the_cap_is_checked_before_the_ack(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    log = log_of(inbox)
    for uid in range(WHATIF_DAILY_CAP):
        log.append_numbers(uid, {"ts": NOW.isoformat(), "event": 2, "chip": WILDCARD, "band": "hold"})

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [CAPPED]
    assert stages.order == []


def test_no_answer_from_the_solver_spends_nothing(tmp_path, monkeypatch):
    Stages(monkeypatch, whatif=None)
    cfg, inbox = setup(tmp_path)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), NO_ANSWER]
    assert log_of(inbox).numbers_today(NOW) == 0


def test_a_send_that_fails_after_the_numbers_spends_nothing(tmp_path, monkeypatch):
    # Message 1 is sent before its line is written, so a Telegram that drops
    # it leaves no line: the cap is not spent and he simply asks again.
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    with pytest.raises(httpx.ConnectError):
        ask(cfg, inbox, phone=Phone(fail_on=2))

    assert log_of(inbox).numbers_today(NOW) == 0


def test_the_opinion_line_is_written_before_message_2(tmp_path, monkeypatch):
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    with pytest.raises(httpx.ConnectError):
        ask(cfg, inbox, phone=Phone(fail_on=3))

    assert log_of(inbox).has_opinion(UID), "a kill here cannot buy a second opinion"


def test_no_manager_means_numbers_only(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path, key=False)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1"]
    assert stages.seen["numbers_only"] is True
    assert "run_chip_opinion" not in stages.order
    assert log_of(inbox).numbers(UID)["briefing"] is None


def test_a_gaffer_stage_that_raises_says_so_and_the_numbers_stand(tmp_path, monkeypatch):
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    def broken(client, cfg, briefing, band):
        raise ValueError("bad words")

    monkeypatch.setattr(whatif_handler, "run_chip_opinion", broken)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1", GAFFER_DOWN.format(reason="ValueError")]


# --- the gates -----------------------------------------------------------------


def gated(tmp_path, monkeypatch, routes=None, statuses=None, kind=WILDCARD, **setup_kw):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path, **setup_kw)
    transport = CountingTransport(routes or pipeline_routes())
    if statuses:
        transport.inner = fake_fpl_transport(routes or pipeline_routes(), statuses)
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)
    phone, _ = ask(cfg, inbox, client=client, kind=kind)
    # No gate spends the cap, and none starts the two-hundred-request fetch.
    assert log_of(inbox).numbers_today(NOW) == 0
    assert "prepare_week" not in stages.order
    return phone.sent, transport


def test_chips_switched_off_is_said_before_any_request(tmp_path, monkeypatch):
    sent, transport = gated(tmp_path, monkeypatch, chips=False)

    assert sent == [ack(), CHIPS_OFF]
    assert sum(transport.counts.values()) == 0


def test_the_single_week_planner_cannot_answer(tmp_path, monkeypatch):
    sent, _ = gated(tmp_path, monkeypatch, planner="single")

    assert sent == [ack(), CHIPS_OFF]


def test_no_gameweek_ahead(tmp_path, monkeypatch):
    bootstrap = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    for event in bootstrap["events"]:
        event.update(is_next=False, finished=True)
    sent, _ = gated(tmp_path, monkeypatch, routes=pipeline_routes(bootstrap=bootstrap))

    assert sent == [ack(), NO_GAMEWEEK]


def test_no_squad_yet(tmp_path, monkeypatch):
    routes = pipeline_routes()
    del routes[PICKS_PATH]  # the fake answers 404, as the API does for a new entry
    sent, _ = gated(tmp_path, monkeypatch, routes=routes)

    assert sent == [ack(), NO_SQUAD]


def test_a_picks_fetch_that_fails(tmp_path, monkeypatch):
    sent, _ = gated(tmp_path, monkeypatch, statuses={PICKS_PATH: 503})

    assert sent == [ack(), PICKS_FAILED]


def test_the_gates_ask_three_things_not_two_hundred(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    transport = CountingTransport(pipeline_routes())
    client = FplClient(http=httpx.Client(transport=transport), sleep=lambda _: None)

    ask(cfg, inbox, client=client)

    # prepare_week is stubbed, so these are the gates' requests alone.
    assert dict(transport.counts) == {
        "/api/bootstrap-static/": 1, PICKS_PATH: 1, HISTORY_PATH: 1,
    }


def test_another_chip_entered_this_week(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    Store(cfg.state_dir / DB_NAME).save_executed(make_executed(chip="bench_boost"))

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), ONE_CHIP_A_WEEK.format(entered="bench boost", gw=2)]
    assert stages.order == []


def test_a_free_hit_after_entered_moves_is_not_simulated(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    Store(cfg.state_dir / DB_NAME).save_executed(make_executed())  # Grant for Reyes

    phone, _ = ask(cfg, inbox, kind=FREE_HIT)

    assert phone.sent == [ack(FREE_HIT), FH_AFTER_MOVES.format(gw=2)]
    assert stages.order == []


def test_a_wildcard_after_entered_moves_is_simulated(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    Store(cfg.state_dir / DB_NAME).save_executed(make_executed())

    phone, _ = ask(cfg, inbox)

    assert "simulate" in stages.order


def test_a_chip_not_held(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    monkeypatch.setattr(whatif_handler, "held_in_week", lambda *args, **kw: ())
    cfg, inbox = setup(tmp_path)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), NOT_HELD.format(chip="wildcard", gw=2)]
    assert stages.order == []


def test_a_free_hit_the_week_after_a_free_hit(tmp_path, monkeypatch):
    # held_in_week (Task 3) applies the rule; the handler only names it.
    Stages(monkeypatch)
    monkeypatch.setattr(whatif_handler, "held_in_week", lambda *args, **kw: ())
    routes = pipeline_routes()
    routes[HISTORY_PATH] = {**routes[HISTORY_PATH], "chips": [{"name": "freehit", "event": 1}]}
    cfg, inbox = setup(tmp_path)

    phone, _ = ask(cfg, inbox, client=make_client(routes), kind=FREE_HIT)

    assert phone.sent == [ack(FREE_HIT), FH_BACK_TO_BACK.format(gw=2)]


def after_the_deadline() -> dict:
    """GW2's deadline has gone: GW2 is current, GW3 next."""
    bootstrap = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    first, second = bootstrap["events"]
    first.update(is_current=False, is_previous=True)
    second.update(is_current=True, is_next=False)
    third = {**second, "id": 3, "name": "Gameweek 3",
             "deadline_time": "2025-08-29T17:30:00Z", "is_current": False, "is_next": True}
    bootstrap["events"].append(third)
    routes = pipeline_routes(bootstrap=bootstrap)
    routes[f"/api/entry/{TEAM_ID}/event/2/picks/"] = PICKS_15_JSON
    return routes


def test_after_a_deadline_the_question_is_about_the_next_gameweek(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    # Last week's recorded bench boost is last week's: it bars nothing now.
    Store(cfg.state_dir / DB_NAME).save_executed(make_executed(chip="bench_boost"))

    phone, _ = ask(cfg, inbox, client=make_client(after_the_deadline()))

    assert phone.sent[:2] == [ack(), "message 1"]
    assert stages.seen["verdict_gw"] == 3
    assert stages.seen["executed_for"](3) is None


# --- a killed what-if ----------------------------------------------------------


def killed(inbox, uid=UID) -> None:
    inbox.mkdir(parents=True, exist_ok=True)
    mark_failed(inbox, uid)


def test_killed_before_the_numbers_says_so_and_moves_on(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    killed(inbox)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [DIDNT_FINISH]
    assert stages.order == []
    assert (inbox / "offset").read_text() == str(UID)


def test_killed_after_the_numbers_runs_only_the_opinion(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    killed(inbox)
    log_of(inbox).append_numbers(UID, {
        "ts": NOW.isoformat(), "event": 2, "chip": WILDCARD, "band": "hold",
        "briefing": "the logged briefing",
    })

    phone, _ = ask(cfg, inbox)

    assert phone.sent == ["message 2"], "message 1 is never sent twice"
    assert stages.order == ["run_chip_opinion", "render_opinion", "opinion_entry"]
    assert stages.seen["briefing"] == "the logged briefing"
    assert stages.seen["band"] == "hold"
    assert log_of(inbox).has_opinion(UID)
    assert (inbox / "offset").read_text() == str(UID)


def test_killed_after_both_does_nothing(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    killed(inbox)
    log = log_of(inbox)
    log.append_numbers(UID, {"ts": NOW.isoformat(), "event": 2, "chip": WILDCARD,
                             "band": "hold", "briefing": "b"})
    log.append_opinion(UID, {"ts": NOW.isoformat(), "verdict": "hold"})

    phone, _ = ask(cfg, inbox)

    assert phone.sent == []
    assert stages.order == []
    assert (inbox / "offset").read_text() == str(UID)


def test_killed_after_the_numbers_with_no_manager_does_nothing(tmp_path, monkeypatch):
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path, key=False)
    killed(inbox)
    log_of(inbox).append_numbers(UID, {"ts": NOW.isoformat(), "event": 2, "chip": WILDCARD,
                                       "band": "hold", "briefing": None})

    phone, _ = ask(cfg, inbox)

    assert phone.sent == []
    assert stages.order == []


def test_a_marker_for_another_update_is_not_a_kill(tmp_path, monkeypatch):
    # The marker is the transfers-made retry's (Spec 1) for update 40; this is
    # update 41, a fresh what-if.
    Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)
    killed(inbox, uid=UID - 1)

    phone, _ = ask(cfg, inbox)

    assert phone.sent[0] == ack()
    assert (inbox / "failed").read_text() == str(UID), "the what-if's own marker"


def test_a_briefing_that_cannot_be_built_says_so_and_skips_the_opinion(tmp_path, monkeypatch):
    # Spec §11: anything raised after message 1 is caught here. The numbers
    # line is still written (cap spent: message 1 went), with no briefing.
    stages = Stages(monkeypatch)
    cfg, inbox = setup(tmp_path)

    def broken(week, whatif, verdict, **kwargs):
        raise ValueError("bad lineup id")

    monkeypatch.setattr(whatif_handler, "build_chip_briefing", broken)

    phone, _ = ask(cfg, inbox)

    assert phone.sent == [ack(), "message 1", GAFFER_DOWN.format(reason="ValueError")]
    assert log_of(inbox).numbers(UID)["briefing"] is None
    assert not log_of(inbox).has_opinion(UID)
    assert "run_chip_opinion" not in stages.order
