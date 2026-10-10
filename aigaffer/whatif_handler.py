"""The chip what-if: "Wildcard?" or "Free hit?" from the owner, answered.

The inbox hands a chip word here (:func:`aigaffer.inbox.dispatch`), with a
``reply`` to answer through, and this does the whole of spec §4 in order: the
failure marker first, the day's cap, an ack, the cheap gates, a snapshot of
the state under the shared lock, the run's own preamble on that snapshot, the
two window solves, message 1 and its log line, then the gaffer, his log line
and message 2. Nothing it computes is saved anywhere but the what-if log in
the inbox directory — never the repo, which is public.

**Order matters, and each step is where it is for a reason.**

* The marker is the first act, before anything that can raise. Then an
  exception anywhere finds it already set, so the inbox lets the update go
  at once and the owner asks again (a what-if is never re-run behind his
  back: it costs minutes and a gaffer's bill). And a marker that is still
  there at the start of a later minute can only mean this process was
  killed — a timeout, an OOM — and the next one picks up what is missing.
* The cap is a file read, so it goes before the ack: a capped request gets
  one message, not two.
* The ack goes before any request or lock: a tick holding the lock on a
  deadline day holds it for most of an hour, and silence that long reads
  as a dead bot.
* The gates make the recorder's three requests, never the preamble's two
  hundred, and spend nothing: no line is written, so the cap is untouched.
* The lock is taken only to copy the store — seconds — and the copy is
  what everything after reads, so a tick's ``git pull --rebase`` can never
  rewrite a file under a what-if, and the tick never waits for one. Taking
  it at all also puts a what-if behind a running report rather than beside
  it on the box's one CPU, and the process is niced for the same reason.
* Message 1 is sent before its line is written, so a Telegram that drops
  it spends nothing; the opinion's line is written before message 2, so a
  kill between them cannot buy a second opinion. The numbers line carries
  the gaffer's briefing, so a resumed what-if runs the opinion alone, on
  exactly the numbers message 1 showed, rather than recomputing them.

The heavy stages are imported into this module's namespace and called by
name, which is how the tests replace them without a solver or a network.
"""

import os
import sqlite3
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aigaffer.chips import CHIP_API_NAMES, CHIP_ORDER, FREE_HIT, held_for, held_in_week
from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.executed import Executed
from aigaffer.inbox import Reply, confirm_handled, mark_failed, marked_failed
from aigaffer.manager.chip_opinion import build_chip_briefing, render_opinion, run_chip_opinion
from aigaffer.manager.client import build_client
from aigaffer.news_ledger import evaluate_ledger, ledger_path, read_ledger
from aigaffer.orchestrator import SINGLE, prepare_week
from aigaffer.report.render import chip_label
from aigaffer.report.whatif import render_numbers
from aigaffer.statesync import REPORT_RUNNING, state_lock
from aigaffer.store import DB_NAME, EXECUTED_DIR, Store, read_executed
from aigaffer.whatif import simulate, verdict_minutes
from aigaffer.whatif_log import LOG_FILE, WhatIfLog, numbers_entry, opinion_entry

WHATIF_DAILY_CAP = 6
# How much lower than the hourly tick a what-if runs: the box has one CPU,
# and a deadline report must never be the one that waits.
WHATIF_NICE = 10

# Every sentence the owner can be sent from here. The ack names no gameweek:
# it goes before the bootstrap is fetched, and message 1's header names it.
CAPPED = "That's six what-ifs today — ask again tomorrow."
ACK = "Looking at a {chip} for the coming gameweek — numbers in a few minutes, the gaffer's view after."
CHIPS_OFF = "What-ifs need the chip planner, which is switched off on this box."
NO_GAMEWEEK = "No what-if: the API has no gameweek ahead."
NO_SQUAD = "No what-if: the API shows no squad to work from."
PICKS_FAILED = "No what-if: the API wouldn't give me your squad just now — try again in a few minutes."
ONE_CHIP_A_WEEK = "You've entered a {entered} for GW{gw} — one chip a week."
FH_AFTER_MOVES = "You've entered moves for GW{gw}; a free hit would undo them. Not simulated."
FH_BACK_TO_BACK = "No free hit in GW{gw}: one can't follow last week's."
NOT_HELD = "You don't hold a {chip} you can play in GW{gw}."
NO_ANSWER = "The solver couldn't answer in time — try again later."
DIDNT_FINISH = "That what-if didn't finish — ask again."
GAFFER_DOWN = "The gaffer couldn't be reached ({reason}) — the numbers above stand."


def _utcnow() -> datetime:
    return datetime.now(UTC)


def chip_whatif(
    cfg: Config,
    client: FplClient,
    directory: Path,
    lock_path: Path,
    kind: str,
    update_id: int,
    sent_at: datetime,
    reply: Reply,
    *,
    now: Callable[[], datetime] = _utcnow,
    anthropic_factory: Callable[[Config], object] = build_client,
    renice: Callable[[int], object] = os.nice,
) -> None:
    """Answer one chip what-if through ``reply``; always None (it has replied).

    Bound in :func:`aigaffer.__main__._inbox` with the first four arguments;
    the inbox supplies the rest. ``sent_at`` is unused today and kept for the
    handler shape the inbox calls. May raise — the inbox apologises and,
    because the marker is already this update's, lets it go.

    The directory is made here as well as by the inbox: the first act is the
    marker, and the marker must never be the thing that raises.
    """
    directory.mkdir(parents=True, exist_ok=True)
    log = WhatIfLog(directory / LOG_FILE)
    if marked_failed(directory) == update_id:
        # Only a kill leaves our own marker behind (see the module docstring).
        confirm_handled(directory, update_id)
        _resume(cfg, log, update_id, reply, anthropic_factory, now)
        return None
    mark_failed(directory, update_id)
    started = time.monotonic()

    if log.numbers_today(now()) >= WHATIF_DAILY_CAP:
        reply(CAPPED)
        return None
    reply(ACK.format(chip=chip_label(kind).lower()))

    gate = _gate(cfg, client, kind)
    if isinstance(gate, str):
        reply(gate)
        return None
    event_id, row = gate

    with tempfile.TemporaryDirectory(prefix="aigaffer-whatif-") as scratch:
        snapshot = Path(scratch) / DB_NAME
        with state_lock(lock_path, on_wait=lambda: reply(REPORT_RUNNING)):
            _copy_store(cfg.state_dir / DB_NAME, snapshot)
            # The news ledger is read under the same lock and never written:
            # a what-if is a question, and what it learns is not news.
            raw_ledger = read_ledger(ledger_path(cfg.state_dir))
        store = Store(snapshot)
        overrides, source, verdict = verdict_minutes(store, event_id)
        renice(WHATIF_NICE)
        week = prepare_week(
            cfg, client, store,
            executed_for=lambda gw: row if gw == event_id else None,
            minute_overrides=overrides,
        )
        whatif = simulate(week, cfg, kind, minutes_source=source)
        if whatif is None:
            reply(NO_ANSWER)
            return None
        reply(render_numbers(
            whatif, week, verdict=verdict, now=now(),
            numbers_only=not cfg.manager_enabled,
        ))
        try:
            # Judged inside this try: the ledger is memory, but a shape nobody
            # foresaw is raised after message 1 like any other, and reads as
            # "the gaffer couldn't be reached" while the numbers stand.
            news = evaluate_ledger(
                raw_ledger, week.effective.bootstrap, week.effective.fixtures,
                week.effective.event, now(),
            )
            briefing = (
                build_chip_briefing(week, whatif, verdict, ledger=news)
                if cfg.manager_enabled else None
            )
        except Exception as error:
            # Message 1 has gone, so the numbers line is still owed (the cap
            # is spent and a resume must find it) — with no briefing, since
            # there is none to keep. Spec §11: whatever is raised after
            # message 1 is caught here, and message 2 says the gaffer
            # couldn't be reached. The opinion is not attempted: it has
            # nothing to read.
            entry = numbers_entry(
                whatif, ts=now().isoformat(), total_seconds=time.monotonic() - started
            )
            log.append_numbers(update_id, {**entry, "briefing": None})
            reply(GAFFER_DOWN.format(reason=type(error).__name__))
            return None
        entry = numbers_entry(
            whatif, ts=now().isoformat(), total_seconds=time.monotonic() - started
        )
        log.append_numbers(update_id, {**entry, "briefing": briefing})
        if briefing is not None:
            _opinion(cfg, log, update_id, reply, anthropic_factory, now, briefing, whatif.band)
    return None


def _gate(cfg: Config, client: FplClient, kind: str) -> str | tuple[int, Executed | None]:
    """The reply that ends a what-if before it starts, or the gameweek it is
    about and the recorded week to solve from.

    The recorder's three requests (:func:`aigaffer.recording.transfers_made`)
    and no more. The squad is the current gameweek's picks: after a deadline
    that is the squad he entered, and the question is about the next one.
    The recorded week is read from its text file without the database
    (:func:`aigaffer.store.read_executed`) — an atomic file, written only by
    this inbox, one message at a time.
    """
    if not cfg.chips or cfg.planner == SINGLE:
        return CHIPS_OFF
    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        return NO_GAMEWEEK
    current = bootstrap.current_event()
    if current is None:
        return NO_SQUAD
    try:
        squad = client.picks(cfg.team_id, current.id)
    except httpx.HTTPStatusError as error:
        # A 404 is the API's way of saying there is no squad (a new entry);
        # anything else is the API having a bad minute.
        return NO_SQUAD if error.response.status_code == 404 else PICKS_FAILED
    except httpx.TransportError:
        return PICKS_FAILED
    chips_used = client.chips_used(cfg.team_id)
    row = read_executed(cfg.state_dir / EXECUTED_DIR, event.id)

    if row is not None and row.chip in CHIP_ORDER:
        return ONE_CHIP_A_WEEK.format(entered=chip_label(row.chip).lower(), gw=event.id)
    if kind == FREE_HIT and row is not None and row.transfers_in:
        return FH_AFTER_MOVES.format(gw=event.id)
    held = held_in_week(bootstrap, chips_used, event.id, row)
    if held_for(held, kind, event.id) is None:
        if kind == FREE_HIT and _free_hit_played(chips_used, event.id - 1):
            return FH_BACK_TO_BACK.format(gw=event.id)
        return NOT_HELD.format(chip=chip_label(kind).lower(), gw=event.id)
    return event.id, row


def _free_hit_played(chips_used: list[dict], event: int) -> bool:
    """Whether the chip history has a free hit played in ``event``."""
    return any(
        str(entry.get("name", "")).strip().lower() == CHIP_API_NAMES[FREE_HIT]
        and entry.get("event") == event
        for entry in chips_used
    )


def _copy_store(live: Path, snapshot: Path) -> None:
    """A consistent copy of the store, by SQLite's own backup.

    A file copy could catch a database mid-write; the backup API reads it as
    one transaction. A box with no database yet copies nothing, and the
    :class:`Store` opened on the snapshot path creates an empty one.

    The path is resolved first: ``as_uri`` refuses a relative one, and the
    box's state directory is the relative ``state`` (the tick runs from the
    clone's root), so an unresolved path would fail every real what-if.
    """
    if not live.exists():
        return
    source = sqlite3.connect(f"{live.resolve().as_uri()}?mode=ro", uri=True)
    target = sqlite3.connect(snapshot)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()


def _opinion(
    cfg: Config,
    log: WhatIfLog,
    update_id: int,
    reply: Reply,
    anthropic_factory: Callable[[Config], object],
    now: Callable[[], datetime],
    briefing: str,
    band: str,
) -> None:
    """The gaffer's view, its log line, then message 2 — in that order.

    :func:`~aigaffer.manager.chip_opinion.run_chip_opinion` never raises; the
    broad catch is for everything around it (building the client, rendering,
    the log), and its message carries the exception's class, never its
    words. Message 1 stands either way.
    """
    try:
        opinion = run_chip_opinion(anthropic_factory(cfg), cfg, briefing, band)
        text = render_opinion(opinion, band)
        log.append_opinion(
            update_id, opinion_entry(opinion, ts=now().isoformat(), model=cfg.manager_model)
        )
    except Exception as error:  # the gaffer is advice; message 1 stands
        text = GAFFER_DOWN.format(reason=type(error).__name__)
    reply(text)


def _resume(
    cfg: Config,
    log: WhatIfLog,
    update_id: int,
    reply: Reply,
    anthropic_factory: Callable[[Config], object],
    now: Callable[[], datetime],
) -> None:
    """Pick up a what-if a killed process left: only what is missing.

    No numbers line: message 1 may or may not have gone, and nothing it
    computed is kept, so he is told to ask again — re-running it here would
    be a second what-if he did not ask for. A numbers line and no opinion:
    the gaffer is asked on the briefing the line kept, so message 2 speaks
    to the numbers message 1 showed. Both lines, or a box with no manager:
    nothing was left to do.
    """
    numbers = log.numbers(update_id)
    if numbers is None:
        reply(DIDNT_FINISH)
        return
    if log.has_opinion(update_id) or not cfg.manager_enabled:
        return
    briefing = numbers.get("briefing")
    if not briefing:
        return
    _opinion(cfg, log, update_id, reply, anthropic_factory, now, briefing, numbers["band"])
