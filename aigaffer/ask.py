"""Ask the gaffer: the owner's question, answered from this week's report.

The inbox hands a question here — any text that is not a command and reads as
a question (:func:`aigaffer.inbox.is_question`) — as the owner typed it, and
sends whatever this returns. The answer is the gaffer's, from the newest
detailed report and its snapshot, through the short conversation in
:mod:`aigaffer.manager.ask_writer`; this module is the handler around it: the
order things happen in, the day's cap, the thread, and the one retry.

**Order matters, and each step is where it is for a reason** — the chip
what-if's (:mod:`aigaffer.whatif_handler`), cut down to a question:

* The failed marker is the first act. A marker that already holds this very
  update means an earlier try at it was killed mid-question, or its answer
  was paid for and the reply then lost; either way it may have been billed,
  so it is not asked again behind the owner's back — he is told to ask again.
  Set before anything that can raise, so a kill anywhere after it is found.
* The cap is a file read, so it goes before any lock or request: a capped
  question costs nothing. Only an answer spends one; a question with no
  report, or none answered, is free.
* The lock is the hourly tick's (``state.lock`` in the inbox dir, flocked by
  ``scripts/run-tick.sh``), taken only to read the gameweek's snapshots into
  memory, because the tick's ``git pull --rebase`` can rewrite
  ``state/detailed/`` under a reader. The report is rendered from that copy
  outside the lock, in memory, and the ``reports/`` file is never read: it is
  the same text, and a second file is a second thing that could be caught
  mid-write. A wait for the lock is said, as the what-if says it — a tick
  holds it for a whole report.
* No snapshot at all (the season's first days, a fresh box) is a plain
  sentence and no request.
* The thread is the gameweek's (``ask/gw{n}.jsonl``); its last ten
  exchanges go into the conversation, so "and Saka?" follows on.
* One retry, in-process, within the question's six turns: the six are
  metered across both tries, so a first try that spent them leaves the retry
  nothing to spend and a question never costs more than six requests.
* It never raises (spec R15). A failure twice is the "ask again" line and a
  normal return, so the inbox never sees an exception and never re-runs a
  paid question on its next minute.

**What it writes, and where.** Only the inbox dir: ``ask/cap.json``, the
thread, and the marker the inbox itself keeps. Never the repo, which is
public, and never ``state/``: the owner's questions and the answers are his.
The store, the snapshots and the reports are read, at most, and never opened
for writing.
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aigaffer.config import Config
from aigaffer.detailed.render import render_detailed
from aigaffer.detailed.snapshot import latest_snapshot, load_snapshots
from aigaffer.inbox import ASK_WRONG, FAILED_FILE, Reply, mark_failed, marked_failed
from aigaffer.manager import ask_writer
from aigaffer.manager.ask_writer import MAX_TURNS, AskUnavailable, plan_moves

# Bound at module scope so a test can replace it; called only when a question
# first needs the gaffer, so a box that is never asked never builds a client.
# The review's client: three minutes a request and no retry of its own — the
# handler's one retry is the retry.
from aigaffer.manager.client import build_review_client
from aigaffer.statesync import REPORT_RUNNING, STATE_LOCK, state_lock

__all__ = [
    "ASK_WRONG", "CAP_REACHED", "DAILY_CAP", "MAX_TURNS", "NO_SNAPSHOT",
    "THREAD_KEEP", "make_ask",
]

DAILY_CAP = 30
THREAD_KEEP = 10
ATTEMPTS = 2

NO_SNAPSHOT = "I haven't written this week's detailed report yet — ask after the next run."
CAP_REACHED = "That's today's questions — ask again tomorrow."
PLAN_CHANGED = "The plan changed since your last question. "

# The snapshot keeps no deadline (it is the run's, not the gameweek's), and
# fetching the bootstrap for one line of a report the owner already has is a
# request a question does not need. The report's header says so instead.
NO_DEADLINE = "as in the GW report"

ASK_DIR = "ask"
CAP_FILE = "cap.json"

NO_ANSWER = "aigaffer ask: no answer ({reason})"
NOT_RECORDED = "aigaffer ask: answered, not recorded ({reason})"


def make_ask(cfg: Config, inbox_dir: Path, client: Any = None) -> Callable[[str, int, datetime, Reply], str | None]:
    """The inbox's ``ask``: a question in, the reply to send back out.

    ``client`` is a scripted one in the tests; on the box it is None and the
    review's client is built on the first question that needs it, once, and
    kept for the rest of the inbox tick.
    """
    clients: list[Any] = [] if client is None else [client]

    def gaffer() -> Any:
        if not clients:
            clients.append(build_review_client(cfg))
        return clients[0]

    def ask(raw: str, update_id: int, sent_at: datetime, reply: Reply) -> str | None:
        try:
            if marked_failed(inbox_dir) == update_id:
                text = ASK_WRONG
            else:
                inbox_dir.mkdir(parents=True, exist_ok=True)
                mark_failed(inbox_dir, update_id)
                text = _handle(cfg, inbox_dir, gaffer, raw, sent_at, reply)
        except Exception as error:  # R15: never raises; the class, never the words
            print(NO_ANSWER.format(reason=type(error).__name__))
            text = ASK_WRONG
        _clear_marker(inbox_dir, update_id)
        return text

    return ask


def _handle(
    cfg: Config, inbox_dir: Path, gaffer: Callable[[], Any], raw: str, sent_at: datetime, reply: Reply
) -> str:
    """Steps 2 to 7 of the order in the module docstring, after the marker."""
    folder = inbox_dir / ASK_DIR
    day = sent_at.astimezone(UTC).date().isoformat()
    if _count(folder / CAP_FILE, day) >= DAILY_CAP:
        return CAP_REACHED

    with state_lock(inbox_dir / STATE_LOCK, on_wait=lambda: _say(reply, REPORT_RUNNING)):
        newest = latest_snapshot(cfg.state_dir)
        snapshots = [] if newest is None else load_snapshots(cfg.state_dir, newest["gw"])
    if not snapshots:
        return NO_SNAPSHOT
    snapshot = snapshots[-1]
    report_md = render_detailed(snapshots, NO_DEADLINE)

    thread_path = folder / f"gw{snapshot['gw']}.jsonl"
    thread = _thread(thread_path)[-THREAD_KEEP:]

    budget = _Metered(gaffer, MAX_TURNS)
    text = None
    for _ in range(ATTEMPTS):
        try:
            text = ask_writer.answer(cfg, snapshot, report_md, thread, raw, budget)
            break
        except Exception as error:  # an API error, an invalid answer, a fault: retried once
            reason = str(error) if isinstance(error, AskUnavailable) else type(error).__name__
            print(NO_ANSWER.format(reason=reason))
    if text is None:
        return ASK_WRONG

    moves = plan_moves(snapshot)
    if thread and thread[-1].get("plan_moves") not in (None, moves):
        text = PLAN_CHANGED + text
    try:
        # Answered is answered: a disk that will not take the cap or the
        # thread costs the bookkeeping, never the answer already paid for.
        _write_json(folder / CAP_FILE, {"day": day, "n": _count(folder / CAP_FILE, day) + 1})
        _append(thread_path, {"q": raw, "a": text, "ts": sent_at.isoformat(), "plan_moves": moves})
    except Exception as error:
        print(NOT_RECORDED.format(reason=type(error).__name__))
    return text


class _Metered:
    """The client, with the question's six turns metered across both tries.

    Built on first use, so a question turned away before the gaffer is asked
    never builds a client at all.
    """

    def __init__(self, gaffer: Callable[[], Any], turns: int) -> None:
        self._gaffer = gaffer
        self.left = turns
        self.messages = self

    def create(self, **request: Any) -> Any:
        if self.left <= 0:
            raise AskUnavailable("turns")
        self.left -= 1
        return self._gaffer().messages.create(**request)


def _say(reply: Reply, text: str) -> None:
    """A courtesy message: one that fails is not the question failing."""
    try:
        reply(text)
    except Exception:
        pass


def _count(path: Path, day: str) -> int:
    """Questions answered on ``day``: the cap file's count when it is that
    day's, else none. A file that will not read counts none, as a new day."""
    try:
        cap = json.loads(path.read_text(encoding="utf-8"))
        return int(cap["n"]) if cap.get("day") == day else 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return 0


def _thread(path: Path) -> list[dict]:
    """The gameweek's exchanges, oldest first. A line that will not parse —
    a write cut off by a kill — is passed over, not the whole thread."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("q"), str) and isinstance(entry.get("a"), str):
            entries.append(entry)
    return entries


def _write_json(path: Path, value: dict) -> None:
    """Atomically: a kill mid-write leaves the old count, not half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temporary, path)


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _clear_marker(inbox_dir: Path, update_id: int) -> None:
    """Take this update's marker away: it was handled, one way or another.
    Best effort, since a marker left behind only means "ask again"."""
    try:
        if marked_failed(inbox_dir) == update_id:
            (inbox_dir / FAILED_FILE).unlink(missing_ok=True)
    except OSError:
        pass
