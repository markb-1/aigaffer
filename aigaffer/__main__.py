"""``python -m aigaffer`` — the one way to run the bot.

``auto`` is what the schedule calls: it looks at the deadline, works out
which report the hour calls for and runs it, or says nothing and exits
cleanly, which is what it does most of the time. ``early``, ``scout``,
``deadline`` and ``reminder`` run a report by name for someone at a keyboard, and
unlike ``auto`` they explain themselves when they decide not to.

A gameweek gets one of each report. The store is what remembers that, so an
hourly schedule inside a late-open window does not send the same thing twice;
``--force`` is how a person overrules it — the store's memory, and the waiting
a run does for a manager who did not decide (:mod:`aigaffer.orchestrator`).
``--dry-run`` prints the report and writes nothing anywhere, which is what
makes it safe against the live API.

``backtest`` is the odd one out: it grades the model against a gameweek that
has already been played instead of advising on one to come, so it wants no
team, no store and no schedule — only a gameweek and the API.

``review`` looks back on a finished gameweek (:mod:`aigaffer.review`); ``auto``
runs it on its own once a week is final.

``replay`` is the other look back, and the slowest command here: it plays
real managers' squads through each way of valuing the bench and the free
transfers over a finished range of gameweeks (:mod:`aigaffer.replay`), and
prints the gate that says whether a candidate ships. Like ``backtest`` it
wants only the API — and the team id, if set, to replay this squad too. Its
rows name other managers, so they go to the git-ignored ``state/replay/``
and never to stdout.

``inbox`` is the VM's other timer, once a minute: it reads what the owner has
texted the bot and answers (:mod:`aigaffer.inbox`). It wants no schedule and no
report; it does want the team and the store, because "Transfers made" is
written down.

Failure is loud here and nowhere else. A run on a schedule has no one
watching it, so a run that dies has to say so twice: on stdout for the log,
and — when there is a phone configured and this was not a dry run — as one
line to Telegram, because a silent failure and a quiet week look identical
from the outside. What it must never say is the exception's own words:
``httpx`` names the URL it was calling in every message it raises, and for
the Telegram leg of a run that URL has the bot token in it.
"""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import httpx

from aigaffer.ask import make_ask
from aigaffer.backtest import backtest_gw, finished_gameweeks
from aigaffer.chip_forecast import chip_forecast
from aigaffer.config import SCOUT_HORIZON_HOURS, Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, Event
from aigaffer.inbox import inbox_dir, run_inbox
from aigaffer.orchestrator import (
    DEADLINE_MODE,
    EARLY_MODE,
    REMINDER_MODE,
    PipelineError,
    decide_mode,
    last_kickoff,
    run_pipeline,
)
from aigaffer.peer import peer_has_run
from aigaffer.recording import transfers_made
from aigaffer.replay import (
    BASELINE,
    DEEP_PAGES,
    POLICIES,
    TOP_PAGES,
    gate,
    render_summary,
    replay,
)
from aigaffer.report.telegram import send_report
from aigaffer.review import REVIEW_MODE, ReviewUnavailable, review_target, review_tick, run_review
from aigaffer.statesync import STATE_LOCK, GitStateSync
from aigaffer.store import DB_NAME, EXECUTED_DIR, Store
from aigaffer.whatif_handler import chip_whatif
from aigaffer.whatif_log import LOG_FILE, WhatIfLog

AUTO = "auto"
BACKTEST = "backtest"
INBOX = "inbox"
REVIEW = "review"
REPLAY = "replay"
COMMANDS = (AUTO, "early", "scout", "deadline", "reminder", BACKTEST, INBOX, REVIEW, REPLAY)

NEEDS_MANAGER = "aigaffer: the review needs the manager (ANTHROPIC_API_KEY, AIGAFFER_MANAGER not 0)"
SCHEDULE_STAGE = "reading the schedule"
BACKTEST_STAGE = "the backtest"
REPLAY_STAGE = "the replay"

# Where the replay's rows go, under the state directory: git-ignored, because
# every row names a real manager's entry.
REPLAY_DIR = "replay"

# The manager switch, and what a run says when it was thrown on and there is
# nothing to reach the manager with. The name is Config's; it is read here too
# because only the command line knows what was asked for, as against what the
# configuration could deliver.
MANAGER_ENV = "AIGAFFER_MANAGER"
MANAGER_OFF = "0"
NO_MANAGER = (
    "the manager was asked for but ANTHROPIC_API_KEY is not set —"
    " running the solver alone"
)


class _OneBootstrap:
    """The client, asking the API for the bootstrap once per tick.

    The schedule's question and the review's both start from it, and the
    review must not cost the hour a second copy of the game. The report
    gets the client itself: its run fetches a bootstrap of its own, fresh,
    as it always has.
    """

    def __init__(self, client: FplClient) -> None:
        self._client = client
        self._bootstrap: Bootstrap | None = None

    def bootstrap(self) -> Bootstrap:
        if self._bootstrap is None:
            self._bootstrap = self._client.bootstrap()
        return self._bootstrap

    def __getattr__(self, name: str):
        return getattr(self._client, name)


def main(argv: list[str] | None = None) -> int:
    """Run one command and return the process exit code."""
    args = _parse_args(argv)
    client = FplClient()
    if args.command == BACKTEST:
        try:
            return _backtest(client, args.gw)
        except httpx.HTTPError as error:
            return _failed(None, error, event_id=None, stage=BACKTEST_STAGE)
    if args.command == INBOX:
        return _inbox(client)
    if args.command == REVIEW:
        return _review(args, client)
    if args.command == REPLAY:
        return _replay(args, client)

    cfg = Config.from_env()
    _manager_notice(cfg)
    store = Store(cfg.state_dir / DB_NAME)
    alert_to = None if args.dry_run else cfg

    ticking = _OneBootstrap(client)
    try:
        mode, event_id = _mode(args, ticking, store)
    except httpx.HTTPError as error:
        return _failed(alert_to, error, event_id=None, stage=SCHEDULE_STAGE)

    # The week just finished, looked back on (aigaffer.review) — on the ticks
    # with time to spare: one that stands down, or the early scout's, which
    # then reads the lessons. A scout, deadline or reminder tick keeps its hour
    # for its report; every window has quiet ticks after it. Never on a dry
    # run, which writes nothing anywhere, and never fatal.
    if args.command == AUTO and not args.dry_run and mode in (None, EARLY_MODE):
        review_tick(
            cfg,
            ticking,
            store,
            overtaken=None if args.force else partial(peer_has_run, cfg.state_dir.parent),
        )
    if mode is None:
        return 0

    # The schedule's ticks ask the other scheduler before the phone: the
    # store in hand said nothing had run, but a manager run is minutes long
    # and the peer may have pushed this very report meanwhile. A named mode
    # or a forced run is a person at the keyboard, who gets what they asked
    # for. The repository is the state directory's parent — where the diary
    # and the homepage verdict live too — which with the default state
    # directory is the working directory both runners start in.
    overtaken = None
    if args.command == AUTO and not args.force:
        overtaken = partial(peer_has_run, cfg.state_dir.parent)

    try:
        report = run_pipeline(
            cfg,
            client,
            store,
            mode,
            send=not args.dry_run,
            save=not args.dry_run,
            force=args.force,
            overtaken=overtaken,
        )
    except (PipelineError, httpx.HTTPError) as error:
        return _failed(alert_to, error, event_id, stage=f"the {mode} run")

    if args.dry_run:
        print(report)
    return 0


def _manager_notice(cfg: Config) -> None:
    """Say so when the manager was asked for and cannot run.

    ``AIGAFFER_MANAGER`` is an opt-out, so a run that sets it to anything but
    ``0`` has asked for the manager, and :class:`Config` turns him off anyway
    when there is no key to reach him with. That combination is what a missing
    repository secret looks like from the outside, and without a line here it
    looks like nothing at all: the report comes out, a little worse, and says
    nothing about what it did not do.

    Not an alert. The run is going to succeed, and a phone that buzzes for a
    successful run is a phone that gets ignored on the week it matters.
    """
    asked = os.environ.get(MANAGER_ENV)
    if asked is not None and asked != MANAGER_OFF and not cfg.manager_enabled:
        print(f"aigaffer: {NO_MANAGER}")


def _failed(
    cfg: Config | None, error: Exception, event_id: int | None, stage: str
) -> int:
    """Report a run that did not happen, and return its exit code.

    A :class:`PipelineError` is ours: it was raised about our own data, in
    words written to be read, so it is printed as it stands. Anything from
    ``httpx`` is not — its message carries the URL, and a bot token with it —
    so only the class name gets out, which says as much as a log needs.
    """
    detail = str(error) if isinstance(error, PipelineError) else type(error).__name__
    print(f"aigaffer: {stage} failed: {detail}")
    if cfg is not None:
        _alert(cfg, error, event_id)
    return 1


def _alert(cfg: Config, error: Exception, event_id: int | None) -> None:
    """Best effort: tell the phone that was expecting a report.

    Nothing here may raise. The exit code is already 1 and the reason is
    already on stdout; an alert that cannot be delivered is not a second
    failure to handle, and it certainly must not print what it tried.
    """
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        return
    gameweek = "" if event_id is None else f" (gw {event_id})"
    message = f"aigaffer run failed: {type(error).__name__}{gameweek}"
    try:
        send_report(cfg.telegram_token, cfg.telegram_chat_id, message)
    except Exception as sending:  # any failure; the alert is the last resort
        print(f"aigaffer: the alert failed too: {type(sending).__name__}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m aigaffer",
        description=(
            "Report on the next Fantasy Premier League gameweek, "
            "or grade the model against a finished one."
        ),
    )
    parser.add_argument(
        "command",
        choices=COMMANDS,
        help="auto runs whichever report the deadline calls for, if any",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="run even if this gameweek has already had this report",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the report instead of saving or sending it",
    )
    parser.add_argument(
        "--gw",
        type=int,
        help="the finished gameweek to backtest or review (default: the most recent)",
    )
    parser.add_argument(
        "--from",
        dest="from_gw",
        type=int,
        default=2,
        help="replay: the first gameweek to replay (default: 2)",
    )
    parser.add_argument(
        "--to",
        dest="to_gw",
        type=int,
        help="replay: the last gameweek to replay (default: the latest finished)",
    )
    parser.add_argument(
        "--managers",
        type=_managers,
        default=(100, 50),
        help="replay: squads to sample from the top and deep in the league, as TOP,DEEP",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="replay: the sampling seed (default: 1)"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="replay: solver processes to run at once",
    )
    return parser.parse_args(argv)


def _managers(text: str) -> tuple[int, int]:
    """``--managers``: "TOP,DEEP", two counts."""
    try:
        top, deep = (int(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected TOP,DEEP, e.g. 100,50") from None
    return top, deep


def _replay(args: argparse.Namespace, client: FplClient) -> int:
    """``replay``: play sampled squads through every policy over a finished
    range and print the gate.

    The range must be finished, and start no earlier than GW2 — the starts are
    the picks for the week before it. The team id, when set, is replayed in
    the top group beside the sample. The rows are written as JSON lines to
    ``state/replay/<timestamp>.jsonl`` and stdout gets the summary, which
    holds counts and means and no entry id. An HTTP failure is reported by
    its class alone, as everywhere here.

    Two mistakes in the asking are answered in a line rather than a
    traceback. ``--managers`` draws one squad from each of that many distinct
    pages, so it can ask for no more squads than its band has pages — 200 at
    the top, 18000 deep — and is refused, before anything is fetched, when it
    does. A team id that is not a number is not an entry anybody can replay:
    the run goes ahead on the sample alone and says so on stderr, without
    repeating what was set.
    """
    top, deep = args.managers
    top_most = TOP_PAGES[1] - TOP_PAGES[0] + 1
    deep_most = DEEP_PAGES[1] - DEEP_PAGES[0] + 1
    if not (0 <= top <= top_most and 0 <= deep <= deep_most):
        print(
            f"aigaffer: --managers draws one squad a page, so at most {top_most}"
            f" from the top and {deep_most} from deep"
        )
        return 1
    own = _own_entry()
    ticking = _OneBootstrap(client)
    try:
        finished = finished_gameweeks(ticking.bootstrap())
        last = args.to_gw if args.to_gw is not None else (finished[-1] if finished else None)
        if last is None:
            print("aigaffer: no gameweek has finished yet")
            return 1
        if last not in finished:
            print(f"aigaffer: GW{last} has not finished")
            return 1
        if not 2 <= args.from_gw <= last:
            print(f"aigaffer: the replay runs from GW2 at the earliest, to GW{last}")
            return 1
        rows, meta = replay(
            ticking, args.from_gw, last, top, deep, args.seed, args.jobs, own
        )
    except httpx.HTTPError as error:
        return _failed(None, error, event_id=None, stage=REPLAY_STAGE)

    directory = Path(os.environ.get("AIGAFFER_STATE_DIR", "state")) / REPLAY_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    # Every policy played is named to the gate, so one whose every run failed
    # is judged incomplete rather than left out of the verdicts.
    candidates = [name for name in POLICIES if name != BASELINE]
    print(render_summary(gate(rows, candidates=candidates), meta))
    print(f"aigaffer: {len(rows)} rows written to {path}")
    return 0


def _own_entry() -> int | None:
    """The replay's own entry from ``FPL_TEAM_ID``: None when it is unset, and
    None with a note on stderr when it is not a number. The note does not say
    what the variable held — whatever it was, it is somebody's setting."""
    text = os.environ.get("FPL_TEAM_ID")
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        print(
            "aigaffer: FPL_TEAM_ID is not a number, so the replay runs without your squad",
            file=sys.stderr,
        )
        return None


def _review(args: argparse.Namespace, client: FplClient) -> int:
    """``review``: look back on a finished week by hand — the latest final
    one, or ``--gw``. ``--force`` writes it again over the old section;
    ``--dry-run`` prints it and writes nothing."""
    cfg = Config.from_env()
    if not cfg.manager_enabled:
        print(NEEDS_MANAGER)
        return 1
    store = Store(cfg.state_dir / DB_NAME)
    try:
        bootstrap = client.bootstrap()
        gw = args.gw if args.gw is not None else review_target(bootstrap)
        if gw is None:
            print("aigaffer: no finished gameweek to review yet")
            return 0
        if store.has_run(gw, REVIEW_MODE) and not args.force and not args.dry_run:
            print(f"aigaffer: GW{gw} review has run already — use --force")
            return 0
        section = run_review(cfg, client, store, bootstrap, gw, dry_run=args.dry_run)
    except Exception as error:  # by hand or not, a failed review says so and exits 1
        print(f"aigaffer: review failed ({error if isinstance(error, ReviewUnavailable) else type(error).__name__})")
        return 1
    if args.dry_run:
        print(section)
    return 0


def _backtest(client: FplClient, gw: int | None) -> int:
    """Print how well the model ranked one finished gameweek.

    A gameweek still being played has no scores to grade against, and before
    the first deadline of a season there is nothing to grade at all. Both say
    so and stop, because a person asked and is owed an answer.
    """
    bootstrap = client.bootstrap()
    finished = finished_gameweeks(bootstrap)
    if not finished:
        print("aigaffer: no gameweek has finished yet")
        return 1

    gameweek = finished[-1] if gw is None else gw
    if gameweek not in finished:
        print(f"aigaffer: GW{gameweek} has not finished")
        return 1

    print(backtest_gw(client, bootstrap, client.fixtures(), gameweek))
    return 0


def _inbox(client: FplClient) -> int:
    """Read what the owner texted and answer it — the VM's minute timer.

    The handler for "Transfers made" is bound here, with the clone that holds
    the state (the state directory's parent, as for the peer check), the one
    path its commits may touch — the recorded weeks' text files, never the
    database — and the lock it shares with the hourly tick, so
    :mod:`aigaffer.inbox` knows nothing of git or of the store. The chip
    what-if is bound beside it with the inbox directory, where its marker and
    its log live, and the same lock, which it takes only to copy the store.
    Questions go to the gaffer (:mod:`aigaffer.ask`) when the manager is on.
    The chip forecast needs only the configuration and the client: it reads
    the store without the lock, a few rows and nothing written.
    """
    cfg = Config.from_env()
    directory = inbox_dir()
    recordings = f"{cfg.state_dir.name}/{EXECUTED_DIR}/"
    handler = partial(
        transfers_made,
        cfg,
        client,
        GitStateSync(cfg.state_dir.parent, (recordings,)),
        directory / STATE_LOCK,
        log=WhatIfLog(directory / LOG_FILE),
    )
    whatif = partial(chip_whatif, cfg, client, directory, directory / STATE_LOCK)
    forecast = partial(chip_forecast, cfg, client)
    # The gaffer answers questions only where there is a manager to answer
    # them; without one a question gets the help, as before. The client is
    # built on the first question, so a box that is never asked builds none.
    ask = make_ask(cfg, directory) if cfg.manager_enabled else None
    return run_inbox(
        cfg.telegram_token, cfg.telegram_chat_id, directory, handler, whatif,
        forecast, ask=ask,
    )


def _mode(
    args: argparse.Namespace, client: FplClient, store: Store
) -> tuple[str | None, int | None]:
    """The mode to run and the gameweek it is for, or ``(None, ...)`` to
    stand down.

    Deciding costs a bootstrap fetch — for the deadline ``auto`` reads and
    the gameweek the store is asked about — so a mode named on the command
    line with ``--force`` skips the question entirely and gets on with it.
    That is also the one path where the gameweek is not known yet, which is
    why it comes back alongside the mode: a failure alert says which week it
    was about when anything has bothered to ask.
    """
    if args.command != AUTO and args.force:
        return args.command, None

    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        _explain(args.command, "the API has no gameweek ahead")
        return None, None

    mode = args.command
    if mode == AUTO:
        now = datetime.now(UTC)
        # The round's end is only worth asking after while the early scout it
        # anchors has not gone: four days of hourly ticks sit beyond the
        # horizon after it has, and none of them needs the fixtures to know.
        ended = (
            None
            if store.has_run(event.id, EARLY_MODE)
            else _round_end(client, bootstrap, event, now)
        )
        chosen = decide_mode(now, event.deadline_time, last_kickoff=ended)
        if chosen is None:
            return None, event.id
        mode = chosen
        # The reminder checks the full report against the morning's news —
        # but when every tick since T-18h was dropped there is no full report
        # to check, and a solver-only alert is a poor substitute for the one
        # report the week is actually about. So the reminder's hour runs the
        # missing report instead, and the reminder gets a later tick — when
        # one lands in time; a promotion in the final hour may be the week's
        # last word, and a manager run started there can even finalize past
        # the deadline. Late beats never, and that close to it nothing beats
        # the full verdict. Auto's business only: a person naming the
        # reminder gets the reminder.
        if mode == REMINDER_MODE and not store.has_run(event.id, DEADLINE_MODE):
            mode = DEADLINE_MODE

    if store.has_run(event.id, mode) and not args.force:
        _explain(args.command, f"GW{event.id} {mode} has run already — use --force")
        return None, event.id
    return mode, event.id


def _round_end(
    client: FplClient, bootstrap: Bootstrap, event: Event, now: datetime
) -> datetime | None:
    """When the round just played had its last kickoff — the early scout's
    anchor — or None when the clock could not want it anyway.

    The fixtures are a second request, and most ticks have no use for them:
    inside the scout horizon the clock has its answer from the deadline
    alone, so the request is only made beyond it. No current round (the
    season's first week) is no round end.
    """
    if (event.deadline_time - now).total_seconds() / 3600 <= SCOUT_HORIZON_HOURS:
        return None
    current = bootstrap.current_event()
    if current is None:
        return None
    return last_kickoff(client.fixtures(), current=current.id, before=event.deadline_time)


def _explain(command: str, reason: str) -> None:
    """Say why nothing happened — unless nothing happening is the normal
    course of events, which for a job that runs every hour it is."""
    if command != AUTO:
        print(f"aigaffer: {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
