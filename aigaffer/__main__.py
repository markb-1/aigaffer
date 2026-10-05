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
import os
from datetime import UTC, datetime
from functools import partial

import httpx

from aigaffer.backtest import backtest_gw, finished_gameweeks
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
from aigaffer.report.telegram import send_report
from aigaffer.statesync import STATE_LOCK, GitStateSync
from aigaffer.store import DB_NAME, EXECUTED_DIR, Store

AUTO = "auto"
BACKTEST = "backtest"
INBOX = "inbox"
COMMANDS = (AUTO, "early", "scout", "deadline", "reminder", BACKTEST, INBOX)

SCHEDULE_STAGE = "reading the schedule"
BACKTEST_STAGE = "the backtest"

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

    cfg = Config.from_env()
    _manager_notice(cfg)
    store = Store(cfg.state_dir / DB_NAME)
    alert_to = None if args.dry_run else cfg

    try:
        mode, event_id = _mode(args, client, store)
    except httpx.HTTPError as error:
        return _failed(alert_to, error, event_id=None, stage=SCHEDULE_STAGE)
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
        help="which finished gameweek to backtest (default: the most recent)",
    )
    return parser.parse_args(argv)


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
    :mod:`aigaffer.inbox` knows nothing of git or of the store.
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
    )
    return run_inbox(cfg.telegram_token, cfg.telegram_chat_id, directory, handler)


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
        # but when every tick since T-24h was dropped there is no full report
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
