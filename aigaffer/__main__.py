"""``python -m aigaffer`` — the one way to run the bot.

``auto`` is what the schedule calls: it looks at the deadline, works out
which report the hour calls for and runs it, or says nothing and exits
cleanly, which is what it does most of the time. ``scout`` and ``deadline``
run a report by name for someone sitting at a keyboard, and unlike ``auto``
they explain themselves when they decide not to.

A gameweek gets one of each report. The store is what remembers that, so a
schedule that fires every six hours does not send the same thing four times;
``--force`` is how a person overrules it. ``--dry-run`` prints the report and
writes nothing anywhere, which is what makes it safe against the live API.
"""

import argparse
from datetime import UTC, datetime

from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.orchestrator import PipelineError, decide_mode, run_pipeline
from aigaffer.store import Store

AUTO = "auto"
COMMANDS = (AUTO, "scout", "deadline")
DB_NAME = "aigaffer.db"


def main(argv: list[str] | None = None) -> int:
    """Run one command and return the process exit code."""
    args = _parse_args(argv)
    cfg = Config.from_env()
    client = FplClient()
    store = Store(cfg.state_dir / DB_NAME)

    mode = _mode(args, client, store)
    if mode is None:
        return 0

    try:
        report = run_pipeline(
            cfg, client, store, mode, send=not args.dry_run, save=not args.dry_run
        )
    except PipelineError as error:
        print(f"aigaffer: {error}")
        return 1

    if args.dry_run:
        print(report)
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m aigaffer",
        description="Report on the next Fantasy Premier League gameweek.",
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
    return parser.parse_args(argv)


def _mode(args: argparse.Namespace, client: FplClient, store: Store) -> str | None:
    """The mode to run, or None to stand down.

    Deciding costs a bootstrap fetch — for the deadline ``auto`` reads and
    the gameweek the store is asked about — so a mode named on the command
    line with ``--force`` skips the question entirely and gets on with it.
    """
    if args.command != AUTO and args.force:
        return args.command

    event = client.bootstrap().next_event()
    if event is None:
        _explain(args.command, "the API has no gameweek ahead")
        return None

    mode = args.command
    if mode == AUTO:
        chosen = decide_mode(datetime.now(UTC), event.deadline_time)
        if chosen is None:
            return None
        mode = chosen

    if store.has_run(event.id, mode) and not args.force:
        _explain(args.command, f"GW{event.id} {mode} has run already — use --force")
        return None
    return mode


def _explain(command: str, reason: str) -> None:
    """Say why nothing happened — unless nothing happening is the normal
    course of events, which for a job that runs every six hours it is."""
    if command != AUTO:
        print(f"aigaffer: {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
