"""The inbox: what the owner texts the bot, read once a minute.

``python -m aigaffer inbox`` is run by a one-minute systemd timer on the VM
(``deploy/aigaffer-inbox.timer``) and by nothing else: GitHub Actions never
reads Telegram. Each tick asks Telegram for the messages since the last one
handled, obeys the configured chat and nobody else, answers, and exits — one
HTTPS call on a quiet minute, and silence on stdout.

Three things can be said to it. "Transfers made" records the latest verdict
as entered (:mod:`aigaffer.recording`). The chip words are the chip
what-ifs' (a later change); until then they get a line saying so. Anything
else — "help", a typo, a sticker — gets the help.

**Where it is.** The last handled ``update_id`` lives in
``$AIGAFFER_INBOX_DIR/offset`` (default ``~/.aigaffer/``), outside the
checkout and never committed. It is written after an update is fully
handled, so a crash re-handles that update — which is safe because every
handler is idempotent or reply-only. Losing the file re-reads at most the
day of updates Telegram keeps.

**One bad message.** An update that raises is retried once: the first
failure writes its id to ``$AIGAFFER_INBOX_DIR/failed`` and leaves the offset,
so the next minute tries again; a second failure on the same id lets it go
and carries on. The marker is a file because consecutive minutes are
separate processes. Either way the owner is told something went wrong, and
the log gets the exception's class and never its words.
"""

import os
import string
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aigaffer.report.telegram import get_updates, send_message

INBOX_ENV = "AIGAFFER_INBOX_DIR"
OFFSET_FILE = "offset"
FAILED_FILE = "failed"

TRANSFERS_MADE = "transfers made"
CHIP_WORDS = frozenset({"wildcard", "wc", "free hit", "freehit", "fh"})

HELP = (
    "Tap a button below, or text me:\n"
    "• Transfers made — straight after you enter the latest recommendation"
    " exactly, its transfers and its chip. Later reports this gameweek then"
    " work from that squad. Don't send it if you changed anything.\n"
    "• wildcard / free hit — chip what-ifs (coming soon).\n"
    "• help — this message."
)
CHIPS_SOON = "Chip what-ifs (wildcard, free hit) are coming soon — nothing to show yet."
SOMETHING_WRONG = "Something went wrong recording that — try again in a minute."

NOT_CONFIGURED = "aigaffer inbox: telegram not configured — nothing to read"
UNREACHABLE = "aigaffer inbox: telegram unreachable ({reason})"
FAILED = "aigaffer inbox: an update failed ({reason})"
SKIPPED = "aigaffer inbox: an update skipped ({reason})"


def inbox_dir() -> Path:
    """Where the inbox keeps its offset, marker and locks: never the repo."""
    configured = os.environ.get(INBOX_ENV)
    return Path(configured) if configured else Path.home() / ".aigaffer"


def normalise(text: str) -> str:
    """Lowercase, one space between words, no trailing punctuation."""
    collapsed = " ".join(text.lower().split())
    return collapsed.rstrip(string.punctuation + " ")


def dispatch(
    command: str, sent_at: datetime, transfers_made: Callable[[datetime], str]
) -> str:
    """The reply to one normalised message."""
    if command == TRANSFERS_MADE:
        return transfers_made(sent_at)
    if command in CHIP_WORDS:
        return CHIPS_SOON
    return HELP


def run_inbox(
    token: str | None,
    chat_id: str | None,
    directory: Path,
    transfers_made: Callable[[datetime], str],
    *,
    get: Callable[[str, int | None], list[dict]] = get_updates,
    send: Callable[[str, str, str], None] = send_message,
) -> int:
    """Handle every waiting message once, in order, and return 0.

    Always 0: a minute tick that failed is retried by the next one, and a
    systemd unit that fails every minute while Telegram is down is a log
    full of noise saying nothing new.
    """
    if not (token and chat_id):
        print(NOT_CONFIGURED)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    last = _read_int(directory / OFFSET_FILE)
    try:
        # Broad on purpose: a transport error (httpx) or a malformed payload
        # (ValueError, AttributeError, TypeError, KeyError) both mean "try
        # next minute", and httpx's own message carries the bot-token URL, so
        # only the class name is ever printed.
        fetched = list(get(token, None if last is None else last + 1))
    except Exception as error:
        print(UNREACHABLE.format(reason=type(error).__name__))
        return 0

    # An update without an integer id cannot be handled, retried or passed —
    # the id is what the offset is made of — so it is skipped with one line,
    # and the rest are handled in id order. Sorting or reading the id inside
    # the loop instead would fail the whole batch on it, every minute, for
    # ever: the offset could never move past it.
    numbered = []
    for update in fetched:
        fault = _id_fault(update)
        if fault is None:
            numbered.append(update)
        else:
            print(SKIPPED.format(reason=fault))
    for update in sorted(numbered, key=lambda u: u["update_id"]):
        uid = update["update_id"]
        message = update.get("message")
        try:
            ours = (
                message is not None
                and str(message.get("chat", {}).get("id")) == str(chat_id)
            )
            if ours:
                sent_at = datetime.fromtimestamp(message["date"], UTC)
                reply = dispatch(
                    normalise(message.get("text") or ""), sent_at, transfers_made
                )
                send(token, chat_id, reply)
        except Exception as error:  # any failure; one bad text must not wedge us
            print(FAILED.format(reason=type(error).__name__))
            _apologise(token, chat_id, send)
            if _read_int(directory / FAILED_FILE) != uid:
                _write_int(directory / FAILED_FILE, uid)
                return 0
        _write_int(directory / OFFSET_FILE, uid)
        (directory / FAILED_FILE).unlink(missing_ok=True)
    return 0


def _id_fault(update: object) -> str | None:
    """None for an update with an integer ``update_id``, else the class of
    fault to log: KeyError for a dict without one, TypeError for anything
    else (not a dict at all, or an id that is not an int — a bool counts as
    not one)."""
    if not isinstance(update, dict):
        return TypeError.__name__
    if "update_id" not in update:
        return KeyError.__name__
    uid = update.get("update_id")
    if not isinstance(uid, int) or isinstance(uid, bool):
        return TypeError.__name__
    return None


def _apologise(
    token: str, chat_id: str, send: Callable[[str, str, str], None]
) -> None:
    """Best effort: a failed apology is not a second failure to handle."""
    try:
        send(token, chat_id, SOMETHING_WRONG)
    except Exception:  # the apology is the last resort
        pass


def _read_int(path: Path) -> int | None:
    """The integer in ``path``, or None for a file missing, empty or garbled."""
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _write_int(path: Path, value: int) -> None:
    """Write atomically: a crash mid-write must not leave half a number."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(str(value))
    os.replace(temporary, path)
