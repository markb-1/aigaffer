"""The inbox: what the owner texts the bot, read once a minute.

``python -m aigaffer inbox`` is run by a one-minute systemd timer on the VM
(``deploy/aigaffer-inbox.timer``) and by nothing else: GitHub Actions never
reads Telegram. Each tick asks Telegram for the messages since the last one
handled, obeys the configured chat and nobody else, answers, and exits — one
HTTPS call on a quiet minute, and silence on stdout.

Four things can be said to it. "Transfers made" records the latest verdict
as entered (:mod:`aigaffer.recording`). "Wildcard?" and "Free hit?" — and
the words they normalise to — run a chip what-if
(:mod:`aigaffer.whatif_handler`), which replies for itself, in more than one
message. "Chip forecast" answers at once, from the newest saved report, with
when the bot currently plans to play each chip (:mod:`aigaffer.chip_forecast`):
no solve and no gaffer, so a read of a few rows and one text back. Anything else — "help", a typo, a sticker — gets the help.

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

A chip what-if is the exception: it writes the marker itself, first thing,
so an exception anywhere inside it finds the marker already set and the
update is let go at once — a what-if costs minutes and a gaffer's bill, and
is never re-run behind the owner's back; the apology tells him to ask again.
A marker that is still there when the next minute starts can then only mean
the process was killed mid-what-if, and the handler picks up what it left
(see :mod:`aigaffer.whatif_handler`).
"""

import os
import string
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.report.telegram import get_updates, send_message

INBOX_ENV = "AIGAFFER_INBOX_DIR"
OFFSET_FILE = "offset"
FAILED_FILE = "failed"

TRANSFERS_MADE = "transfers made"
CHIP_FORECAST = "chip forecast"
# Every word that asks for a chip what-if, normalised, and the chip it asks
# about. The keyboard's "Wildcard?" and "Free hit?" normalise to the first and
# third (``normalise`` strips the question mark).
CHIP_KINDS = {
    "wildcard": WILDCARD,
    "wc": WILDCARD,
    "free hit": FREE_HIT,
    "freehit": FREE_HIT,
    "fh": FREE_HIT,
}
CHIP_WORDS = frozenset(CHIP_KINDS)

HELP = (
    "Tap a button below, or text me:\n"
    "• Transfers made — straight after you enter the latest recommendation"
    " exactly, its transfers and its chip. Later reports this gameweek then"
    " work from that squad. Don't send it if you changed anything.\n"
    "• Wildcard? / Free hit? — what playing that chip this gameweek would do:"
    " numbers in a few minutes, the gaffer's view after. Nothing is recorded —"
    " if you play it, don't send Transfers made.\n"
    "• Chip forecast — when the bot currently plans to play each chip, from the"
    " latest report. Instant.\n"
    "• help — this message."
)
SOMETHING_WRONG = "Something went wrong recording that — try again in a minute."
WHATIF_WRONG = "Something went wrong with that what-if — ask again in a minute."
WHATIFS_OFF = "Chip what-ifs aren't set up on this box."
CHIP_FORECAST_OFF = "Chip forecast is not configured."
FORECAST_WRONG = "Something went wrong with that forecast — try again in a minute."

# What a handler answers with: one text, sent with the keyboard like every
# reply (send_message always attaches it, hence no keyboard keyword). A
# what-if is handed one of these and sends its own messages through it — an
# ack, the numbers, the gaffer's view — because it has more than one thing to
# say and minutes between them.
Reply = Callable[[str], None]
ChipWhatIf = Callable[[str, int, datetime, Reply], str | None]

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
    command: str,
    update_id: int,
    sent_at: datetime,
    reply: Reply,
    transfers_made: Callable[[datetime], str],
    chip_whatif: ChipWhatIf | None = None,
    chip_forecast: Callable[[datetime], str] | None = None,
) -> str | None:
    """The reply to one normalised message, or None when the handler has
    already replied for itself — which is what a chip what-if does.

    ``chip_whatif`` None is a box wired without what-ifs (and most tests):
    the chip words then get a line saying so, never the help, so a tap on the
    keyboard is not answered with a list that offers the very button tapped.
    The same goes for ``chip_forecast`` None and the forecast.
    """
    if command == TRANSFERS_MADE:
        return transfers_made(sent_at)
    if command == CHIP_FORECAST:
        if chip_forecast is None:
            return CHIP_FORECAST_OFF
        return chip_forecast(sent_at)
    kind = CHIP_KINDS.get(command)
    if kind is not None:
        if chip_whatif is None:
            return WHATIFS_OFF
        return chip_whatif(kind, update_id, sent_at, reply)
    return HELP


def run_inbox(
    token: str | None,
    chat_id: str | None,
    directory: Path,
    transfers_made: Callable[[datetime], str],
    chip_whatif: ChipWhatIf | None = None,
    chip_forecast: Callable[[datetime], str] | None = None,
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

    def reply(text: str) -> None:
        send(token, chat_id, text)

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
        # The apology fits the command: a what-if that fell over is asked
        # again, a recording is tried again. Set before anything can raise,
        # so a message too malformed to read still gets the general one.
        apology = SOMETHING_WRONG
        try:
            ours = (
                message is not None
                and str(message.get("chat", {}).get("id")) == str(chat_id)
            )
            if ours:
                sent_at = datetime.fromtimestamp(message["date"], UTC)
                command = normalise(message.get("text") or "")
                if command in CHIP_KINDS:
                    apology = WHATIF_WRONG
                elif command == CHIP_FORECAST:
                    apology = FORECAST_WRONG
                answer = dispatch(
                    command, uid, sent_at, reply, transfers_made, chip_whatif,
                    chip_forecast,
                )
                if answer is not None:
                    send(token, chat_id, answer)
        except Exception as error:  # any failure; one bad text must not wedge us
            print(FAILED.format(reason=type(error).__name__))
            _apologise(token, chat_id, send, apology)
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
    token: str, chat_id: str, send: Callable[[str, str, str], None], text: str
) -> None:
    """Best effort: a failed apology is not a second failure to handle."""
    try:
        send(token, chat_id, text)
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


def mark_failed(directory: Path, update_id: int) -> None:
    """Set the failure marker to ``update_id`` — how a what-if says, before it
    does anything that can raise, "if you find this again, I was killed"."""
    _write_int(directory / FAILED_FILE, update_id)


def marked_failed(directory: Path) -> int | None:
    """The update id the failure marker holds, or None."""
    return _read_int(directory / FAILED_FILE)


def confirm_handled(directory: Path, update_id: int) -> None:
    """Move the offset to ``update_id``: Telegram stops handing it out. A
    resumed what-if does this itself before resuming, so a second kill
    cannot bring it round a third time; the inbox's own write of the same
    value afterwards is harmless."""
    _write_int(directory / OFFSET_FILE, update_id)
