"""The message log: what the owner texted the bot, kept on his own box.

``$AIGAFFER_INBOX_DIR/messages/YYYY-MM.jsonl`` (default ``~/.aigaffer/``),
beside the offset, the marker and the gaffer's threads: outside the checkout,
never committed. The repo is public, and these are the owner's own words —
which is also why the journal's line for an update (:data:`aigaffer.inbox.
ROUTED`) carries the route and never the text: this file is where the text
is, and only here.

**One line per update from the owner's chat**, written before the update is
handled, so a message whose handler falls over is still on record; a second
line follows if it does. Another chat's updates are never written: the
journal says they were ignored, and that is all. A crash that re-handles an
update writes its line again, under the same ``update_id`` — readers dedupe.

**A whitelist, never the update.** A line is built field by field
(:meth:`MessageLog.received`): when it was read, when it was sent, the update
id, what kind of message it was, the text if it was text, and the route and
reason code the inbox chose. Nothing else from the message — not the sender,
not the chat, not a caption, not a file id — is ever serialised. A
non-text message is recorded by its kind alone.

**Never in a checkout.** A directory that resolves (symlinks followed) inside
this repository, or any other checkout it is told of, is refused with one
journal line: a misconfigured ``AIGAFFER_INBOX_DIR`` must not put the owner's
words where a commit could pick them up.

**Private.** The folder is kept at 0700 (made so, or put back if it was
looser) and each file is created 0600 through ``os.open`` — never through a
symlink — so there is no moment when it exists with looser permissions.

**Kept three months** (:data:`RETENTION_MONTHS`): this month's file and the
three before it, so at least three full months. Each write drops older month
files; a name that is not ``YYYY-MM.jsonl`` is never touched.

**Never the inbox's problem.** Every write swallows its own failure and says
so in one journal line naming the class — never the words — so a full disk
or a bad permission costs the record, never the reply, the offset or the
retry.
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

MESSAGES_DIR = "messages"
RETENTION_MONTHS = 3

# This repository, found from this file: the checkout the inbox runs from on
# the box, and the one place the log must never land.
REPO_ROOT = Path(__file__).resolve().parents[1]

NOT_LOGGED = "aigaffer inbox: message not logged ({reason})"
INSIDE_CHECKOUT = "inside-checkout"

RECEIVED = "received"
FAILED = "failed"
TEXT = "text"
OTHER = "other"
# The kinds of non-text message Telegram sends, in the order they are looked
# for; anything else is "other". Only the kind is kept, never its contents.
KINDS = (
    "photo", "sticker", "voice", "video", "video_note", "animation", "audio",
    "document", "location", "venue", "contact", "poll", "dice",
)


def content_of(message: dict) -> tuple[str, str | None]:
    """What kind of message this is, and its text when it is text."""
    text = message.get("text")
    if isinstance(text, str):
        return TEXT, text
    return next((kind for kind in KINDS if kind in message), OTHER), None


class MessageLog:
    """The owner's texts, one JSON line each, in month files."""

    def __init__(
        self,
        directory: Path,
        *,
        checkouts: tuple[Path, ...] = (),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.directory = directory
        self.checkouts = (REPO_ROOT, *checkouts)
        self.now = now

    def received(
        self, update_id: int, sent_at: datetime, message: dict, route: str, reason: str
    ) -> None:
        """Log one update from the owner's chat, before it is handled."""
        try:
            now = self.now()
            content, text = content_of(message)
            entry: dict = {
                "event": RECEIVED,
                "ts": now.isoformat(),
                "sent_at": sent_at.isoformat(),
                "update_id": update_id,
                "content": content,
            }
            if text is not None:
                entry["text"] = text
            entry["route"] = route
            entry["reason"] = reason
            self._append(entry, now)
        except Exception as error:  # the record is lost, never the reply
            print(NOT_LOGGED.format(reason=type(error).__name__))

    def failed(self, update_id: int, error: str) -> None:
        """Log that handling the update raised ``error`` (a class name)."""
        try:
            now = self.now()
            self._append({
                "event": FAILED, "ts": now.isoformat(),
                "update_id": update_id, "error": error,
            }, now)
        except Exception as failure:
            print(NOT_LOGGED.format(reason=type(failure).__name__))

    def _append(self, entry: dict, now: datetime) -> None:
        """One line into ``now``'s month file — the same ``now`` as the
        line's own timestamp, so a line never lands in the wrong month."""
        folder = self.directory / MESSAGES_DIR
        resolved = folder.resolve()
        if any(resolved.is_relative_to(checkout.resolve()) for checkout in self.checkouts):
            print(NOT_LOGGED.format(reason=INSIDE_CHECKOUT))
            return
        os.makedirs(folder, mode=0o700, exist_ok=True)
        os.chmod(folder, 0o700)
        path = folder / f"{now:%Y-%m}.jsonl"
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        # backslashreplace: a lone surrogate in a text is kept escaped, not
        # the reason the whole line is lost.
        with os.fdopen(descriptor, "a", encoding="utf-8", errors="backslashreplace") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        try:
            prune(folder, now)
        except Exception:  # an old month left behind is retried on the next write
            pass


def prune(folder: Path, now: datetime) -> None:
    """Drop month files more than :data:`RETENTION_MONTHS` before ``now``'s
    month: November keeps itself and August to October. A file whose name is
    not a real ``YYYY-MM.jsonl`` is left alone."""
    oldest = now.year * 12 + now.month - 1 - RETENTION_MONTHS
    for path in folder.glob("*.jsonl"):
        month = _month_index(path.stem)
        if month is not None and month < oldest:
            path.unlink()


def _month_index(stem: str) -> int | None:
    """``YYYY-MM`` as a count of months, or None for any other name."""
    year, dash, month = stem.partition("-")
    if not (dash and len(year) == 4 and len(month) == 2 and year.isdigit() and month.isdigit()):
        return None
    if not 1 <= int(month) <= 12:
        return None
    return int(year) * 12 + int(month) - 1
