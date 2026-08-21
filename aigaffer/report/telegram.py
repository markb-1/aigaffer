"""Delivery: the report, cut into messages Telegram will accept.

A message tops out at 4096 characters, so a report goes as several. The cut
falls on a line boundary and never inside one, because the report is a list of
lines and half a line at the top of a message reads as a mistake; 4000 rather
than 4096 leaves room for the counting here to be about characters while
Telegram's is about UTF-16 units. Nothing is marked up: no ``parse_mode``, so
the report arrives as the plain text it was written as, and a stray asterisk in
a player's name cannot break a message or, worse, get it rejected.

The token is part of the URL Telegram publishes for its API, so it travels in
every request line and comes back inside any ``httpx`` error. That is httpx's
to say; this module never puts it anywhere else.
"""

from collections.abc import Iterator

import httpx

BASE_URL = "https://api.telegram.org"
MAX_CHARS = 4000


def send_report(
    token: str, chat_id: str, text: str, http: httpx.Client | None = None
) -> None:
    """Send ``text`` to ``chat_id`` as one message per chunk, in order.

    Raises ``httpx.HTTPStatusError`` on the first message Telegram refuses,
    which leaves the ones before it sent — a report that arrives truncated is
    more use than one that does not arrive.
    """
    client = http or httpx.Client(timeout=30.0)
    url = f"{BASE_URL}/bot{token}/sendMessage"
    for chunk in _chunks(text):
        response = client.post(url, json={"chat_id": chat_id, "text": chunk})
        response.raise_for_status()


def _chunks(text: str) -> list[str]:
    """``text`` as whole lines packed into pieces of at most ``MAX_CHARS``.

    Blank pieces are dropped: Telegram rejects an empty message, and a report
    that ends on a newline can otherwise leave one behind.
    """
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for line in _lines(text):
        if current and length + 1 + len(line) > MAX_CHARS:
            chunks.append("\n".join(current))
            current, length = [], 0
        length += len(line) + (1 if current else 0)
        current.append(line)
    chunks.append("\n".join(current))
    return [chunk for chunk in chunks if chunk.strip()]


def _lines(text: str) -> Iterator[str]:
    """The lines of ``text``, with any line too long for one message cut up.

    A single line longer than a message has to be broken somewhere; breaking
    it at the limit is the one cut that is guaranteed to make progress.
    """
    for line in text.split("\n"):
        if len(line) <= MAX_CHARS:
            yield line
            continue
        for start in range(0, len(line), MAX_CHARS):
            yield line[start : start + MAX_CHARS]
