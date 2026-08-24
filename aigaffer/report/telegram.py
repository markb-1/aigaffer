"""Delivery: the report, cut into messages Telegram will accept.

A message tops out at 4096 characters, so a report goes as several. The cut
falls on a line boundary and never inside one, because the report is a list of
lines and half a line at the top of a message reads as a mistake; 4000 rather
than 4096 leaves room for the counting here to be about characters while
Telegram's is about UTF-16 units — and, now that the chunks carry markup, for
a heading's tags on top of that.

The markup is Telegram HTML, made one line at a time: every line is escaped —
``&`` first, then ``<`` and ``>``, so an entity is never escaped twice — and a
markdown heading becomes a ``<b>`` line with its hashes dropped, which is the
difference between a report that reads like a report on the phone and one that
opens with a row of ``#``. Escaping is not optional prudence: the live report
contains the literal ``<- recommended``, which unescaped is an open tag and a
message Telegram rejects outright. Because every tag opens and closes on its
own line, no cut — between lines, or inside a line too long for one message,
which is cut by escaped cost so an entity is never severed either — can leave
a tag half-sent.

Markup is also a way to be rejected that plain text never had, so each chunk
remembers the plain lines it was made from: if Telegram turns the HTML down
with a 4xx, the same chunk goes again as that plain text, no ``parse_mode`` —
a report that arrives ugly beats one that does not arrive. Only if that copy
fails too does the error propagate, which leaves the chunks before it sent —
a report that arrives truncated is still more use than one that never comes.

The token is part of the URL Telegram publishes for its API, so it travels in
every request line and comes back inside any ``httpx`` error. That is httpx's
to say; this module never puts it anywhere else.
"""

from collections.abc import Iterator

import httpx

BASE_URL = "https://api.telegram.org"
MAX_CHARS = 4000

_HEADINGS = ("### ", "## ", "# ")
_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def send_report(
    token: str, chat_id: str, text: str, http: httpx.Client | None = None
) -> None:
    """Send ``text`` to ``chat_id`` as one message per chunk, in order.

    Each chunk goes as HTML; a 4xx answer to that is retried once, as the
    chunk's own plain text. Raises ``httpx.HTTPStatusError`` on the first
    chunk Telegram refuses both ways, which leaves the ones before it sent —
    a report that arrives truncated is more use than one that does not arrive.
    """
    client = http or httpx.Client(timeout=30.0)
    url = f"{BASE_URL}/bot{token}/sendMessage"
    for plain, html in _chunks(text):
        response = client.post(
            url, json={"chat_id": chat_id, "text": html, "parse_mode": "HTML"}
        )
        if 400 <= response.status_code < 500:
            response = client.post(url, json={"chat_id": chat_id, "text": plain})
        response.raise_for_status()


def _chunks(text: str) -> list[tuple[str, str]]:
    """``text`` as whole lines packed into ``(plain, html)`` pieces.

    Both renderings of a line stay in the same chunk, so a fallback resends
    exactly what the rejected message would have said. The budget is measured
    against the HTML, tags and entities included — conservative, since
    Telegram counts the text after parsing, but a count that can never send
    an oversized message is worth the handful of characters it wastes.

    Blank pieces are dropped: Telegram rejects an empty message, and a report
    that ends on a newline can otherwise leave one behind.
    """
    chunks: list[tuple[str, str]] = []
    plain_lines: list[str] = []
    html_lines: list[str] = []
    length = 0
    for line in _lines(text):
        html = _as_html(line)
        if plain_lines and length + 1 + len(html) > MAX_CHARS:
            chunks.append(("\n".join(plain_lines), "\n".join(html_lines)))
            plain_lines, html_lines, length = [], [], 0
        length += len(html) + (1 if plain_lines else 0)
        plain_lines.append(line)
        html_lines.append(html)
    chunks.append(("\n".join(plain_lines), "\n".join(html_lines)))
    return [chunk for chunk in chunks if chunk[0].strip()]


def _as_html(line: str) -> str:
    """``line`` as Telegram HTML: escaped, and bold if it was a heading.

    ``###`` is not something the renderer writes today, but a heading gaining
    a level should degrade to bold text, not to a line of hashes.
    """
    for hashes in _HEADINGS:
        if line.startswith(hashes):
            return f"<b>{_escape(line[len(hashes):])}</b>"
    return _escape(line)


def _escape(text: str) -> str:
    """``text`` with HTML's three special characters as entities.

    ``&`` goes first, or the escapes themselves would be escaped again.
    """
    for char, entity in _ESCAPES.items():
        text = text.replace(char, entity)
    return text


def _lines(text: str) -> Iterator[str]:
    """The lines of ``text``, with any line too long for one message cut up.

    A single line longer than a message has to be broken somewhere; breaking
    it at the limit is the one cut that is guaranteed to make progress. The
    limit is measured in escaped characters — an ``&`` costs five — so the
    cut can never land inside an entity: each piece escapes to at most
    ``MAX_CHARS``, and only a heading's ``<b></b>`` can carry a piece past
    that, into slack the 4096 ceiling still covers.
    """
    for line in text.split("\n"):
        if len(_escape(line)) <= MAX_CHARS:
            yield line
            continue
        piece: list[str] = []
        cost = 0
        for char in line:
            escaped = len(_ESCAPES.get(char, char))
            if cost + escaped > MAX_CHARS:
                yield "".join(piece)
                piece, cost = [], 0
            piece.append(char)
            cost += escaped
        yield "".join(piece)
