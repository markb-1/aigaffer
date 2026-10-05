"""Delivery: the report, cut into messages Telegram will accept.

A message tops out at 4096 characters, so a report goes as several. The cut
falls on a line boundary and never inside one, because the report is a list of
lines and half a line at the top of a message reads as a mistake; the budget
is measured in UTF-16 units, because that is what Telegram counts — an emoji
outside the Basic Multilingual Plane is one Python character but two units —
and 4000 rather than 4096 leaves room, now that the chunks carry markup, for
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
with a 400 — "can't parse entities" comes back as exactly that — the same
chunk goes again as that plain text, no ``parse_mode``. A report that arrives
ugly beats one that does not arrive. Any other status raises as it always
did: a 401 or 404 would fail the resend identically, and a 429 resent is a
second request straight into the rate limiter. Only if the plain copy fails
too does the error propagate, which leaves the chunks before it sent — a
report that arrives truncated is still more use than one that never comes.

The token is part of the URL Telegram publishes for its API, so it travels in
every request line and comes back inside any ``httpx`` error. That is httpx's
to say; this module never puts it anywhere else.

The bot also reads. :func:`get_updates` is the VM inbox's one-minute poll
(``python -m aigaffer inbox``), and :func:`send_message` is how it answers; the
GitHub workflow never calls either.
"""

from collections.abc import Iterator

import httpx

BASE_URL = "https://api.telegram.org"
MAX_CHARS = 4000

# The owner's four commands as buttons under his message box: a Telegram
# *reply keyboard*, kept on screen (``is_persistent``) and sized to its labels
# (``resize_keyboard``). Tapping a button sends its label as an ordinary text
# message, which is exactly what the inbox already reads — so there is no
# callback handling and no new update type, and the labels have to stay words
# the inbox's dispatcher understands.
KEYBOARD: dict = {
    "keyboard": [
        [{"text": "Transfers made"}],
        [{"text": "Wildcard?"}, {"text": "Free hit?"}],
        [{"text": "Help"}],
    ],
    "is_persistent": True,
    "resize_keyboard": True,
}

_HEADINGS = ("### ", "## ", "# ")
_ESCAPES = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def send_report(
    token: str,
    chat_id: str,
    text: str,
    http: httpx.Client | None = None,
    reply_markup: dict | None = None,
) -> None:
    """Send ``text`` to ``chat_id`` as one message per chunk, in order.

    Each chunk goes as HTML; a 400 — the status "can't parse entities" comes
    back as — is retried once, as the chunk's own plain text, while any other
    status raises untried: a plain copy cannot fix a bad token or appease a
    rate limiter. Raises ``httpx.HTTPStatusError`` on the first chunk Telegram
    refuses for good, which leaves the ones before it sent — a report that
    arrives truncated is more use than one that does not arrive.

    ``reply_markup``, when given, rides on the last chunk only — the phone
    redraws a keyboard each time it arrives, and one per report is enough —
    on the HTML attempt and on its plain-text resend alike, so a chunk that
    fell back to plain text does not lose the buttons. None leaves every
    request body exactly as it always was.
    """
    client = http or httpx.Client(timeout=30.0)
    url = f"{BASE_URL}/bot{token}/sendMessage"
    chunks = _chunks(text)
    for index, (plain, html) in enumerate(chunks):
        extra = {"reply_markup": reply_markup} if reply_markup and index == len(chunks) - 1 else {}
        response = client.post(
            url,
            json={"chat_id": chat_id, "text": html, "parse_mode": "HTML", **extra},
        )
        if response.status_code == 400:
            response = client.post(
                url, json={"chat_id": chat_id, "text": plain, **extra}
            )
        response.raise_for_status()


def get_updates(
    token: str, offset: int | None, http: httpx.Client | None = None
) -> list[dict]:
    """The messages waiting for the bot, oldest first, from ``offset`` on.

    ``offset`` is one past the last update already handled: Telegram forgets
    everything before it, which is how the inbox confirms what it has done.
    None asks for everything still held (about a day's worth). Messages only
    — an edit to an old message is not a new instruction — and ``timeout``
    0, because a systemd tick that long-polls is a tick that overlaps the
    next one. Raises ``httpx.HTTPStatusError`` on any refusal; the caller
    reads that as "unreachable this minute" and says nothing more, because
    the error text carries the token in its URL.
    """
    client = http or httpx.Client(timeout=30.0)
    payload: dict = {"timeout": 0, "allowed_updates": ["message"]}
    if offset is not None:
        payload["offset"] = offset
    response = client.post(f"{BASE_URL}/bot{token}/getUpdates", json=payload)
    response.raise_for_status()
    return list(response.json().get("result", []))


def send_message(
    token: str, chat_id: str, text: str, http: httpx.Client | None = None
) -> None:
    """A reply to the owner, sent the way a report is.

    The inbox's replies are short, but they carry names and arrows and the
    odd ``<``, so they go through exactly the chunking, escaping and plain
    fallback :func:`send_report` already proves — one path, not two. Every
    reply carries the :data:`KEYBOARD`, so the buttons are there whenever he
    has just been answered.
    """
    send_report(token, chat_id, text, http=http, reply_markup=KEYBOARD)


def _utf16_len(text: str) -> int:
    """``text`` measured as Telegram measures it: in UTF-16 code units.

    A character inside the Basic Multilingual Plane is one unit; an emoji
    beyond it is a surrogate pair, two. Counting Python characters instead
    would let a chunk heavy with emoji pass the budget here and blow the
    4096 there — and the plain fallback, same text, would fail identically.
    """
    return len(text.encode("utf-16-le")) // 2


def _chunks(text: str) -> list[tuple[str, str]]:
    """``text`` as whole lines packed into ``(plain, html)`` pieces.

    Both renderings of a line stay in the same chunk, so a fallback resends
    exactly what the rejected message would have said. The budget is measured
    against the HTML in UTF-16 units, tags and entities included —
    conservative, since Telegram counts the text after parsing, but a count
    that can never send an oversized message is worth the handful of
    characters it wastes.

    Blank pieces are dropped: Telegram rejects an empty message, and a report
    that ends on a newline can otherwise leave one behind.
    """
    chunks: list[tuple[str, str]] = []
    plain_lines: list[str] = []
    html_lines: list[str] = []
    length = 0
    for line in _lines(text):
        html = _as_html(line)
        if plain_lines and length + 1 + _utf16_len(html) > MAX_CHARS:
            chunks.append(("\n".join(plain_lines), "\n".join(html_lines)))
            plain_lines, html_lines, length = [], [], 0
        length += _utf16_len(html) + (1 if plain_lines else 0)
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
    limit is measured in escaped UTF-16 units — an ``&`` costs five, an
    astral emoji two, and the two never compound because nothing that needs
    escaping lies outside the Basic Multilingual Plane — so the cut can never
    land inside an entity, and every character crosses whole: each piece
    escapes to at most ``MAX_CHARS`` units, and only a heading's ``<b></b>``
    can carry a piece past that, into slack the 4096 ceiling still covers.
    """
    for line in text.split("\n"):
        if _utf16_len(_escape(line)) <= MAX_CHARS:
            yield line
            continue
        piece: list[str] = []
        cost = 0
        for char in line:
            escaped = _utf16_len(_ESCAPES.get(char, char))
            if cost + escaped > MAX_CHARS:
                yield "".join(piece)
                piece, cost = [], 0
            piece.append(char)
            cost += escaped
        yield "".join(piece)
