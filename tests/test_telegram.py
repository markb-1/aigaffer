"""Tests for Telegram delivery.

Every request is served by an ``httpx.MockTransport`` that keeps what it was
handed, so the assertions are about the wire: one POST per chunk, to the
sendMessage URL for the token, carrying the chat id and a piece of the report
marked up as Telegram HTML — headings bold, everything else escaped — and,
when Telegram turns a chunk down with a 400, the same chunk again as the
plain text it came from. Only a 400 earns that resend; any other status
raises as it stands.

The markup is line-by-line, which is what the chunking tests lean on: a tag
opened on one line is closed on the same line, so no cut — between lines or
inside one — can sever a tag or an entity, and the tests hold the converter
to that where it is tightest, a line that is nothing but ampersands.

The report itself is lines, and the split has to keep them: a message that
begins mid-sentence is worse than one more message. The exception is a line
longer than a whole message, which can only be cut.
"""

import json

import httpx
import pytest

from aigaffer.report.telegram import (
    KEYBOARD,
    _utf16_len,
    get_updates,
    send_message,
    send_report,
)

TOKEN = "123456:fake-bot-token"
CHAT_ID = "42"
SEND_MESSAGE_URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
LIMIT = 4000

# 90 lines of 99 characters: 8999 characters once joined, which is three
# messages' worth and no line long enough to need cutting.
LONG_LINES = [f"{i:03d} " + "x" * 95 for i in range(90)]
LONG_REPORT = "\n".join(LONG_LINES)


def send(text: str, status: int = 200) -> list[httpx.Request]:
    """Send ``text`` through a mock transport; return the requests it saw."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={"ok": status == 200})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_report(TOKEN, CHAT_ID, text, http=http)
    return seen


def send_html_rejected(text: str, plain_status: int = 200) -> list[httpx.Request]:
    """Send ``text`` while Telegram 400s every HTML chunk; return the requests.

    The plain resends answer ``plain_status``, so the same helper covers both
    the fallback that lands and the one that fails too.
    """
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if json.loads(request.content).get("parse_mode") == "HTML":
            return httpx.Response(400, json={"ok": False})
        return httpx.Response(plain_status, json={"ok": plain_status == 200})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_report(TOKEN, CHAT_ID, text, http=http)
    return seen


def payloads(requests: list[httpx.Request]) -> list[dict]:
    return [json.loads(request.content) for request in requests]


def texts(requests: list[httpx.Request]) -> list[str]:
    return [payload["text"] for payload in payloads(requests)]


def test_short_report_is_one_post():
    requests = send("Hello, gaffer.")
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == SEND_MESSAGE_URL
    assert json.loads(requests[0].content) == {
        "chat_id": CHAT_ID,
        "text": "Hello, gaffer.",
        "parse_mode": "HTML",
    }


def test_report_is_sent_verbatim():
    assert texts(send("Hi\n")) == ["Hi\n"]


def test_every_chunk_is_posted_as_html():
    assert all(p["parse_mode"] == "HTML" for p in payloads(send(LONG_REPORT)))


def test_headings_arrive_bold_with_the_hashes_dropped():
    report = "# AI Gaffer — GW2 scout\n## Do this\n### Fine print\nA plain line"
    assert texts(send(report)) == [
        "<b>AI Gaffer — GW2 scout</b>\n<b>Do this</b>\n<b>Fine print</b>\nA plain line"
    ]


def test_a_hash_without_a_space_is_not_a_heading():
    # "#1 in the form table" is prose that happens to open with a hash.
    assert texts(send("#1 in the form table")) == ["#1 in the form table"]


def test_the_recommended_arrow_is_escaped_rather_than_read_as_a_tag():
    # The live report marks the picked plan with this exact string; unescaped,
    # "<- recommended" opens a tag Telegram rejects the whole message over.
    line = "- Plan A: +4.2 xP  <- recommended"
    assert texts(send(line)) == ["- Plan A: +4.2 xP  &lt;- recommended"]


def test_an_ampersand_in_a_name_is_escaped_first():
    # Escaping & after < would turn &lt; into &amp;lt; — the order is the test.
    assert texts(send("Brighton & Hove Albion <3")) == [
        "Brighton &amp; Hove Albion &lt;3"
    ]


def test_a_heading_is_escaped_inside_its_bold_tags():
    assert texts(send("# Ins & outs <- do these")) == [
        "<b>Ins &amp; outs &lt;- do these</b>"
    ]


def test_long_report_is_split_into_three_posts():
    requests = send(LONG_REPORT)
    assert len(requests) == 3
    assert all(str(request.url) == SEND_MESSAGE_URL for request in requests)
    assert all(json.loads(r.content)["chat_id"] == CHAT_ID for r in requests)
    assert all(len(chunk) <= LIMIT for chunk in texts(requests))


def test_long_report_is_split_on_line_boundaries():
    chunks = texts(send(LONG_REPORT))
    assert [line for chunk in chunks for line in chunk.split("\n")] == LONG_LINES
    assert "\n".join(chunks) == LONG_REPORT


def test_line_longer_than_a_message_is_cut_to_fit():
    chunks = texts(send("y" * 9000))
    assert [len(chunk) for chunk in chunks] == [LIMIT, LIMIT, 1000]
    assert "".join(chunks) == "y" * 9000


def test_a_cut_never_severs_an_entity():
    # The worst case for the cutter: a line that is nothing but ampersands,
    # where every character becomes a five-character entity. Every chunk must
    # be whole entities, and the plain text they stand for must survive intact.
    chunks = texts(send("&" * 5000))
    assert len(chunks) > 1
    assert all(set(chunk.split("&amp;")) == {""} for chunk in chunks)
    assert "".join(chunks).replace("&amp;", "&") == "&" * 5000


def test_a_cut_never_severs_a_heading_tag():
    # A heading longer than a message is cut like any line; only the piece
    # that still starts with the hashes is a heading, so every <b> that opens
    # in a chunk closes in the same chunk.
    chunks = texts(send("# " + "H" * 4500))
    assert len(chunks) > 1
    assert all(chunk.count("<b>") == chunk.count("</b>") for chunk in chunks)
    assert chunks[0].startswith("<b>") and chunks[0].endswith("</b>")


def test_an_emoji_heavy_line_is_budgeted_in_utf16_units():
    # 🔥 is one Python character but two UTF-16 units, and UTF-16 units are
    # what Telegram counts: 3000 of them measured as characters would look
    # like one message and arrive as 6000 units, over the ceiling — and the
    # plain fallback, same text, would be rejected identically. Every chunk
    # must fit the budget as Telegram measures it, and a Python string cannot
    # hold half a surrogate pair, so fitting is proof no cut split one.
    line = "🔥" * 3000
    chunks = texts(send(line))
    assert len(chunks) > 1
    assert all(_utf16_len(chunk) <= LIMIT for chunk in chunks)
    assert "".join(chunks) == line


def test_blank_report_sends_nothing():
    assert send("") == []
    assert send("\n\n") == []


def test_a_rejected_chunk_is_resent_as_readable_plain_text():
    report = "# Do this\nSign Ferrer  <- recommended\nBrighton & Hove away"
    requests = send_html_rejected(report)

    assert [p.get("parse_mode") for p in payloads(requests)] == ["HTML", None]
    # The resend is the original lines — hashes, arrow and ampersand as
    # written, no entities — because this copy is for reading, not parsing.
    assert texts(requests)[1] == report


def test_a_rejected_chunk_falls_back_before_the_next_chunk_is_sent():
    marked = "\n".join(f"<{line}>" for line in LONG_LINES)
    requests = send_html_rejected(marked)

    modes = [p.get("parse_mode") for p in payloads(requests)]
    assert modes == ["HTML", None] * (len(requests) // 2)
    # Each plain resend carries the same lines as the HTML chunk it replaces,
    # so the report still arrives whole and in order.
    plain = [text for p, text in zip(payloads(requests), texts(requests)) if "parse_mode" not in p]
    assert "\n".join(plain) == marked


def test_a_plain_fallback_that_also_fails_still_raises():
    with pytest.raises(httpx.HTTPStatusError):
        send_html_rejected("Hello, gaffer.", plain_status=400)


def test_a_rate_limited_chunk_is_not_resent_into_the_limiter():
    # 429 is Telegram saying stop sending; the plain fallback exists for
    # markup rejections, which arrive as 400. A resend here would be a second
    # request aimed straight at the rate limiter.
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(429, json={"ok": False})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        send_report(TOKEN, CHAT_ID, "Hello, gaffer.", http=http)
    assert len(seen) == 1


def test_non_200_raises():
    with pytest.raises(httpx.HTTPStatusError):
        send("Hello, gaffer.", status=500)


# --- reading -----------------------------------------------------------------
#
# The inbox polls once a minute and must never hang a systemd tick: the
# request asks for messages only (no edits, no channel posts) and for an
# answer now (timeout 0), and confirms everything before ``offset``.

GET_UPDATES_URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"


def polled(
    status: int = 200, result: list[dict] | None = None, offset: int | None = None
) -> tuple[list[httpx.Request], list[dict]]:
    """Poll through a mock transport; return the requests and the updates."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            status, json={"ok": status == 200, "result": result or []}
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return seen, get_updates(TOKEN, offset, http=http)


def test_get_updates_asks_for_messages_only_and_never_waits():
    seen, _ = polled(offset=11)

    assert len(seen) == 1
    assert str(seen[0].url) == GET_UPDATES_URL
    assert json.loads(seen[0].content) == {
        "offset": 11,
        "timeout": 0,
        "allowed_updates": ["message"],
    }


def test_get_updates_with_no_offset_sends_none():
    seen, _ = polled(offset=None)

    assert "offset" not in json.loads(seen[0].content)


def test_get_updates_hands_back_the_result_list():
    update = {"update_id": 7, "message": {"chat": {"id": 42}, "text": "help", "date": 1}}

    _, updates = polled(result=[update])

    assert updates == [update]


def test_get_updates_raises_on_a_refusal():
    # A 409 is what Telegram says when a webhook is set; the inbox treats any
    # status error as "unreachable this minute".
    with pytest.raises(httpx.HTTPStatusError):
        polled(status=409)


def test_send_message_is_the_report_path_escaped():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_message(TOKEN, CHAT_ID, "Bank <£0.5m> & 1 free transfer", http=http)

    assert payloads(seen) == [
        {
            "chat_id": CHAT_ID,
            "text": "Bank &lt;£0.5m&gt; &amp; 1 free transfer",
            "parse_mode": "HTML",
            "reply_markup": KEYBOARD,
        }
    ]


# --- the keyboard ---------------------------------------------------------------


def send_with_keyboard(text: str, statuses: list[int], **kwargs) -> list[dict]:
    """Send ``text`` with each request answered by the next of ``statuses``;
    return the JSON bodies, in order. ``kwargs`` go to :func:`send_report`."""
    bodies: list[dict] = []
    replies = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(next(replies), json={})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_report(TOKEN, CHAT_ID, text, http=http, **kwargs)
    return bodies


def test_the_keyboard_is_the_five_commands_in_three_rows():
    assert KEYBOARD == {
        "keyboard": [
            [{"text": "Transfers made"}],
            [{"text": "Wildcard?"}, {"text": "Free hit?"}],
            [{"text": "Chip forecast"}, {"text": "Help"}],
        ],
        "is_persistent": True,
        "resize_keyboard": True,
    }


def test_the_keyboard_rides_the_last_chunk_only():
    # LONG_REPORT is three chunks: only the third may carry the keyboard, or
    # the phone would redraw it three times for one report.
    bodies = send_with_keyboard(LONG_REPORT, [200] * 3, reply_markup=KEYBOARD)

    assert len(bodies) == 3
    assert ["reply_markup" in body for body in bodies] == [False, False, True]
    assert bodies[-1]["reply_markup"] == KEYBOARD


def test_the_plain_fallback_of_the_last_chunk_keeps_the_keyboard():
    # Chunks 1 and 2 go fine; chunk 3's HTML is refused with a 400 and goes
    # again as plain text, which must still bring the keyboard.
    bodies = send_with_keyboard(LONG_REPORT, [200, 200, 400, 200], reply_markup=KEYBOARD)

    assert len(bodies) == 4
    assert bodies[2]["parse_mode"] == "HTML" and bodies[2]["reply_markup"] == KEYBOARD
    assert "parse_mode" not in bodies[3]
    assert bodies[3]["reply_markup"] == KEYBOARD


def test_no_keyboard_means_no_key_in_any_body():
    bodies = send_with_keyboard(LONG_REPORT, [200, 200, 400, 200])

    assert not any("reply_markup" in body for body in bodies)


def test_send_message_attaches_the_keyboard():
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_message(TOKEN, CHAT_ID, "hello", http=http)

    assert [body["reply_markup"] for body in bodies] == [KEYBOARD]
