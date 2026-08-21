"""Tests for Telegram delivery.

Every request is served by an ``httpx.MockTransport`` that keeps what it was
handed, so the assertions are about the wire: one POST per chunk, to the
sendMessage URL for the token, carrying the chat id and a piece of the report
and nothing else — no ``parse_mode``, because the report is written to read as
plain text.

The report itself is lines, and the split has to keep them: a message that
begins mid-sentence is worse than one more message. The exception is a line
longer than a whole message, which can only be cut.
"""

import json

import httpx
import pytest

from aigaffer.report.telegram import send_report

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


def texts(requests: list[httpx.Request]) -> list[str]:
    return [json.loads(request.content)["text"] for request in requests]


def test_short_report_is_one_post():
    requests = send("Hello, gaffer.")
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert str(requests[0].url) == SEND_MESSAGE_URL
    assert json.loads(requests[0].content) == {
        "chat_id": CHAT_ID,
        "text": "Hello, gaffer.",
    }


def test_report_is_sent_verbatim():
    assert texts(send("Hi\n")) == ["Hi\n"]


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


def test_blank_report_sends_nothing():
    assert send("") == []
    assert send("\n\n") == []


def test_non_200_raises():
    with pytest.raises(httpx.HTTPStatusError):
        send("Hello, gaffer.", status=500)
