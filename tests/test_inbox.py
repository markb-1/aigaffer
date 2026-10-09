"""Tests for the inbox: what the owner texts the bot, read once a minute.

Telegram is a fake ``Phone`` that serves updates and keeps the replies;
the "Transfers made" handler is a stub, because recording has its own tests
(tests/test_recording.py). What is pinned here is the inbox's own contract:
who is obeyed, what each word does, when the offset moves, and how one bad
message is retried once and then let go — across two processes, which is
why the marker is a file.
"""

import json
from datetime import UTC, datetime

import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import inbox
from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.inbox import (
    CHIP_KINDS,
    HELP,
    NOT_CONFIGURED,
    SOMETHING_WRONG,
    WHATIF_WRONG,
    WHATIFS_OFF,
    confirm_handled,
    dispatch,
    mark_failed,
    marked_failed,
    normalise,
    run_inbox,
)
from aigaffer.report.telegram import KEYBOARD

TOKEN = "1234:super-secret-bot-token"
CHAT = "42"
DATE = 1755800000  # 2025-08-21T18:13:20Z


def update(uid: int, text: str | None = "help", chat: str = CHAT, kind: str = "message") -> dict:
    message = {"message_id": uid, "date": DATE, "chat": {"id": int(chat)}}
    if text is not None:
        message["text"] = text
    return {"update_id": uid, kind: message}


class Phone:
    """Telegram, faked: serves ``updates`` from the offset asked, keeps replies."""

    def __init__(self, updates: list[dict], unreachable: bool = False) -> None:
        self.updates = updates
        self.unreachable = unreachable
        self.asked: list[int | None] = []
        self.sent: list[str] = []

    def get(self, token: str, offset: int | None) -> list[dict]:
        self.asked.append(offset)
        if self.unreachable:
            raise httpx.ConnectError("no route")
        return [u for u in self.updates if offset is None or u["update_id"] >= offset]

    def send(self, token: str, chat_id: str, text: str) -> None:
        assert chat_id == CHAT, "replies only ever go to the configured chat"
        self.sent.append(text)


def recorded(sent_at: datetime) -> str:
    return f"Recorded at {sent_at.isoformat()}"


def no_reply(text: str) -> None:
    pytest.fail(f"dispatch itself never replies: {text!r}")


class WhatIfStub:
    """The chip what-if handler, faked: records what it was handed, replies
    through ``reply`` itself and returns None, as the real one does."""

    def __init__(self, replies: tuple[str, ...] = ("numbers", "opinion")) -> None:
        self.calls: list[tuple[str, int, datetime]] = []
        self.replies = replies

    def __call__(self, kind, update_id, sent_at, reply):
        self.calls.append((kind, update_id, sent_at))
        for text in self.replies:
            reply(text)
        return None


def run_with(tmp_path, phone: Phone, chip_whatif, handler=recorded) -> int:
    return run_inbox(
        TOKEN, CHAT, tmp_path, handler, chip_whatif, get=phone.get, send=phone.send
    )


def run(tmp_path, phone: Phone, handler=recorded) -> int:
    return run_inbox(TOKEN, CHAT, tmp_path, handler, get=phone.get, send=phone.send)


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Transfers made", "transfers made"),
        ("  TRANSFERS   made!! ", "transfers made"),
        ("transfers made.", "transfers made"),
        ("Free  Hit?", "free hit"),
        ("", ""),
    ],
)
def test_normalise_lowercases_collapses_and_strips_the_tail(text, command):
    assert normalise(text) == command


@pytest.mark.parametrize(
    ("command", "kind"),
    [("wildcard", WILDCARD), ("wc", WILDCARD), ("free hit", FREE_HIT),
     ("freehit", FREE_HIT), ("fh", FREE_HIT)],
)
def test_the_chip_words_go_to_the_what_if_with_their_kind(command, kind):
    stub = WhatIfStub(replies=())
    sent_at = datetime.now(UTC)

    assert dispatch(command, 7, sent_at, no_reply, recorded, stub) is None
    assert stub.calls == [(kind, 7, sent_at)]


def test_the_chip_words_without_a_what_if_say_so():
    assert dispatch("wildcard", 7, datetime.now(UTC), no_reply, recorded) == WHATIFS_OFF


@pytest.mark.parametrize("command", ["help", "hello", "", "transfers"])
def test_anything_else_gets_the_help(command):
    assert dispatch(command, 7, datetime.now(UTC), no_reply, recorded) == HELP


def test_the_chip_words_are_exactly_the_kinds():
    assert set(CHIP_KINDS) == {"wildcard", "wc", "free hit", "freehit", "fh"}
    assert set(CHIP_KINDS.values()) == {WILDCARD, FREE_HIT}


def test_transfers_made_is_handed_the_messages_own_time(tmp_path):
    phone = Phone([update(10, "Transfers made!")])

    assert run(tmp_path, phone) == 0

    assert phone.sent == [f"Recorded at {datetime.fromtimestamp(DATE, UTC).isoformat()}"]
    assert (tmp_path / "offset").read_text() == "10"


def test_the_help_lists_the_commands():
    assert "Transfers made" in HELP and "help" in HELP
    assert "Wildcard? / Free hit?" in HELP
    assert "Nothing is recorded" in HELP and "don't send Transfers made" in HELP
    assert HELP.startswith("Tap a button below")
    assert "coming soon" not in HELP


def test_another_chat_is_ignored_silently_and_passed(tmp_path):
    phone = Phone([update(10, "Transfers made", chat="7")])

    run(tmp_path, phone, handler=lambda _: pytest.fail("a stranger is never obeyed"))

    assert phone.sent == []
    assert (tmp_path / "offset").read_text() == "10"


def test_an_edited_message_is_ignored(tmp_path):
    phone = Phone([update(10, "Transfers made", kind="edited_message")])

    run(tmp_path, phone, handler=lambda _: pytest.fail("an edit is not a new text"))

    assert phone.sent == []
    assert (tmp_path / "offset").read_text() == "10"


def test_a_sticker_gets_the_help(tmp_path):
    phone = Phone([update(10, text=None)])

    run(tmp_path, phone)

    assert phone.sent == [HELP]


def test_the_offset_asked_is_one_past_the_last_handled(tmp_path):
    (tmp_path / "offset").write_text("9")
    phone = Phone([update(9), update(10)])

    run(tmp_path, phone)

    assert phone.asked == [10]
    assert phone.sent == [HELP]


def test_a_garbage_offset_file_reads_as_none(tmp_path):
    (tmp_path / "offset").write_text("not a number")
    phone = Phone([])

    run(tmp_path, phone)

    assert phone.asked == [None]


def test_the_offset_moves_only_after_the_update_is_handled(tmp_path):
    seen = []

    def handler(sent_at):
        seen.append((tmp_path / "offset").exists())
        return "Recorded"

    run(tmp_path, Phone([update(10, "transfers made")]), handler)

    assert seen == [False]
    assert (tmp_path / "offset").read_text() == "10"


def test_nothing_to_do_is_quiet(tmp_path, capsys):
    assert run(tmp_path, Phone([])) == 0

    assert capsys.readouterr().out == ""
    assert not (tmp_path / "offset").exists()


def test_telegram_unreachable_is_one_line_and_a_clean_exit(tmp_path, capsys):
    assert run(tmp_path, Phone([], unreachable=True)) == 0

    out = capsys.readouterr().out
    assert out.count("\n") == 1 and "ConnectError" in out
    assert TOKEN not in out


def test_missing_secrets_stand_down_without_asking(tmp_path, capsys):
    phone = Phone([update(10)])

    assert run_inbox(None, CHAT, tmp_path, recorded, get=phone.get, send=phone.send) == 0
    assert run_inbox(TOKEN, None, tmp_path, recorded, get=phone.get, send=phone.send) == 0

    assert phone.asked == []
    assert capsys.readouterr().out == f"{NOT_CONFIGURED}\n{NOT_CONFIGURED}\n"


def broken(sent_at):
    raise RuntimeError("boom with details nobody should log")


def test_one_bad_update_is_retried_once_then_let_go(tmp_path, capsys):
    phone = Phone([update(10, "Transfers made"), update(11, "help")])

    # The first minute: the failure is marked, the offset stays, he is told.
    run(tmp_path, phone, broken)
    assert not (tmp_path / "offset").exists()
    assert (tmp_path / "failed").read_text() == "10"
    assert phone.sent == [SOMETHING_WRONG]
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "boom" not in out

    # The next minute is a new process; the marker matches, so the update is
    # let go and the inbox carries on to the next one.
    run(tmp_path, phone, broken)
    assert (tmp_path / "offset").read_text() == "11"
    assert not (tmp_path / "failed").exists()
    assert phone.sent == [SOMETHING_WRONG, SOMETHING_WRONG, HELP]


def test_a_retry_that_succeeds_clears_the_marker(tmp_path):
    phone = Phone([update(10, "Transfers made")])
    run(tmp_path, phone, broken)

    run(tmp_path, phone, recorded)

    assert (tmp_path / "offset").read_text() == "10"
    assert not (tmp_path / "failed").exists()


def test_a_reply_that_fails_to_send_re_handles_the_update(tmp_path):
    # The recording itself is idempotent (tests/test_recording.py), so a
    # handled-but-unanswered text is simply handled again next minute.
    calls = []
    phone = Phone([update(10, "Transfers made")])
    sends = iter([httpx.ConnectError("down"), None, None])

    def flaky(token, chat_id, text):
        outcome = next(sends)
        if outcome is not None:
            raise outcome
        phone.sent.append(text)

    def handler(sent_at):
        calls.append(sent_at)
        return "Recorded"

    run_inbox(TOKEN, CHAT, tmp_path, handler, get=phone.get, send=flaky)
    run_inbox(TOKEN, CHAT, tmp_path, handler, get=phone.get, send=flaky)

    assert len(calls) == 2
    assert phone.sent[-1] == "Recorded"
    assert (tmp_path / "offset").read_text() == "10"


def test_a_malformed_payload_is_one_line_with_no_words_from_the_error(tmp_path, capsys):
    class Garbled(Phone):
        def get(self, token, offset):
            raise ValueError(f"bad json from https://api.telegram.org/bot{TOKEN}/x")

    assert run(tmp_path, Garbled([])) == 0

    out = capsys.readouterr().out
    assert "ValueError" in out and TOKEN not in out


def test_a_malformed_update_is_a_failure_not_a_crash(tmp_path, capsys):
    phone = Phone([{"update_id": 10, "message": {"chat": "not-a-dict"}}])

    assert run(tmp_path, phone) == 0

    assert (tmp_path / "failed").read_text() == "10"
    assert "AttributeError" in capsys.readouterr().out


class Unnumbered(Phone):
    """Telegram serving updates some of which carry no usable ``update_id``:
    served whole, whatever the offset, as a garbled payload would be."""

    def get(self, token, offset):
        self.asked.append(offset)
        return list(self.updates)


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        ({"message": {"chat": {"id": int(CHAT)}, "date": DATE, "text": "help"}}, "KeyError"),
        ({"update_id": "11", "message": None}, "TypeError"),
        ("not-an-update", "TypeError"),
    ],
)
def test_an_update_without_an_id_is_skipped_and_the_rest_handled(
    tmp_path, capsys, bad, reason
):
    # The id is what the offset is made of, so an update without an integer
    # one cannot be handled, retried or passed: it is skipped with one line
    # naming the class of the fault, and the good update behind it is
    # answered and moves the offset — rather than the whole batch failing
    # every minute, for ever, on the same unreadable update.
    phone = Unnumbered([bad, update(12, "Transfers made")])

    assert run(tmp_path, phone) == 0

    assert phone.sent == [recorded(datetime.fromtimestamp(DATE, UTC))]
    assert (tmp_path / "offset").read_text() == "12"
    out = capsys.readouterr().out
    assert out.strip().splitlines() == [f"aigaffer inbox: an update skipped ({reason})"]


def test_a_what_if_replies_for_itself_and_the_inbox_adds_nothing(tmp_path):
    stub = WhatIfStub()
    phone = Phone([update(10, "Wildcard?")])

    assert run_with(tmp_path, phone, stub) == 0

    assert phone.sent == ["numbers", "opinion"], "no third message from the inbox"
    assert stub.calls == [(WILDCARD, 10, datetime.fromtimestamp(DATE, UTC))]
    assert (tmp_path / "offset").read_text() == "10"
    assert not (tmp_path / "failed").exists()


def test_two_taps_in_one_batch_are_answered_in_order(tmp_path):
    # A wildcard and a free hit waiting in one getUpdates batch. Each is
    # handled to the end before the next starts, in id order.
    stub = WhatIfStub()
    phone = Phone([update(11, "fh"), update(10, "wc")])

    run_with(tmp_path, phone, stub)

    assert [call[:2] for call in stub.calls] == [(WILDCARD, 10), (FREE_HIT, 11)]
    assert phone.sent == ["numbers", "opinion", "numbers", "opinion"]
    assert (tmp_path / "offset").read_text() == "11"


def test_a_what_if_that_raises_gets_its_own_apology_and_is_not_retried(tmp_path, capsys):
    # The real handler writes the marker before anything that can raise
    # (spec section 8), so the inbox finds it already set and lets the update
    # go: the what-if is never re-run behind the owner's back.
    def failing(kind, update_id, sent_at, reply):
        mark_failed(tmp_path, update_id)
        raise RuntimeError("solver blew up with words nobody should log")

    phone = Phone([update(10, "free hit"), update(11, "help")])

    run_with(tmp_path, phone, failing)

    assert phone.sent == [WHATIF_WRONG, HELP]
    assert (tmp_path / "offset").read_text() == "11"
    assert not (tmp_path / "failed").exists()
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "words" not in out


def test_a_transfers_made_failure_keeps_its_own_apology(tmp_path):
    phone = Phone([update(10, "Transfers made")])

    run_with(tmp_path, phone, WhatIfStub(), handler=broken)

    assert phone.sent == [SOMETHING_WRONG]


def test_the_marker_and_offset_helpers_are_the_inboxs_own_files(tmp_path):
    assert marked_failed(tmp_path) is None
    mark_failed(tmp_path, 12)
    confirm_handled(tmp_path, 11)

    assert marked_failed(tmp_path) == 12
    assert (tmp_path / "failed").read_text() == "12"
    assert (tmp_path / "offset").read_text() == "11"


# --- the command line ---------------------------------------------------------


def test_the_cli_inbox_without_telegram_stands_down(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert cli.main(["inbox"]) == 0
    assert NOT_CONFIGURED in capsys.readouterr().out


def test_the_cli_inbox_reads_the_inbox_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    seen = {}

    def fake_run_inbox(token, chat_id, directory, handler):
        seen.update(token=token, chat_id=chat_id, directory=directory, handler=handler)
        return 0

    monkeypatch.setattr(cli, "run_inbox", fake_run_inbox)

    assert cli.main(["inbox"]) == 0
    assert seen["directory"] == tmp_path / "inbox"
    assert seen["token"] == TOKEN and seen["chat_id"] == CHAT
    assert callable(seen["handler"])
    sync = seen["handler"].args[2]
    assert sync.paths == ("state/executed/",), "the inbox commits its text files only"
    assert seen["handler"].args[3] == tmp_path / "inbox" / "state.lock"


def test_the_inbox_directory_defaults_to_the_home_dot_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("AIGAFFER_INBOX_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert inbox.inbox_dir() == tmp_path / ".aigaffer"


# The four buttons of the persistent keyboard send their labels as plain text,
# so each must normalise to the command the dispatcher already acts on.
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Transfers made", "the transfers-made handler ran"),
        ("Wildcard?", f"what-if {WILDCARD}"),
        ("Free hit?", f"what-if {FREE_HIT}"),
        ("Help", HELP),
    ],
)
def test_each_keyboard_button_reaches_its_command(label, expected):
    reply = dispatch(
        normalise(label), 7, datetime.now(UTC), no_reply,
        lambda sent_at: "the transfers-made handler ran",
        lambda kind, uid, sent_at, reply: f"what-if {kind}",
    )

    assert reply == expected


def test_the_keyboard_labels_are_exactly_the_buttons_tested_above():
    labels = [button["text"] for row in KEYBOARD["keyboard"] for button in row]

    assert labels == ["Transfers made", "Wildcard?", "Free hit?", "Help"]


def test_the_inboxs_default_reply_path_sends_with_the_keyboard(monkeypatch, tmp_path):
    # run_inbox's default ``send`` is send_message; the keyboard rides on that.
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    phone = Phone([update(10, "Help")])
    run_inbox(
        TOKEN, CHAT, tmp_path, recorded, get=phone.get,
        send=lambda token, chat_id, text: inbox.send_message(token, chat_id, text, http=http),
    )

    assert [body["reply_markup"] for body in bodies] == [KEYBOARD]
    assert bodies[0]["chat_id"] == CHAT
