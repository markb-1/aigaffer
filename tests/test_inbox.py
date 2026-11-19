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
from pathlib import Path

import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer import inbox
from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.inbox import (
    ASK_LINE,
    ASK_WRONG,
    CHIP_FORECAST,
    CHIP_FORECAST_OFF,
    CHIP_KINDS,
    FORECAST_WRONG,
    HELP,
    WEEK_PLAN_OFF,
    WEEK_PLAN_WORDS,
    WEEK_PLAN_WRONG,
    NOT_CONFIGURED,
    SOMETHING_WRONG,
    WHATIF_WRONG,
    WHATIFS_OFF,
    confirm_handled,
    dispatch,
    mark_failed,
    marked_failed,
    normalise,
    route,
    run_inbox,
    simulate,
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


def test_another_chat_is_ignored_and_passed(tmp_path):
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
    assert out.strip().splitlines() == [
        f"aigaffer inbox: an update skipped ({reason})",
        "aigaffer inbox: update 12 -> transfers-made (command)",
    ]


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

    def fake_run_inbox(
        token, chat_id, directory, handler, chip_whatif=None, chip_forecast=None,
        ask=None, week_plan=None, log=None,
    ):
        seen.update(token=token, chat_id=chat_id, directory=directory,
                    handler=handler, chip_whatif=chip_whatif,
                    chip_forecast=chip_forecast, ask=ask, week_plan=week_plan,
                    log=log)
        return 0

    monkeypatch.setattr(cli, "run_inbox", fake_run_inbox)

    assert cli.main(["inbox"]) == 0
    assert seen["directory"] == tmp_path / "inbox"
    assert seen["token"] == TOKEN and seen["chat_id"] == CHAT
    assert callable(seen["handler"])
    sync = seen["handler"].args[2]
    assert sync.paths == ("state/executed/",), "the inbox commits its text files only"
    assert seen["handler"].args[3] == tmp_path / "inbox" / "state.lock"
    assert seen["handler"].keywords["log"].path == tmp_path / "inbox" / "whatifs.jsonl"
    whatif = seen["chip_whatif"]
    assert whatif.func.__name__ == "chip_whatif"
    assert whatif.args[2] == tmp_path / "inbox"
    assert whatif.args[3] == tmp_path / "inbox" / "state.lock"
    forecast = seen["chip_forecast"]
    assert forecast.func.__name__ == "chip_forecast"
    assert len(forecast.args) == 2, "bound with the config and the client only"
    plan = seen["week_plan"]
    assert plan.func.__name__ == "week_plan"
    assert plan.args[1] == tmp_path / "inbox" / "state.lock", "read under the tick's lock"
    log = seen["log"]
    assert log.directory == tmp_path / "inbox"
    assert Path.cwd().resolve() in log.checkouts, "the clone holding state/ is refused too"


def test_the_inbox_wires_ask_only_with_a_manager(monkeypatch, tmp_path):
    # With a key the manager is on and the gaffer answers questions; with none
    # (or AIGAFFER_MANAGER=0) there is nobody to ask, so ``ask`` stays None and
    # a question gets the help, as it did before the gaffer existed.
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    seen = []

    def fake_run_inbox(*args, ask=None, **kwargs):
        seen.append(ask)
        return 0

    monkeypatch.setattr(cli, "run_inbox", fake_run_inbox)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert cli.main(["inbox"]) == 0
    assert callable(seen[-1]), "a manager on: the gaffer answers"

    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert cli.main(["inbox"]) == 0
    assert seen[-1] is None, "no key, no manager, no gaffer"

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("AIGAFFER_MANAGER", "0")
    assert cli.main(["inbox"]) == 0
    assert seen[-1] is None, "the manager switched off, no gaffer"


def test_the_inbox_directory_defaults_to_the_home_dot_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("AIGAFFER_INBOX_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    assert inbox.inbox_dir() == tmp_path / ".aigaffer"


# The six buttons of the persistent keyboard send their labels as plain text,
# so each must normalise to the command the dispatcher already acts on.
@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Transfers made", "the transfers-made handler ran"),
        ("Wildcard?", f"what-if {WILDCARD}"),
        ("Free hit?", f"what-if {FREE_HIT}"),
        ("Chip forecast", "the chip forecast ran"),
        ("Week plan", None),
        ("Help", HELP),
    ],
)
def test_each_keyboard_button_reaches_its_command(label, expected):
    ran = []
    reply = dispatch(
        normalise(label), 7, datetime.now(UTC), no_reply,
        lambda sent_at: "the transfers-made handler ran",
        lambda kind, uid, sent_at, reply: f"what-if {kind}",
        lambda sent_at: "the chip forecast ran",
        week_plan=lambda sent_at, reply: ran.append("the week plan ran"),
    )

    assert reply == expected
    assert ran == (["the week plan ran"] if label == "Week plan" else [])


def test_the_keyboard_labels_are_exactly_the_buttons_tested_above():
    labels = [button["text"] for row in KEYBOARD["keyboard"] for button in row]

    assert labels == [
        "Transfers made", "Week plan", "Wildcard?", "Free hit?", "Chip forecast",
        "Help",
    ]


def test_chip_forecast_is_the_normalised_words_and_the_button(tmp_path):
    assert CHIP_FORECAST == "chip forecast"
    assert normalise("Chip  Forecast!") == CHIP_FORECAST
    seen = []

    def forecast(sent_at):
        seen.append(sent_at)
        return "the plan"

    phone = Phone([update(10, "Chip forecast"), update(11, "chip forecast.")])
    run_inbox(
        TOKEN, CHAT, tmp_path, recorded, chip_forecast=forecast,
        get=phone.get, send=phone.send,
    )

    assert phone.sent == ["the plan", "the plan"]
    assert seen == [datetime.fromtimestamp(DATE, UTC)] * 2


def test_an_unconfigured_forecast_says_so_not_the_help():
    reply = dispatch(
        CHIP_FORECAST, 7, datetime.now(UTC), no_reply, lambda sent_at: "recorded"
    )

    assert reply == CHIP_FORECAST_OFF == "Chip forecast is not configured."


def test_a_forecast_that_raises_gets_its_own_apology(tmp_path):
    def broken(sent_at):
        raise httpx.ConnectError("fpl down")

    phone = Phone([update(10, "Chip forecast")])
    run_inbox(
        TOKEN, CHAT, tmp_path, recorded, chip_forecast=broken,
        get=phone.get, send=phone.send,
    )

    assert phone.sent == [FORECAST_WRONG]


def test_help_lists_the_chip_forecast():
    assert (
        "• Chip forecast — when the bot currently plans to play each chip,"
        " from the latest report. Instant."
    ) in HELP.splitlines()


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


# --- Phase 3: a question goes to the gaffer ---------------------------------


class AskStub:
    """The gaffer's answerer, faked: records what it was handed and returns a
    canned answer, as the real one will."""

    def __init__(self, answer: str | None = "because fixtures") -> None:
        self.calls: list[tuple[str, int, datetime]] = []
        self.answer = answer

    def __call__(self, raw, update_id, sent_at, reply):
        self.calls.append((raw, update_id, sent_at))
        return self.answer


def run_asking(tmp_path, phone: Phone, ask, chip_whatif=None) -> int:
    return run_inbox(
        TOKEN, CHAT, tmp_path, recorded, chip_whatif,
        get=phone.get, send=phone.send, ask=ask,
    )


def test_a_question_mark_routes_to_the_gaffer(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "Is Saka a good captain?")])
    run_asking(tmp_path, phone, ask)
    # The gaffer gets the RAW text (case and mark intact), not the normalised.
    assert [c[0] for c in ask.calls] == ["Is Saka a good captain?"]
    assert ask.calls[0][1] == 1
    assert phone.sent == ["because fixtures"]


def test_an_interrogative_start_routes_without_a_mark(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "why saka")])
    run_asking(tmp_path, phone, ask)
    assert [c[0] for c in ask.calls] == ["why saka"]


def test_a_trailing_question_mark_alone_routes(tmp_path):
    ask = AskStub()
    run_asking(tmp_path, Phone([update(1, "Saka vs Palmer?  ")]), ask)
    assert len(ask.calls) == 1


def test_a_typo_of_a_command_still_gets_help(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "transfer made")])
    run_asking(tmp_path, phone, ask)
    assert ask.calls == []
    assert phone.sent == [HELP]


def test_existing_commands_win_over_questions(tmp_path):
    ask = AskStub()
    whatif = WhatIfStub(replies=())
    phone = Phone([update(1, "Wildcard?")])
    run_asking(tmp_path, phone, ask, whatif)
    assert ask.calls == []
    assert whatif.calls == [(WILDCARD, 1, whatif.calls[0][2])]


def test_help_names_how_to_ask():
    # Any text that is not a button goes to the gaffer now, so the help no
    # longer asks for a question mark.
    assert ASK_LINE in HELP
    assert "question mark" not in ASK_LINE
    assert "anything else" in ASK_LINE.lower()


def test_ask_off_gives_help():
    sent_at = datetime.now(UTC)
    assert dispatch("why saka", 7, sent_at, no_reply, recorded, raw="Why Saka?") == HELP


def test_a_question_that_raises_gets_the_ask_apology(tmp_path):
    def broken_ask(raw, update_id, sent_at, reply):
        raise RuntimeError("gaffer down")

    phone = Phone([update(1, "why saka")])
    run_asking(tmp_path, phone, broken_ask)
    assert phone.sent == [ASK_WRONG]


def test_a_transfers_made_failure_keeps_its_apology_even_with_a_mark(tmp_path):
    # "Transfers made?" is still the command, so its failure is the recording's
    # apology, not the gaffer's.
    def broken_handler(sent_at):
        raise RuntimeError("db down")

    phone = Phone([update(1, "Transfers made?")])
    run_inbox(
        TOKEN, CHAT, tmp_path, broken_handler,
        get=phone.get, send=phone.send, ask=AskStub(),
    )
    assert phone.sent == [SOMETHING_WRONG]


# --- The Week plan button ----------------------------------------------------


class WeekPlanStub:
    """The week plan handler, faked: records the time it was handed, replies
    through ``reply`` and returns None, as the real one does."""

    def __init__(self, texts: tuple[str, ...] = ("one", "two")) -> None:
        self.calls: list[datetime] = []
        self.texts = texts

    def __call__(self, sent_at, reply):
        self.calls.append(sent_at)
        for text in self.texts:
            reply(text)


def run_planning(tmp_path, phone: Phone, week_plan, ask=None) -> int:
    return run_inbox(
        TOKEN, CHAT, tmp_path, recorded, get=phone.get, send=phone.send,
        ask=ask, week_plan=week_plan,
    )


def test_the_week_plan_words_are_exactly_two():
    assert WEEK_PLAN_WORDS == {"week plan", "plan"}


@pytest.mark.parametrize("text", ["Week plan", "week plan", "Week  plan!", "plan", "Plan"])
def test_week_plan_and_plan_reach_the_handler_and_send_its_texts(tmp_path, text):
    stub = WeekPlanStub()
    phone = Phone([update(10, text)])

    run_planning(tmp_path, phone, stub)

    assert stub.calls == [datetime.fromtimestamp(DATE, UTC)]
    assert phone.sent == ["one", "two"], "the handler's own texts, and no extra help"
    assert (tmp_path / "offset").read_text() == "10"


def test_plan_with_a_question_mark_is_the_week_plan_not_a_question(tmp_path):
    # ``normalise`` strips the "?", and the commands are routed before the
    # question mark is looked for, so "plan?" never reaches the gaffer.
    stub, ask = WeekPlanStub(), AskStub()
    phone = Phone([update(10, "plan?")])

    run_planning(tmp_path, phone, stub, ask=ask)

    assert len(stub.calls) == 1
    assert ask.calls == []


def test_a_multi_word_question_with_plan_still_goes_to_the_gaffer(tmp_path):
    stub, ask = WeekPlanStub(), AskStub()
    phone = Phone([update(10, "Why this plan?")])

    run_planning(tmp_path, phone, stub, ask=ask)

    assert stub.calls == []
    assert [c[0] for c in ask.calls] == ["Why this plan?"]


def test_the_week_plan_without_a_handler_says_so_and_never_the_help():
    assert (
        dispatch("week plan", 7, datetime.now(UTC), no_reply, recorded)
        == WEEK_PLAN_OFF
        == "Week plan is not configured."
    )


def test_a_week_plan_that_raises_gets_its_own_apology(tmp_path):
    def broken(sent_at, reply):
        raise RuntimeError("snapshot unreadable")

    phone = Phone([update(10, "Week plan")])

    run_planning(tmp_path, phone, broken)

    # Not the recording's apology and not the ask apology; and the first
    # failure leaves the offset, so the next minute tries once more.
    assert phone.sent == [WEEK_PLAN_WRONG]
    assert WEEK_PLAN_WRONG not in (SOMETHING_WRONG, ASK_WRONG)
    assert not (tmp_path / "offset").exists()
    assert marked_failed(tmp_path) == 10


def test_the_help_lists_the_week_plan():
    assert any(
        line.startswith("• Week plan —") and "five short texts" in line
        for line in HELP.splitlines()
    )


def test_the_inbox_docstring_counts_five_things():
    assert "Five things can be said to it" in inbox.__doc__
    assert "Four things" not in inbox.__doc__


# --- Routing: one function decides, for dispatch and the apology alike ------
#
# The order (each step only for what the steps before it left):
#   1. an exact command          -> its button
#   2. no text at all, or "help" -> the help
#   3. a "?" anywhere ("？" too)  -> the gaffer
#   4. a near-miss of a command  -> the help (difflib, cutoff 0.8, same word count)
#   5. a single word             -> the help
#   6. anything else             -> the gaffer
# With no gaffer wired, a text that would go to him gets the help instead.
# All messages here are invented.


def routed(text: str, ask_wired: bool = True) -> tuple[str, str]:
    return route(normalise(text), text, ask_wired)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Transfers made", ("transfers-made", "command")),
        ("Transfers made?", ("transfers-made", "command")),
        ("Chip forecast", ("chip-forecast", "command")),
        ("Week plan", ("week-plan", "command")),
        ("plan?", ("week-plan", "command")),
        ("plan？", ("week-plan", "command")),
        ("Wildcard?", ("whatif", "command")),
        ("Wildcard？", ("whatif", "command")),
        ("Free hit?", ("whatif", "command")),
        ("fh", ("whatif", "command")),
    ],
)
def test_the_commands_route_first(text, expected):
    assert routed(text) == expected


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "no-text"),
        ("   ", "no-text"),
        ("?!", "no-text"),
        ("？", "no-text"),
        ("help", "help-word"),
        ("Help?", "help-word"),
    ],
)
def test_no_text_and_the_help_word_get_the_help(text, reason):
    assert routed(text) == ("help", reason)


@pytest.mark.parametrize(
    "text",
    [
        "Why is Saka benched?",
        "Lineup?",
        "Lineup ok?!",
        "Lineup ok？",
        "Lineup ok? 🤔",
        "Is Palmer fit? thinking of benching him",
        "Wildcard now?",
        "Transfers made yet?",
        "transfer made?",
    ],
)
def test_a_question_mark_anywhere_goes_to_the_gaffer(text):
    assert routed(text) == ("ask", "question-mark")


@pytest.mark.parametrize(
    "text",
    [
        "What's the lineup",
        "What’s the lineup",
        "whats the lineup",
        "Who's captain",
        "How's the bench looking",
        "Where's Palmer",
        "Are we starting Haaland",
        "Do I start Mbeumo",
        "Does Rice start",
        "Will Haaland play",
        "Did you consider Saka",
        "Has Haaland recovered",
        "Am I right to bench Konsa",
        "Were you sure about Konsa",
        "Shouldn't Saka start",
        "Isn't Konsa a doubt",
        "Can't we bench Konsa",
        "Any reason Konsa starts over Guehi",
        "Thoughts on the lineup",
        "Explain the lineup",
        "Tell me why Konsa starts",
        "Konsa over Guehi",
        "Lineup looks odd, why Konsa",
        "Hi, why is Konsa in",
        "Quick question: why Konsa",
        "Re the lineup - why Konsa",
        "Haaland or Mbeumo captain",
        "why saka",
        "wildcard now",
    ],
)
def test_ordinary_text_goes_to_the_gaffer(text):
    assert routed(text) == ("ask", "free-text")


@pytest.mark.parametrize(
    "text", ["transfer made", "chip forcast", "week pln", "free hits", "transfer mode", "wildcrad"]
)
def test_a_near_miss_of_a_command_gets_the_help(text):
    assert routed(text) == ("help", "command-typo")


@pytest.mark.parametrize("text", ["the plan", "new plan", "keep plan"])
def test_a_phrase_merely_like_a_command_goes_to_the_gaffer(text):
    # The other side of the 0.8 cutoff: these score below it against
    # "week plan", so they are text for the gaffer, not typos.
    assert routed(text) == ("ask", "free-text")


@pytest.mark.parametrize("text", ["hello", "thanks", "Konsa", "lineup"])
def test_a_single_word_gets_the_help(text):
    assert routed(text) == ("help", "one-word")


@pytest.mark.parametrize("text", ["Why Saka?", "What's the lineup"])
def test_without_a_gaffer_what_would_go_to_him_gets_the_help(text):
    assert routed(text, ask_wired=False) == ("help", "ask-off")


def test_without_a_gaffer_a_typo_still_says_typo():
    assert routed("chip forcast", ask_wired=False) == ("help", "command-typo")


def test_the_reasons_are_a_fixed_set_and_never_the_messages_words():
    texts = ["", "help", "Why Zorblax?", "zorblax plinth", "zorblax", "chip forcast", "Wildcard?"]
    for wired in (True, False):
        for text in texts:
            name, reason = routed(text, wired)
            assert reason in inbox.REASONS
            assert name in inbox.ROUTES
            assert "zorblax" not in reason and "plinth" not in reason


# --- Routing, end to end through run_inbox -----------------------------------


def test_a_question_without_a_mark_or_a_question_word_reaches_the_gaffer(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "What's the lineup")])
    run_asking(tmp_path, phone, ask)
    assert [c[0] for c in ask.calls] == ["What's the lineup"]
    assert phone.sent == ["because fixtures"]


def test_hello_gets_the_help_with_a_gaffer_wired(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "hello")])
    run_asking(tmp_path, phone, ask)
    assert ask.calls == []
    assert phone.sent == [HELP]


def test_a_newly_routed_question_that_raises_gets_the_ask_apology(tmp_path):
    def broken_ask(raw, update_id, sent_at, reply):
        raise RuntimeError("gaffer down")

    phone = Phone([update(1, "Thoughts on the lineup")])
    run_asking(tmp_path, phone, broken_ask)
    assert phone.sent == [ASK_WRONG]


def test_a_sticker_gets_the_help_with_a_gaffer_wired(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, None)])
    run_asking(tmp_path, phone, ask)
    assert ask.calls == []
    assert phone.sent == [HELP]


def test_dispatch_keeps_its_signature_and_routes_a_question():
    ask = AskStub()
    sent_at = datetime.now(UTC)
    assert dispatch("whats the lineup", 7, sent_at, no_reply, recorded, ask=ask,
                    raw="What's the lineup") == "because fixtures"


# --- One journal line per update: the route, never the words -----------------


def test_each_update_logs_its_route_and_reason_and_never_its_words(tmp_path, capsys):
    ask = AskStub()
    phone = Phone([update(1, "Zorblax over Plinth"), update(2, "zorblax")])
    run_asking(tmp_path, phone, ask)
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "aigaffer inbox: update 1 -> ask (free-text)",
        "aigaffer inbox: update 2 -> help (one-word)",
    ]
    assert "orblax" not in out and "linth" not in out


def test_another_chat_logs_ignored_without_its_chat_id(tmp_path, capsys):
    phone = Phone([update(5, "Zorblax?", chat="777")])
    run(tmp_path, phone)
    out = capsys.readouterr().out
    assert out.splitlines() == ["aigaffer inbox: update 5 -> ignored (other-chat)"]
    assert "777" not in out and "orblax" not in out


def test_the_route_is_logged_before_a_handler_that_fails(tmp_path, capsys):
    phone = Phone([update(10, "Transfers made")])
    run(tmp_path, phone, broken)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "aigaffer inbox: update 10 -> transfers-made (command)"
    assert "RuntimeError" in lines[1]


# --- The simulation: route a made-up text, send nothing, write nothing --------


def test_a_simulation_prints_the_route_and_asks_nobody_by_default(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert simulate("Thoughts on the lineup", ask_wired=True) == 0
    out = capsys.readouterr().out
    assert "route: ask" in out and "reason: free-text" in out
    assert "--ask" in out, "says the gaffer was not asked, and how to ask him"
    assert list(tmp_path.iterdir()) == []


def test_a_simulation_shows_the_help_it_would_send(capsys):
    simulate("hello", ask_wired=True)
    out = capsys.readouterr().out
    assert "route: help" in out and "reason: one-word" in out
    assert HELP in out


def test_a_simulated_command_is_never_run(capsys):
    simulate("Transfers made", ask_wired=True)
    out = capsys.readouterr().out
    assert "route: transfers-made" in out and "reason: command" in out
    assert "not run" in out
    simulate("Wildcard?", ask_wired=True)
    assert "route: whatif" in capsys.readouterr().out


def test_a_simulation_without_a_gaffer_routes_as_the_box_would(capsys):
    simulate("Why Saka?", ask_wired=False)
    out = capsys.readouterr().out
    assert "route: help" in out and "reason: ask-off" in out


def test_a_simulation_with_ask_hands_the_raw_text_to_the_gaffer(capsys):
    ask = AskStub()
    simulate("Thoughts on the lineup", ask_wired=True, ask=ask)
    assert [c[0] for c in ask.calls] == ["Thoughts on the lineup"]
    assert "reply: because fixtures" in capsys.readouterr().out


def test_a_simulation_prints_what_the_gaffer_says_for_himself(capsys):
    def chatty(raw, update_id, sent_at, reply):
        reply("waiting for the report")
        return "the answer"

    simulate("Why Saka?", ask_wired=True, ask=chatty)
    out = capsys.readouterr().out
    assert out.index("reply: waiting for the report") < out.index("reply: the answer")


def test_the_cli_simulation_never_reads_telegram_or_writes(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("AIGAFFER_MANAGER", raising=False)
    monkeypatch.setattr(cli, "run_inbox", lambda *a, **k: pytest.fail("no inbox run"))
    monkeypatch.setattr(inbox, "send_message", lambda *a, **k: pytest.fail("nothing sent"))
    monkeypatch.setattr(inbox, "get_updates", lambda *a, **k: pytest.fail("nothing read"))

    assert cli.main(["inbox", "--simulate", "Thoughts on the lineup"]) == 0

    out = capsys.readouterr().out
    assert "route: ask" in out and "reason: free-text" in out
    assert not (tmp_path / "inbox").exists()


def test_the_cli_simulation_follows_the_manager_switch(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert cli.main(["inbox", "--simulate", "Why Saka?"]) == 0
    assert "reason: ask-off" in capsys.readouterr().out


def test_the_cli_simulation_with_ask_uses_the_in_memory_gaffer(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    monkeypatch.setenv("AIGAFFER_INBOX_DIR", str(tmp_path / "inbox"))
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("AIGAFFER_MANAGER", raising=False)
    built = []

    def fake_simulated_ask(cfg, directory):
        built.append(directory)
        return AskStub("simulated answer")

    monkeypatch.setattr(cli, "make_simulated_ask", fake_simulated_ask)

    assert cli.main(["inbox", "--simulate", "Why Saka?", "--ask"]) == 0
    assert built == [tmp_path / "inbox"]
    assert "reply: simulated answer" in capsys.readouterr().out


def test_ask_without_simulate_is_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    with pytest.raises(SystemExit):
        cli.main(["inbox", "--ask"])


def test_simulate_is_only_for_the_inbox(monkeypatch):
    monkeypatch.setenv("FPL_TEAM_ID", "99")
    with pytest.raises(SystemExit):
        cli.main(["auto", "--simulate", "Why Saka?"])



# --- The message log: the owner's texts, on his box, never the repo ----------


def logged(directory) -> list[dict]:
    folder = directory / "messages"
    return [
        json.loads(line)
        for path in sorted(folder.glob("*.jsonl"))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_each_owner_text_is_logged_with_its_route(tmp_path):
    ask = AskStub()
    phone = Phone([update(1, "Zorblax over Plinth"), update(2, "hello"), update(3, None)])
    run_asking(tmp_path, phone, ask)

    entries = logged(tmp_path)
    assert [(e["update_id"], e["route"], e["reason"], e["content"]) for e in entries] == [
        (1, "ask", "free-text", "text"),
        (2, "help", "one-word", "text"),
        (3, "help", "no-text", "other"),
    ]
    assert entries[0]["text"] == "Zorblax over Plinth"
    assert "text" not in entries[2]


def test_another_chats_text_is_never_logged(tmp_path):
    phone = Phone([update(5, "Zorblax?", chat="777")])
    run(tmp_path, phone)
    assert not (tmp_path / "messages").exists()


def test_the_line_is_written_before_a_handler_that_fails(tmp_path):
    phone = Phone([update(10, "Transfers made")])
    run(tmp_path, phone, broken)

    assert [(e["event"], e["update_id"]) for e in logged(tmp_path)] == [
        ("received", 10), ("failed", 10),
    ]
    assert logged(tmp_path)[1]["error"] == "RuntimeError"


def test_a_log_that_cannot_be_written_never_reaches_the_inbox(tmp_path, capsys):
    (tmp_path / "messages").write_text("a file where the folder should be")
    phone = Phone([update(10, "help")])

    run(tmp_path, phone)

    assert phone.sent == [HELP], "answered, and no apology"
    assert (tmp_path / "offset").read_text() == "10"
    assert not (tmp_path / "failed").exists()
    assert "message not logged (FileExistsError)" in capsys.readouterr().out


def test_a_re_handled_update_is_logged_again_under_the_same_id(tmp_path):
    calls = []

    def flaky(token, chat_id, text):
        calls.append(text)
        if len(calls) == 1:
            raise httpx.ConnectError("down")

    phone = Phone([update(10, "help")])
    run_inbox(TOKEN, CHAT, tmp_path, recorded, get=phone.get, send=flaky)
    run_inbox(TOKEN, CHAT, tmp_path, recorded, get=phone.get, send=flaky)

    received = [e["update_id"] for e in logged(tmp_path) if e["event"] == "received"]
    assert received == [10, 10]


def test_the_journal_still_never_carries_the_words(tmp_path, capsys):
    run_asking(tmp_path, Phone([update(1, "Zorblax over Plinth")]), AskStub())
    out = capsys.readouterr().out
    assert "orblax" not in out and "linth" not in out
