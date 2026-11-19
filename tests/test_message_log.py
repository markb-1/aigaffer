"""The message log: what the owner texted, kept on his own box, never the repo.

``$AIGAFFER_INBOX_DIR/messages/YYYY-MM.jsonl``, one line per update from the
owner's chat. What is pinned here is the writer's own contract: which fields a
line may carry (a whitelist, never the update itself), where it refuses to
write, the file's permissions, how long it keeps a month, and that no failure
of its own ever reaches the inbox. All messages are invented.
"""

import json
import os
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from aigaffer.message_log import (
    MESSAGES_DIR,
    RETENTION_MONTHS,
    MessageLog,
    content_of,
    prune,
)

NOW = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
SENT = datetime(2026, 11, 19, 19, 59, 30, tzinfo=UTC)


def log_in(directory: Path, checkouts: tuple[Path, ...] = ()) -> MessageLog:
    return MessageLog(directory, checkouts=checkouts, now=lambda: NOW)


def lines(directory: Path, month: str = "2026-11") -> list[dict]:
    path = directory / MESSAGES_DIR / f"{month}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def text_message(text: str) -> dict:
    return {"message_id": 1, "date": int(SENT.timestamp()), "chat": {"id": 42}, "text": text}


def test_a_text_is_logged_with_exactly_the_whitelisted_fields(tmp_path):
    log_in(tmp_path).received(7, SENT, text_message("Zorblax over Plinth"), "ask", "free-text")

    assert lines(tmp_path) == [{
        "event": "received",
        "ts": NOW.isoformat(),
        "sent_at": SENT.isoformat(),
        "update_id": 7,
        "content": "text",
        "text": "Zorblax over Plinth",
        "route": "ask",
        "reason": "free-text",
    }]


def test_nothing_but_the_whitelist_is_ever_written(tmp_path):
    # A message carries the sender, the chat, the reply it answers and more:
    # none of it is the log's to keep.
    message = {
        **text_message("Why Zorblax?"),
        "from": {"id": 99, "first_name": "Someone", "username": "someone"},
        "reply_to_message": {"text": "an earlier bot reply"},
        "entities": [{"type": "mention"}],
    }
    log_in(tmp_path).received(7, SENT, message, "ask", "question-mark")

    written = (tmp_path / MESSAGES_DIR / "2026-11.jsonl").read_text(encoding="utf-8")
    for leaked in ("Someone", "someone", "99", "earlier bot reply", "mention", "chat"):
        assert leaked not in written


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ({"text": "hello there"}, ("text", "hello there")),
        ({"sticker": {"emoji": "x"}}, ("sticker", None)),
        ({"photo": [{}], "caption": "a caption with words"}, ("photo", None)),
        ({"voice": {"duration": 3}}, ("voice", None)),
        ({"location": {}}, ("location", None)),
        ({"something_new": {}}, ("other", None)),
    ],
)
def test_a_non_text_message_is_recorded_by_its_kind_only(message, expected):
    assert content_of(message) == expected


def test_a_photo_with_a_caption_keeps_neither(tmp_path):
    message = {"date": int(SENT.timestamp()), "chat": {"id": 42},
               "photo": [{"file_id": "abc"}], "caption": "Zorblax caption"}
    log_in(tmp_path).received(8, SENT, message, "help", "no-text")

    (line,) = lines(tmp_path)
    assert line["content"] == "photo"
    assert "text" not in line
    assert "Zorblax" not in json.dumps(line) and "abc" not in json.dumps(line)


def test_a_failure_is_one_more_line_with_the_class_only(tmp_path):
    log = log_in(tmp_path)
    log.received(9, SENT, text_message("Zorblax?"), "ask", "question-mark")
    log.failed(9, "RuntimeError")

    assert lines(tmp_path)[1] == {
        "event": "failed", "ts": NOW.isoformat(), "update_id": 9, "error": "RuntimeError",
    }


def test_unicode_and_emoji_round_trip(tmp_path):
    text = "Zorblax’s knee? 🤔 ？ café"
    log_in(tmp_path).received(1, SENT, text_message(text), "ask", "question-mark")

    assert lines(tmp_path)[0]["text"] == text
    assert "🤔" in (tmp_path / MESSAGES_DIR / "2026-11.jsonl").read_text(encoding="utf-8")


def test_the_month_file_is_the_writes_month(tmp_path):
    log = MessageLog(tmp_path, checkouts=(), now=lambda: datetime(2027, 1, 2, tzinfo=UTC))
    log.received(1, SENT, text_message("Zorblax?"), "ask", "question-mark")

    assert [p.name for p in (tmp_path / MESSAGES_DIR).iterdir()] == ["2027-01.jsonl"]


def test_the_directory_is_private_and_the_file_owner_only(tmp_path):
    log_in(tmp_path).received(1, SENT, text_message("Zorblax?"), "ask", "question-mark")

    folder = tmp_path / MESSAGES_DIR
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    assert stat.S_IMODE((folder / "2026-11.jsonl").stat().st_mode) == 0o600


def test_it_refuses_to_write_inside_a_checkout(tmp_path, capsys):
    checkout = tmp_path / "repo"
    (checkout / ".git").mkdir(parents=True)
    inside = checkout / "inbox"

    log_in(inside, checkouts=(checkout,)).received(
        1, SENT, text_message("Zorblax?"), "ask", "question-mark"
    )

    assert not (inside / MESSAGES_DIR).exists()
    out = capsys.readouterr().out
    assert out.splitlines() == ["aigaffer inbox: message not logged (inside-checkout)"]
    assert "orblax" not in out


def test_it_refuses_a_path_that_only_resolves_inside_a_checkout(tmp_path):
    checkout = tmp_path / "repo"
    checkout.mkdir()
    (tmp_path / "sneaky").symlink_to(checkout, target_is_directory=True)

    log_in(tmp_path / "sneaky" / "inbox", checkouts=(checkout,)).received(
        1, SENT, text_message("Zorblax?"), "ask", "question-mark"
    )

    assert not (checkout / "inbox").exists()


def test_the_default_checkout_is_this_repository(tmp_path):
    # Without being told, it still refuses the repository it was imported from.
    from aigaffer import message_log

    repo = Path(message_log.__file__).resolve().parents[1]
    assert (repo / "aigaffer" / "message_log.py").exists()
    assert repo in MessageLog(tmp_path).checkouts


def test_a_write_that_fails_is_one_line_and_never_raises(tmp_path, capsys):
    (tmp_path / MESSAGES_DIR).write_text("a file where the folder should be")
    log = log_in(tmp_path)

    log.received(1, SENT, text_message("Zorblax?"), "ask", "question-mark")
    log.failed(1, "RuntimeError")

    out = capsys.readouterr().out
    assert out.splitlines() == [
        "aigaffer inbox: message not logged (FileExistsError)",
        "aigaffer inbox: message not logged (FileExistsError)",
    ]


def test_retention_is_three_months():
    assert RETENTION_MONTHS == 3


def test_prune_drops_months_older_than_the_retention(tmp_path):
    folder = tmp_path / MESSAGES_DIR
    folder.mkdir()
    for name in ("2026-07.jsonl", "2026-08.jsonl", "2026-10.jsonl", "2026-11.jsonl",
                 "2025-12.jsonl", "notes.txt", "2026-13.jsonl", "abcd-ef.jsonl"):
        (folder / name).write_text("{}\n")

    prune(folder, NOW)

    # November 2026 keeps itself and the three months before it.
    assert sorted(p.name for p in folder.iterdir()) == [
        "2026-08.jsonl", "2026-10.jsonl", "2026-11.jsonl",
        "2026-13.jsonl", "abcd-ef.jsonl", "notes.txt",
    ]


def test_prune_across_a_year_boundary(tmp_path):
    folder = tmp_path / MESSAGES_DIR
    folder.mkdir()
    for name in ("2026-09.jsonl", "2026-10.jsonl", "2027-01.jsonl"):
        (folder / name).write_text("{}\n")

    prune(folder, datetime(2027, 1, 5, tzinfo=UTC))

    assert sorted(p.name for p in folder.iterdir()) == ["2026-10.jsonl", "2027-01.jsonl"]


def test_a_write_prunes_old_months(tmp_path):
    folder = tmp_path / MESSAGES_DIR
    folder.mkdir(mode=0o700)
    (folder / "2026-01.jsonl").write_text("{}\n")

    log_in(tmp_path).received(1, SENT, text_message("Zorblax?"), "ask", "question-mark")

    assert sorted(p.name for p in folder.iterdir()) == ["2026-11.jsonl"]


def test_a_prune_that_fails_never_loses_the_line(tmp_path, monkeypatch):
    def refuses(self, *args, **kwargs):
        raise PermissionError("no")

    folder = tmp_path / MESSAGES_DIR
    folder.mkdir(mode=0o700)
    (folder / "2026-01.jsonl").write_text("{}\n")
    monkeypatch.setattr(Path, "unlink", refuses)

    log_in(tmp_path).received(1, SENT, text_message("Zorblax?"), "ask", "question-mark")

    assert lines(tmp_path)[0]["update_id"] == 1


def test_an_existing_file_keeps_its_permissions_and_is_appended(tmp_path):
    log = log_in(tmp_path)
    log.received(1, SENT, text_message("First?"), "ask", "question-mark")
    log.received(2, SENT, text_message("Second?"), "ask", "question-mark")

    assert [line["update_id"] for line in lines(tmp_path)] == [1, 2]
    path = tmp_path / MESSAGES_DIR / "2026-11.jsonl"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
