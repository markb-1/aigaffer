"""Ask the gaffer: the question handler and its conversation, against a
scripted client and tmp directories.

The board is the detailed writer's (tests/test_detailed_writer.py): GW8, a
window of gameweeks 8 and 9, and a chosen plan that sells Justin (4) for
Cucurella (7). Four more players are added for the name lookup, none of them
in the squad, so the report renders exactly as the writer's tests see it:

==  ==================  ===  ====  =================  ==========  =======
id  name                pos  club  fixtures           xP (8, 9)   window
==  ==================  ===  ====  =================  ==========  =======
10  Gabriel Jesus       FWD  ARS   FUL(a), LEE(h)     3.00, 4.00   7.00
11  Gabriel Martinelli  MID  ARS   FUL(a), LEE(h)     4.00, 4.00   8.00
12  Palmer              MID  CHE   LEE(a), WOL(h)+BOU(a)  5.00, 6.50  11.50
13  Wood                FWD  NFO   blank, ARS(h)      0.00, 4.20   4.20
14  Woodman             GKP  NEW   MCI(h), ARS(a)     2.00, 2.00   4.00
==  ==================  ===  ====  =================  ==========  =======

So "Gabriel" is a prefix of two names and an exact match of none (the list
comes back); "Wood" is exact for one and a prefix of another (the exact one
wins); "Palm" is a prefix of one (Palmer); and "Palmerr" is a prefix of no
one, so it is not guessed to be Palmer.

The question is always answered on the newest snapshot. A good answer here is
:data:`GOOD` — 26 words, plain, and naming GW8.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import httpx
import pytest

from aigaffer import ask as ask_module
from aigaffer.ask import (
    CAP_REACHED,
    DAILY_CAP,
    NO_SNAPSHOT,
    THREAD_KEEP,
    MAX_TURNS,
    make_ask,
)
from aigaffer.config import Config
from aigaffer.detailed.render import render_detailed
from aigaffer.detailed.snapshot import write_snapshot
from aigaffer.inbox import ASK_WRONG, FAILED_FILE, mark_failed
from aigaffer.manager.ask_writer import (
    ANSWER,
    NOT_PROJECTED,
    PLAN,
    PLAYER,
    SYSTEM_PROMPT,
    AskUnavailable,
    answer,
    find_player,
    validate_answer,
)
from aigaffer.manager.tools import ToolError
from tests.test_detailed_writer import PLAYERS as WRITER_PLAYERS
from tests.test_detailed_writer import row, snap
from tests.test_manager_agent import FakeClient, reply, text, use

CFG = Config(team_id=42, anthropic_api_key="sk-test", manager_model="claude-opus-5-5")

PLAYERS = {
    **WRITER_PLAYERS,
    10: row("Gabriel Jesus", "FWD", "ARS", "Arsenal", 68, [["FUL", "a"]], [["LEE", "h"]], 3.0, 4.0),
    11: row("Gabriel Martinelli", "MID", "ARS", "Arsenal", 72, [["FUL", "a"]], [["LEE", "h"]], 4.0, 4.0),
    12: row("Palmer", "MID", "CHE", "Chelsea", 105,
            [["LEE", "a"]], [["WOL", "h"], ["BOU", "a"]], 5.0, 6.5),
    13: row("Wood", "FWD", "NFO", "Nott'm Forest", 70, [], [["ARS", "h"]], 0.0, 4.2),
    14: row("Woodman", "GKP", "NEW", "Newcastle", 40, [["MCI", "h"]], [["ARS", "a"]], 2.0, 2.0),
}

SENT = datetime(2026, 10, 18, 9, 30, tzinfo=UTC)
TODAY = "2026-10-18"
QUESTION = "Why Cucurella?"
GOOD = (
    "For GW8's plan: Cucurella has WOL (h) and BOU (a) in GW9 for 4.00 points,"
    " where Justin blanks after Arsenal (h). That double is the move."
)
CHANGED = "The plan changed since your last question. "


def board(**over) -> dict:
    return snap(players=PLAYERS, **over)


def setup(tmp_path: Path, *snapshots: dict) -> tuple[Config, Path]:
    """A repo with ``state/`` beside ``reports/``, and an inbox dir outside it."""
    state = tmp_path / "repo" / "state"
    state.mkdir(parents=True)
    for snapshot in snapshots:
        write_snapshot(state, snapshot["gw"], snapshot["mode"], snapshot)
    cfg = Config(
        team_id=42, anthropic_api_key="sk-test", manager_model="claude-opus-5-5",
        state_dir=state,
    )
    return cfg, tmp_path / "inbox"


def ask(cfg, inbox, client, question=QUESTION, uid=1, sent=SENT):
    replies: list[str] = []
    result = make_ask(cfg, inbox, client)(question, uid, sent, replies.append)
    return result, replies


def answered(body: str = GOOD, block_id: str = "tu_a"):
    return reply(use(ANSWER, {"text": body}, block_id))


def cap(inbox: Path) -> dict | None:
    path = inbox / "ask" / "cap.json"
    return json.loads(path.read_text()) if path.exists() else None


def thread(inbox: Path, gw: int = 8) -> list[dict]:
    path = inbox / "ask" / f"gw{gw}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_thread(inbox: Path, entries: list[dict], gw: int = 8) -> None:
    path = inbox / "ask" / f"gw{gw}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))


def request_text(request: dict) -> str:
    """Every text and tool result the request carried, as one string."""
    parts = []
    for message in request["messages"]:
        content = message["content"]
        for block in content if isinstance(content, list) else [content]:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")) + str(block.get("content", "")))
    return "\n".join(parts)


def connection_error() -> Exception:
    return anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))


# --- the handler's order ---------------------------------------------------------


def test_no_snapshot_replies_without_a_model_call_or_cap(tmp_path):
    cfg, inbox = setup(tmp_path)
    client = FakeClient([])

    result, _ = ask(cfg, inbox, client)

    assert result == NO_SNAPSHOT
    assert client.requests == []
    assert cap(inbox) is None
    assert not (inbox / FAILED_FILE).exists()


def test_the_cap_stops_the_thirty_first_question(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    (inbox / "ask").mkdir(parents=True)
    (inbox / "ask" / "cap.json").write_text(json.dumps({"day": TODAY, "n": DAILY_CAP - 1}))
    client = FakeClient([answered(), answered()])

    # The thirtieth is answered and spends the last one…
    assert ask(cfg, inbox, client, uid=1)[0] == GOOD
    assert cap(inbox) == {"day": TODAY, "n": DAILY_CAP}
    # …and the thirty-first is turned away before any request.
    assert ask(cfg, inbox, client, uid=2)[0] == CAP_REACHED
    assert len(client.requests) == 1
    assert cap(inbox) == {"day": TODAY, "n": DAILY_CAP}
    assert len(thread(inbox)) == 1


def test_the_cap_is_the_questions_utc_day(tmp_path):
    # Thirty yesterday is nothing today: the count starts again at 1.
    cfg, inbox = setup(tmp_path, board())
    (inbox / "ask").mkdir(parents=True)
    (inbox / "ask" / "cap.json").write_text(json.dumps({"day": "2026-10-17", "n": DAILY_CAP}))

    assert ask(cfg, inbox, FakeClient([answered()]))[0] == GOOD
    assert cap(inbox) == {"day": TODAY, "n": 1}


def test_an_answer_is_appended_to_the_thread_and_counts(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    client = FakeClient([answered()])

    result, replies = ask(cfg, inbox, client)

    assert result == GOOD
    # No send of its own: the inbox sends what comes back.
    assert replies == []
    assert cap(inbox) == {"day": TODAY, "n": 1}
    assert thread(inbox) == [
        {"q": QUESTION, "a": GOOD, "ts": SENT.isoformat(),
         "plan_moves": {"out": [4], "in": [7]}}
    ]
    assert not (inbox / FAILED_FILE).exists()


def test_the_request_follows_the_model_rules(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    client = FakeClient([answered()])
    ask(cfg, inbox, client)

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert "tool_choice" not in request and "thinking" not in request
    assert [t["name"] for t in request["tools"]] == [PLAYER, PLAN, ANSWER]
    # No web search: the three tools are all there is.
    assert all("type" not in t for t in request["tools"])
    assert request["system"][0]["text"] == SYSTEM_PROMPT
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    # The report is rendered in memory from the snapshot, and cached as the
    # first block; the question comes after it.
    first = request["messages"][0]["content"]
    assert first[0]["cache_control"] == {"type": "ephemeral"}
    # board() has no "deadline" key, as a snapshot written before the key
    # existed: the header falls back to NO_DEADLINE.
    report = render_detailed([board()], ask_module.NO_DEADLINE)
    assert report in first[0]["text"]
    assert QUESTION in first[-1]["text"]


def test_the_report_carries_the_snapshots_deadline(tmp_path):
    # A snapshot that keeps its deadline text is rendered with it, exactly as
    # the run rendered the reports/ file: the header is the same line.
    deadline = "Sat 18 Oct 2026 10:00 UTC"
    cfg, inbox = setup(tmp_path, board(deadline=deadline))
    client = FakeClient([answered()])
    ask(cfg, inbox, client)

    sent = client.requests[0]["messages"][0]["content"][0]["text"]
    assert f"Deadline: {deadline} · Last updated: deadline run " in sent
    assert ask_module.NO_DEADLINE not in sent
    assert render_detailed([board(deadline=deadline)], deadline) in sent


def test_a_question_with_no_scripted_client_is_refused_by_the_guard(tmp_path, capsys):
    # conftest replaces the asker's client builder with one that refuses
    # every request: a test that forgets to script a client cannot reach
    # Anthropic, and the question fails the way it would on a dead network.
    cfg, inbox = setup(tmp_path, board())

    result = make_ask(cfg, inbox)(QUESTION, 1, SENT, lambda _: None)

    assert result == ASK_WRONG
    assert "AnthropicUnreachable" in capsys.readouterr().out
    assert cap(inbox) is None
    assert thread(inbox) == []


def test_the_report_is_rendered_and_never_read_from_reports(tmp_path):
    # A reports/ file that says something else is not what the gaffer reads.
    cfg, inbox = setup(tmp_path, board())
    reports = cfg.state_dir.parent / "reports"
    reports.mkdir()
    (reports / "GW8 Detailed Report.md").write_text("STALE REPORT")
    client = FakeClient([answered()])
    ask(cfg, inbox, client)
    assert "STALE REPORT" not in request_text(client.requests[0])
    assert "# GW8 Detailed Report" in request_text(client.requests[0])


def test_the_newest_gameweek_is_the_one_answered(tmp_path):
    # GW8's deadline snapshot and GW9's early one: GW9 is the newest, and its
    # thread is gw9.jsonl. The answer must name GW9, so one that names only
    # GW8 is refused once and a GW9 answer then accepted.
    gw9 = board(gw=9, mode="early", events=[9])
    gw9["players"] = {
        pid: {**r, "f": {9: r["f"][9]}, "x": {9: r["x"][9]}, "t": r["x"][9]}
        for pid, r in PLAYERS.items()
    }
    cfg, inbox = setup(tmp_path, board(), gw9)
    nine = "For GW9's plan: Cucurella has WOL (h) and BOU (a) for 4.00 points."
    eight = "For GW8's plan: Cucurella has the double."
    client = FakeClient([answered(eight, "tu_1"), answered(nine, "tu_2")])

    assert ask(cfg, inbox, client)[0] == nine
    assert "GW9" in client.requests[1]["messages"][-1]["content"][0]["content"]
    assert [e["a"] for e in thread(inbox, 9)] == [nine]
    assert thread(inbox, 8) == []


def test_the_thread_carries_the_last_ten(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    old = [
        {"q": f"Old question {i:02d}?", "a": f"Old answer {i:02d}.", "ts": SENT.isoformat(),
         "plan_moves": {"out": [4], "in": [7]}}
        for i in range(12)
    ]
    write_thread(inbox, old)
    client = FakeClient([answered()])

    ask(cfg, inbox, client)

    sent = request_text(client.requests[0])
    # Twelve on file; the last THREAD_KEEP (10) go in: 02 to 11, oldest first.
    assert THREAD_KEEP == 10
    assert "Old question 00?" not in sent and "Old question 01?" not in sent
    kept = [f"Old question {i:02d}?" for i in range(2, 12)]
    assert all(q in sent for q in kept)
    assert [sent.index(q) for q in kept] == sorted(sent.index(q) for q in kept)
    assert "Old answer 11." in sent
    # The file keeps everything; only the conversation is capped.
    assert len(thread(inbox)) == 13


def test_a_changed_plan_is_flagged(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    # The last answer was given on a plan that sold Haaland (3) for Isak (8);
    # the snapshot's chosen plan now sells Justin (4) for Cucurella (7).
    write_thread(inbox, [{"q": "Why Isak?", "a": "For GW8's plan: Isak.", "ts": SENT.isoformat(),
                          "plan_moves": {"out": [3], "in": [8]}}])
    client = FakeClient([answered(), answered()])

    first, _ = ask(cfg, inbox, client, uid=1)
    assert first == CHANGED + GOOD
    # The gaffer is told too, so he does not defend the old plan.
    assert "changed" in request_text(client.requests[0]).lower()
    assert thread(inbox)[-1]["plan_moves"] == {"out": [4], "in": [7]}

    # Asked again on the same plan: no flag.
    second, _ = ask(cfg, inbox, client, uid=2)
    assert second == GOOD


def test_no_flag_on_the_first_question(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    assert ask(cfg, inbox, FakeClient([answered()]))[0] == GOOD


def test_two_failures_reply_ask_again_and_do_not_raise(tmp_path, capsys):
    cfg, inbox = setup(tmp_path, board())
    client = FakeClient([connection_error(), RuntimeError("boom with private words")])

    result, _ = ask(cfg, inbox, client)

    assert result == ASK_WRONG
    # One try and one retry, each one request; nothing spent or kept.
    assert len(client.requests) == 2
    assert cap(inbox) is None
    assert thread(inbox) == []
    assert not (inbox / FAILED_FILE).exists()
    # The log names the class, never the words.
    out = capsys.readouterr().out
    assert "RuntimeError" in out and "private words" not in out


def test_one_failure_is_retried_in_process(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    client = FakeClient([connection_error(), answered()])

    assert ask(cfg, inbox, client)[0] == GOOD
    assert len(client.requests) == 2
    assert cap(inbox) == {"day": TODAY, "n": 1}


def test_a_marker_for_this_update_is_never_answered_twice(tmp_path):
    # The marker holding this very update means a try at it was killed (or
    # its reply was lost) after it may have been paid for: no second bill.
    cfg, inbox = setup(tmp_path, board())
    inbox.mkdir()
    mark_failed(inbox, 7)
    client = FakeClient([answered()])

    assert ask(cfg, inbox, client, uid=7)[0] == ASK_WRONG
    assert client.requests == []
    assert cap(inbox) is None


def test_the_marker_is_set_while_the_gaffer_is_asked(tmp_path):
    cfg, inbox = setup(tmp_path, board())
    seen = []

    class Watching(FakeClient):
        def _create(self, **kwargs):
            seen.append((inbox / FAILED_FILE).read_text())
            return super()._create(**kwargs)

    ask(cfg, inbox, Watching([answered()]), uid=5)
    assert seen == ["5"]
    assert not (inbox / FAILED_FILE).exists()


def test_the_client_is_built_lazily(tmp_path, monkeypatch):
    built = []

    def factory(cfg):
        built.append(cfg)
        return FakeClient([answered()])

    monkeypatch.setattr(ask_module, "build_review_client", factory)
    cfg, inbox = setup(tmp_path, board())
    handler = make_ask(cfg, inbox)
    assert built == []
    # No snapshot needed for that: but with one, the first question builds it.
    assert handler(QUESTION, 1, SENT, lambda _: None) == GOOD
    assert len(built) == 1


def test_a_long_wait_for_the_lock_is_said(tmp_path, monkeypatch):
    # A tick holding the lock holds it for a whole report; the owner is told.
    cfg, inbox = setup(tmp_path, board())
    waits = []

    from contextlib import contextmanager

    @contextmanager
    def held(path, on_wait=None):
        waits.append(path)
        on_wait()
        yield

    monkeypatch.setattr(ask_module, "state_lock", held)
    result, replies = ask(cfg, inbox, FakeClient([answered()]))
    assert waits == [inbox / "state.lock"]
    assert replies == [ask_module.REPORT_RUNNING]
    assert result == GOOD


def test_nothing_is_written_outside_the_inbox_dir(tmp_path):
    cfg, inbox = setup(tmp_path, board(mode="scout"), board())
    repo = cfg.state_dir.parent
    (repo / "reports").mkdir()
    (repo / "reports" / "GW8 Detailed Report.md").write_text("the report\n")
    (cfg.state_dir / "aigaffer.db").write_bytes(b"not really sqlite")
    old = 1_700_000_000
    for path in repo.rglob("*"):
        os.utime(path, (old, old))

    def tree():
        return {
            str(p.relative_to(repo)): (p.stat().st_mtime, p.read_bytes() if p.is_file() else b"")
            for p in repo.rglob("*")
        }

    before = tree()
    question = "Why Cucurella over the Palmer idea?"
    result, _ = ask(cfg, inbox, FakeClient([answered()]), question=question)
    assert result == GOOD

    assert tree() == before
    # And the Q&A lives only in the inbox's ask/ folder (and the lock file).
    assert question.encode() not in b"".join(content for _, content in tree().values())
    written = {str(p.relative_to(inbox)) for p in inbox.rglob("*") if p.is_file()}
    assert written == {"ask/cap.json", "ask/gw8.jsonl", "state.lock"}


# --- the conversation --------------------------------------------------------------


def test_ambiguous_and_unknown_names():
    snapshot = board()
    # A prefix of two and an exact match of none: the candidates, never a guess.
    found = find_player(snapshot, "Gabriel")
    assert set(found) == {"matches"}
    assert len(found["matches"]) == 2
    assert any("Gabriel Jesus" in m and "10" in m for m in found["matches"])
    assert any("Gabriel Martinelli" in m and "11" in m for m in found["matches"])
    # A typo is no one: not silently Palmer.
    assert find_player(snapshot, "Palmerr") == NOT_PROJECTED
    assert find_player(snapshot, "Haaland Jr") == NOT_PROJECTED
    assert find_player(snapshot, 999) == NOT_PROJECTED
    # Exact first: "Wood" is Wood, though it starts "Woodman" too.
    assert find_player(snapshot, "Wood")["name"] == "Wood"
    # A unique prefix, any case and spacing, is that player.
    assert find_player(snapshot, "  palm ")["name"] == "Palmer"
    # An id, as a number or digits, is that player; his row reads plainly.
    cucurella = find_player(snapshot, "7")
    assert cucurella == find_player(snapshot, 7)
    assert cucurella["name"] == "Cucurella"
    assert cucurella["club"] == "CHE"
    assert cucurella["fixtures"] == {"GW8": "LEE(a)", "GW9": "WOL(h)+BOU(a)"}
    assert cucurella["xp"] == {"GW8": 2.0, "GW9": 4.0}
    assert cucurella["window_xp"] == 6.0  # 2.00 + 4.00
    assert cucurella["price"] == "£6.1m"
    # In the XI, not the fifteen: the chosen plan brings him in.
    assert cucurella["role"] == "XI"
    assert find_player(snapshot, "Saka")["role"] == "captain"
    assert find_player(snapshot, "Xhaka")["role"] == "bench"
    assert find_player(snapshot, "Palmer")["role"] == "not in the squad"


def test_the_player_tool_answers_inside_the_conversation():
    client = FakeClient([
        reply(use(PLAYER, {"name_or_id": "Gabriel"}, "tu_1")),
        reply(use(PLAYER, {"name_or_id": "Palmerr"}, "tu_2")),
        answered(),
    ])
    assert answer(CFG, board(), "report", [], QUESTION, client) == GOOD
    second = client.requests[1]["messages"][-1]["content"]
    assert second[0]["tool_use_id"] == "tu_1"
    assert "Gabriel Jesus" in second[0]["content"] and "Gabriel Martinelli" in second[0]["content"]
    third = client.requests[2]["messages"][-1]["content"]
    assert third[0]["content"] == NOT_PROJECTED
    assert "is_error" not in third[0]


def test_the_plan_tool_shows_a_plan_by_id():
    client = FakeClient([
        reply(use(PLAN, {"id": 0}, "tu_1"), use(PLAN, {"id": 5}, "tu_2")),
        answered(),
    ])
    answer(CFG, board(), "report", [], QUESTION, client)
    results = client.requests[1]["messages"][-1]["content"]
    # Every tool_use gets its own result, in one user message.
    assert [r["tool_use_id"] for r in results] == ["tu_1", "tu_2"]
    shown = json.loads(results[0]["content"])
    assert shown["in"] == ["Cucurella"] and shown["out"] == ["Justin"]
    assert shown["chosen"] is True and shown["net"] == 19.0
    # Plan 5 is not one he weighed: an error that names the ones he did.
    assert results[1]["is_error"] is True
    assert "0" in results[1]["content"] and "1" in results[1]["content"]


def test_an_answer_must_name_its_gameweek_and_fit_120_words():
    assert validate_answer(GOOD, 8) == GOOD
    assert validate_answer("  For gw8:\n Cucurella's   double.  ", 8) == "For gw8: Cucurella's double."
    with pytest.raises(ToolError, match="GW8"):
        validate_answer("Cucurella has the double.", 8)
    # GW80 and GW18 are not GW8.
    with pytest.raises(ToolError, match="GW8"):
        validate_answer("In GW80 and GW18 Cucurella has a double.", 8)
    # Under 120: 119 words pass, 120 do not.
    words = ["GW8"] + ["word"] * 118
    assert validate_answer(" ".join(words), 8)
    with pytest.raises(ToolError, match="120"):
        validate_answer(" ".join(words + ["more"]), 8)
    # Plain text: no tables, headings, bold or code.
    for marked in (
        "| GW8 | Cucurella |\n|---|---|",
        "# GW8\nCucurella.",
        "For GW8: **Cucurella**.",
        "For GW8: `Cucurella`.",
    ):
        with pytest.raises(ToolError, match="plain"):
            validate_answer(marked, 8)
    with pytest.raises(ToolError):
        validate_answer("", 8)


def test_an_invalid_answer_gets_one_retry_in_the_conversation():
    # Wrong once (no gameweek), then right: answered, two requests.
    client = FakeClient([answered("Cucurella has the double.", "tu_1"), answered(GOOD, "tu_2")])
    assert answer(CFG, board(), "report", [], QUESTION, client) == GOOD
    retry = client.requests[1]["messages"][-1]["content"]
    assert retry[0]["is_error"] is True and "GW8" in retry[0]["content"]

    # Wrong twice: unavailable, with no third request.
    client = FakeClient([answered("No week.", "tu_1"), answered("Still no week.", "tu_2"), answered()])
    with pytest.raises(AskUnavailable):
        answer(CFG, board(), "report", [], QUESTION, client)
    assert len(client.requests) == 2


def test_six_turns_at_most(tmp_path):
    # A gaffer who only looks players up: six requests, then unavailable.
    assert MAX_TURNS == 6
    lookups = [reply(use(PLAYER, {"name_or_id": "Saka"}, f"tu_{i}")) for i in range(20)]
    client = FakeClient(list(lookups))
    with pytest.raises(AskUnavailable):
        answer(CFG, board(), "report", [], QUESTION, client)
    assert len(client.requests) == MAX_TURNS
    # The last turn is told it is the last.
    assert ANSWER in client.requests[-1]["messages"][-1]["content"][-1]["text"]

    # Through the handler the six are the question's, retry included: the
    # retry has no turns left and asks for nothing.
    cfg, inbox = setup(tmp_path, board())
    client = FakeClient(list(lookups))
    assert ask(cfg, inbox, client)[0] == ASK_WRONG
    assert len(client.requests) == MAX_TURNS


def test_a_max_tokens_turn_is_dropped():
    client = FakeClient([reply(text("For GW8's plan: Cucu…"), stop="max_tokens"), answered()])
    assert answer(CFG, board(), "report", [], QUESTION, client) == GOOD
    messages = client.requests[1]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert ANSWER in messages[0]["content"][-1]["text"]


def test_talking_instead_of_answering_gets_a_plain_ask():
    client = FakeClient([reply(text("Let me think."), stop="end_turn"), answered()])
    assert answer(CFG, board(), "report", [], QUESTION, client) == GOOD
    messages = client.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]


def test_an_api_error_is_unavailable():
    with pytest.raises(AskUnavailable, match="APIConnectionError"):
        answer(CFG, board(), "report", [], QUESTION, FakeClient([connection_error()]))


def test_the_system_prompt_keeps_the_plan_and_names_the_week():
    prompt = SYSTEM_PROMPT.lower()
    for needed in ("never change", "fixtures", "numbers", "fair point", "change your mind",
                   "fresh news", "gameweek", "120 words", "plain text", ANSWER):
        assert needed in prompt
