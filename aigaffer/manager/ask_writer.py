"""The gaffer, answering the owner's question about a plan he has made.

"Why Cucurella?", "Should Saka not be captain?" — a text from the owner, routed
here by the inbox (:func:`aigaffer.inbox.is_question`) through the handler in
:mod:`aigaffer.ask`, which owns the order, the cap and the thread. This module
is only the conversation: the gameweek's detailed report goes in, rendered in
memory from its snapshots, with the earlier questions this gameweek and the
new one; one short plain-text answer comes out through the ``answer`` tool.

**Explain-only.** The plan is final and the conversation cannot touch it:
the only tools are two lookups into the snapshot — ``player`` for one man's
fixtures and numbers, ``plan`` for one of the plans he weighed — and the
answer. No web search, so a question that turns on fresh news (a knock, a
press conference) is told so rather than guessed at; no re-solve, no
finalize, nothing that writes. A fair point from the owner is conceded in
words, with what would change the gaffer's mind, and the plan still stands.

**What an answer must be,** checked here rather than hoped for in the prompt:
plain text (Telegram shows it as typed, so a table or a heading arrives as
pipes and hashes), under 120 words, and naming its gameweek as ``GW{n}`` —
between a deadline and the next early run the newest snapshot is the finished
gameweek's, and an answer that does not say which week it means can be read
as being about the wrong one. An answer that fails gets one more go inside the
conversation, with the error in words; a second failure is
:class:`AskUnavailable`.

**Six turns, tools included** (:data:`MAX_TURNS`). Each request is a turn,
whether it answers or looks someone up, so a gaffer who looks players up for
ever is stopped at six; the last turn is told it is the last, and asked for
the answer. The handler meters the same six across its one retry, so a
question costs six requests at the outside, whatever happens.

The model's rules are the manager's, as the review writer
(:mod:`aigaffer.manager.review_writer`) keeps them: no ``tool_choice``, no
thinking parameter, history append-only, every ``tool_use`` answered with its
own ``tool_result`` in one user message, a turn cut off at ``max_tokens``
dropped as if it never happened with its ask joined to the still-open user
message (never two user turns in a row), and the system prompt and the report
cached (:func:`~aigaffer.manager.agent._cached`) with the growing end marked
per request (:func:`~aigaffer.manager.agent._rolling`). The report is the same
block for every question until the next run rewrites it, so a second question
within the cache's few minutes reads it from the cache.
"""

import json
import re
import unicodedata
from typing import Any

import anthropic

from aigaffer.config import Config
from aigaffer.detailed.render import _moves, _name
from aigaffer.inbox import normalise
from aigaffer.manager.agent import EFFORT, MAX_RESUMPTIONS, MAX_TOKENS, _cached, _result, _rolling, _text
from aigaffer.manager.tools import MARKUP, ToolError
from aigaffer.report.render import price

PLAYER = "player"
PLAN = "plan"
ANSWER = "answer"

MAX_TURNS = 6
# The answer tool may be wrong once; the second wrong answer ends it.
ANSWER_TRIES = 2
# "Under 120 words": 119 is the most an answer may have.
ANSWER_WORDS = 120

NOT_PROJECTED = "not projected this week"

SYSTEM_PROMPT = (
    "You are the gaffer of a Fantasy Premier League team, answering the"
    " owner's question about this gameweek's plan. The plan is made and it"
    " stands: explain it as it was decided, and never change it, reopen it or"
    " offer another — he sets the team himself if he disagrees. Use only the"
    " detailed report, the earlier questions and your two lookups: player"
    " gives one player's fixtures and numbers by name or id, and plan gives"
    " one of the plans you weighed by its id. Quote the fixtures and the"
    " numbers that decided it — opponents, points, minutes, prices. The numbers"
    " are yours: never hide behind \"the projection\" or \"the model\". If the"
    " owner makes a fair point, say so plainly and say what would change your"
    " mind. You have no news beyond what the report says and cannot search:"
    " when the answer turns on fresh news — an injury, a press conference, a"
    " team sheet — say that you would need fresh news to be sure. Name the"
    " gameweek you are talking about, as the report does (\"For GW8's plan:"
    " …\" in gameweek 8), since the newest report can be the week just played."
    f" Answer in under {ANSWER_WORDS} words of plain text: no markdown,"
    " headings, bullets, bold or tables."
    f" Give the answer by calling {ANSWER}."
)

TOOLS = [
    {
        "name": PLAYER,
        "description": (
            "One player's row from this week's projections: club, position,"
            " price, status and news, expected minutes, each window gameweek's"
            " fixtures (h home, a away; a blank has none, a double two) and"
            " xP, the window total, and his place in the squad. By his name as"
            " FPL spells it (a unique start of it will do) or his id. Several"
            " names that fit come back as a list to choose from by id; a player"
            f" outside the projected pool is \"{NOT_PROJECTED}\"."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_or_id": {"type": "string", "description": "A name, or an id as digits."},
            },
            "required": ["name_or_id"],
        },
    },
    {
        "name": PLAN,
        "description": (
            "One of the plans weighed this week, by the id \"The options weighed\""
            " gives it: its moves, hits, xP and net over the window, its path"
            " through the window, and whether it is the chosen one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": ANSWER,
        "description": (
            f"Your answer to the owner: plain text, under {ANSWER_WORDS} words,"
            " naming the gameweek as GW and its number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
]

ASK = f"Call {ANSWER} now with your answer to his question — no more lookups."
LAST_TURN = f"That was your last lookup. Call {ANSWER} now with your answer to his question."

# Markdown that would arrive in Telegram as its raw characters: a heading or a
# table row at the start of a line, bold, code, an HTML comment.
_MARKDOWN = re.compile(r"(?m)^\s*(?:#|\|)|\*\*|__|`|<!--")


class AskUnavailable(Exception):
    """No valid answer within the turns, or the API could not be reached. The
    message is ours — a stop reason or an error's class name — never the
    API's own words, so it is safe for the log."""


# --- the lookups -----------------------------------------------------------------


def _fold(text: str) -> str:
    """A name as the owner might type it: the inbox's normalising, with the
    accents taken off, so "Gyokeres" finds "Gyökeres". A letter that is not
    an accented one (the "Ø" of "Ødegaard") is left as it is."""
    decomposed = unicodedata.normalize("NFKD", normalise(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def find_player(snapshot: dict, name_or_id: Any) -> dict | str:
    """The player the owner means, his row; several who fit, by name; or
    :data:`NOT_PROJECTED`.

    An id (a number, or digits) is that player or no one. A name is matched
    on its folded ``web_name``: an exact match first — "Wood" is Wood though
    "Woodman" starts the same — then a start of exactly one name ("Cucu").
    Two or more that fit, exact or by their start, come back as a list with
    their clubs and ids, for the gaffer to choose from or ask about; one is
    never picked silently, which is how "Gabriel" stays a question rather
    than becomes the wrong Arsenal player. A typo that starts no name
    ("Palmerr") is no one: a guess would be an answer about someone else.
    """
    players = snapshot["players"]
    if isinstance(name_or_id, bool):
        return NOT_PROJECTED
    if isinstance(name_or_id, int) or (isinstance(name_or_id, str) and name_or_id.strip().isdigit()):
        pid = int(name_or_id)
        return _row(snapshot, pid) if pid in players else NOT_PROJECTED
    wanted = _fold(name_or_id) if isinstance(name_or_id, str) else ""
    if not wanted:
        return NOT_PROJECTED
    exact = [pid for pid, row in players.items() if _fold(row["n"]) == wanted]
    fits = exact or [pid for pid, row in players.items() if _fold(row["n"]).startswith(wanted)]
    if not fits:
        return NOT_PROJECTED
    if len(fits) == 1:
        return _row(snapshot, fits[0])
    return {"matches": [f"{players[pid]['n']} ({players[pid]['c']}, id {pid})" for pid in sorted(fits)]}


def _row(snapshot: dict, pid: int) -> dict:
    """A snapshot row in words a reader needs no key for: the gameweeks by
    their names, a fixture as the report writes it, prices in millions."""
    row = snapshot["players"][pid]
    return {
        "id": pid,
        "name": row["n"],
        "position": row["p"],
        "club": row["c"],
        "club_name": row["cf"],
        "price": price(row["pr"]),
        "selling_price": None if row.get("sp") is None else price(row["sp"]),
        "status": row["st"],
        "news": row["nw"],
        "expected_minutes": row["xm"],
        "fixtures": {
            f"GW{gw}": "+".join(f"{opp}({side})" for opp, side in games) or "blank"
            for gw, games in sorted(row["f"].items())
        },
        "xp": {f"GW{gw}": x for gw, x in sorted(row["x"].items())},
        "window_xp": row["t"],
        "role": _role(snapshot, pid),
    }


def _role(snapshot: dict, pid: int) -> str:
    """His place in the team that plays: the armbands first, then the XI,
    the bench, or none of it."""
    if pid == snapshot.get("captain"):
        return "captain"
    if pid == snapshot.get("vice"):
        return "vice-captain"
    if pid in snapshot.get("xi", []):
        return "XI"
    if pid in snapshot.get("bench", []):
        return "bench"
    return "not in the squad"


def plan_detail(snapshot: dict, plan_id: Any) -> dict:
    """One plan of the registry, with names for ids, or a ToolError naming
    the ids there are."""
    plans = snapshot["plans"]
    if isinstance(plan_id, bool) or not isinstance(plan_id, int) or plan_id not in plans:
        raise ToolError(f"There is no plan {plan_id!r}; the plans weighed are {sorted(plans)}.")
    plan = plans[plan_id]
    return {
        "id": plan_id,
        "chosen": plan_id == snapshot["chosen"],
        "in": [_name(snapshot, p) for p in plan["in"]],
        "out": [_name(snapshot, p) for p in plan["out"]],
        "hits": plan["hits"],
        "xp": plan["xp"],
        "net": plan["net"],
        "path": [
            {
                "gw": move["gw"],
                "in": [_name(snapshot, p) for p in move["in"]],
                "out": [_name(snapshot, p) for p in move["out"]],
                "chip": move.get("chip"),
            }
            for move in plan.get("path", [])
        ],
        "free_transfers_kept": plan.get("ft_kept"),
        "from_a_resolve": plan.get("resolved", False),
    }


# --- the answer ------------------------------------------------------------------


def validate_answer(text: Any, gw: int) -> str:
    """The answer, flattened to one line, or a ToolError that says what to fix."""
    if not isinstance(text, str) or not text.strip():
        raise ToolError("The answer must be a few plain sentences.")
    if _MARKDOWN.search(text) or MARKUP.search(text):
        raise ToolError("Write plain text: no tables, headings, bold or code — Telegram shows them raw.")
    flat = " ".join(text.split())
    words = len(flat.split())
    if words >= ANSWER_WORDS:
        raise ToolError(f"That is {words} words; keep it under {ANSWER_WORDS}.")
    if not re.search(rf"\bGW{gw}\b", flat, re.IGNORECASE):
        raise ToolError(f"Name the gameweek: say GW{gw} in the answer, e.g. \"For GW{gw}'s plan: …\".")
    return flat


# --- the briefing ----------------------------------------------------------------


def ask_briefing(snapshot: dict, report_md: str) -> str:
    """The cached block: which week, then the report as the owner can read it."""
    return "\n".join([
        f"# This week's plan: GW{snapshot['gw']}",
        "",
        f"The owner is asking about GW{snapshot['gw']}'s plan, from the"
        f" {snapshot['mode']} run. Here is the detailed report he can read:",
        "",
        report_md,
    ])


def question_block(snapshot: dict, thread: list[dict], question: str) -> str:
    """The earlier questions this gameweek, oldest first, then this one.

    When the plan the last answer was given on is not the plan now — a run
    since has changed the moves — the gaffer is told, so he does not defend
    a plan that is no longer his.
    """
    lines = []
    if thread:
        lines.append("Earlier questions this gameweek, oldest first:")
        for entry in thread:
            lines += [f"Q: {entry.get('q', '')}", f"A: {entry.get('a', '')}"]
        lines.append("")
        if _changed(snapshot, thread):
            lines += [
                "The plan has changed since those answers: a later run chose"
                " different moves. Answer on the plan in the report.",
                "",
            ]
    lines.append(f"His question now: {question}")
    return "\n".join(lines)


def plan_moves(snapshot: dict) -> dict:
    """The chosen plan's moves as the thread keeps them."""
    out, came = _moves(snapshot)
    return {"out": out, "in": came}


def _changed(snapshot: dict, thread: list[dict]) -> bool:
    """Whether the last answer was given on other moves than the chosen
    plan's now. A thread line with no moves on it says nothing."""
    if not thread:
        return False
    before = thread[-1].get("plan_moves")
    return before is not None and before != plan_moves(snapshot)


# --- the conversation ------------------------------------------------------------


def answer(cfg: Config, snapshot: dict, report_md: str, thread: list[dict], question: str, client: Any) -> str:
    """The gaffer's answer to ``question``, or AskUnavailable.

    ``thread`` is the earlier exchanges this gameweek, already capped by the
    handler; ``client`` is the handler's, built lazily there.
    """
    conversation = _Conversation(client, cfg, snapshot, report_md, thread, question)
    try:
        return conversation.run()
    except anthropic.AnthropicError as error:
        raise AskUnavailable(type(error).__name__) from error


class _Conversation:
    """The review writer's loop with two lookups beside the answer, inside six
    requests."""

    def __init__(
        self, client: Any, cfg: Config, snapshot: dict, report_md: str,
        thread: list[dict], question: str,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.snapshot = snapshot
        self.messages: list[dict] = [{
            "role": "user",
            "content": [
                _cached(ask_briefing(snapshot, report_md)),
                _text(question_block(snapshot, thread, question)),
            ],
        }]
        self.requests = 0
        self.wrong = 0

    def run(self) -> str:
        reason = "no answer"
        while self.requests < MAX_TURNS:
            response = self._ask()
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                raise AskUnavailable("refusal")
            if stop == "max_tokens":
                # Cut off mid-sentence: never happened. The ask joins the
                # still-open user message, so no two user turns run together.
                reason = "max_tokens"
                self._join_ask()
                continue
            self._append(response)
            text, results = self._act(response)
            if text is not None:
                return text
            if self.wrong >= ANSWER_TRIES:
                raise AskUnavailable("invalid answer")
            if not results:
                reason = "no tool call"
                self._join_ask()
                continue
            # Every tool_use needs its own tool_result, in one user message,
            # or the API answers 400. The last request is told it is the last.
            reason = "turns"
            if self.requests == MAX_TURNS - 1:
                results = [*results, _text(LAST_TURN)]
            self.messages.append({"role": "user", "content": results})
        raise AskUnavailable(reason)

    def _act(self, response: Any) -> tuple[str | None, list[dict]]:
        """Every tool call in the turn, in order; the first valid answer wins."""
        results: list[dict] = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            given = getattr(block, "input", None)
            given = given if isinstance(given, dict) else {}
            block_id = getattr(block, "id", "")
            try:
                if name == ANSWER:
                    try:
                        return validate_answer(given.get("text"), self.snapshot["gw"]), results
                    except ToolError:
                        self.wrong += 1
                        raise
                if name == PLAYER:
                    found = find_player(self.snapshot, given.get("name_or_id"))
                    content = found if isinstance(found, str) else json.dumps(found, ensure_ascii=False)
                elif name == PLAN:
                    content = json.dumps(plan_detail(self.snapshot, given.get("id")), ensure_ascii=False)
                else:
                    raise ToolError(f"There is no tool called '{name}'. Use {PLAYER}, {PLAN} or {ANSWER}.")
                results.append(_result(block_id, content))
            except ToolError as error:
                results.append(_result(block_id, str(error), failed=True))
        return None, results

    def _join_ask(self) -> None:
        last = self.messages[-1]
        block = _text(ASK)
        if last["role"] == "user" and last["content"][-1] != block:
            self.messages[-1] = {"role": "user", "content": [*last["content"], block]}
        elif last["role"] != "user":
            self.messages.append({"role": "user", "content": [block]})

    def _ask(self) -> Any:
        """One assistant turn, resumed as often as the server pauses it; every
        request, a resumption too, is one of the six."""
        for _ in range(MAX_RESUMPTIONS + 1):
            if self.requests >= MAX_TURNS:
                raise AskUnavailable("turns")
            self.requests += 1
            response = self.client.messages.create(**self._request())
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
            self._append(response)
        raise AskUnavailable("pause_turn")

    def _request(self) -> dict:
        messages = list(self.messages)
        messages[-1] = _rolling(messages[-1])
        return {
            "model": self.cfg.manager_model,
            "max_tokens": MAX_TOKENS,
            "output_config": EFFORT,
            "system": [_cached(SYSTEM_PROMPT)],
            "tools": TOOLS,
            "messages": messages,
        }

    def _append(self, response: Any) -> None:
        content = getattr(response, "content", None)
        if content:
            self.messages.append({"role": "assistant", "content": content})
