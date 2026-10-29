"""The gaffer, explaining a decision he has already made.

The detailed report (:mod:`aigaffer.detailed.render`) has a prose slot beside
every table: why this transfer, why this captain, why this chip or none, why
this plan over the others, and a short note on each player passed over. The
numbers in those tables are the run's; the words are the gaffer's, and this is
where he writes them — once, after the verdict is final, in one short
conversation that goes into the snapshot's ``reasons`` and nowhere else.

The decision is not reopened. He is told it is final, he is given only what
the snapshot holds, and the one thing he can do is call the answer tool,
``write_detailed_reasons``. Two requests at most, as the gameweek review
(:mod:`aigaffer.manager.review_writer`) has: an answer that is missing or does
not validate gets one more turn with the error in words, and then the reasons
are unavailable (:class:`ReasonsUnavailable`) and the report says so in each
slot. The run never fails for want of them.

What a reason must be, checked here rather than hoped for in the prompt:

* about the games. A transfer reason names an opponent from the window
  fixtures of either man in the move — the man out's or the man in's, since
  "Justin blanks in GW9" and "Cucurella has Wolves at home" are both the
  reason — by its short name (``ARS``) or its full one (``Arsenal``). A double
  gameweek's two opponents both count. And it quotes a number, because "better
  fixtures" with no points beside it is a slogan;
* his own. "The projection says" and "the model says" are refused: the
  projections are the bot's own numbers, and a gaffer who hides behind them is
  telling the owner nothing the table beside the words does not;
* short and plain. 60 words a transfer, 40 a skipped player, 80 each for the
  captain, the chips and the plan; one line each, no headings or markup;
* about the snapshot's own moves, by their ids, each exactly once.

The skipped lines are the exception, and on purpose. A week passes over a
dozen players or more, and one line that runs long or names a man who is not
on the list should cost that line, not the transfer, captain and plan
reasons beside it, which a retry would put at risk. So a bad skipped line is
dropped and the rest kept, and the report leaves that row's note blank. And
on a week with no chip the chips reason may be empty: the report prints it
only beside a chip played.

The model's rules are the manager's: no ``tool_choice`` (Opus 5.5 answers it
with a 400), no thinking parameter, history append-only, every ``tool_use``
answered with a ``tool_result``, and a turn cut off at ``max_tokens`` dropped
as if it never happened.
"""

import re
from typing import Any

import anthropic

from aigaffer.config import Config
from aigaffer.detailed.render import name, _pairs, window, captain_rivals
from aigaffer.manager.agent import EFFORT, MAX_RESUMPTIONS, MAX_TOKENS, _cached, _result, _rolling, _text

# Bound at module scope, not imported inside the function as the review
# writer does: the suite's conftest replaces this one name so that no
# pipeline test with a stubbed gaffer can send the writer's request for real,
# and the client module's own tests still build the real thing.
from aigaffer.manager.client import build_review_client
from aigaffer.manager.tools import MARKUP, ToolError
from aigaffer.report.render import chip_label, plural, price

ANSWER = "write_detailed_reasons"
MAX_REQUESTS = 2

TRANSFER_WORDS = 60
SKIPPED_WORDS = 40
PROSE_WORDS = 80

SYSTEM_PROMPT = (
    "You are the gaffer of a Fantasy Premier League team. Your decision for"
    " this gameweek is final; now write the reasons for it, for the owner's"
    " detailed report. Explain the decision as it was made: do not change it,"
    " reopen it, or suggest another. Use only the facts in the briefing."
    " Name the opponents and the numbers. Each transfer reason names at least"
    " one opponent of the man out or the man in, by the club's short name"
    " (ARS) or its full name (Arsenal) as the briefing gives them, and quotes"
    " the points that decided it. The numbers are yours: never write \"the"
    " projection says\" or \"the model says\". The captain reason says what"
    " the armband was chosen on — his total over the window against his"
    " rivals', or one gameweek's fixture — and names the game that decided it."
    " The chips reason says why this chip, or why none this week. The plan"
    " reason says why the chosen plan beat the others weighed. Each skipped"
    " player gets one short line on why he was passed over. On a week with no"
    " chip the chips reason may be left empty."
    f" Word limits: {TRANSFER_WORDS} words per transfer reason, {SKIPPED_WORDS}"
    f" words per skipped player, and {PROSE_WORDS} words each for the captain,"
    " the chips and the plan. Plain sentences, no headings, bullets or markup."
    " Give one reason for every transfer, by its out and in ids, and one for"
    " every skipped player, by his id."
    f" Answer by calling {ANSWER}."
)

_REASON = {"type": "string"}

TOOLS = [
    {
        "name": ANSWER,
        "description": (
            "Your reasons for this week's decision, one per slot of the detailed"
            " report. Ids are the integers the briefing gives."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transfers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "out": {"type": "integer"},
                            "in": {"type": "integer"},
                            "reason": _REASON,
                        },
                        "required": ["out", "in", "reason"],
                    },
                },
                "captain": _REASON,
                "chips": _REASON,
                "plan": _REASON,
                "skipped": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}, "reason": _REASON},
                        "required": ["id", "reason"],
                    },
                },
            },
            "required": ["transfers", "captain", "chips", "plan", "skipped"],
        },
    }
]

ASK = f"Call {ANSWER} now with your reasons — no more discussion."

# Hiding behind the numbers instead of owning them: "the projection says",
# "the model thinks", "according to the solver". The bare words are fine —
# "Saka's projection of 7.30" is a number with its source — and only the
# phrasing that hands the judgement to something else is refused.
_VOICE = re.compile(
    r"\b(?:projections?|models?|solvers?|optimi[sz]ers?|algorithms?)\s+"
    r"(?:says?|said|thinks?|thought|predicts?|suggests?|wants?|likes?|prefers?|expects?|rates?)\b"
    r"|\baccording to (?:the |my )?(?:projections?|models?|solver|optimi[sz]er|numbers)\b",
    re.IGNORECASE,
)


class ReasonsUnavailable(Exception):
    """The writer gave no valid answer in its two requests, or could not be
    reached. The message is ours — a stop reason or an error's class name —
    never the API's own words, so it is safe for the log."""


# --- validation --------------------------------------------------------------


def validate_reasons(args: dict, snapshot: dict) -> dict:
    """The reasons, cleaned, in the shape the renderer reads, or a ToolError
    that says what to fix.

    The transfers come back in the order the report pairs them, and the
    skipped lines in the snapshot's row order, whatever order they arrived in.
    """
    if not isinstance(args, dict):
        args = {}
    return {
        "transfers": _transfers(args.get("transfers"), snapshot),
        "captain": _prose(args.get("captain"), "captain", PROSE_WORDS),
        "chips": _chips(args.get("chips"), snapshot),
        "plan": _prose(args.get("plan"), "plan", PROSE_WORDS),
        "skipped": _skipped(args.get("skipped"), snapshot),
    }


def _chips(value: Any, snapshot: dict) -> str:
    """The chips reason: required beside a chip played, and allowed to be
    empty on a week with none, where the report has no line to put it on."""
    if (snapshot["decision"].get("chip") or "none") == "none":
        if value is None or (isinstance(value, str) and not value.strip()):
            return ""
    return _prose(value, "chips", PROSE_WORDS)


def _is_id(value: Any) -> bool:
    # A bool is an int to Python and never a player.
    return isinstance(value, int) and not isinstance(value, bool)


def _prose(value: Any, label: str, limit: int) -> str:
    """One reason, flattened to one line, plain, in his own voice, in bounds."""
    text = " ".join(value.split()) if isinstance(value, str) else ""
    if not text:
        raise ToolError(f"{label} must be a sentence or two of plain text.")
    # "<!--" too: the report is markdown, and a comment opened inside a table
    # cell swallows everything after it.
    if text.startswith("#") or "<!--" in text or MARKUP.search(text):
        raise ToolError(f"{label} has a heading or markup in it; write plain sentences.")
    if _VOICE.search(text):
        raise ToolError(
            f"{label} hands the judgement to the projection or the model; the numbers"
            " are your own — say what they are and why they decided it."
        )
    words = len(text.split())
    if words > limit:
        raise ToolError(f"{label} is {words} words; keep it to {limit}.")
    return text


def _move(snapshot: dict, out: int, came: int) -> str:
    return f"out {out} {name(snapshot, out)}, in {came} {name(snapshot, came)}"


def _transfers(items: Any, snapshot: dict) -> list[dict]:
    pairs = _pairs(snapshot)
    listing = "; ".join(_move(snapshot, o, i) for o, i in pairs) or "none — give an empty list"
    if not isinstance(items, list):
        raise ToolError(f"transfers must be a list, one entry per move. The moves are: {listing}.")
    reasons: dict[tuple[int, int], str] = {}
    for item in items:
        out = item.get("out") if isinstance(item, dict) else None
        came = item.get("in") if isinstance(item, dict) else None
        if not (_is_id(out) and _is_id(came)) or (out, came) not in pairs:
            raise ToolError(
                f"transfers has a move that is not this week's (out {out!r}, in {came!r})."
                f" The moves are: {listing}. Use those integer ids."
            )
        if (out, came) in reasons:
            raise ToolError(f"transfers gives {_move(snapshot, out, came)} twice; one reason per move.")
        reasons[(out, came)] = _transfer_reason(item.get("reason"), snapshot, out, came)
    missing = [pair for pair in pairs if pair not in reasons]
    if missing:
        named = "; ".join(_move(snapshot, o, i) for o, i in missing)
        raise ToolError(f"transfers has no reason for {named}; give every move one.")
    return [{"out": o, "in": i, "reason": reasons[(o, i)]} for o, i in pairs]


def _transfer_reason(value: Any, snapshot: dict, out: int, came: int) -> str:
    label = f"the reason for {name(snapshot, out)} → {name(snapshot, came)}"
    text = _prose(value, label, TRANSFER_WORDS)
    opponents = _opponents(snapshot, (out, came))
    # Both men without a game all window has no opponent to name; the check
    # is about naming the games that decided it, and there were none.
    if opponents and not _names_an_opponent(text, opponents):
        names = ", ".join(f"{short} ({full})" if full else short for short, full in opponents.items())
        raise ToolError(
            f"{label} names no opponent. Name at least one of the games that decided"
            f" it, by short or full club name: {names}."
        )
    if not re.search(r"\d", text):
        raise ToolError(f"{label} quotes no number; give the points that decided it.")
    return text


def _full_names(snapshot: dict) -> dict[str, str]:
    """Club short name to full name, from the rows on the board. A club with
    no projected player has no row to read it from, and only its short name
    is known — which is the name the fixture cells print anyway."""
    return {
        row["c"]: row["cf"]
        for row in snapshot["players"].values()
        if row.get("cf") and row["cf"] != row["c"]
    }


def _opponents(snapshot: dict, pids: tuple[int, ...]) -> dict[str, str | None]:
    """Every opponent these men play in the window, short name to full (None
    where the board does not know it), in the order they are played."""
    full = _full_names(snapshot)
    found: dict[str, str | None] = {}
    for pid in pids:
        row = snapshot["players"].get(pid)
        if row is None:
            continue
        for gw in snapshot["events"]:
            for opponent, _ in row["f"].get(gw) or []:
                found.setdefault(opponent, full.get(opponent))
    return found


def _names_an_opponent(text: str, opponents: dict[str, str | None]) -> bool:
    """Whether ``text`` names one of ``opponents``.

    A short name counts in capitals as a word of its own (``ARS``, ``ARS(h)``)
    and in any case only with a venue after it (``wol(a)``, the report's own
    away spelling): FPL's short names include NEW and SUN, which in lower case
    are English words, and BOU is the start of "bout". A full name counts in
    any case, as a whole word.
    """
    for short, full in opponents.items():
        escaped = re.escape(short)
        if re.search(rf"(?<![A-Za-z]){escaped}(?![A-Za-z])", text):
            return True
        if re.search(rf"(?<![A-Za-z]){escaped}\s?\([ha]\)", text, re.IGNORECASE):
            return True
        if full and re.search(rf"(?<![A-Za-z]){re.escape(full)}(?![A-Za-z])", text, re.IGNORECASE):
            return True
    return False


def _skipped(items: Any, snapshot: dict) -> list[dict]:
    """The skipped lines that hold up, in the snapshot's row order.

    Never a ToolError (see the module docstring): a line that is not a dict,
    names an id that is not a skipped row, says it twice, runs past the word
    limit, carries markup or hands the judgement to the model is dropped, and
    the rest are kept. A missing line is a blank note in the report.
    """
    ids = [row["id"] for row in snapshot.get("skipped") or []]
    reasons: dict[int, str] = {}
    for item in items if isinstance(items, list) else []:
        pid = item.get("id") if isinstance(item, dict) else None
        if not _is_id(pid) or pid not in ids or pid in reasons:
            continue
        try:
            reasons[pid] = _prose(item.get("reason"), "skipped", SKIPPED_WORDS)
        except ToolError:
            continue
    return [{"id": pid, "reason": reasons[pid]} for pid in ids if pid in reasons]


# --- the briefing ----------------------------------------------------------------


def reasons_briefing(snapshot: dict) -> str:
    """What the writer reads: his decision and its words, the moves by the ids
    the answer must use, the men involved with their games, the captain's
    rivals, the skipped rows, the plans weighed and the chips."""
    decision = snapshot["decision"]
    pairs = _pairs(snapshot)
    rivals = captain_rivals(snapshot)
    lines = [
        f"# GW{snapshot['gw']}: the reasons for your {snapshot['mode']} decision",
        "",
        "Your decision below is final. Write the reasons for it; do not change it.",
        "",
        "## Your decision",
        f"Rationale: {decision.get('rationale') or '(none recorded)'}",
    ]
    for adjustment in decision.get("adjustments") or []:
        lines.append(
            f"- {name(snapshot, adjustment.get('player_id'))}:"
            f" {adjustment.get('expected_minutes', 0):g} expected minutes"
            f" ({adjustment.get('reason', '')})"
        )
    chip = decision.get("chip") or "none"
    if chip == "none":
        lines.append("Chip: none this week.")
    else:
        justification = decision.get("chip_justification") or ""
        lines.append(f"Chip: {chip_label(chip)}" + (f" — {justification}" if justification else ""))
    captain, vice = snapshot.get("captain"), snapshot.get("vice")
    if captain is not None:
        lines.append(
            f"Captain: {name(snapshot, captain)} ({window(snapshot, captain):.2f} over the window);"
            f" vice: {name(snapshot, vice)}"
        )

    lines += ["", f"## The transfers (one reason each, by these ids; {TRANSFER_WORDS} words at most)"]
    if not pairs:
        lines.append("No transfers this week: give transfers as an empty list.")
    for out, came in pairs:
        players = snapshot["players"]
        gap = players[came]["t"] - players[out]["t"]
        plain = window(snapshot, came) - window(snapshot, out)
        lines.append(
            f"- out {out} {name(snapshot, out)} → in {came} {name(snapshot, came)}:"
            f" {gap:+.2f} weighted ({plain:+.2f} plain) over the window"
        )

    involved = [pid for pair in pairs for pid in pair]
    involved += [p for p in (captain, vice) if p is not None] + rivals
    for row in snapshot.get("skipped") or []:
        involved += [row["id"], row["replaces"]]
    events = ", ".join(f"GW{gw}" for gw in snapshot["events"])
    lines += [
        "",
        "## The players involved",
        f"The window is {events}. Each gameweek: the opponent, h or a, and the"
        " player's xP; then his plain total and his weighted one (later"
        " gameweeks count for less, as the planner does).",
    ]
    lines += [_player_line(snapshot, pid) for pid in dict.fromkeys(involved) if pid in snapshot["players"]]

    lines += ["", "## The captain's rivals (by total xP over the window)"]
    lines.append(
        ", ".join(f"{name(snapshot, p)} ({window(snapshot, p):.2f})" for p in rivals) or "None."
    )

    lines += ["", f"## Transfers skipped (one line each, by id; {SKIPPED_WORDS} words at most)"]
    rows = snapshot.get("skipped") or []
    if not rows:
        lines.append("None: give skipped as an empty list.")
    for row in rows:
        lines.append(
            f"- id {row['id']} {name(snapshot, row['id'])} ({row['pos']}) instead of"
            f" {name(snapshot, row['replaces'])}: {row['gap']:+.2f} weighted,"
            f" {row['price_diff'] / 10:+.1f}m, {row['tag']}"
        )

    lines += ["", "## The plans weighed"]
    for pid, plan in sorted(snapshot["plans"].items()):
        marks = [m for m, on in (("chosen", pid == snapshot["chosen"]), ("re-solved", plan.get("resolved"))) if on]
        mark = f" ({', '.join(marks)})" if marks else ""
        lines.append(
            f"- Plan {pid}{mark}: in {_names(snapshot, plan['in'])}; out {_names(snapshot, plan['out'])};"
            f" {plural(plan['hits'], 'hit')}; xP {plan['xp']:.2f}; net {plan['net']:.2f}"
        )

    lines += ["", "## Chips", "Points each chip would add this week:"]
    for kind, value in (snapshot.get("chip_evs") or {}).items():
        label = chip_label(kind) + (" (over the window)" if kind == "wildcard" else "")
        lines.append(f"- {label}: {value:+.2f}")
    if snapshot.get("calendar"):
        lines.append("Calendar:")
        lines += [f"- {line}" for line in snapshot["calendar"]]
    return "\n".join(lines)


def _names(snapshot: dict, pids: list[int]) -> str:
    return ", ".join(name(snapshot, p) for p in pids) or "none"


def _player_line(snapshot: dict, pid: int) -> str:
    """``- 4 Justin (DEF, FUL) £4.6m, xMins 90: GW8 ARS (Arsenal) h 3.00; GW9
    no game 0.00; plain 3.00, weighted 3.00`` — the opponent's full name beside
    the short one where the board knows it, so the writer has both to choose
    from and the check accepts either."""
    row = snapshot["players"][pid]
    full = _full_names(snapshot)
    cells = []
    for gw in snapshot["events"]:
        games = row["f"].get(gw) or []
        xp = row["x"].get(gw, 0.0)
        if not games:
            cells.append(f"GW{gw} no game {xp:.2f}")
            continue
        named = " + ".join(
            f"{opp} ({full[opp]}) {where}" if opp in full else f"{opp} {where}" for opp, where in games
        )
        cells.append(f"GW{gw} {named} {xp:.2f}")
    line = (
        f"- {pid} {row['n']} ({row['p']}, {row['c']}) {price(row['pr'])}, xMins {row['xm']:g}: "
        + "; ".join(cells)
        + f"; plain {window(snapshot, pid):.2f}, weighted {row['t']:.2f}"
    )
    if row.get("nw"):
        line += f"; news: {row['nw']}"
    return line


# --- the conversation ------------------------------------------------------------


def write_reasons(cfg: Config, snapshot: dict, client: Any = None) -> dict:
    """The gaffer's reasons for ``snapshot``'s decision, or ReasonsUnavailable.

    The client is the review's (:func:`~aigaffer.manager.client.build_review_client`):
    one short request with no searches, three minutes and no retry, for the
    same reason — a run that cannot get the reasons goes on without them.
    """
    if client is None:
        client = build_review_client(cfg)
    conversation = _Conversation(client, cfg, snapshot)
    try:
        return conversation.run()
    except anthropic.AnthropicError as error:
        raise ReasonsUnavailable(type(error).__name__) from error


class _Conversation:
    """The review writer's loop (:mod:`aigaffer.manager.review_writer`), on a
    different answer: the same two requests, the same dropped ``max_tokens``
    turn, the same one tool result per tool call."""

    def __init__(self, client: Any, cfg: Config, snapshot: dict) -> None:
        self.client = client
        self.cfg = cfg
        self.snapshot = snapshot
        self.messages: list[dict] = [{"role": "user", "content": [_cached(reasons_briefing(snapshot))]}]

    def run(self) -> dict:
        reason = "no answer"
        for _ in range(MAX_REQUESTS):
            response = self._ask()
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                raise ReasonsUnavailable("refusal")
            if stop == "max_tokens":
                # Cut off mid-sentence: never happened. The ask joins the
                # still-open user message, so no two user turns run together.
                reason = "max_tokens"
                self._join_ask()
                continue
            self._append(response)
            answer, results = self._act(response)
            if answer is not None:
                return answer
            if not results:
                reason = "no tool call"
                self._join_ask()
                continue
            # Every tool_use needs its own tool_result, in one user message,
            # or the API answers 400.
            reason = "invalid answer"
            self.messages.append({"role": "user", "content": results})
        raise ReasonsUnavailable(reason)

    def _act(self, response: Any) -> tuple[dict | None, list[dict]]:
        """Every tool call in the turn; the first valid answer wins."""
        results: list[dict] = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            given = getattr(block, "input", None)
            try:
                if name != ANSWER:
                    raise ToolError(f"There is no tool called '{name}'. Call {ANSWER}.")
                return validate_reasons(given if isinstance(given, dict) else {}, self.snapshot), results
            except ToolError as error:
                results.append(_result(getattr(block, "id", ""), str(error), failed=True))
        return None, results

    def _join_ask(self) -> None:
        last = self.messages[-1]
        block = _text(ASK)
        if last["role"] == "user" and last["content"][-1] != block:
            self.messages[-1] = {"role": "user", "content": [*last["content"], block]}
        elif last["role"] != "user":
            self.messages.append({"role": "user", "content": [block]})

    def _ask(self) -> Any:
        """One assistant turn, resumed as often as the server pauses it."""
        for _ in range(MAX_RESUMPTIONS + 1):
            response = self.client.messages.create(**self._request())
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
            self._append(response)
        raise ReasonsUnavailable("pause_turn")

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
