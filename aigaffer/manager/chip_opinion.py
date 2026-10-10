"""The gaffer's opinion on a chip what-if: one short conversation, one answer.

A what-if (:mod:`aigaffer.whatif`) is numbers: two window solves, one playing
the chip at the coming deadline and one not, and a band read off the
difference. What the numbers cannot know is Friday's press conference, so the
owner's second message is a manager's reading of them — the same model as the
weekly manager, handed a briefing of the numbers, a web search, and one tool
to answer with.

It is deliberately smaller than :mod:`aigaffer.manager.agent`. There is
nothing to adjust and nothing to re-solve: he cannot move the numbers, only
judge them, and news that would move them goes in his text. So the loop is
that module's loop with the decision tools taken away — the same pause
resumption, the same container carry, the same moving cache marker, the same
append-only history (Opus 5.5 binds its thinking blocks to the conversation),
and the same way of asking for the answer in words rather than with a forced
``tool_choice``, which this model rejects.

Only the small block helpers are shared: ``_cached``, ``_rolling``, ``_text``
and ``_result`` are imported from there. The loop itself — ``_Conversation``
here, with its turn, pause-resumption, container and answer-asking handling —
mirrors :class:`aigaffer.manager.agent._Conversation` and is deliberately
copied, not shared, for now: pulling a common base out of the manager's loop
is a refactor outside this feature. Until that is done, a fix to either loop's
turn, pause or container handling must be made in both.

And like that module it never raises. A refusal, a rate limit, a dead
connection, six turns without an answer, eight minutes on the clock — every
one is a :class:`ChipOpinion` whose ``verdict`` is None and whose ``source``
says why, because message 1 has already gone and stands without this.
"""

from dataclasses import dataclass
from datetime import date
from time import monotonic
from typing import TYPE_CHECKING, Any

import anthropic

from aigaffer.chips import CHIP_ORDER, held_by_rules, held_for
from aigaffer.config import Config
from aigaffer.manager.agent import (
    EFFORT,
    MAX_RESUMPTIONS,
    MAX_TOKENS,
    _cached,
    _result,
    _rolling,
    _text,
)
from aigaffer.manager.briefing import (
    _Board,
    _by_position,
    _described,
    _planned,
    _player_line,
    _safe,
    _squad,
)
from aigaffer.manager.playbook import CHIP_PLAYBOOK
from aigaffer.manager.tools import MARKUP, ToolError
from aigaffer.report.render import chip_label, horizon_of, price, render_chip_calendar
from aigaffer.whatif import HOLD, MARGINAL, PLAY, next_break

if TYPE_CHECKING:
    from aigaffer.orchestrator import PreparedWeek
    from aigaffer.whatif import PathSide, WhatIf

# Six turns is a long conversation for an opinion on numbers already worked
# out: a turn to search, a turn to search again, a turn to answer, and room for
# a correction or two. The sixth asks for the answer in words.
MAX_TURNS = 6
# Eight minutes, checked before every request. The what-if has a deadline of
# its own — the owner is waiting on his phone — and the inbox unit's timeout
# has to cover the solves and this together.
TIME_BUDGET_SECONDS = 480
OUT_OF_TIME = "out of time"
# Per request, like the manager's: what a turn may spend, not a conversation's
# budget. Five is the handful of players the gain actually rests on.
MAX_SEARCHES = 5

ANSWER = "give_chip_opinion"
GAFFER = "gaffer"
FAILED = "failed: {reason}"
UNREACHABLE = "The gaffer couldn't be reached ({reason}) — the numbers above stand."

# A synopsis: two or three sentences, because the lineup and the numbers have
# already gone to his phone in message 1 and this is only what they cannot
# see. Long enough not to be a one-liner, short enough to read at a glance.
MIN_OPINION = 80
MAX_OPINION = 400

VERDICTS = [PLAY, MARGINAL, HOLD]
# How far apart two bands are: Play and Hold are two steps, either and
# Marginal one.
_STEP = {PLAY: 0, MARGINAL: 1, HOLD: 2}
_SHOUTED = {PLAY: "PLAY", MARGINAL: "MARGINAL", HOLD: "HOLD"}
_SPOKEN = {PLAY: "Play", MARGINAL: "Marginal", HOLD: "Hold"}
# The words on the owner's phone: the band constant stays "marginal" in the
# code, but he reads MAYBE.
_PHONE = {PLAY: "PLAY", MARGINAL: "MAYBE", HOLD: "HOLD"}

# Turned back once if he answers having searched nothing. The first live run
# answered in two turns with no searches at all, judging from the numbers
# alone, which defeats the point of asking him.
SEARCH_FIRST = (
    "Search first: you have not checked any team news. Search the new"
    " signings and flagged players, then answer."
)

NUDGE = (
    "That is not an answer yet. If you ended your turn to refresh your search"
    " allowance, it is fresh now — continue the research you paused for."
    f" Otherwise call {ANSWER} with your verdict and your opinion."
)
ANSWER_NOW = (
    f"Answer now. This turn must be exactly one {ANSWER} call and nothing"
    " else: no searches, no other tools, and no reply that is only text. Give"
    " your verdict (play, marginal or hold), the better week or null, the new"
    " fact or null, and the opinion."
)

_BASE = f"""\
You are the gaffer of a Fantasy Premier League team, asked for an opinion on a \
chip what-if. The owner tapped a button on his phone asking what playing a \
chip at the coming deadline would do. The solver has answered with numbers, \
which he already has. You are asked to read them against the news.

What a what-if is. Two window solves on the same board, identical except for \
one thing: in one the chip is played at the coming deadline ("on"), in the \
other it is not played this week ("off") and may still be played later in the \
window. Both get the same future free transfers, and hits count. Net is the \
on-path's objective minus the off-path's; both objectives carry hits and the \
chip calendar's bars — the value of keeping each chip they play — so net is \
already "now against keeping it". The band is read off net: wildcard ±8 xP, \
free hit ±4 xP. Play above the band, Hold below it, Marginal inside it. A \
solve that stopped on its time limit is marked; then the band needs twice the \
margin.

What it is not. Not a decision, not a recommendation to act, and never \
recorded. You cannot change the numbers: you have no tools to re-project or \
re-solve. If the news would move them, say so in your text and say which way.

Your verdict uses the same yardstick as the numbers. You may move the band one \
step on judgement (Play to Marginal, Marginal to Hold, and back). Moving it two \
steps — Play to Hold, or Hold to Play — needs a named fact the numbers do not \
contain, such as an injury or press-conference news, given in new_fact. If \
your view differs from the latest report's chip plan, say why.

Before you give your answer, use web_search on current team news for the \
chip's new signings and for any flagged or doubtful players (the briefing \
names them): at least one search, at most {MAX_SEARCHES} a turn. The numbers cannot \
see Friday's press conference; you can. Do not answer from the numbers alone.

Check, in this order, and spend your searches on what matters:
1. Where the gain comes from. If more than about 40% of it comes from one or \
two players, or from this week's captain, it is fragile: search those \
players' news first.
2. Hot form and thin minutes data in the new picks: points above their xG, new \
signings, players just back from injury. This is where a projection is most \
optimistic.
3. Flagged players in the current squad and in the simulated one, and for how \
long they are out. Rotation, and European midweeks.
4. Is it a wildcard problem — three or more gameweeks of squad issues — or a \
free-hit one? For a free hit, is the squad it reverts to fine for the week after?
5. Sequencing. Is the bench boost still held (a wildcard a week or three \
before a bench-boost week builds the bench)? A free hit and a bench boost never \
share a week, and a free hit in GW19 rules out one in GW20.
6. A chip cannot be undone once confirmed. Advise setting it late, after the \
press conferences.
7. The expiry countdown, and the next international break.
8. Team value only when the band is Marginal, and only as a tiebreak.

Search the simulated squad's players and the flagged ones, not general \
previews. A turn boundary refreshes your search allowance.

Finish with {ANSWER}, exactly once: the verdict, the better week (a gameweek \
number, or null), the new fact (or null), and the opinion — a synopsis of two \
or three sentences ({MIN_OPINION} to {MAX_OPINION} characters of plain prose): \
the key reasons and anything the numbers cannot see (injury news, a better \
week). Not the lineup or the numbers: he already has both, in the message \
before yours. He reads it on his phone. Anything you write outside that call \
is discarded."""

SYSTEM_PROMPT = _BASE + "\n\n" + CHIP_PLAYBOOK

TOOLS: list[dict] = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES},
    {
        "name": ANSWER,
        "description": (
            "Give your opinion on the what-if and end the conversation. The"
            " verdict may differ from the numbers' band by one step on"
            " judgement; two steps needs new_fact."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": VERDICTS,
                    "description": "play, marginal or hold.",
                },
                # A value or null, never the empty string: the manager's
                # chip_justification taught us what an empty value invites.
                "better_week": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "The gameweek you would play it in instead, or null.",
                },
                "new_fact": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": (
                        "A fact the numbers do not contain — an injury, press-"
                        "conference news — with its source. Needed only when"
                        " the verdict is two steps from the band; else null."
                    ),
                },
                "opinion": {
                    "type": "string",
                    "description": (
                        f"A synopsis for the owner of two or three sentences,"
                        f" {MIN_OPINION}-{MAX_OPINION} characters of plain prose:"
                        " the key reasons and anything the numbers cannot see"
                        " (injury news, a better week), not the lineup or"
                        " numbers he already has."
                    ),
                },
            },
            "required": ["verdict", "better_week", "new_fact", "opinion"],
            "additionalProperties": False,
        },
    },
]

# $ per million tokens: input, output, cache read, cache write (5-minute TTL).
# Anthropic's list prices as the senior checked them on 2026-10-08 (Opus 5.5:
# $4/$20, cache reads $0.20; writes 1.25x input). Feeds only the local log's
# estimate; update the table, not the test, when prices change.
PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-opus-5": (5.0, 25.0, 0.50, 6.25),
}


@dataclass(frozen=True)
class Usage:
    """Tokens the conversation was billed for, summed over every response —
    paused ones included, because a paused turn was run and paid for."""

    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0

    def est_usd(self, model: str) -> float | None:
        """List-price dollars for ``model``, or None for a model the table does
        not know: an estimate nobody can check is worse than none."""
        rates = PRICES.get(model)
        if rates is None:
            return None
        per_input, per_output, per_read, per_write = rates
        dollars = (
            self.input * per_input
            + self.output * per_output
            + self.cache_read * per_read
            + self.cache_write * per_write
        ) / 1_000_000
        return round(dollars, 4)


@dataclass(frozen=True)
class ChipOpinion:
    verdict: str | None
    better_week: int | None
    new_fact: str | None
    opinion: str
    searches: int
    turns: int
    seconds: float
    source: str
    usage: Usage


def run_chip_opinion(client: Any, cfg: Config, briefing: str, band: str) -> ChipOpinion:
    """Put the what-if to the gaffer and come back with an opinion, always.

    ``band`` is the numbers' band, which the two-step rule is measured from.
    ``client`` is injected and only ``messages.create`` is called on it, so the
    suite runs the whole loop against a scripted stand-in.
    """
    conversation = None
    try:
        conversation = _Conversation(client, cfg, briefing, band)
        return conversation.run()
    except (
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError,
        anthropic.AnthropicError,
    ) as error:
        return _failed(conversation, FAILED.format(reason=type(error).__name__))
    except Exception as error:  # ours, then — and still not the owner's problem
        return _failed(conversation, FAILED.format(reason=f"unexpected {type(error).__name__}"))


def _failed(conversation: "_Conversation | None", source: str) -> ChipOpinion:
    """An opinion that never came, with what it cost on the way."""
    if conversation is None:
        return ChipOpinion(None, None, None, "", 0, 0, 0.0, source, Usage())
    return conversation.finish(None, source)


def validate_opinion(args: dict, band: str) -> tuple[str, int | None, str | None, str]:
    """The answer, if it is one the owner can read; otherwise the correction.

    Every error says what was wrong and what would be right — it goes back to
    him as a tool result and is his only chance to fix it.
    """
    verdict = args.get("verdict")
    if verdict not in VERDICTS:
        raise ToolError(f"verdict must be one of play, marginal or hold, not {verdict!r}.")

    better = args.get("better_week")
    if better is not None and (
        isinstance(better, bool) or not isinstance(better, int) or not 1 <= better <= 38
    ):
        raise ToolError(f"better_week must be a gameweek number from 1 to 38, or null — not {better!r}.")

    fact = args.get("new_fact")
    fact = fact.strip() if isinstance(fact, str) else ""
    opinion = args.get("opinion")
    opinion = opinion.strip() if isinstance(opinion, str) else ""

    if MARKUP.search(opinion) or MARKUP.search(fact):
        raise ToolError(
            "That call came back with parameter markup inside a field. Call"
            f" {ANSWER} again with each field holding only its own plain text."
        )
    if len(opinion) < MIN_OPINION:
        raise ToolError(
            f"That opinion is {len(opinion)} characters; the owner reads it as"
            f" your whole view. Write a synopsis of two or three sentences,"
            f" {MIN_OPINION} to {MAX_OPINION} characters: the key reasons and"
            " what the numbers cannot see."
        )
    if len(opinion) > MAX_OPINION:
        raise ToolError(
            f"That opinion is {len(opinion)} characters; it goes to a phone."
            f" Keep it to {MAX_OPINION}, two or three sentences: the key reasons"
            " and anything the numbers cannot see, not the lineup or numbers."
        )
    if abs(_STEP[verdict] - _STEP.get(band, _STEP[MARGINAL])) == 2 and not fact:
        raise ToolError(
            f"The numbers say {_SPOKEN.get(band, band)} and you said"
            f" {_SPOKEN[verdict]}: that is two steps. Move one step"
            " (to marginal) on judgement, or name the fact the numbers do not"
            " contain in new_fact — an injury, press-conference news — with its"
            " source."
        )
    return verdict, better, fact or None, opinion


class _Conversation:
    """One run of the loop and what accumulates across its turns."""

    def __init__(self, client: Any, cfg: Config, briefing: str, band: str) -> None:
        self.client = client
        self.cfg = cfg
        self.band = band
        # The briefing never changes once written, so it is cached with the
        # system prompt.
        self.messages: list[dict] = [{"role": "user", "content": [_cached(briefing)]}]
        self.searches = 0
        # Whether the search-first guard has fired: it fires once, and the
        # next answer stands whatever the count, since search may really be
        # unavailable.
        self.search_first_given = False
        # Set once the answer has been asked for in words. A flag, not a look
        # at the last message: a paused final turn leaves an assistant message
        # at the tail, and the guard must still stand down.
        self.answer_asked = False
        self.turns = 0
        self.container: str | None = None
        self.tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        self.started = monotonic()

    def run(self) -> ChipOpinion:
        nudged = False
        for turn in range(1, MAX_TURNS + 1):
            if self._expired():
                return self.finish(None, OUT_OF_TIME)
            if turn == MAX_TURNS:
                self._ask_for_answer()
            response = self._ask()
            if response is None:
                return self.finish(None, OUT_OF_TIME if self._expired() else FAILED.format(reason="pause_turn"))
            self.turns = turn

            # Before the content, always: a refusal may have none worth reading.
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                return self.finish(None, FAILED.format(reason="refusal"))
            if stop == "max_tokens":
                # A turn cut off mid-sentence never happened: dropped, and the
                # next turn asks for the answer.
                self._ask_for_answer()
                continue

            answer, results = self._act(response)
            if answer is not None:
                return self.finish(answer, GAFFER)
            if results:
                self._append(response)
                self.messages.append({"role": "user", "content": results})
                nudged = False
                continue

            # He talked instead of answering: a word, then the ask.
            self._append(response)
            if nudged:
                self._ask_for_answer()
            else:
                self.messages.append({"role": "user", "content": [_text(NUDGE)]})
            nudged = True
        return self.finish(None, f"no decision in {MAX_TURNS} turns")

    def finish(self, answer: tuple | None, source: str) -> ChipOpinion:
        verdict, better, fact, opinion = answer if answer is not None else (None, None, None, "")
        return ChipOpinion(
            verdict=verdict,
            better_week=better,
            new_fact=fact,
            opinion=opinion,
            searches=self.searches,
            turns=self.turns,
            seconds=round(monotonic() - self.started, 1),
            source=source,
            usage=Usage(**self.tokens),
        )

    def _ask_for_answer(self) -> None:
        """:data:`ANSWER_NOW` at the end of the conversation, once — joined to
        an unanswered user message as a copy, or a message of its own after an
        assistant turn. The manager's rule, for the manager's reasons
        (:meth:`aigaffer.manager.agent._Conversation._ask_for_decision`)."""
        self.answer_asked = True
        block = _text(ANSWER_NOW)
        last = self.messages[-1]
        if last["role"] == "user":
            content = last["content"]
            if content and content[-1] == block:
                return
            self.messages[-1] = {"role": "user", "content": [*content, block]}
        else:
            self.messages.append({"role": "user", "content": [block]})

    def _ask(self) -> Any:
        """One assistant turn, resumed as often as the server pauses it."""
        for attempt in range(MAX_RESUMPTIONS + 1):
            if attempt and self._expired():
                return None
            response = self.client.messages.create(**self._request())
            self._meter(response)
            self._note_container(response)
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
            self._count(response)
            self._append(response)
        return None

    def _expired(self) -> bool:
        return monotonic() - self.started > TIME_BUDGET_SECONDS

    def _request(self) -> dict:
        """System prompt and briefing cached, the marker rolling on a copy of
        the last message, no ``tool_choice`` ever."""
        messages = list(self.messages)
        messages[-1] = _rolling(messages[-1])
        request = {
            "model": self.cfg.manager_model,
            "max_tokens": MAX_TOKENS,
            "output_config": EFFORT,
            "system": [_cached(SYSTEM_PROMPT)],
            "tools": TOOLS,
            "messages": messages,
        }
        if self.container is not None:
            request["container"] = self.container
        return request

    def _act(self, response: Any) -> tuple[tuple | None, list[dict]]:
        """Every tool call in the turn; a valid answer ends the conversation."""
        self._count(response)
        results: list[dict] = []
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            block_id = getattr(block, "id", "")
            given = getattr(block, "input", None)
            args = given if isinstance(given, dict) else {}
            try:
                if name != ANSWER:
                    raise ToolError(
                        f"There is no tool called '{name}'. The tools are"
                        f" web_search and {ANSWER}."
                    )
                if self._must_search_first():
                    self.search_first_given = True
                    raise ToolError(SEARCH_FIRST)
                return validate_opinion(args, self.band), results
            except ToolError as error:
                results.append(_result(block_id, str(error), failed=True))
        return None, results

    def _must_search_first(self) -> bool:
        """True for an answer that arrives with nothing searched, once.

        Not on a turn where the answer was asked for in words: that text
        tells him not to search, and the guard must not contradict it.
        """
        return not (self.searches or self.search_first_given or self.answer_asked)

    def _count(self, response: Any) -> None:
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "web_search_tool_result":
                self.searches += 1

    def _meter(self, response: Any) -> None:
        """Add one response's usage. Fields the SDK leaves None count nothing."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        for ours, theirs in (
            ("input", "input_tokens"),
            ("cache_read", "cache_read_input_tokens"),
            ("cache_write", "cache_creation_input_tokens"),
            ("output", "output_tokens"),
        ):
            value = getattr(usage, theirs, None)
            if isinstance(value, int) and not isinstance(value, bool):
                self.tokens[ours] += value

    def _note_container(self, response: Any) -> None:
        container = getattr(response, "container", None)
        identifier = getattr(container, "id", None)
        if isinstance(identifier, str) and identifier:
            self.container = identifier

    def _append(self, response: Any) -> None:
        content = getattr(response, "content", None)
        if content:
            self.messages.append({"role": "assistant", "content": content})


# --- message 2 ----------------------------------------------------------------


def render_opinion(opinion: ChipOpinion, band: str) -> str:
    """Message 2: the gaffer's brief synopsis, or why there is none.

    One line: ``🧠 Gaffer: HOLD — <synopsis>``, with the better week in
    brackets when he named one. Message 1 already carries the numbers' word,
    the points and the lineup, so this only adds what he thinks; when he
    departs from the numbers' band it says so in the same line, ``HOLD, not
    MAYBE``. The words are the phone's: PLAY, MAYBE (the code's marginal),
    HOLD. No search count: the owner does not read it, and the log has it.
    """
    if opinion.verdict is None:
        reason = opinion.source.removeprefix("failed: ")
        return UNREACHABLE.format(reason=reason)
    better = f" (GW{opinion.better_week})" if opinion.better_week else ""
    dissent = f", not {_PHONE.get(band, band.upper())}" if opinion.verdict != band else ""
    return f"🧠 Gaffer: {_PHONE[opinion.verdict]}{better}{dissent} — {opinion.opinion}"


# --- the briefing ---------------------------------------------------------------

_CHIP_NAMES = {chip: chip_label(chip) for chip in CHIP_ORDER}


def build_chip_briefing(
    week: "PreparedWeek",
    whatif: "WhatIf",
    verdict: dict | None,
    today: date | None = None,
) -> str:
    """Everything he needs to read the numbers against the news. Pure; no I/O.

    The squad lines and the plan path lines are the weekly briefing's own
    (:mod:`aigaffer.manager.briefing`), names flattened by the same
    ``_safe``, so the two documents read the same players the same way.
    """
    inputs = week.effective
    board = _Board(
        players=inputs.players,
        clubs={team.id: team.short_name for team in inputs.bootstrap.teams},
        projections=week.projections,
        horizon=horizon_of(week.projections),
        xmins=week.xmins,
    )
    event = whatif.event
    deadlines = {e.id: e.deadline_time for e in inputs.bootstrap.events}
    chip = _CHIP_NAMES.get(whatif.kind, whatif.kind)
    today = today or date.today()

    sections = [
        f"# AI Gaffer — chip what-if: {chip} in GW{event}\n\n"
        f"Today is {today:%a %d %b %Y}. The GW{event} deadline is"
        f" {_when(deadlines.get(event))}. You are asked for an opinion on the"
        " numbers below, not for a decision.",
        _numbers(whatif),
        "## Chip plays by path\n\n"
        f"- On: {_plays(whatif.on.chips)}\n- Off: {_plays(whatif.off.chips)}",
        _if_played(whatif, board),
        _without(whatif, board),
        _squad(inputs.squad.player_ids, board, event, False),
        _position(week, whatif, verdict, deadlines, board),
    ]
    calendar = render_chip_calendar(week.calendar, event)
    if calendar is not None:
        sections.append(calendar)
    return "\n\n".join(sections)


def _numbers(whatif: "WhatIf") -> str:
    window = f"GW{whatif.window[0]}–{whatif.window[-1]}"
    lean = f", lean {whatif.lean}" if whatif.lean else ""
    band = f"{_SHOUTED.get(whatif.band, whatif.band)}{lean}"
    lines = [
        "## The numbers",
        "",
        f"- Band: {band} (noise ±{whatif.margin:g} xP)",
        f"- Net: {whatif.net:+.1f} xP over {window} — the number the band reads",
        f"- Gain: {whatif.gain:+.1f} xP, decayed, after hits, before chip bars",
        f"- Chip bars: on pays {whatif.on.path.bars_paid:.1f}, off pays"
        f" {whatif.off.path.bars_paid:.1f} (difference {whatif.bars_diff:+.1f})",
        "- Gain by week (on − off): "
        + ", ".join(f"GW{w} {g:+.1f}" for w, g in sorted(whatif.weekly_gain.items())),
        f"- Solver: {_proven(whatif)}",
    ]
    if whatif.refunded_hits:
        lines.append(f"- Hits refunded by the wildcard: {whatif.refunded_hits}")
    terms = whatif.fh_terms
    if terms is not None:
        lines.append(
            f"- Free hit: team vs your GW{whatif.event} XI {terms.one_week:+.1f}"
            f" + knock-on {terms.knock_on:+.1f} − keeping it {terms.keeping:.1f}"
            f" = net {whatif.net:+.1f}"
        )
    lines.append(
        "- Minutes: "
        + (whatif.minutes_source or "the model's own — no report has run this gameweek")
    )
    return "\n".join(lines)


def _proven(whatif: "WhatIf") -> str:
    stopped = [
        name
        for name, side in (("on", whatif.on), ("off", whatif.off))
        if not side.path.proven
    ]
    if not stopped:
        return "both solves proven optimal"
    return " and ".join(f"the {name} solve" for name in stopped) + " stopped on its time limit"


def _plays(chips: dict[int, str]) -> str:
    if not chips:
        return "none in the window"
    return ", ".join(f"GW{w} {_CHIP_NAMES.get(c, c)}" for w, c in sorted(chips.items()))


def _if_played(whatif: "WhatIf", board: _Board) -> str:
    on = whatif.on
    event = whatif.event
    lines = [f"## If played: GW{event}", ""]
    if on.path.week1_freehit_squad:
        lines.append("The free-hit fifteen, reverting after the week:")
        team = on.path.week1_freehit_squad
    else:
        moves = len(on.plan.transfers_in)
        lines.append(
            f"{moves} {'move' if moves == 1 else 'moves'} — out "
            + ", ".join(_described(p, board) for p in on.plan.transfers_out)
            + "; in "
            + ", ".join(_described(p, board) for p in on.plan.transfers_in)
        )
        team = on.plan.squad
    lines += [f"- {_player_line(pid, board, event)}" for pid in _by_position(team, board, event)]
    captain, vice = on.lineup.captain, on.lineup.vice
    lines.append(
        f"Captain {_safe(board.players[captain].web_name)} (id {captain}),"
        f" vice {_safe(board.players[vice].web_name)} (id {vice})"
    )
    return "\n".join(lines)


def _without(whatif: "WhatIf", board: _Board) -> str:
    off = whatif.off
    week1 = (
        "roll — no transfers"
        if not off.plan.transfers_in and not off.plan.transfers_out
        else "out " + ", ".join(_described(p, board) for p in off.plan.transfers_out)
        + "; in " + ", ".join(_described(p, board) for p in off.plan.transfers_in)
    )
    lines = ["## Without it: the best path that does not play it this week", "", f"GW{whatif.event}: {week1}"]
    lines += [_planned(move, board) for move in off.path.moves]
    return "\n".join(lines)


def _position(week, whatif, verdict, deadlines, board) -> str:
    inputs = week.effective
    lines = [
        "## Position",
        "",
        f"Bank {price(inputs.squad.bank)} · free transfers banked: {inputs.free_transfers}",
    ]
    held = held_for(held_by_rules(inputs), whatif.kind, whatif.event)
    if held is not None:
        left = held.stop_event - whatif.event + 1
        lines.append(
            f"{_CHIP_NAMES.get(whatif.kind, whatif.kind)} held until the"
            f" GW{held.stop_event} deadline ({_when(deadlines.get(held.stop_event))})"
            f" — {left} {'gameweek' if left == 1 else 'gameweeks'} left in its window"
        )
    gap = next_break(inputs.bootstrap.events, whatif.event)
    if gap is not None:
        lines.append(f"Next international break: between GW{gap[0]} and GW{gap[1]}")
    lines.append(_latest(verdict, whatif.event, board))
    return "\n".join(lines)


def _latest(verdict: dict | None, event: int, board: _Board) -> str:
    if verdict is None:
        return f"Latest report for GW{event}: none yet"
    ins, outs = verdict.get("transfers_in") or [], verdict.get("transfers_out") or []
    moves = (
        "roll — no transfers"
        if not ins and not outs
        else "out " + ", ".join(_described(p, board) for p in outs)
        + "; in " + ", ".join(_described(p, board) for p in ins)
    )
    captain, vice = verdict.get("captain"), verdict.get("vice")
    armbands = ""
    if captain in board.players and vice in board.players:
        armbands = (
            f"; captain {_safe(board.players[captain].web_name)} (id {captain}),"
            f" vice {_safe(board.players[vice].web_name)} (id {vice})"
        )
    return (
        f"Latest report for GW{event} ({verdict.get('mode', 'report')}): {moves}"
        f"{armbands}; chip {verdict.get('chip', 'none')}"
    )


def _when(deadline) -> str:
    return "unknown" if deadline is None else f"{deadline:%a %d %b %Y %H:%M} UTC"

