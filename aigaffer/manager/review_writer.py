"""The gaffer, looking back at a finished gameweek.

One short conversation: the week's facts go in, three short lists come out
through one answer tool, ``write_review``. No web search — the facts are
final and the job is judgement, not news. Two requests at most: an answer
that is missing or does not validate gets one more turn, with the error in
words, and then the review waits for the next tick
(:class:`~aigaffer.review.ReviewUnavailable`).

The model's rules are the manager's: no ``tool_choice`` (Opus 5.5 answers it
with a 400), no thinking parameter, history append-only, and a turn cut off
at ``max_tokens`` is dropped as if it never happened.
"""

from typing import Any

import anthropic

from aigaffer.config import Config
from aigaffer.manager.agent import EFFORT, MAX_RESUMPTIONS, MAX_TOKENS, _cached, _result, _rolling, _text
from aigaffer.manager.tools import MARKUP, ToolError
from aigaffer.review import ReviewUnavailable, WeekFacts, Written, review_facts

ANSWER = "write_review"
MAX_POINTS = 3
REVIEW_MAX_WORDS = 350
MAX_REQUESTS = 2

SYSTEM_PROMPT = (
    "You are the gaffer of a Fantasy Premier League team, looking back at a"
    " gameweek that has finished. Use only the facts you are given; do not"
    " guess at anything they do not say. Be honest and brief: 150 to 250"
    " words in total. Give up to three points each for what went well, what"
    " went wrong, and lessons for the coming weeks. A lesson is something a"
    " future decision could use — captaincy, minutes risk, hits, bench order,"
    " chip timing — not a restatement of the score. Bad luck is not a"
    " mistake: say which it was. One plain sentence or two per point, no"
    f" headings or bullets of your own. Answer by calling {ANSWER}."
)

TOOLS = [
    {
        "name": ANSWER,
        "description": "Your look back at the gameweek: up to three short points in each list.",
        "input_schema": {
            "type": "object",
            "properties": {
                "went_well": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_POINTS},
                "went_wrong": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_POINTS},
                "lessons": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_POINTS},
            },
            "required": ["went_well", "went_wrong", "lessons"],
        },
    }
]

ASK = f"Call {ANSWER} now with your three lists — no more discussion."


def validate_review(args: dict) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """The three lists, cleaned, or a ToolError that says what to fix."""
    lists = []
    for key in ("went_well", "went_wrong", "lessons"):
        items = args.get(key)
        if not isinstance(items, list):
            raise ToolError(f"{key} must be a list of short strings (it may be empty).")
        if len(items) > MAX_POINTS:
            raise ToolError(f"{key} has {len(items)} points; give at most {MAX_POINTS}.")
        cleaned = []
        for item in items:
            point = " ".join(item.split()) if isinstance(item, str) else ""
            point = point.removeprefix("- ").removeprefix("* ").strip()
            if not point:
                raise ToolError(f"{key} has an empty point; drop it or say something.")
            # "<!--" too: a forced re-review cuts the report at the first
            # review marker, so a point holding one would cut it in the wrong place.
            if point.startswith("#") or "<!--" in point or MARKUP.search(point):
                raise ToolError(f"{key} has a heading or markup in it; write plain sentences.")
            cleaned.append(point)
        lists.append(tuple(cleaned))
    if not lists[0] and not lists[1]:
        raise ToolError("Say at least one thing that went well or went wrong.")
    words = sum(len(point.split()) for points in lists for point in points)
    if words > REVIEW_MAX_WORDS:
        raise ToolError(f"That is {words} words; keep it near 250, and {REVIEW_MAX_WORDS} at most.")
    return lists[0], lists[1], lists[2]


def review_briefing(facts: WeekFacts) -> str:
    """What the writer reads: the facts, then whose verdict it was and why."""
    lines = [f"# Looking back at GW{facts.gw}", "", review_facts(facts), "", "## The verdict the week was played on"]
    if facts.verdict_mode is None:
        lines.append("No verdict on record for this week.")
    elif not facts.gaffer_decided:
        lines.append(
            f"The solver's verdict — the gaffer did not decide this week ({facts.verdict_mode} report)."
            " Do not judge whether the gaffer was followed."
        )
    else:
        lines.append(f"The gaffer's verdict ({facts.verdict_mode} report).")
        if facts.rationale:
            lines.append(f"Rationale: {facts.rationale}")
        if facts.chip_justification:
            lines.append(f"Chip justification: {facts.chip_justification}")
    text = "\n".join(lines)
    if facts.verdict_mode is not None and not facts.gaffer_decided:
        # A solver week is not a question about following the gaffer.
        text = "\n".join(line for line in text.splitlines() if not line.startswith("Verdict followed"))
    return text


def write_review(cfg: Config, facts: WeekFacts, client: Any = None) -> Written:
    """The gaffer's review of ``facts``, or ReviewUnavailable."""
    if client is None:
        from aigaffer.manager.client import build_review_client

        client = build_review_client(cfg)
    conversation = _Conversation(client, cfg, review_briefing(facts))
    try:
        return conversation.run()
    except anthropic.AnthropicError as error:
        raise ReviewUnavailable(type(error).__name__) from error


class _Conversation:
    def __init__(self, client: Any, cfg: Config, briefing: str) -> None:
        self.client = client
        self.cfg = cfg
        self.messages: list[dict] = [{"role": "user", "content": [_cached(briefing)]}]
        self.tokens = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0}
        self.turns = 0

    def run(self) -> Written:
        reason = "no answer"
        for _ in range(MAX_REQUESTS):
            response = self._ask()
            self.turns += 1
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                raise ReviewUnavailable("refusal")
            if stop == "max_tokens":
                # Cut off mid-sentence: never happened. The ask joins the
                # still-open user message, so no two user turns run together.
                reason = "max_tokens"
                self._join_ask()
                continue
            self._append(response)
            answer, results = self._act(response)
            if answer is not None:
                well, wrong, lessons = answer
                return Written(well, wrong, lessons, dict(self.tokens), self.turns)
            if not results:
                reason = "no tool call"
                self._join_ask()
                continue
            # Every tool_use needs its own tool_result, in one user message,
            # or the API answers 400.
            reason = "invalid answer"
            self.messages.append({"role": "user", "content": results})
        raise ReviewUnavailable(reason)

    def _act(self, response: Any) -> tuple[tuple | None, list[dict]]:
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
                return validate_review(given if isinstance(given, dict) else {}), results
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
            self._meter(response)
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
            self._append(response)
        raise ReviewUnavailable("pause_turn")

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

    def _meter(self, response: Any) -> None:
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

    def _append(self, response: Any) -> None:
        content = getattr(response, "content", None)
        if content:
            self.messages.append({"role": "assistant", "content": content})
