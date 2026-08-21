"""The conversation with the manager, and the week it either produces or does not.

This is the only module in the project that talks to Anthropic, and the client
is handed to it rather than built here, so every test in the suite runs the
whole loop against a scripted one and none of them can reach the network.

The loop is manual — the SDK's tool runner would drive it in fewer lines — for
three reasons that all come down to the same thing. Every tool call has to be
checked before it takes effect, an illegal one has to go back as a correction
the model can act on rather than as an exception, and the run has to end with a
team sheet whatever the API does. The runner is beta, does not resume a paused
turn, and would put the guardrails inside the tool functions where a raised
exception ends the conversation instead of continuing it.

What the manager may do is deliberately narrow. He may change what the model
believes about a player's minutes, ask for the solver's answer again with those
beliefs in it, and choose one of the plans the solver has actually reached. He
may not name a squad. Everything he chooses from is minted here — the plan-id
registry seeds from the ids the briefing printed and goes on counting across
re-solves — and ``finalize_decision`` is validated against that registry and
against nothing else. The conversation is full of text that looks like plans:
the briefing quotes a public API, a tool result quotes a solver run, and a web
page quotes whoever wrote it. None of it is a plan. The registry is.

And the whole thing is optional. Every failure — a refusal, a rate limit, a
connection that dies, twelve turns that reach no decision — degrades to the
solver's own recommendation with the reason recorded in ``source``, because a
deadline is never silently missed and a manager who cannot be reached is not a
reason to miss one.
"""

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import anthropic

from aigaffer.config import Config
from aigaffer.manager.briefing import format_plans, initial_plan_ids
from aigaffer.manager.tools import (
    NO_CHIP,
    SYSTEM_PROMPT,
    TOOLS,
    ToolError,
    validate_adjustments,
    validate_finalize,
)
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.lineup import Lineup, pick_lineup
from aigaffer.solver.optimizer import Plan

if TYPE_CHECKING:  # the orchestrator imports the manager, so never the reverse
    from aigaffer.orchestrator import PipelineInputs, SolveResult

# Twelve assistant turns is a long conversation for a decision with this much
# already worked out for it, and the twelfth is spent asking for the decision
# rather than hoping for it.
MAX_TURNS = 12

# A turn the server pauses is re-sent, and three times is generous: a fourth
# identical pause is a turn that is not coming back.
MAX_RESUMPTIONS = 3

# Non-streaming, so the ceiling stays under the SDK's HTTP timeout. Effort is
# the depth control on this model; thinking is adaptive by default and an
# explicit configuration risks a 400, so the parameter is not sent at all.
MAX_TOKENS = 16000
EFFORT = {"effort": "high"}

FORCE_FINALIZE = {"type": "tool", "name": "finalize_decision"}

MANAGER = "manager"
FALLBACK = "solver-fallback"

NUDGE = (
    "That is not a decision yet. Finish the job: call finalize_decision with"
    " the plan id you have settled on, your captain and vice from that plan's"
    " eleven, a chip (or 'none'), and the rationale."
)

# What the report says when there was no managerial view to print. The reason
# lives in ``source``; this is the sentence beside it, and it never carries the
# text of an exception — a bot token or an API key can be inside one.
NO_VIEW = (
    "No managerial view this week: the solver's own recommendation stands,"
    " unadjusted."
)

# What Task 5 builds as a closure over build_projections and solve: expected
# minutes in, a fresh shortlist and fresh projections out.
Resolver = Callable[
    [dict[int, float]], tuple["SolveResult", dict[int, PlayerProjection]]
]


@dataclass
class ManagerDecision:
    """The week's decision and where it came from.

    ``lineup`` is picked from the chosen plan's squad on the projections in
    force when it was chosen — the adjusted ones, if he re-solved — with the
    captain and vice overwritten by his, which is why they are also carried
    beside it: the two must never be read apart.

    ``source`` is ``"manager"`` or ``"solver-fallback: <reason>"``, and it is
    printed in the report, because "the solver picked this" and "the manager
    picked this" are different claims and Mark is entitled to know which he is
    reading.
    """

    plan: Plan
    lineup: Lineup
    captain: int
    vice: int
    chip: str
    chip_justification: str
    rationale: str
    adjustments: list[dict]
    searches: int
    source: str


def run_manager(
    client: anthropic.Anthropic,
    cfg: Config,
    inputs: "PipelineInputs",
    solve0: "SolveResult",
    projections0: dict[int, PlayerProjection],
    briefing: str,
    resolver: Resolver,
) -> ManagerDecision:
    """Put the week to the manager and come back with a decision, always.

    ``client`` is injected rather than built, and nothing here touches it but
    ``messages.create``: the suite runs the whole loop against a scripted stand-in
    and no test can reach the network by accident.

    ``solve0`` and ``projections0`` are the run as the solver left it: the
    plans the briefing numbered, and the projections it numbered them on. They
    are also the fallback — every failure below returns the solver's own
    recommendation built from them.

    This never raises. The exception chain names the failures the SDK actually
    has, and the bare ``except`` under it is not a lapse: an unexpected error in
    this module is a bug in this module, and losing the week's report to it
    would be a second, worse one.
    """
    conversation = _Conversation(
        client, cfg, inputs, solve0, projections0, briefing, resolver
    )
    try:
        return conversation.run()
    except (
        anthropic.RateLimitError,  # too many requests, or too many tokens
        anthropic.APIStatusError,  # a 4xx or 5xx: bad request, overloaded, refused
        anthropic.APIConnectionError,  # the network, or a request that timed out
        anthropic.AnthropicError,  # anything else the SDK raises on its own
    ) as error:
        return conversation.fallback(type(error).__name__)
    except Exception as error:  # ours, then — and still not the run's problem
        return conversation.fallback(f"unexpected {type(error).__name__}")


class _Conversation:
    """One run of the loop, and the state that accumulates across its turns.

    The state is the point: the plan-id registry, the minute adjustments, the
    projections currently in force and the count of searches spent all outlive
    a single turn, and all of them are read by the fallback if the next request
    is the one that fails.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        cfg: Config,
        inputs: "PipelineInputs",
        solve0: "SolveResult",
        projections: dict[int, PlayerProjection],
        briefing: str,
        resolver: Resolver,
    ) -> None:
        self.client = client
        self.cfg = cfg
        self.inputs = inputs
        self.solve0 = solve0
        self.projections = projections
        self.resolver = resolver

        # The ids the briefing printed, minted in the one place that mints them.
        self.registry: dict[int, Plan] = dict(initial_plan_ids(solve0))
        self.messages: list[dict] = [{"role": "user", "content": briefing}]
        self.adjustments: dict[int, float] = {}
        self.record: list[dict] = []
        self.searches = 0

        self.clubs = {team.id: team.short_name for team in inputs.bootstrap.teams}
        self.positions = {pid: p.element_type for pid, p in inputs.players.items()}

    def run(self) -> ManagerDecision:
        """Turn after turn until he decides, or until we stop asking."""
        forced = False
        nudged = False

        for turn in range(1, MAX_TURNS + 1):
            response = self._ask(forced or turn == MAX_TURNS)
            # Forcing is one turn's worth of insistence, not a mode: a turn cut
            # off mid-sentence should not cost him the rest of his research.
            forced = False
            if response is None:
                return self.fallback("pause_turn")

            # Before the content, always: on a refusal there may be nothing in
            # it, and reading it first is how that becomes a crash.
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                return self.fallback("refusal")
            if stop == "max_tokens":
                # A turn cut off mid-sentence is a turn that never happened: its
                # half-written tool call cannot be answered, so it is dropped
                # rather than sent back, and the next turn asks for the decision.
                forced = True
                continue

            decision, results = self._act(response)
            if decision is not None:
                return decision
            if results:
                self._append(response)
                self.messages.append({"role": "user", "content": results})
                continue

            # He talked instead of deciding. Once is worth a word; after that
            # the decision is taken out of his hands with tool_choice, and stays
            # out of them for as long as he keeps talking.
            self._append(response)
            self.messages.append({"role": "user", "content": NUDGE})
            if nudged:
                forced = True
            nudged = True

        return self.fallback(f"no decision in {MAX_TURNS} turns")

    def fallback(self, reason: str) -> ManagerDecision:
        """The solver's own week, with the reason it is being read instead.

        The adjustments go with the manager: they were never applied to the
        plan this returns, and listing them under it would be a report claiming
        a minutes model it did not use. The searches stay, because they happened
        and they were paid for.
        """
        lineup = self.solve0.lineup
        return ManagerDecision(
            plan=self.solve0.choice,
            lineup=lineup,
            captain=lineup.captain,
            vice=lineup.vice,
            chip=NO_CHIP,
            chip_justification="",
            rationale=NO_VIEW,
            adjustments=[],
            searches=self.searches,
            source=f"{FALLBACK}: {reason}",
        )

    def _ask(self, forced: bool) -> Any:
        """One assistant turn, resumed if the server pauses it. None if it stays
        paused: a turn that will not come back is not a turn to keep paying for."""
        for _ in range(MAX_RESUMPTIONS + 1):
            response = self.client.messages.create(**self._request(forced))
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
        return None

    def _request(self, forced: bool) -> dict:
        """The request, byte-stable where it can be: the system block never
        moves, so the prefix caches, and the messages are copied so that a
        request already sent cannot be edited by the turn after it."""
        request = {
            "model": self.cfg.manager_model,
            "max_tokens": MAX_TOKENS,
            "output_config": EFFORT,
            "system": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "tools": TOOLS,
            "messages": list(self.messages),
        }
        if forced:
            request["tool_choice"] = FORCE_FINALIZE
        return request

    def _act(self, response: Any) -> tuple[ManagerDecision | None, list[dict]]:
        """Carry out every tool call in one assistant turn.

        All of them, and their results go back in one user message: results
        split across two messages teach the model not to ask for two things at
        once, and a call left unanswered is a request the API will reject.

        A legal ``finalize_decision`` ends the conversation where it stands.
        Anything after it in the same turn is a tool call whose result nobody
        will ever read, and running it would let a re-solve rewrite the
        projections the decision was just built on.
        """
        results: list[dict] = []
        for block in getattr(response, "content", None) or []:
            kind = getattr(block, "type", None)
            if kind == "web_search_tool_result":
                # The server ran it and paid for it; we only count it.
                self.searches += 1
            if kind != "tool_use":
                continue

            name = getattr(block, "name", "")
            block_id = getattr(block, "id", "")
            given = getattr(block, "input", None)
            args = given if isinstance(given, dict) else {}
            try:
                if name == "finalize_decision":
                    return self._decide(args), results
                results.append(_result(block_id, self._tool(name, args)))
            except ToolError as error:
                results.append(_result(block_id, str(error), failed=True))
        return None, results

    def _tool(self, name: str, args: dict) -> str:
        if name == "adjust_players":
            return self._adjust(args)
        if name == "resolve":
            return self._resolve()
        raise ToolError(
            f"There is no tool called '{name}'. The tools are adjust_players,"
            " resolve, web_search and finalize_decision."
        )

    def _adjust(self, args: dict) -> str:
        """Record what he now believes about somebody's minutes.

        Nothing is projected here — the adjustments accumulate and are spent by
        the next ``resolve`` — and the result names players by id and by nothing
        else. Every other field we hold about a player came off a public API,
        and this document's structure is its own.
        """
        records = validate_adjustments(args, self.inputs.players)
        self.record.extend(records)
        for record in records:
            self.adjustments[record["player_id"]] = record["expected_minutes"]

        return "\n".join(
            [
                "Recorded: " + _minutes(records) + ".",
                "In force for the next resolve: " + _in_force(self.adjustments) + ".",
                "Nothing has changed yet — call resolve to re-project and"
                " re-solve with these.",
            ]
        )

    def _resolve(self) -> str:
        """Ask the solver the week's question again, with his minutes in it.

        The fresh plans are numbered from where the registry has got to and
        added to it, so an id means one plan for the whole conversation. The
        lines are rendered by the briefing's own formatter: it is the one place
        that knows how a plan reads, and the one place that sanitizes the names
        it reads them with.
        """
        try:
            solved, projections = self.resolver(dict(self.adjustments))
        except Exception as error:  # the solver's bad week, not the manager's
            raise ToolError(
                f"The re-solve failed ({type(error).__name__}) and nothing"
                " changed. The plans you have already been shown are still on"
                " the board — finalize one of those, or adjust and try again."
            ) from error

        if not solved.plans:
            raise ToolError(
                "The re-solve reached no legal squad at all, so nothing changed."
                " The plans you have already been shown are still on the board."
            )

        self.projections = projections
        first = len(self.registry)
        fresh = [(first + offset, plan) for offset, plan in enumerate(solved.plans)]
        self.registry.update(fresh)
        pick = next((pid for pid, plan in fresh if plan is solved.choice), None)

        return "\n\n".join(
            [
                _resolved(self.adjustments),
                format_plans(
                    fresh, self.inputs.players, self.clubs, projections, pick
                ),
                f"{_earlier(first)} still on the board and can still be"
                " finalized, but the numbers on them were computed before these"
                " adjustments.",
            ]
        )

    def _decide(self, args: dict) -> ManagerDecision:
        """His decision, if it is a legal one; otherwise the correction.

        The armbands are checked against the same eleven the decision is built
        with — :func:`pick_lineup` on the chosen plan's squad, on the
        projections now in force — so a captain who passes the check is a
        captain who is in the team we enter.
        """
        final = validate_finalize(
            args, self.registry, lambda plan: self._lineup(plan).xi
        )
        lineup = self._lineup(final.plan)
        return ManagerDecision(
            plan=final.plan,
            lineup=replace(lineup, captain=final.captain, vice=final.vice),
            captain=final.captain,
            vice=final.vice,
            chip=final.chip,
            chip_justification=final.chip_justification,
            rationale=final.rationale,
            adjustments=list(self.record),
            searches=self.searches,
            source=MANAGER,
        )

    def _lineup(self, plan: Plan) -> Lineup:
        """The eleven ``plan`` fields next gameweek, on the projections in force."""
        event = self.inputs.event.id
        gw_xp = {
            pid: projection.per_gw.get(event, 0.0)
            for pid, projection in self.projections.items()
        }
        return pick_lineup(plan.squad, self.positions, gw_xp)

    def _append(self, response: Any) -> None:
        """Put his turn into the record, unless he said nothing at all: an empty
        content block is a message the API will not accept back."""
        content = getattr(response, "content", None)
        if content:
            self.messages.append({"role": "assistant", "content": content})


def _result(tool_use_id: str, content: str, failed: bool = False) -> dict:
    """One tool result. ``is_error`` is present only when it is true, which is
    how the API reads it and how a reader of the log reads it too."""
    result = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if failed:
        result["is_error"] = True
    return result


def _minutes(records: list[dict]) -> str:
    """``id 7 at 0 minutes; id 4 at 60 minutes`` — what this call just set."""
    return "; ".join(
        f"id {record['player_id']} at {record['expected_minutes']:.0f} minutes"
        for record in records
    )


def _in_force(adjustments: dict[int, float]) -> str:
    """Every adjustment made so far, in id order. Callers check first that there
    is at least one: "in force: nothing" is a sentence about nothing."""
    return "; ".join(
        f"id {pid} at {minutes:.0f} minutes"
        for pid, minutes in sorted(adjustments.items())
    )


def _resolved(adjustments: dict[int, float]) -> str:
    """What the re-solve was run on.

    A re-solve with nothing adjusted returns the plans the briefing already
    showed, under new ids — legal, occasionally sensible, and baffling unless
    the result says plainly that nothing was changed.
    """
    if not adjustments:
        return (
            "Re-solved with no adjustments in force, so the plans below are"
            " costed on the same projections as the briefing's."
        )
    return (
        f"Re-solved with your adjustments in force: {_in_force(adjustments)}."
        " The plans below are costed on the adjusted projections."
    )


def _earlier(first: int) -> str:
    """How to name the plans that were on the board before this re-solve."""
    if first == 1:
        return "Plan 0 is"
    return f"Plans numbered 0-{first - 1} are"
