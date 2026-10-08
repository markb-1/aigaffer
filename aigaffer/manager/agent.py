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
connection that dies, twelve turns or twelve minutes that reach no decision —
degrades to the solver's own recommendation with the reason recorded in
``source``, because a deadline is never silently missed and a manager who
cannot be reached is not a reason to miss one.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from time import monotonic
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
from aigaffer.solver.lineup import Lineup, attacking_evs, pick_lineup
from aigaffer.solver.optimizer import Plan

if TYPE_CHECKING:  # the orchestrator imports the manager, so never the reverse
    from aigaffer.orchestrator import PipelineInputs, SolveResult

# Twelve assistant turns is a long conversation for a decision with this much
# already worked out for it, and the twelfth is spent asking for the decision
# rather than hoping for it.
MAX_TURNS = 12

# A turn the server pauses is resumed, and three times is generous: a fourth
# pause on the same turn is a turn that is not coming back.
MAX_RESUMPTIONS = 3

# And the same conversation against the clock, because a cap on turns is not a
# cap on minutes. Twelve unstreamed turns of an agent that searches the web
# between them can run long, and the run they are inside has a deadline of its
# own: the report is due before the gameweek's, and the workflow gives the whole
# job thirty. Twelve minutes leaves room for the turn already in flight — the
# client the orchestrator builds bounds that one at five minutes and one retry —
# and for the solver's own week to be rendered and sent after it.
#
# The budget is checked before a request and never during one, so the whole of
# the last turn falls outside it, tools and all. A hung request is ten minutes,
# and what that turn asked for is run after it comes back: a resolve is a sweep
# of six windowed solves at :data:`~aigaffer.solver.plans.SWEEP_TIME_LIMIT`
# seconds each, so two minutes more. Twelve, ten and two is twenty-four minutes
# inside the manager, which is why the job is given thirty and not fifteen.
#
# A worse report on time beats a better one that missed, every week.
TIME_BUDGET_SECONDS = 720
OUT_OF_TIME = "out of time"

# Non-streaming, because the loop reads whole messages and shows nobody a
# partial one. What that costs is a request the client has to wait out in
# silence: a turn at this effort can run its searches on the server before a
# single token comes back, and it is minutes rather than seconds. The orchestrator's
# five-minute timeout is what makes it safe, and the two-minute one this was
# first written with is what made it fail every time. Effort is the depth
# control on this model; thinking is adaptive by default and an explicit
# configuration risks a 400, so the parameter is not sent at all.
#
# Medium is Opus 5.5's own default and is set explicitly anyway, so that a
# change of default on the platform's side is never a change of behaviour on
# ours.
MAX_TOKENS = 16000
EFFORT = {"effort": "medium"}

MANAGER = "manager"
FALLBACK = "solver-fallback"

# The refresh clause first, because the live GW3 scout showed what happens
# without it: the gaffer ended a turn exactly as the prompt tells it to —
# to refresh a spent search allowance — and this message's bare demand
# talked it into finalizing instead, after which it wrote "my allowance
# ran out and it did not refresh" into the rationale, wrongly. The demand
# stays: a model that merely talks, twice running, is still told outright to decide.
NUDGE = (
    "That is not a decision yet. If you ended your turn to refresh your"
    " search allowance, it is fresh now — continue the research you paused"
    " for. Otherwise finish the job: call finalize_decision with the plan id"
    " you have settled on, your captain and vice from that plan's eleven, a"
    " chip (or 'none'), and the rationale."
)

# The decision, asked for in words, because Opus 5.5 answers a forced
# ``tool_choice`` ("tool" or "any") with an HTTP 400 and so the request never
# carries one. This is what stands in for it on the three turns that used to be
# forced: a second consecutive stall, the turn after one cut off at max_tokens,
# and the last turn. It is strict about the shape of the turn — one call, no
# searches, no other tool, no prose-only reply — because those are the ways an
# unforced model can still avoid deciding, and ``finalize_decision`` is a
# strict tool, so a call that does come is well-formed.
#
# It is appended to the conversation and stays there. Opus 5.5 binds its
# thinking blocks to the history they were produced in, so an earlier message
# is never edited or removed; the instruction is one more user turn at the end,
# and "one turn's worth of insistence, not a mode" is now a matter of not
# appending it again rather than of leaving a parameter off.
FINALIZE_NOW = (
    "Decide now. This turn must be exactly one finalize_decision call and"
    " nothing else: no searches, no other tools, and no reply that is only"
    " text. Give the plan id you have settled on (or the solver's pick, if you"
    " have settled on none), your captain and vice from that plan's eleven, a"
    " chip (or 'none'), and the rationale."
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
    picked this" are different claims and the owner is entitled to know which
    he is reading.

    ``projections`` is what the decision was costed on — the ones in force
    when he finalized, which are the re-solve's if he re-solved. The report is
    written on them, so that the eleven it prints and the numbers beside it
    describe the same week. None on a fallback: that is the solver's own week
    and the pipeline already holds the projections it was solved on.

    ``adjustments`` and ``unapplied`` are the two halves of the record he kept,
    and the line between them is the last ``resolve``. Only what that re-solve
    was run on reached the projections this decision is costed against;
    anything he wrote down after it — or never re-solved on at all — changed
    nothing, and saying otherwise would be a report claiming a minutes model it
    did not use. Both halves are kept because both happened: one is what he
    did, the other is what he only said.
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
    projections: dict[int, PlayerProjection] | None = None
    unapplied: list[dict] = field(default_factory=list)


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

    Given a ``solve0`` the solver produced, this never raises. The exception
    chain names the failures the SDK actually has, and the bare ``except`` under
    it is not a lapse: an unexpected error in this module is a bug in this
    module, and losing the week's report to it would be a second, worse one.

    Setting the conversation up is inside the try for the same reason. It reads
    the fetch — clubs off the bootstrap, positions off the board — and a fetch
    that came back malformed would otherwise raise past every handler here and
    take the report with it, which is exactly the failure this is for.
    """
    conversation = None
    try:
        conversation = _Conversation(
            client, cfg, inputs, solve0, projections0, briefing, resolver
        )
        return conversation.run()
    except (
        anthropic.RateLimitError,  # too many requests, or too many tokens
        anthropic.APIStatusError,  # a 4xx or 5xx: bad request, overloaded, refused
        anthropic.APIConnectionError,  # the network, or a request that timed out
        anthropic.AnthropicError,  # anything else the SDK raises on its own
    ) as error:
        return _abandoned(solve0, conversation, type(error).__name__)
    except Exception as error:  # ours, then — and still not the run's problem
        return _abandoned(solve0, conversation, f"unexpected {type(error).__name__}")


def _abandoned(
    solve0: "SolveResult", conversation: "_Conversation | None", reason: str
) -> ManagerDecision:
    """The solver's own week, with the reason it is being read instead.

    ``conversation`` is None when there was never one: the failure happened
    while the loop was being built, and then not even a search count survives
    it. Everything this returns comes from ``solve0``, which is the one thing
    that cannot have been the problem — it was in hand before the manager was
    asked anything.

    The adjustments go with the manager. They were never applied to the plan
    this returns, and listing them under it would be a report claiming a
    minutes model it did not use. The projections go with them, and for the
    same reason: this is the solver's own week, and the caller is holding the
    projections it was solved on. The searches stay: they happened, and they
    were paid for.
    """
    lineup = solve0.lineup
    return ManagerDecision(
        plan=solve0.choice,
        lineup=lineup,
        captain=lineup.captain,
        vice=lineup.vice,
        chip=NO_CHIP,
        chip_justification="",
        rationale=NO_VIEW,
        adjustments=[],
        searches=conversation.searches if conversation is not None else 0,
        source=f"{FALLBACK}: {reason}",
    )


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
        # The briefing is the longest thing in the conversation and never
        # changes once written, so it is cached with the system prompt: the
        # loop is up to a dozen turns and every one of them resends it.
        self.messages: list[dict] = [{"role": "user", "content": [_cached(briefing)]}]
        self.adjustments: dict[int, float] = {}
        # What the last re-solve was actually run on, as it stood then. The
        # accumulated adjustments above go on changing after it; this does not,
        # which is what makes it the answer to "which of these took effect?".
        self.spent: dict[int, float] = {}
        self.record: list[dict] = []
        self.searches = 0
        # The server-side container his searches ran in, once there has been
        # one. The web_search tool this loop declares filters its results by
        # running code on Anthropic's servers, so a turn that searches comes
        # back with ``server_tool_use``/``code_execution_tool_result`` blocks in
        # it and a container id beside them — and a request that replays those
        # blocks without the id is rejected outright: *"container_id is required
        # when there are pending tool uses generated by code execution with
        # tools"*. That 400 is what stopped every live conversation dead on its
        # second request, because the first request never has one to send.
        #
        # It is carried rather than renewed: the container is good for an hour
        # against a budget of twelve minutes, so no run this loop can have
        # outlives it.
        self.container: str | None = None
        # When the clock started. Here rather than in :meth:`run`, because a
        # conversation is built to be run at once and the one thing the budget
        # must not depend on is how many places read it.
        self.started = monotonic()

        self.clubs = {team.id: team.short_name for team in inputs.bootstrap.teams}
        self.positions = {pid: p.element_type for pid, p in inputs.players.items()}

    def run(self) -> ManagerDecision:
        """Turn after turn until he decides, or until we stop asking.

        Two things stop us asking: :data:`MAX_TURNS`, and the clock. The clock
        is read before every request — the turns here and the resumptions in
        :meth:`_ask` — because a budget that is already spent buys nothing by
        being spent again. A request already in flight is left to come back:
        its answer is paid for either way, and the client's own timeout is what
        bounds it.
        """
        # Whether FINALIZE_NOW is already the last thing in the conversation,
        # put there by the turn before. The last turn asks for the decision
        # whether or not anything else has, and must not ask twice in a row.
        asked = False
        nudged = False

        for turn in range(1, MAX_TURNS + 1):
            if self._expired():
                return self.fallback(OUT_OF_TIME)
            if turn == MAX_TURNS and not asked:
                self._ask_for_decision()
            response = self._ask()
            # Asking is one turn's worth of insistence, not a mode: a turn cut
            # off mid-sentence should not cost him the rest of his research.
            # The words stay in the history, but nothing repeats them.
            asked = False
            if response is None:
                # Either the pauses ran out or the clock did, and the two are
                # different weeks to explain: one is a turn that would not come
                # back, the other is a turn there was no longer time for.
                return self.fallback(OUT_OF_TIME if self._expired() else "pause_turn")

            # Before the content, always: on a refusal there may be nothing in
            # it, and reading it first is how that becomes a crash.
            stop = getattr(response, "stop_reason", None)
            if stop == "refusal":
                return self.fallback("refusal")
            if stop == "max_tokens":
                # A turn cut off mid-sentence is a turn that never happened: its
                # half-written tool call cannot be answered, so it is dropped
                # rather than sent back, and the next turn asks for the decision.
                self._ask_for_decision()
                asked = True
                continue

            decision, results = self._act(response)
            if decision is not None:
                return decision
            if results:
                self._append(response)
                self.messages.append({"role": "user", "content": results})
                # Real work buys back the right to pause: the prompt tells him
                # a turn boundary refreshes his search allowance, so a bare
                # turn-end between productive turns is sometimes the correct
                # move, and only *consecutive* stalling earns the escalation.
                nudged = False
                continue

            # He talked instead of deciding. Once is worth a word; after that
            # the decision is asked for outright, in place of the word and not
            # on top of it, and is asked for again after every further stall.
            self._append(response)
            if nudged:
                self._ask_for_decision()
                asked = True
            else:
                # A block list rather than a bare string, like every other
                # message this loop writes: the request marks the last block of
                # the last message, and a string has no blocks to mark.
                self.messages.append({"role": "user", "content": [_text(NUDGE)]})
            nudged = True

        return self.fallback(f"no decision in {MAX_TURNS} turns")

    def _ask_for_decision(self) -> None:
        """Append :data:`FINALIZE_NOW` as the next user turn, and change nothing else.

        Append-only on purpose: an earlier message is never edited or removed,
        because Opus 5.5 binds its thinking blocks to the conversation they
        were written in. A user turn straight after another is fine for the
        API, which reads the two as one.
        """
        self.messages.append({"role": "user", "content": [_text(FINALIZE_NOW)]})

    def fallback(self, reason: str) -> ManagerDecision:
        """Give the week back to the solver, from inside the loop."""
        return _abandoned(self.solve0, self, reason)

    def _ask(self) -> Any:
        """One assistant turn, resumed as often as the server pauses it.

        A pause is not a failure and not a fresh start: the server has stopped
        part-way through a turn it means to finish, and the way to finish it is
        to put what it has already produced into the conversation and ask
        again. Sending the same request again instead would only buy the same
        pause, at the price of running its searches twice.

        So the paused content is appended — searches counted, because they were
        run and paid for — and the next request continues from it. This is the
        one place a request may end on an assistant turn, and the only one: for
        a paused turn the API reads that as the continuation it is, rather than
        as the prefill this model refuses.

        None if it is still paused after :data:`MAX_RESUMPTIONS`, or if the
        time budget goes while it is paused: a turn that will not come back is
        not a turn to keep paying for, and neither is one there is no longer
        time to finish. The caller reads the clock again to tell them apart.
        """
        for attempt in range(MAX_RESUMPTIONS + 1):
            if attempt and self._expired():
                return None
            response = self.client.messages.create(**self._request())
            # Before the stop reason is read, and on every answer: a paused turn
            # opens the container just as a finished one does, and the request
            # that resumes it is already replaying the blocks that need it.
            self._note_container(response)
            if getattr(response, "stop_reason", None) != "pause_turn":
                return response
            self._count(response)
            self._append(response)
        return None

    def _expired(self) -> bool:
        """Has the conversation outrun :data:`TIME_BUDGET_SECONDS`?"""
        return monotonic() - self.started > TIME_BUDGET_SECONDS

    def _request(self) -> dict:
        """The request, byte-stable where it can be: the system block never
        moves, so the prefix caches, and the messages are copied so that a
        request already sent cannot be edited by the turn after it.

        Three cache breakpoints, of the four the API allows. Two never move —
        the system prompt and the briefing, which is the whole of what cannot
        change once the conversation has started. The third rides the end of
        the messages, and it is the one that matters by the twelfth turn: the
        conversation is resent in full every time, so the prefix that grows is
        the expensive one, and a marker only on the fixed head of it leaves
        every turn's searches and tool results to be paid for again on the next.

        It moves rather than multiplies. The marker is put on the copy this
        request sends and never on ``self.messages``, so nothing accumulates,
        nothing already sent is edited afterwards, and the count stays at three.

        ``container`` goes on every request after a turn has opened one, and it
        is not optional then: see :meth:`_note_container` for what the
        conversation is carrying by that point and why the API refuses to take
        it without being told which container it belongs to.

        There is no ``tool_choice``, on any request: Opus 5.5 rejects a forced
        one, and the decision is asked for in words instead (see
        :data:`FINALIZE_NOW`).
        """
        messages = list(self.messages)
        if messages:
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
        overrides = dict(self.adjustments)
        try:
            solved, projections = self.resolver(overrides)
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
        # Spent, now that it has worked: these minutes are in the projections
        # every plan below is costed on, and a decision taken from here is
        # taken on them. A re-solve that raised or reached nothing gets no
        # further than the guards above and spends nothing.
        self.spent = overrides
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
        applied, unapplied = self._split()
        return ManagerDecision(
            plan=final.plan,
            lineup=replace(lineup, captain=final.captain, vice=final.vice),
            captain=final.captain,
            vice=final.vice,
            chip=final.chip,
            chip_justification=final.chip_justification,
            rationale=final.rationale,
            adjustments=applied,
            unapplied=unapplied,
            searches=self.searches,
            source=MANAGER,
            # The ones in force: the re-solve's if he re-solved, the
            # briefing's if he did not. The lineup above was picked from them
            # and the report is written on them.
            projections=self.projections,
        )

    def _split(self) -> tuple[list[dict], list[dict]]:
        """His record, cut into what took effect and what did not.

        An adjustment does nothing on its own — it is spent by the next
        ``resolve`` — so the only ones the decision is costed on are the ones
        the *last* re-solve was run on. Everything else is a note: minutes he
        wrote down after that re-solve, minutes a failed re-solve never
        reached, minutes he superseded before one, and every minute at all if
        he finalized without ever asking the solver again.

        An entry counts as applied when the re-solve was run with that player
        at that number. Both halves keep the order he made them in, and
        together they are the whole record.
        """
        applied: list[dict] = []
        unapplied: list[dict] = []
        for record in self.record:
            spent = self.spent.get(record["player_id"])
            side = applied if spent == record["expected_minutes"] else unapplied
            side.append(record)
        return applied, unapplied

    def _lineup(self, plan: Plan) -> Lineup:
        """The eleven ``plan`` fields next gameweek, on the projections in force.

        The armband is chosen on the same ceiling — goals and assists — the
        solver uses, read off the projections now in force so a re-solve on the
        manager's own minutes moves the captain the same way it moves the team.
        """
        event = self.inputs.event.id
        gw_xp = {
            pid: projection.per_gw.get(event, 0.0)
            for pid, projection in self.projections.items()
        }
        return pick_lineup(
            plan.squad, self.positions, gw_xp, attacking_evs(self.projections, event)
        )

    def _count(self, response: Any) -> None:
        """Note the searches in one turn. The server ran them and billed them;
        counting the results rather than the requests counts the ones that came
        back, which is what the report means by a search."""
        for block in getattr(response, "content", None) or []:
            if getattr(block, "type", None) == "web_search_tool_result":
                self.searches += 1

    def _note_container(self, response: Any) -> None:
        """Keep the container his searches ran in, if this turn opened one.

        The web search tool filters its own results by running code on the
        server, so a turn that searches is a turn that used a code-execution
        container: its content carries ``server_tool_use`` and
        ``code_execution_tool_result`` blocks, and the message carries the id of
        the container that produced them. :meth:`_append` puts those blocks back
        into the conversation, as it must — a turn is replayed whole or not at
        all — and from then on every request is replaying work that belongs to a
        container. The API will not take one without being told which:

            container_id is required when there are pending tool uses
            generated by code execution with tools

        Which is a 400 in a third of a second, and it is what the live manager
        hit on the second request of every conversation he ever had.

        The latest id wins, and a turn that opened no container leaves the last
        one standing: a conversation only ever has the one, and forgetting it
        because he happened not to search this turn would break the turn after.
        """
        container = getattr(response, "container", None)
        identifier = getattr(container, "id", None)
        if isinstance(identifier, str) and identifier:
            self.container = identifier

    def _append(self, response: Any) -> None:
        """Put his turn into the record, unless he said nothing at all: an empty
        content block is a message the API will not accept back."""
        content = getattr(response, "content", None)
        if content:
            self.messages.append({"role": "assistant", "content": content})


def _rolling(message: dict) -> dict:
    """``message`` with the moving cache marker on its last block.

    A copy, always: the marker belongs to one request and the next one puts it
    somewhere else, so writing it into the conversation would leave a trail of
    breakpoints behind and edit requests that have already gone.

    The last block of an assistant turn is the SDK's own object rather than a
    dict of ours, and that is the one case this leaves alone — a resumed pause
    is the only request that ends on one, it is a request in the middle of a
    turn, and rewriting somebody else's model to save a cache read is not a
    trade worth making. Marking the briefing block twice is not a case at all:
    it already carries the marker, and setting it again sets the same value.
    """
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return message
    last = content[-1]
    if not isinstance(last, dict):
        return message
    marked = list(content)
    marked[-1] = {**last, "cache_control": {"type": "ephemeral"}}
    return {**message, "content": marked}


def _text(body: str) -> dict:
    """A plain text block: what this loop writes instead of a bare string."""
    return {"type": "text", "text": body}


def _cached(text: str) -> dict:
    """A text block the server is asked to cache the prefix up to.

    Caching is a prefix match, so the two blocks this marks — the system
    prompt and the briefing, in that order — are exactly the part of the
    request that cannot change once the conversation has started. The third
    breakpoint, on the growing end of the conversation, is placed per request
    by :func:`_rolling` and never written into the conversation itself.
    """
    return {"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}


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
