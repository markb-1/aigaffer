"""What the manager is allowed to do, and who checks that he did it.

The four tools are declared here and so is every rule about their arguments,
because a guardrail written into the system prompt is a request and a guardrail
written into a validator is a boundary. The model may be persuaded, distracted
or wrong; the code that turns his answer into a team cannot be.

Three boundaries matter more than the rest.

* **A plan is an id from the registry, and nothing else.** Everything else in
  the conversation is text — the briefing quotes an API that serves us player
  names, and a tool result quotes a solver run — and text can be made to read
  like a plan line that was never solved. :func:`validate_finalize` is handed
  the registry the loop minted and consults nothing but it.
* **An armband belongs to an eleven.** The briefing shows one plan's XI, so a
  captain who is illegal for the plan he actually chose is an ordinary mistake
  rather than a lie, and the error result carries the eleven he may pick from.
* **A chip is a season's worth of points.** It costs nothing to ask for one and
  it can only be played once, so the justification is measured before it is
  accepted: long enough to be an argument, and about the chip it is playing.

Everything here is pure. Nothing imports the orchestrator, nothing talks to an
API, and nothing renders a string that came off the FPL payload — plan lines go
through :func:`aigaffer.manager.briefing.format_plans`, which sanitizes, and
these validators name players by the only handle that cannot forge a document:
the id.
"""

from collections.abc import Callable, Container
from dataclasses import dataclass
from math import isfinite

from aigaffer.solver.optimizer import Plan

# The gaffer, as he is briefed once and cached. Byte-stable by contract: the
# date, the gameweek and the squad are all briefing lines, and one interpolated
# variable here would be a cache that never hits and a prompt nobody can diff.
SYSTEM_PROMPT = """\
You are the manager of a Fantasy Premier League side. A solver has already done \
the arithmetic: it has projected every player, priced every legal transfer and \
handed you a short list of plans it can actually reach. Your job is the half it \
cannot do — what has happened since the numbers were computed, and what a \
person who watches football knows that a spreadsheet does not.

Work in this order.

1. Read the briefing. It carries the deadline, the date it was written, your \
squad, the candidate plans and the players any of them would buy or sell.
2. Search for what the model cannot know: injuries, suspensions, rotation, \
press-conference minutes, a manager saying somebody is a doubt. Search only the \
players the briefing lists as relevant — a search spent on anyone else is a \
search you do not have for the players you might own. Check the date on every \
report against the date in the briefing: a fitness update from three weeks ago \
is not team news, and a headline about last season is not news at all.
3. Where you have learned something the projection does not know, call \
adjust_players. It sets a player's expected minutes for the coming gameweek \
absolutely — 0 for a player who is out, 20 for a substitute, 90 for a starter — \
and the briefing shows you the number you are overwriting. Say why, in the \
reason: it goes into the written record.
4. Call resolve to re-run the projection and the solver with your adjustments. \
It comes back with fresh plans and fresh ids. The old ids stay valid.
5. Finish with finalize_decision, and finish with it exactly once.

The rules the code enforces, so that you are not surprised by them:

- You may only finalize a plan id that has been presented to you. The solver \
has checked those for budget, club quotas and the hit cap; a squad that is not \
on the list is not reachable, however good it would be.
- Your captain and vice-captain must both belong to the eleven of the plan you \
finalize, and they must be two different players. Choose a different plan and \
the eleven is picked again from that plan's squad.
- A chip is played once a season. To play one you must argue for it properly: \
why this gameweek rather than any other, what the chip EV panel in the briefing \
says it is worth, and what you give up by burning it now. A sentence will be \
rejected.

Two habits, in the order they matter.

Roll when you are not sure. A transfer has to beat doing nothing by more than \
the noise in the projection, and a hit has to beat it by four points more than \
that. "The solver has it a point ahead" is not a reason to move; "he is \
injured and the solver does not know" is.

Be honest in the rationale. It is read by the person whose team this is, after \
the gameweek has been played, next to the score. Say what you did, what you \
learned that made you do it, and what you were unsure about."""

# What the server tool is allowed to cost us. Eight is enough for the handful of
# players a shortlist actually turns on, and few enough that the model has to
# choose which ones those are.
MAX_SEARCHES = 8

CHIPS = ["none", "bench_boost", "triple_captain", "free_hit", "wildcard"]
NO_CHIP = CHIPS[0]

# What an argument for a chip has to be, at the very least. Neither number is
# clever: they are the two vacuous answers — the one-liner, and the essay about
# something else — and nothing longer or better-aimed is being claimed for them.
MIN_JUSTIFICATION = 200

# A match, and what a manager who is not playing plays. Expected minutes are
# clamped here as well as in the projection so that the tool result can say what
# was actually recorded rather than what was asked for.
NO_MINUTES, FULL_MATCH = 0.0, 90.0

TOOLS: list[dict] = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": MAX_SEARCHES},
    {
        "name": "adjust_players",
        "description": (
            "Overwrite the expected minutes the projection assumed for one or"
            " more players, for the coming gameweek. Minutes are absolute, not"
            " a nudge: 0 for a player who is out, 90 for a nailed starter. The"
            " adjustments accumulate and change nothing until you call resolve."
            " A player id that is not in the briefing is rejected, and one bad"
            " id voids the whole call."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "adjustments": {
                    "type": "array",
                    "description": "One entry per player. At least one.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "player_id": {
                                "type": "integer",
                                "description": "The id the briefing gives him.",
                            },
                            "expected_minutes": {
                                "type": "number",
                                "description": "Minutes he is expected to play, 0-90.",
                            },
                            "reason": {
                                "type": "string",
                                "description": (
                                    "What you learned, and where. It goes into"
                                    " the written record of the decision."
                                ),
                            },
                        },
                        "required": ["player_id", "expected_minutes", "reason"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["adjustments"],
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve",
        "description": (
            "Re-project and re-solve with every adjustment you have made so"
            " far. Comes back with a fresh set of candidate plans, numbered"
            " with new ids that continue from the ones you have already been"
            " shown. The earlier ids remain valid."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "finalize_decision",
        "description": (
            "Commit to one plan and end the conversation. The plan id must be"
            " one you have been shown; the captain and vice must both be in"
            " that plan's eleven and must differ from each other; a chip other"
            " than 'none' needs a justification that argues for it properly."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "plan_id": {
                    "type": "integer",
                    "description": "The id of the plan to play, as presented.",
                },
                "captain_id": {
                    "type": "integer",
                    "description": "Captain. Must be in the chosen plan's XI.",
                },
                "vice_id": {
                    "type": "integer",
                    "description": (
                        "Vice-captain. In the XI too, and not the captain."
                    ),
                },
                "chip": {
                    "type": "string",
                    "enum": CHIPS,
                    "description": "The chip to play, or 'none'.",
                },
                "chip_justification": {
                    "type": "string",
                    "description": (
                        "Why this chip, this week: what the EV panel says, and"
                        " what is lost by burning it now. Empty when the chip"
                        " is 'none'."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "The decision in plain words, for the manager whose"
                        " team this is: what you did, what you learned, and"
                        " what you were unsure about."
                    ),
                },
            },
            "required": [
                "plan_id",
                "captain_id",
                "vice_id",
                "chip",
                "chip_justification",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """A tool call the code will not carry out.

    The message is not a log line: it goes straight back to the model as the
    content of an ``is_error`` tool result, and it is the only chance to correct
    him. So every one of them says what was wrong *and* what would be right —
    the ids that exist, the eleven he may captain from, the checklist a chip
    has to clear. A message that only says "invalid" spends a turn saying it.
    """


@dataclass(frozen=True)
class Finalized:
    """A ``finalize_decision`` that passed every check. The plan is the object
    from the registry, never one rebuilt from the arguments."""

    plan_id: int
    plan: Plan
    captain: int
    vice: int
    chip: str
    chip_justification: str
    rationale: str


def validate_adjustments(args: dict, known: Container[int]) -> list[dict]:
    """The adjustments in ``args``, checked, clamped and in the order given.

    ``known`` is the player board — anything keyed by player id — and a player
    who is not on it is refused rather than silently dropped: an id nobody
    recognises usually means the model has invented a transfer target, and
    quietly ignoring it would leave him believing the minutes were set.

    One bad entry voids the whole call. A tool result that reported half a
    batch applied would leave the model to work out which half, and the cheap
    fix — call it again with the good ones — is one he can always make.
    """
    entries = args.get("adjustments")
    if not isinstance(entries, list) or not entries:
        raise ToolError(
            "adjust_players needs at least one adjustment:"
            " {'adjustments': [{'player_id': int, 'expected_minutes': number,"
            " 'reason': string}]}."
        )

    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ToolError("Every adjustment must be an object with the three keys.")
        pid = _whole(entry.get("player_id"), "player_id")
        if pid not in known:
            raise ToolError(
                f"There is no player {pid} in this gameweek's board. Use the ids"
                " the briefing prints beside each name, and adjust only players"
                " it lists. Nothing was recorded."
            )
        records.append(
            {
                "player_id": pid,
                "expected_minutes": _minutes(entry.get("expected_minutes")),
                "reason": _words(entry.get("reason"), "reason"),
            }
        )
    return records


def validate_finalize(
    args: dict, registry: dict[int, Plan], xi_for: Callable[[Plan], list[int]]
) -> Finalized:
    """The decision in ``args``, or the correction that has to go back.

    ``registry`` is the only authority on which plans exist: the ids the
    briefing printed and the ids every re-solve added, mapped to the plan
    objects the solver produced. Nothing is matched against the conversation.

    ``xi_for`` is handed the chosen plan and returns the eleven that plan would
    field — the loop passes the same lineup picker that builds the decision, so
    the armbands are checked against the team that will actually be entered.
    """
    plan_id = _whole(args.get("plan_id"), "plan_id")
    if plan_id not in registry:
        raise ToolError(
            f"Plan {plan_id} is not on the board. Finalize one of these ids"
            f" and no other: {_listed(sorted(registry))}."
        )
    plan = registry[plan_id]

    chip = args.get("chip", NO_CHIP)
    if chip not in CHIPS:
        raise ToolError(f"'{chip}' is not a chip. Choose one of: {', '.join(CHIPS)}.")
    justification = _text(args.get("chip_justification"))
    if chip != NO_CHIP:
        _check_chip(chip, justification)

    rationale = _words(args.get("rationale"), "rationale")
    captain = _whole(args.get("captain_id"), "captain_id")
    vice = _whole(args.get("vice_id"), "vice_id")
    _check_armbands(captain, vice, plan_id, xi_for(plan))

    return Finalized(
        plan_id=plan_id,
        plan=plan,
        captain=captain,
        vice=vice,
        chip=chip,
        chip_justification=justification,
        rationale=rationale,
    )


def _check_chip(chip: str, justification: str) -> None:
    """A chip is played once a season, so the argument for it is measured.

    Both tests are crude on purpose. Length catches the one-liner; the chip's
    own name catches the argument that was written for a different chip, or for
    none — and it is matched loosely, because a manager writing about the bench
    boost writes "bench boost" and the tool calls it ``bench_boost``.
    """
    checklist = (
        f"To play the {_spoken(chip)} you have to argue for it: why this"
        " gameweek and not another, what the chip EV panel in the briefing says"
        " it is worth, and what you give up by burning it now. Name the chip"
        f" and make the case in full ({MIN_JUSTIFICATION} characters at least),"
        " or finalize the same plan with chip 'none'."
    )
    if len(justification.strip()) < MIN_JUSTIFICATION:
        raise ToolError(f"That is not an argument for a chip. {checklist}")
    if _spoken(chip) not in _spoken(justification):
        raise ToolError(
            f"That justification never mentions the {_spoken(chip)}. {checklist}"
        )


def _check_armbands(captain: int, vice: int, plan_id: int, xi: list[int]) -> None:
    """The armbands belong to the eleven of the plan that was chosen.

    The briefing shows one plan's XI and says so; every other plan's is picked
    fresh from its own squad. So the eleven goes back with the error — without
    it the model is being told he is wrong and left to guess at right.
    """
    eleven = f"Plan {plan_id} fields: {_listed(sorted(xi))}."
    if captain == vice:
        raise ToolError(
            "The captain and the vice-captain must be two different players;"
            f" both came back as {captain}. {eleven}"
        )
    for role, pid in (("captain", captain), ("vice-captain", vice)):
        if pid not in xi:
            raise ToolError(
                f"Player {pid} is not in plan {plan_id}'s eleven, so he cannot"
                f" be its {role}. {eleven} Pick from those, or finalize a"
                " different plan."
            )


def _whole(value: object, field: str) -> int:
    """A player or plan id: an integer, and not a boolean wearing one."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolError(f"{field} must be a whole number, not {value!r}.")
    return value


def _minutes(value: object) -> float:
    """Expected minutes, clamped to a match.

    The same clamp :func:`aigaffer.orchestrator.build_projections` applies, done
    here as well and deliberately: the projection would silently correct 120 to
    90, and this way the tool result can tell him it was corrected.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ToolError(f"expected_minutes must be a number, not {value!r}.")
    minutes = float(value)
    if not isfinite(minutes):
        raise ToolError("expected_minutes must be a real number of minutes.")
    return min(FULL_MATCH, max(NO_MINUTES, minutes))


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _words(value: object, field: str) -> str:
    """A field that has to say something. The rationale is the whole point of
    asking a manager rather than a solver, and an empty one is a decision with
    no reasons attached to it."""
    text = _text(value).strip()
    if not text:
        raise ToolError(f"{field} cannot be empty — say it in your own words.")
    return text


def _spoken(text: str) -> str:
    """A chip name as anyone would write it: lower case, one kind of gap.

    The tool spells it ``bench_boost``, a manager writes "bench boost" and a
    newspaper writes "bench-boost", and all three are the same chip. The test
    this feeds is a check that the argument is about the chip being played, so
    the spelling it accepts is every spelling of it.
    """
    return " ".join(text.replace("_", " ").replace("-", " ").lower().split())


def _listed(ids: list[int]) -> str:
    return ", ".join(str(pid) for pid in ids)
