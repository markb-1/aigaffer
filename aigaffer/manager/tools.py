"""What the manager is allowed to do, and who checks that he did it.

The four tools are declared here and so is every rule about their arguments,
because a guardrail written into the system prompt is a request and a guardrail
written into a validator is a boundary. The model may be persuaded, distracted
or wrong; the code that turns his answer into a team cannot be.

Four boundaries matter more than the rest.

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
  All four are now his to finalize — the solver plans chip weeks, so a
  wildcard or a free hit is a squad it actually built and the plans on the
  board carry the road it built them for. The one chip still refused here is
  one the season's history says is already spent, and that boundary lives at
  the integration seam (the orchestrator), not in this validator.
* **The rationale is the report.** It is the only thing the model writes that a
  person ever reads, and the first live decision spent its reasoning on text
  blocks the loop throws away and passed the field a placeholder. So the field
  is measured too — see :data:`MIN_RATIONALE` — and the prompt says where the
  report goes before the validator has to.

Everything here is pure. Nothing imports the orchestrator, nothing talks to an
API, and nothing renders a string that came off the FPL payload — plan lines go
through :func:`aigaffer.manager.briefing.format_plans`, which sanitizes, and
these validators name players by the only handle that cannot forge a document:
the id.
"""

import re
from collections.abc import Callable, Container
from dataclasses import dataclass
from datetime import date
from math import isfinite

from aigaffer.manager.playbook import CHIP_PLAYBOOK
from aigaffer.news_ledger import CATEGORIES, TIERS
from aigaffer.solver.optimizer import Plan

# The gaffer, as he is briefed once and cached. Byte-stable by contract: the
# date, the gameweek and the squad are all briefing lines, and one interpolated
# variable here would be a cache that never hits and a prompt nobody can diff.
_BASE = """\
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
search you do not have for the players you might own. Within that list, verify \
in this order: first the players the plan you are minded to finalize would buy \
or sell — a transfer entered on an unverified target is the one mistake this \
job exists to prevent — then your own likely starters, then the rest. Check \
the date on every report against the date in the briefing: a fitness update \
from three weeks ago is not team news, and a headline about last season is not \
news at all. Your search allowance is per turn, not for the whole job: when \
the tool stops answering mid-turn, end the turn — carrying any adjust_players \
calls for what you have already learned, rather than bare prose — and the \
allowance comes back fresh on the next one, so running dry is never a reason \
to finalize unverified — though it is a reason to spend the fresh allowance on \
the players the plan turns on, not on another round-up.
3. Where you have learned something the projection does not know, call \
adjust_players. It sets a player's expected minutes for the coming gameweek \
absolutely — 0 for a player who is out, 20 for a substitute, 90 for a starter — \
and the briefing shows you the number you are overwriting. It is the coming \
gameweek only: the weeks after it keep the model's minutes. Say why, in the \
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
- Chips come in two sets; the first expires after GW19 (see Chips below). To \
play one you must argue for it properly: why this gameweek rather than the one \
the chip calendar saves it for, what the chip EV panel says it is worth, and \
what you give up by playing it now. A sentence will be rejected.
- The solver plans chips. All four — bench_boost, triple_captain, wildcard \
and free_hit — are yours to finalize, because a plan the solver reached was \
built for the chip it recommends, and its road ahead shows the later gameweeks \
it means to play others. Finalize the chip the plan in hand recommends; a chip \
not held for this gameweek — spent, or the next set's — will be refused \
whatever you argue. If you would rather leave the chip to the person whose \
team this is, finalize with chip 'none' and make the case in your rationale.

Some plans carry a path: what the solver would go on to do in later gameweeks \
if nothing changed, including the gameweeks it plans to play a chip. Read it as \
the argument for the opening move — this is why the transfer, or the chip, is \
worth it now — and never as a commitment. Only the coming gameweek's transfers \
and chip are ever entered, and the path is planned again from scratch every \
run, so a plan you finalize commits its first gameweek and nothing else. \
resolve re-plans the paths along with the plans, on your minutes.

The briefing may carry a price watch: who the market is buying, and whose \
price moved overnight. It is timing information and nothing else — when the \
plan already buys a heavily-bought player, say in the rationale that the move \
may be cheaper tonight than at the deadline, and never let a price talk you \
into a transfer the football does not justify.

Two habits, in the order they matter.

Roll when you are not sure. A transfer has to beat doing nothing by more than \
the noise in the projection, and a hit has to beat it by four points more than \
that. "The solver has it a point ahead" is not a reason to move; "he is \
injured and the solver does not know" is.

Be honest in the rationale. It is read by the person whose team this is, after \
the gameweek has been played, next to the score. Say what you did, what you \
learned that made you do it, and what you were unsure about.

Write it in the tool call itself: the rationale field IS your report, and \
everything you want him to read has to be inside it — the news you found, the \
minutes you adjusted and why, and why this plan rather than the others. \
Anything you write outside a tool call is discarded and reaches nobody, so a \
field that says "placeholder" with the reasoning around it publishes the word \
placeholder and loses the reasoning.

Write it in three parts, under these headings:

WHAT I DID — the transfers, the captain, the chip, in a sentence or two.
WHAT I LEARNED — what the searches turned up, whose expected minutes you \
changed and from what to what, and what you are still unsure about.
WHY THIS PLAN — why this one and not the others on the board, including the \
one that rolls the transfer.

He reads it on a phone, after the gameweek, next to the score. Three headings \
is what makes that readable; a wall of prose is not. Say all three things \
even where one of them is short — "nothing I found changed a minute" is an \
answer, and an empty section is not."""

SYSTEM_PROMPT = _BASE + "\n\n" + CHIP_PLAYBOOK

# What the server tool is allowed to cost us in one request. ``max_uses`` is
# per request — one assistant turn — and not a budget for the conversation: a
# run of a dozen turns could in principle spend eight in each of them, and
# nothing here counts them across turns or stops the loop when they add up.
# What bounds a run is the turn cap and the clock in the loop itself; the
# report prints the total afterwards.
#
# Eight is enough for the handful of players a shortlist actually turns on, and
# few enough that a turn has to choose which ones those are.
MAX_SEARCHES = 8

CHIPS = ["none", "bench_boost", "triple_captain", "free_hit", "wildcard"]
NO_CHIP = CHIPS[0]

# What an argument for a chip has to be, at the very least. Neither number is
# clever: they are the two vacuous answers — the one-liner, and the essay about
# something else — and nothing longer or better-aimed is being claimed for them.
MIN_JUSTIFICATION = 200

# What the rationale has to be, at the very least. The first live decision came
# back with rationale='placeholder' and four paragraphs of real reasoning in the
# text blocks around the tool call — which the loop discards, because a text
# block is not a decision and nothing downstream reads one. The report printed
# the placeholder. So the field is measured too: not because any number of
# characters is a good report, but because a floor is longer than every way of
# not writing one, and the error that comes back says where the report belongs.
#
# A hundred was that floor and it was set against the placeholder alone: it is
# longer than a token and shorter than a sentence about the transfer, which is
# not a report either. The weeks that read well live came back in three parts
# — what he did, what he learned, why this plan — and none of them was near
# two hundred characters. So the floor moves under the shortest report worth
# having rather than over the longest way of not writing one. Nothing parses
# the headings: a rationale that says all three things in prose is the same
# rationale, and the length is the only thing measured.
MIN_RATIONALE = 200

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
            " more players, for the coming gameweek only. Minutes are absolute,"
            " not a nudge: 0 for a player who is out, 90 for a nailed starter."
            " Every later gameweek in the plans' window keeps the model's own"
            " minutes, which already rule out a player the FPL injury flag"
            " has ruled out, so a player out for a month who is not flagged"
            " reads as back after this week; weigh that in your rationale. The"
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
                            "category": {
                                "type": "string",
                                "enum": list(CATEGORIES),
                                "description": (
                                    "What kind of news: nailed, rotation, doubt,"
                                    " test_on_day, injured, suspended or ill."
                                ),
                            },
                            "tier": {
                                "type": "integer",
                                "enum": list(TIERS),
                                "description": (
                                    "How good the source is. 1: the FPL flag, the"
                                    " manager's own words, an official club update."
                                    " 2: a club reporter, The Athletic, BBC, Sky,"
                                    " Premier Injuries. 3: predicted line-ups."
                                ),
                            },
                            "quote_date": {
                                "type": "string",
                                "description": (
                                    "YYYY-MM-DD: the day the quote or evidence was"
                                    " given — not the article's date."
                                ),
                            },
                            "source": {
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                                "description": "Where you read it. null if the FPL flag alone.",
                            },
                            "return_gw": {
                                "anyOf": [{"type": "integer"}, {"type": "null"}],
                                "description": (
                                    "For an injury or a ban, the gameweek he is due"
                                    " back. null otherwise."
                                ),
                            },
                        },
                        "required": [
                            "player_id",
                            "expected_minutes",
                            "reason",
                            "category",
                            "tier",
                            "quote_date",
                            "source",
                            "return_gw",
                        ],
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
                # A string or null, never the empty string: asked for an empty
                # value here, the model reached for its own close-parameter
                # token, the strict grammar substituted a near-miss inside the
                # string, and the rest of the call spilled in after it — every
                # live finalize from 28 Aug 2026 carried the whole rationale
                # in this field behind a garbled tag. Null gives an unplayed
                # chip a value with nothing to write.
                "chip_justification": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": (
                        "Why this chip, this week: what the EV panel says, and"
                        " what is lost by burning it now. Plain prose in this"
                        " field alone. null when the chip is 'none'."
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


# A parameter tag, however the strict grammar mangled its name: ``<parameter``
# opening one, ``</antml…parameter>`` closing one with a near-miss in place of
# the token the grammar would not allow. Anchored to the two shapes so that
# prose which happens to put a ``<`` before a word ending in "parameter"
# does not cost a turn.
MARKUP = re.compile(r"</\S*parameter\b|<parameter\b")

# Null is on offer for the chip nobody played; a value that arrives anyway
# is dropped, and this line is the only trace of it.
CHIP_FIELD_DROPPED = "chip_justification arrived for chip 'none' and was dropped"


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


def validate_adjustments(
    args: dict, known: Container[int], *, today: date, event: int
) -> list[dict]:
    """The adjustments in ``args``, checked, clamped and in the order given.

    ``known`` is the player board — anything keyed by player id — and a player
    who is not on it is refused rather than silently dropped: an id nobody
    recognises usually means the model has invented a transfer target, and
    quietly ignoring it would leave him believing the minutes were set.

    Each record also carries what he found (news-ledger spec §6): the
    category, the source tier, the date of the quote — never after ``today``,
    the briefing's own date — the source, and for an injury or a ban the
    gameweek he is due back, never before ``event``. They become the news
    ledger's entry when the run is saved.

    One bad entry voids the whole call. A tool result that reported half a
    batch applied would leave the model to work out which half, and the cheap
    fix — call it again with the good ones — is one he can always make.
    """
    entries = args.get("adjustments")
    if not isinstance(entries, list) or not entries:
        raise ToolError(
            "adjust_players needs at least one adjustment:"
            " {'adjustments': [{'player_id': int, 'expected_minutes': number,"
            " 'reason': string, 'category': string, 'tier': 1|2|3,"
            " 'quote_date': 'YYYY-MM-DD', 'source': string|null,"
            " 'return_gw': int|null}]}."
        )

    records = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ToolError("Every adjustment must be an object with the eight keys.")
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
                "category": _category(entry.get("category")),
                "tier": _tier(entry.get("tier")),
                "quote_date": _quote_date(entry.get("quote_date"), today),
                "source": _source(entry.get("source")),
                "return_gw": _return_gw(entry.get("return_gw"), event),
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
    # No chip, no argument to keep: whatever was written — or spilled — into
    # the field for a chip nobody played is dropped without spending a turn
    # on it. A chip that is played has to be argued for, in this field, in
    # prose, and that check reads the field as sent.
    justification = ""
    if chip != NO_CHIP:
        justification = _text(args.get("chip_justification"))
        _check_chip(chip, justification)
    elif _text(args.get("chip_justification")).strip():
        print(CHIP_FIELD_DROPPED)

    rationale = _words(args.get("rationale"), "rationale")
    _check_rationale(rationale)
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
    if MARKUP.search(justification):
        # The two tests below pass on a rationale that spilled in behind a
        # parameter tag — it is long, and it mentions every chip — so the tag
        # is refused before either is measured.
        raise ToolError(
            "That justification carries tool-call markup — a parameter tag,"
            " and behind it text that belongs in another field. Write the"
            " argument for the chip in plain prose, in this field alone."
            f" {checklist}"
        )
    if len(justification.strip()) < MIN_JUSTIFICATION:
        raise ToolError(f"That is not an argument for a chip. {checklist}")
    if _spoken(chip) not in _spoken(justification):
        raise ToolError(
            f"That justification never mentions the {_spoken(chip)}. {checklist}"
        )


def _check_rationale(rationale: str) -> None:
    """The rationale is the report, so it has to be long enough to be one.

    The failure this catches is not a short answer but a misplaced one: the
    model writes its reasoning as prose around the tool call and drops a token
    into the field, because from where it sits both look like output. Only the
    field survives. So the error says that, in the words it would have needed
    to read beforehand, and names what belongs in it — a model that is told
    "too short" pads, and a model that is told "this is the report" writes one.

    It names the three parts for the same reason. Asking again for "more
    characters" is asking to be padded; asking for what he did, what he
    learned and why this plan is asking for the report that was missing.
    """
    if len(rationale) < MIN_RATIONALE:
        raise ToolError(
            f"That rationale is {len(rationale)} characters, and it is the"
            " whole of what the manager reads: the rationale field IS your"
            " report. Anything you write outside a tool call is discarded and"
            " never reaches him. Finalize again with the full reasoning in the"
            f" field ({MIN_RATIONALE} characters at least), in three parts:"
            " WHAT I DID — the transfers, the captain, the chip; WHAT I"
            " LEARNED — what the searches turned up, whose minutes you changed"
            " and from what to what, what you are unsure about; WHY THIS PLAN"
            " — why this one and not the others on the board, the one that"
            " rolls included."
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


def _category(value: object) -> str:
    if value not in CATEGORIES:
        raise ToolError(f"category must be one of {', '.join(CATEGORIES)}, not {value!r}.")
    return value


def _tier(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in TIERS:
        raise ToolError(
            f"tier must be 1, 2 or 3, not {value!r}. Social media and unverified"
            " claims are not a tier: search for something better or leave him be."
        )
    return value


def _quote_date(value: object, today: date) -> str:
    if not isinstance(value, str):
        raise ToolError(f"quote_date must be a date like 2026-10-06, not {value!r}.")
    try:
        given = date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"quote_date must be a date like 2026-10-06, not {value!r}.") from None
    if given > today:
        raise ToolError(
            f"quote_date {value} is after today ({today.isoformat()}): give the day"
            " the quote was given."
        )
    return value


def _source(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolError(f"source must be a string or null, not {value!r}.")
    return value.strip() or None


def _return_gw(value: object, event: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < event:
        raise ToolError(
            f"return_gw must be null or a gameweek from GW{event} on, not {value!r}."
        )
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
