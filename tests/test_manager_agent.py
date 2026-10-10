"""Tests for the manager's loop: the twelve turns, and the guardrails on them.

Nothing here talks to Anthropic. ``FakeClient`` is handed a script — a list of
canned responses, or exceptions to raise instead — and records the keyword
arguments of every request, because half of what this module has to get right
is the shape of the request: the cached system block, the absence of any tool
choice (Opus 5.5 rejects a forced one, so the decision is asked for in words), and the rule that every tool result from one assistant turn
comes back in one user message.

The universe is eighteen players over the six clubs of the pipeline fixture,
priced and projected so that every lineup the loop picks can be read straight
off the table below. Ids 1-15 are the squad; 16-18 are the field it may buy from.

==  ======  ====  ===  ======  ===  ==================================
id  name    club  pos  price   xP   role
==  ======  ====  ===  ======  ===  ==================================
1   Alvez   ASH   GKP  £5.5m   5.0  starts
2   Byrne   BRW   GKP  £4.5m   1.0  the substitute keeper
3   Costa   ASH   DEF  £6.0m   6.0  starts
4   Dodd    BRW   DEF  £5.5m   5.5  starts; Fenn takes his place after the re-solve
5   Egan    CRV   DEF  £5.0m   5.2  starts for the roll and the swap, not for the hit
6   Fenn    DUN   DEF  £4.5m   2.0  benched — and 20.0 after the re-solve
7   Gale    EAS   DEF  £4.0m   0.5  injured; every plan that moves sells him
8   Hume    ASH   MID  £12.5m  9.0  the best player on the board, and captain
9   Innes   BRW   MID  £9.0m   7.0  starts
10  Jonker  CRV   MID  £7.5m   6.8  starts
11  Kerr    DUN   MID  £7.0m   6.6  starts
12  Lang    EAS   MID  £5.5m   2.5  benched, and never in anyone's XI
13  Moss    DUN   FWD  £10.5m  8.0  starts, and is vice-captain
14  Nunes   EAS   FWD  £8.0m   6.4  starts
15  Oduya   FAI   FWD  £6.0m   3.0  starts for the roll; benched, then sold
16  Tovey   CRV   GKP  £5.0m   4.0  nobody buys him
17  Quinn   FAI   DEF  £5.5m   6.5  bought in place of Gale
18  Reid    FAI   FWD  £9.5m   7.5  bought after the re-solve
==  ======  ====  ===  ======  ===  ==================================

Four plans exist, in two batches, because the plan-id registry is the thing the
manager finalizes against and it has to go on counting across a re-solve:

* ``SWAP`` (id 0) and ``ROLL`` (id 1) are what the solver reached before the
  conversation started — the ids the briefing printed.
* ``FRESH`` (id 2) and ``FRESH_HIT`` (id 3) are what the fake resolver hands
  back, and they must be numbered 2 and 3 and no other way.
"""

import re
from dataclasses import dataclass, field, replace
from typing import Any

import anthropic
import httpx
import pytest

from aigaffer.config import Config
from aigaffer.data.models import Bootstrap, Pick, Player, Squad
from aigaffer.manager import agent
from aigaffer.manager.agent import FINALIZE_NOW, NUDGE, ManagerDecision, run_manager
from aigaffer.manager.tools import (
    CHIP_FIELD_DROPPED,
    MIN_RATIONALE,
    SYSTEM_PROMPT,
    TOOLS,
)
from aigaffer.model.xp import PlayerProjection
from aigaffer.orchestrator import PipelineInputs, SolveResult
from aigaffer.solver.lineup import ChipEvs, Lineup
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan
from tests.fixtures import PIPELINE_BOOTSTRAP_JSON

GKP, DEF, MID, FWD = 1, 2, 3, 4
ASH, BRW, CRV, DUN, EAS, FAI = 1, 2, 3, 4, 5, 6

UNIVERSE = [
    # id, name, club, position, price, next gameweek's projection
    (1, "Alvez", ASH, GKP, 55, 5.0),
    (2, "Byrne", BRW, GKP, 45, 1.0),
    (3, "Costa", ASH, DEF, 60, 6.0),
    (4, "Dodd", BRW, DEF, 55, 5.5),
    (5, "Egan", CRV, DEF, 50, 5.2),
    (6, "Fenn", DUN, DEF, 45, 2.0),
    (7, "Gale", EAS, DEF, 40, 0.5),
    (8, "Hume", ASH, MID, 125, 9.0),
    (9, "Innes", BRW, MID, 90, 7.0),
    (10, "Jonker", CRV, MID, 75, 6.8),
    (11, "Kerr", DUN, MID, 70, 6.6),
    (12, "Lang", EAS, MID, 55, 2.5),
    (13, "Moss", DUN, FWD, 105, 8.0),
    (14, "Nunes", EAS, FWD, 80, 6.4),
    (15, "Oduya", FAI, FWD, 60, 3.0),
    (16, "Tovey", CRV, GKP, 50, 4.0),
    (17, "Quinn", FAI, DEF, 55, 6.5),
    (18, "Reid", FAI, FWD, 95, 7.5),
]

SQUAD = list(range(1, 16))
EVENT_ID = 2
HORIZON_MULTIPLE = 5


def element(pid: int, name: str, club: int, position: int, cost: int) -> Player:
    """A bootstrap element; the stats the loop never reads are zeroed."""
    return Player(
        id=pid,
        web_name=name,
        team=club,
        element_type=position,
        now_cost=cost,
        status="i" if pid == 7 else "a",
        minutes=0,
        starts=0,
        total_points=0,
        bonus=0,
        saves=0,
    )


FIXTURE = Bootstrap.model_validate(PIPELINE_BOOTSTRAP_JSON)
EVENT = FIXTURE.next_event()
BOOTSTRAP = Bootstrap(
    events=FIXTURE.events,
    teams=FIXTURE.teams,
    elements=[element(*row[:5]) for row in UNIVERSE],
)
PLAYERS = {player.id: player for player in BOOTSTRAP.elements}


def projections(
    overrides: dict[int, float] | None = None,
) -> dict[int, PlayerProjection]:
    """Next gameweek's projection for everyone, with ``overrides`` written over
    the top: the horizon total is five times the week, which keeps the two
    columns of a plan line telling different stories."""
    week = {pid: points for pid, _, _, _, _, points in UNIVERSE} | (overrides or {})
    return {
        pid: PlayerProjection(
            player_id=pid,
            per_gw={EVENT_ID: points},
            total=points * HORIZON_MULTIPLE,
        )
        for pid, points in week.items()
    }


XP = projections()
# What the re-solve hands back: Fenn is suddenly a starter, which is a change no
# adjustment made and therefore proves whose projections the final lineup used.
XP_AFTER = projections({6: 20.0})

SWAP = Plan(
    squad=[pid for pid in SQUAD if pid != 7] + [17],
    xi=[1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 17],
    transfers_in=[17],
    transfers_out=[7],
    hits=0,
    xp_total=248.0,
    objective=248.0,
)
ROLL = Plan(
    squad=SQUAD,
    xi=[1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
    transfers_in=[],
    transfers_out=[],
    hits=0,
    xp_total=245.0,
    objective=245.0,
)
FRESH = Plan(
    squad=[pid for pid in SQUAD if pid != 7] + [17],
    xi=SWAP.xi,
    transfers_in=[17],
    transfers_out=[7],
    hits=0,
    xp_total=250.0,
    objective=250.0,
)
FRESH_HIT = Plan(
    squad=[pid for pid in SQUAD if pid not in (7, 15)] + [17, 18],
    xi=[1, 3, 4, 8, 9, 10, 11, 13, 14, 17, 18],
    transfers_in=[17, 18],
    transfers_out=[7, 15],
    hits=1,
    xp_total=254.0,
    objective=250.0,
)

# The eleven the solver's own choice fields, and the four behind it. Hand
# written rather than picked, because the fallback hands this object straight
# back and a test that computed it would prove nothing about that.
LINEUP0 = Lineup(xi=SWAP.xi, captain=8, vice=13, bench=[2, 15, 12, 6])
CHIPS = ChipEvs(bench_boost=2.5, triple_captain=9.0, free_hit=-3.0, wildcard=4.0)

SOLVE0 = SolveResult(
    plans=[SWAP, ROLL], choice=SWAP, lineup=LINEUP0, chips=CHIPS, draft_mode=False
)
SOLVE1 = SolveResult(
    plans=[FRESH, FRESH_HIT],
    choice=FRESH,
    lineup=Lineup(xi=FRESH.xi, captain=8, vice=13, bench=[2, 15, 12, 6]),
    chips=CHIPS,
    draft_mode=False,
)

# The eleven pick_lineup reaches for each plan, on the projections named — the
# best of the eight legal formations, which for the swap is a back four and for
# the roll is not. The manager's captain and vice have to belong to one of them.
XI_SWAP = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 17]
XI_SWAP_AFTER = [1, 3, 4, 6, 8, 9, 10, 11, 13, 14, 17]
XI_ROLL = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]
XI_FRESH_HIT = [1, 3, 4, 8, 9, 10, 11, 13, 14, 17, 18]
XI_FRESH_HIT_AFTER = [1, 3, 6, 8, 9, 10, 11, 13, 14, 17, 18]

BRIEFING = "# AI Gaffer — manager briefing: GW2\n\n(the week, as he is told it)"
# How it goes into the conversation: one text block, marked as the end of the
# prefix the server may cache.
CACHED_BRIEFING = [
    {"type": "text", "text": BRIEFING, "cache_control": {"type": "ephemeral"}}
]
# A rationale as the report prints it: long enough to clear the floor the
# validator sets, because the field is the whole of what the owner reads. It says
# the three things the prompt asks for — what was done, what was learned and
# what it changed, and why this plan — because a fixture that would be refused
# live is a fixture that proves nothing.
RATIONALE = (
    "Gale is out for a month with a hamstring and the solver did not know it."
    " Quinn comes in: he plays every minute, and he is at home on Saturday."
    " I have taken Gale's minutes to zero and left the rest of the squad as it"
    " was. Rolling instead would leave a hole in the eleven for the sake of a"
    " free transfer I have no better use for next week."
)

CFG = Config(team_id=42, anthropic_api_key="sk-test")

# The rest of an argument for a chip, with the chip's name left out of it: long
# enough to clear the floor, so that the only thing under test is the spelling.
CHIP_CASE = (
    " The panel prices it above anything else on the board this week, all four"
    " of the bench have home fixtures against the bottom three, and what we give"
    " up is the double gameweek in GW34 — eight months of injuries away, against"
    " eleven points that are on offer on Saturday afternoon."
)

# A chip justification that does the work: names the chip, says why this week,
# what the panel is worth and what burning it now costs.
GOOD_CHIP = (
    "Triple captain now: Hume is home to the worst defence in the division, the"
    " panel prices the chip at +9.0 points this week against a season median"
    " nearer +5, and he is the only player on this board projected above nine."
    " What we lose is the double gameweek in GW34, where a triple captain is"
    " ordinarily worth more — but that fixture is eight months of injuries away"
    " and this one is on Saturday, so the certain points win."
)


def squad_held() -> Squad:
    return Squad(
        picks=[
            Pick(
                element=pid,
                position=order,
                is_captain=pid == 8,
                is_vice_captain=pid == 13,
            )
            for order, pid in enumerate(SQUAD, start=1)
        ],
        bank=28,
        event=1,
    )


INPUTS = PipelineInputs(
    bootstrap=BOOTSTRAP,
    fixtures=[],
    event=EVENT,
    squad=squad_held(),
    free_transfers=1,
    histories={},
    players=PLAYERS,
)


# --- the fake API ----------------------------------------------------------


class Block:
    """A content block as the SDK hands one over: attributes, not keys."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)


@dataclass
class Response:
    """A ``Message`` as far as this loop reads one.

    ``container`` is the code-execution container the turn's server-side tools
    ran in, and it is None on a turn that ran none — exactly as the API leaves
    it. A turn that searched has one, because the web search tool filters its
    results by running code.
    """

    content: list[Block]
    stop_reason: str = "tool_use"
    stop_details: Any = None
    container: Any = None


class ScriptFault(BaseException):
    """A test's own script, misused — and deliberately not an ``Exception``.

    The loop's safety net catches ``Exception`` and turns it into a fallback
    decision, which is the whole point of the loop and exactly wrong here: a
    test that over-ran its script, or a loop that read what it must not, would
    launder into ``solver-fallback: unexpected AssertionError`` and pass while
    proving nothing. A ``BaseException`` goes past the net and fails the test.
    """


class Refusal:
    """A refusal whose content is a landmine.

    ``stop_reason`` has to be read before ``content``, because on a refusal
    there may be nothing in the content worth reading — so here there is
    something in it worth not reading.
    """

    stop_reason = "refusal"
    stop_details = Block(type="refusal", category="cyber", explanation="no")

    @property
    def content(self) -> list[Block]:
        raise ScriptFault("the loop read the content of a refusal")


class FakeClient:
    """Answers ``messages.create`` from a script, and remembers what it was
    asked. A scripted entry that is an exception is raised instead."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.messages = _Messages(self)

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if not self.script:
            raise ScriptFault(
                f"the loop asked for turn {len(self.requests)}, past the script"
            )
        answer = self.script.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


@dataclass
class _Messages:
    client: FakeClient

    def create(self, **kwargs: Any) -> Any:
        return self.client._create(**kwargs)


@dataclass
class FakeResolver:
    """The project-and-solve closure Task 5 will build, without the solver."""

    solved: SolveResult = field(default_factory=lambda: SOLVE1)
    projections: dict[int, PlayerProjection] = field(default_factory=lambda: XP_AFTER)
    error: Exception | None = None
    calls: list[dict[int, float]] = field(default_factory=list)

    def __call__(self, overrides: dict[int, float]):
        self.calls.append(dict(overrides))
        if self.error is not None:
            raise self.error
        return self.solved, self.projections


def use(name: str, payload: dict, block_id: str = "tu_1") -> Block:
    return Block(type="tool_use", id=block_id, name=name, input=payload)


def text(body: str = "Reading the news.") -> Block:
    return Block(type="text", text=body)


def searched() -> Block:
    return Block(
        type="web_search_tool_result",
        tool_use_id="srvtoolu_1",
        content=[Block(type="web_search_result", title="Gale out", url="https://x")],
    )


def researched() -> list[Block]:
    """A turn's research, in the blocks the live API actually returns for it.

    This is the shape that broke the manager for the whole of Phase 2, and it
    is not the shape anyone writing this loop would have guessed. The web
    search tool filters its own results by running code on Anthropic's servers,
    so the search arrives wrapped in a code execution: a ``server_tool_use``
    for the container, a ``server_tool_use`` for the query, the search result,
    the execution result, and a thinking block over the lot. The last search
    here is refused because ``max_uses`` is per request and this turn spent all
    eight, so its result block is an error object rather than a list — echoed
    back with everything else, because a turn is replayed whole or not at all.
    """
    return [
        Block(type="thinking", thinking="Who is actually fit?", signature="sig"),
        Block(type="server_tool_use", id="srv_0", name="code_execution", input={}),
        Block(
            type="server_tool_use",
            id="srv_1",
            name="web_search",
            input={"query": "Gale injury"},
        ),
        searched(),
        Block(
            type="code_execution_tool_result",
            tool_use_id="srv_0",
            content=Block(type="code_execution_result", return_code=0, stdout="1 hit"),
        ),
        Block(
            type="server_tool_use",
            id="srv_2",
            name="web_search",
            input={"query": "Hume fitness"},
        ),
        Block(
            type="web_search_tool_result",
            tool_use_id="srv_2",
            content=Block(
                type="web_search_tool_result_error", error_code="max_uses_exceeded"
            ),
        ),
        Block(type="text", text="Gale is out."),
    ]


def opened(identifier: str = "container_1") -> Block:
    """The container a turn's code execution ran in, as the message carries it."""
    return Block(id=identifier, expires_at="2026-08-21T22:18:11Z", skills=None)


def reply(*blocks: Block, stop: str = "tool_use", container: Any = None) -> Response:
    return Response(content=list(blocks), stop_reason=stop, container=container)


def adjust(
    pid: int = 7,
    minutes: float = 0.0,
    reason: str = "hamstring",
    *,
    category: str = "doubt",
    tier: int = 1,
    quote_date: str = "2026-08-20",
    source: str | None = "club update",
    return_gw: int | None = None,
) -> dict:
    return {
        "adjustments": [
            {
                "player_id": pid,
                "expected_minutes": minutes,
                "reason": reason,
                "category": category,
                "tier": tier,
                "quote_date": quote_date,
                "source": source,
                "return_gw": return_gw,
            }
        ]
    }


def finalize(
    plan_id: int = 1,
    captain: int = 8,
    vice: int = 13,
    chip: str = "none",
    justification: str = "",
    rationale: str = RATIONALE,
) -> dict:
    return {
        "plan_id": plan_id,
        "captain_id": captain,
        "vice_id": vice,
        "chip": chip,
        "chip_justification": justification,
        "rationale": rationale,
    }


def converse(script: list[Any], resolver: FakeResolver | None = None):
    """Run the loop over ``script``; hand back the client and the decision."""
    client = FakeClient(script)
    decision = run_manager(
        client, CFG, INPUTS, SOLVE0, XP, BRIEFING, resolver or FakeResolver()
    )
    # Every conversation any test runs must alternate: two user turns running
    # is a request the API is not promised to take.
    assert_roles_alternate(client)
    return client, decision


def sent_results(request: dict) -> list[dict]:
    """The tool results the loop sent back for the turn before ``request``."""
    last = request["messages"][-1]
    assert last["role"] == "user", "tool results must go back as a user message"
    return last["content"]


def only_result(request: dict) -> dict:
    results = sent_results(request)
    assert len(results) == 1
    return results[0]


# --- the happy path --------------------------------------------------------


def test_the_manager_searches_adjusts_resolves_and_decides():
    resolver = FakeResolver()
    client, decision = converse(
        [
            reply(text(), searched(), searched(), use("adjust_players", adjust())),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=3))),
        ],
        resolver,
    )

    assert decision.source == "manager"
    assert decision.plan is FRESH_HIT
    assert decision.captain == 8 and decision.vice == 13
    assert decision.chip == "none"
    assert decision.rationale == RATIONALE
    assert decision.searches == 2
    assert len(client.requests) == 3


def test_the_briefing_opens_the_conversation():
    client, _ = converse([reply(use("finalize_decision", finalize()))])

    assert client.requests[0]["messages"] == [
        {"role": "user", "content": CACHED_BRIEFING}
    ]


def test_the_briefing_is_cached_alongside_the_system_prompt():
    # The two longest things in the request are also the two that never change
    # once the conversation has started, and a dozen turns each resend both.
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize())),
        ]
    )

    for request in client.requests:
        assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
        opening = request["messages"][0]["content"]
        assert opening[0]["text"] == BRIEFING
        assert opening[0]["cache_control"] == {"type": "ephemeral"}


def test_the_adjustments_reach_the_resolver_and_the_decision():
    resolver = FakeResolver()
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 0.0, "out for a month"))),
            reply(use("adjust_players", adjust(4, 60.0, "back from the bench"))),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ],
        resolver,
    )

    assert resolver.calls == [{7: 0.0, 4: 60.0}]
    assert decision.adjustments == [
        {"player_id": 7, "expected_minutes": 0.0, "reason": "out for a month"},
        {"player_id": 4, "expected_minutes": 60.0, "reason": "back from the bench"},
    ]


def test_the_adjustments_a_resolve_spent_are_the_ones_it_reports():
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 0.0, "out for a month"))),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )

    assert decision.adjustments == [
        {"player_id": 7, "expected_minutes": 0.0, "reason": "out for a month"}
    ]
    assert decision.unapplied == []


def test_an_adjustment_made_after_the_last_resolve_is_not_claimed_as_applied():
    # The minutes he set after the last re-solve were never projected on: the
    # plan he finalized was costed before he said it, and a report that listed
    # it beside the decision would claim a squad that was never solved.
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 0.0, "out for a month"))),
            reply(use("resolve", {})),
            reply(use("adjust_players", adjust(4, 60.0, "rested, said the paper"))),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )

    assert [record["player_id"] for record in decision.adjustments] == [7]
    assert decision.unapplied == [
        {"player_id": 4, "expected_minutes": 60.0, "reason": "rested, said the paper"}
    ]


def test_adjustments_with_no_resolve_behind_them_changed_nothing_at_all():
    # He adjusted and then finalized without re-solving, so the plan, the
    # eleven and every number on them are the solver's own — and all he did was
    # write something down.
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 0.0, "out for a month"))),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert decision.adjustments == []
    assert [record["player_id"] for record in decision.unapplied] == [7]
    assert decision.plan is SWAP
    assert decision.projections is XP
    assert decision.lineup.xi == XI_SWAP


def test_a_resolve_that_failed_spends_none_of_his_adjustments():
    resolver = FakeResolver(error=RuntimeError("no legal squad"))
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 0.0, "out for a month"))),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ],
        resolver,
    )

    assert decision.adjustments == []
    assert [record["player_id"] for record in decision.unapplied] == [7]


def test_a_resolve_presents_its_plans_with_ids_that_go_on_counting():
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=3))),
        ]
    )
    presented = only_result(client.requests[1])["content"]

    assert "plan 2: out Gale (id 7," in presented
    assert "plan 3: out Gale (id 7," in presented
    assert "plan 0" not in presented and "plan 1" not in presented


def test_a_resolve_shows_where_its_fresh_plans_are_going():
    # A re-solve runs the same solver the briefing's plans came off, so the
    # plans it hands back carry paths of their own — planned on his minutes,
    # which is the whole reason he re-solved.
    ahead = replace(
        FRESH,
        path=PlannedPath(
            moves=[
                PlannedMove(event=3, transfers_in=[18], transfers_out=[15], hits=0)
            ],
            objective=FRESH.objective,
            weekly_xp={},
        ),
    )
    resolver = FakeResolver(
        solved=replace(SOLVE1, plans=[ahead, FRESH_HIT], choice=ahead)
    )

    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ],
        resolver,
    )
    presented = only_result(client.requests[1])["content"]

    assert "plan 2: out Gale (id 7," in presented
    assert " | path: GW3 +Reid -Oduya" in presented
    assert "plan 3: " in presented and presented.count("| path:") == 1


def test_the_system_prompt_says_a_path_is_advice_and_not_a_commitment():
    # He is choosing between openings on the strength of where they lead, so
    # he has to know that only the opening is ever entered — and that asking
    # for a re-solve plans the rest of it again on his own minutes.
    assert "Some plans carry a path" in SYSTEM_PROMPT
    assert "the coming gameweek's transfers and chip are ever entered" in SYSTEM_PROMPT
    assert "planned again from scratch every run" in SYSTEM_PROMPT
    assert "resolve re-plans the paths" in SYSTEM_PROMPT


def test_a_resolve_says_the_earlier_plans_are_still_on_the_board():
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert "0-1" in only_result(client.requests[1])["content"]


def test_a_plan_from_before_the_resolve_can_still_be_finalized():
    _, decision = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert decision.plan is SWAP


def test_two_resolves_keep_numbering_upwards():
    client, decision = converse(
        [
            reply(use("resolve", {})),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=5))),
        ]
    )

    assert "plan 4:" in only_result(client.requests[2])["content"]
    assert decision.plan is FRESH_HIT


def test_the_lineup_is_repicked_for_the_plan_the_manager_chose():
    _, decision = converse([reply(use("finalize_decision", finalize(plan_id=1)))])

    assert decision.lineup.xi == XI_ROLL
    assert decision.lineup.bench == [2, 12, 6, 7]


def test_the_manager_overrules_the_armbands_the_lineup_would_have_picked():
    _, decision = converse(
        [reply(use("finalize_decision", finalize(plan_id=1, captain=15, vice=9)))]
    )

    assert decision.lineup.captain == 15 and decision.lineup.vice == 9
    assert (decision.captain, decision.vice) == (15, 9)


def test_the_final_lineup_is_built_on_the_projections_the_resolve_returned():
    # Fenn is worth 2.0 on the briefing's projections and 20.0 on the ones the
    # re-solve handed back, and only one of those two puts him in the XI.
    _, decision = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=3))),
        ]
    )

    assert decision.lineup.xi == XI_FRESH_HIT_AFTER
    assert 6 in decision.lineup.xi and 4 not in decision.lineup.xi


def test_the_decision_carries_the_projections_it_was_costed_on():
    # The report is written on these: the eleven is his, and a team sheet
    # ordered by numbers he overruled is a team sheet nobody can check.
    _, decision = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=3))),
        ]
    )

    assert decision.projections is XP_AFTER


def test_a_decision_with_no_resolve_carries_the_ones_it_started_with():
    _, decision = converse([reply(use("finalize_decision", finalize(plan_id=0)))])

    assert decision.projections is XP


def test_a_fallback_carries_no_projections_of_its_own():
    # It is the solver's own week, and the solver's own projections are the
    # ones the pipeline already has in hand.
    _, decision = converse([rate_limited()])

    assert decision.projections is None


def test_without_a_resolve_the_lineup_stands_on_the_briefings_projections():
    # The same plan, on the same two sets of projections: Egan plays on the
    # briefing's, Fenn on the ones no resolve ever asked for.
    _, decision = converse([reply(use("finalize_decision", finalize(plan_id=0)))])

    assert decision.lineup.xi == XI_SWAP != XI_SWAP_AFTER


# --- the shape of the request ----------------------------------------------


def breakpoints(request: dict) -> int:
    """How many cache breakpoints the request spends, the system block included.

    Four is the API's cap, and this loop is meant to spend three: the system
    prompt, the briefing, and one that rides the end of the conversation.
    """
    blocks = [*request["system"]]
    for message in request["messages"]:
        blocks += message["content"]
    return sum(
        isinstance(block, dict) and "cache_control" in block for block in blocks
    )


def test_the_end_of_the_conversation_is_cached_too():
    # The prefix that grows is the expensive one: by the last turn every
    # request is resending the whole conversation, and without a marker on the
    # end of it only the briefing and the prompt are ever read from cache.
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("adjust_players", adjust())),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )

    for request in client.requests:
        last = request["messages"][-1]["content"][-1]
        assert last["cache_control"] == {"type": "ephemeral"}
        assert breakpoints(request) <= 4, "four is the cap, and it is not ours to blow"


def test_the_marker_on_the_end_moves_rather_than_accumulating():
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("adjust_players", adjust())),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )
    third = client.requests[2]["messages"]

    # The briefing, the end of the conversation, and the system block: three,
    # not one more for every turn that has been and gone.
    assert breakpoints(client.requests[2]) == 3
    assert "cache_control" not in third[-3]["content"][-1], "where it used to be"
    assert third[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert client.requests[2]["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_a_turn_that_cannot_be_marked_is_left_alone():
    # A resumed pause ends the request on the assistant's own content blocks,
    # which are the SDK's objects and not ours to rewrite. Two breakpoints then,
    # and no crash.
    paused = reply(text("Half a search in."), searched(), stop="pause_turn")
    client, decision = converse([paused, reply(use("finalize_decision", finalize()))])

    assert breakpoints(client.requests[1]) == 2
    assert decision.source == "manager"


def test_every_request_carries_the_same_cached_system_block():
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize())),
        ]
    )
    blocks = [request["system"] for request in client.requests]

    assert blocks[0] == [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert blocks[0] == blocks[1], "a system block that moves is a cache that misses"


def last_text(request: dict) -> str:
    """The text of the last block of the last message of a request."""
    return request["messages"][-1]["content"][-1].get("text", "")


def _bare(message: dict) -> dict:
    """``message`` with the cache marker taken off its last block."""
    content = message["content"]
    if isinstance(content, list) and content and isinstance(content[-1], dict):
        last = {k: v for k, v in content[-1].items() if k != "cache_control"}
        return {**message, "content": [*content[:-1], last]}
    return message


def assert_append_only(client) -> None:
    """Every request's messages begin with the previous request's, unchanged.

    Opus 5.5 binds thinking blocks to the conversation, so editing or dropping
    an earlier message breaks them. Two allowances, and no more: the cache
    marker rides the last block of each request's last message, so it is
    stripped before comparing; and that last message, not yet answered, may
    have gained FINALIZE_NOW as a trailing text block.
    """
    for before, after in zip(client.requests, client.requests[1:]):
        old, new = before["messages"], after["messages"]
        assert len(new) >= len(old)
        assert new[: len(old) - 1] == old[:-1]
        was = _bare(old[-1])
        # The old last block carried the rolling marker and lost it above; in
        # the new request it is no longer last, so a block that is the briefing
        # may still carry its own permanent one. Strip both the same way.
        grown = _bare({**new[len(old) - 1], "content": new[len(old) - 1]["content"][: len(was["content"])]})
        assert grown["role"] == was["role"]
        assert grown["content"] == was["content"]
        extra = [
            {k: v for k, v in block.items() if k != "cache_control"}
            for block in new[len(old) - 1]["content"][len(was["content"]):]
        ]
        assert extra in ([], [{"type": "text", "text": FINALIZE_NOW}])


def assert_roles_alternate(client) -> None:
    """No request has two user messages in a row.

    User turns only: a resumed pause_turn legitimately puts one assistant turn
    after another, and that is the API's own continuation, not ours.
    """
    for request in client.requests:
        roles = [message["role"] for message in request["messages"]]
        assert not any(a == b == "user" for a, b in zip(roles, roles[1:])), roles


def test_finalize_decision_is_a_strict_tool():
    # Strict is what keeps a schema-valid call once tool_choice is gone.
    tool = next(t for t in TOOLS if t["name"] == "finalize_decision")

    assert tool["strict"] is True
    assert tool["input_schema"]["additionalProperties"] is False


def test_the_request_is_the_one_the_api_reference_specifies():
    client, _ = converse([reply(use("finalize_decision", finalize()))])
    request = client.requests[0]

    assert request["model"] == CFG.manager_model
    assert request["max_tokens"] == 16000
    assert request["output_config"] == {"effort": "medium"}
    assert request["model"] == "claude-opus-5-5"
    assert request["tools"] is TOOLS
    # Adaptive thinking is the default on this model and an explicit config
    # risks a 400; sampling parameters are rejected outright.
    assert "thinking" not in request
    assert "temperature" not in request
    assert "tool_choice" not in request


# --- the container his research ran in -------------------------------------


def researching_turn() -> Response:
    """The live turn 1, in miniature: eight searches spent, then a tool call."""
    return reply(
        *researched(), use("adjust_players", adjust()), container=opened()
    )


def test_the_container_his_research_ran_in_is_carried_into_the_next_request():
    # The defect this pins, in the API's own words: "container_id is required
    # when there are pending tool uses generated by code execution with tools".
    # The first request has no container to send and needs none; the second one
    # replays a turn full of code-execution blocks, and without the id it is a
    # 400 in a third of a second. Live, that was every conversation the manager
    # ever had — turn 1 completed, turn 2 was refused, and he never decided.
    client, decision = converse(
        [researching_turn(), reply(use("finalize_decision", finalize()))]
    )
    first, second = client.requests
    replayed = second["messages"][1]["content"]

    assert "container" not in first, "there is no container until a turn opens one"
    assert second["container"] == "container_1"
    # And what makes it mandatory is in the message it accompanies: the turn
    # goes back whole, code execution, refused search and all.
    kinds = [block.type for block in replayed]
    assert "code_execution_tool_result" in kinds
    refused = [
        block
        for block in replayed
        if block.type == "web_search_tool_result"
        and getattr(block.content, "error_code", None) == "max_uses_exceeded"
    ]
    assert len(refused) == 1, "the search the cap refused goes back untouched too"
    assert decision.source == "manager"


def test_a_conversation_that_ran_no_code_never_invents_a_container():
    # An id we made up would be an id the API has never heard of, and the
    # request would be refused for the opposite reason.
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )

    assert all("container" not in request for request in client.requests)


def test_the_container_outlives_the_turn_that_opened_it():
    # A later turn that searches nothing carries no container of its own, and
    # the conversation still holds every block the first one produced.
    client, _ = converse(
        [
            researching_turn(),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )

    assert [request.get("container") for request in client.requests] == [
        None,
        "container_1",
        "container_1",
    ]


def test_a_paused_turn_is_resumed_inside_its_own_container():
    # The one request that ends on an assistant turn is also the one most
    # certain to be replaying code execution: the server paused it mid-research.
    paused = reply(*researched(), stop="pause_turn", container=opened())
    client, decision = converse([paused, reply(use("finalize_decision", finalize()))])

    assert client.requests[1]["container"] == "container_1"
    assert decision.source == "manager"


def test_the_system_prompt_says_nothing_that_changes_between_runs():
    # Today's date is a briefing line. In the system prompt it would be a cache
    # that never hits, and this is the shape that would have gone in.
    assert not re.search(r"\b(19|20)\d\d\b", SYSTEM_PROMPT)
    assert "finalize_decision" in SYSTEM_PROMPT


def test_the_prompt_orders_the_research_plan_first():
    # GW2 live: the gaffer spent all eight searches on round-ups and a player
    # it decided not to buy, and finalized with its own transfer target
    # unverified. The priority has to be in the prompt — the briefing's list
    # is flat and the model cannot know it from the tool schema.
    assert "the players the plan you are minded to finalize" in SYSTEM_PROMPT


def test_the_prompt_says_the_search_allowance_is_per_turn():
    # max_uses is per request, but the model can only discover the cap by
    # hitting it — and what it concluded live was that research was over.
    assert "per turn, not for the whole job" in SYSTEM_PROMPT
    assert "never a reason to finalize unverified" in SYSTEM_PROMPT


def test_the_tools_are_the_four_the_plan_names():
    names = [tool.get("name") for tool in TOOLS]

    assert names == ["web_search", "adjust_players", "resolve", "finalize_decision"]
    assert TOOLS[0] == {
        "type": "web_search_20260209",
        "name": "web_search",
        "max_uses": 8,
    }


@pytest.mark.parametrize("tool", TOOLS[1:])
def test_every_custom_tool_is_strict_and_closed(tool: dict):
    schema = tool["input_schema"]

    assert tool["strict"] is True
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(schema["properties"])


def test_finalize_takes_every_field_the_decision_needs():
    tool = next(t for t in TOOLS if t.get("name") == "finalize_decision")
    schema = tool["input_schema"]

    assert sorted(schema["properties"]) == [
        "captain_id",
        "chip",
        "chip_justification",
        "plan_id",
        "rationale",
        "vice_id",
    ]
    assert schema["properties"]["chip"]["enum"] == [
        "none",
        "bench_boost",
        "triple_captain",
        "free_hit",
        "wildcard",
    ]


# --- the guardrails --------------------------------------------------------


def test_a_plan_id_nobody_minted_is_refused_and_the_next_one_lands():
    client, decision = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=99))),
            reply(use("finalize_decision", finalize(plan_id=1))),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "99" in refusal["content"]
    assert "0, 1" in refusal["content"], "the error has to name the ids that exist"
    assert decision.plan is ROLL
    assert decision.source == "manager"


def test_a_plan_id_from_a_resolve_that_has_not_happened_is_still_refused():
    # Plans 2 and 3 exist only after the resolver has run.
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=3))),
            reply(use("finalize_decision", finalize(plan_id=1))),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_captain_outside_the_chosen_plans_xi_is_refused_with_the_xi():
    # Lang (12) is in the squad and in nobody's eleven. The briefing only ever
    # showed one plan's XI, so the error has to carry this plan's.
    client, decision = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=0, captain=12))),
            reply(use("finalize_decision", finalize(plan_id=0, captain=8))),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "12" in refusal["content"]
    assert ", ".join(str(pid) for pid in XI_SWAP) in refusal["content"]
    assert decision.captain == 8


def test_a_vice_outside_the_xi_is_refused_too():
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=0, vice=15))),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_captain_who_plays_for_somebody_else_is_refused():
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=0, captain=999))),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_captain_who_is_also_the_vice_is_refused():
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(plan_id=0, captain=8, vice=8))),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_chip_played_on_a_vacuous_justification_is_refused():
    client, _ = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="triple_captain", justification="Hume is good."),
                )
            ),
            reply(use("finalize_decision", finalize())),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "why this gameweek" in refusal["content"]
    assert "EV panel" in refusal["content"]
    assert "burning it now" in refusal["content"]


def test_a_long_justification_that_never_names_the_chip_is_refused():
    client, _ = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="bench_boost", justification=GOOD_CHIP),
                )
            ),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


@pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
def test_a_chip_the_solver_now_plans_can_be_finalized(chip: str):
    # The Phase 2 guard is gone: the solver plans chip weeks, so a wildcard or a
    # free hit is a squad it actually built. A well-argued one is finalized like
    # any other chip — the only remaining refusal is a chip already spent, and
    # that lives at the orchestrator seam, not in this validator.
    argued = f"Playing the {chip.replace('_', ' ')} this week.{CHIP_CASE}"
    client, decision = converse(
        [
            reply(
                use("finalize_decision", finalize(chip=chip, justification=argued))
            )
        ]
    )

    assert decision.chip == chip
    assert decision.chip_justification == argued
    assert decision.source == "manager"
    assert len(client.requests) == 1, "it was accepted the first time"


@pytest.mark.parametrize("chip", ["wildcard", "free_hit"])
def test_a_now_plannable_chip_still_needs_a_real_argument(chip: str):
    # Relaxing the squad-rewrite guard did not relax the justification: a
    # one-liner for a wildcard or a free hit is refused exactly as it is for the
    # other two.
    client, _ = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip=chip, justification="Play it, looks good."),
                )
            ),
            reply(use("finalize_decision", finalize())),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "argue for it" in refusal["content"]


def test_the_system_prompt_says_where_the_report_has_to_be_written():
    # He wrote his reasoning in text blocks and 'placeholder' in the field. The
    # prompt now says the field is the report and that the rest is thrown away.
    assert "the rationale field IS your report" in SYSTEM_PROMPT
    assert "discarded" in SYSTEM_PROMPT


def test_the_system_prompt_asks_for_the_shape_the_report_reads_best_in():
    # Three parts, because that is what the rationales worth reading came back
    # as live. It is asked for and not enforced: nothing parses the headings,
    # and a report that says all three things in prose is the same report.
    for heading in ("WHAT I DID", "WHAT I LEARNED", "WHY THIS PLAN"):
        assert heading in SYSTEM_PROMPT


def test_the_system_prompt_says_the_solver_plans_the_chips():
    # The Phase 2 line naming only two finalizable chips is gone; the prompt now
    # says all four are his and that the road ahead shows the planned chip weeks.
    assert "The only chips you may finalize" not in SYSTEM_PROMPT
    assert "The solver plans chips" in SYSTEM_PROMPT
    assert "not held for this gameweek" in SYSTEM_PROMPT


def test_a_chip_argued_properly_is_played():
    _, decision = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="triple_captain", justification=GOOD_CHIP),
                )
            )
        ]
    )

    assert decision.chip == "triple_captain"


def test_no_chip_needs_no_justification():
    _, decision = converse([reply(use("finalize_decision", finalize(chip="none")))])

    assert decision.chip == "none"


def test_a_decision_with_no_rationale_is_refused():
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(rationale="  "))),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_placeholder_rationale_is_refused_and_the_real_one_accepted():
    # The live incident this guards: the model wrote 'placeholder' into the
    # field and put its actual reasoning in text blocks, which are discarded.
    client, decision = converse(
        [
            reply(
                text("Gale is out; Quinn is the buy. Writing it up below."),
                use("finalize_decision", finalize(rationale="placeholder")),
            ),
            reply(use("finalize_decision", finalize())),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "rationale" in refusal["content"]
    assert "report" in refusal["content"]
    assert str(MIN_RATIONALE) in refusal["content"]
    assert decision.source == "manager"
    assert decision.rationale == RATIONALE


def test_a_single_paragraph_no_longer_clears_the_floor():
    # 150 characters: one sentence about the transfer, which passed the first
    # floor of 100 and is not a report. What came back live and read well ran
    # to three parts — what he did, what he learned, why this plan — and the
    # floor is now under the shortest of those rather than under a sentence.
    paragraph = "Gale is out and Quinn comes in; the fixture is the easiest on"
    paragraph += " the board and he starts every week for them, so this is the"
    paragraph += " move to make."
    assert 100 <= len(paragraph) < MIN_RATIONALE

    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(rationale=paragraph))),
            reply(use("finalize_decision", finalize())),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert str(MIN_RATIONALE) in refusal["content"]
    # The error names the three parts it wants, so that a model told it is too
    # short writes a report rather than padding the one it had.
    for heading in ("did", "learned", "why this plan"):
        assert heading in refusal["content"].lower()


# Whitespace cannot pad the floor: it is measured on the stripped text.
@pytest.mark.parametrize("padding", ["", "   ", "\n\n"])
def test_a_rationale_one_character_short_of_the_floor_is_refused(padding: str):
    short = f"{padding}{'x' * (MIN_RATIONALE - 1)}{padding}"
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(rationale=short))),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


@pytest.mark.parametrize("padding", ["", "   ", "\n\n"])
def test_a_rationale_exactly_at_the_floor_is_accepted(padding: str):
    written = f"{padding}{'x' * MIN_RATIONALE}{padding}"
    client, decision = converse(
        [reply(use("finalize_decision", finalize(rationale=written)))]
    )

    assert decision.source == "manager"
    assert decision.rationale == "x" * MIN_RATIONALE
    assert len(client.requests) == 1, "it was accepted the first time"


@pytest.mark.parametrize("written", ["bench boost", "bench-boost", "BENCH_BOOST"])
def test_the_chip_may_be_spelled_however_it_is_spelled(written: str):
    argued = f"Playing the {written} this week.{CHIP_CASE}"
    _, decision = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="bench_boost", justification=argued),
                )
            )
        ]
    )

    assert decision.chip == "bench_boost"
    assert decision.chip_justification == argued


def test_an_adjustment_with_no_reason_given_is_refused():
    # The reason is what the report prints beside the minutes. Without it the
    # record says a number was changed and nothing about why.
    client, _ = converse(
        [
            reply(use("adjust_players", adjust(reason="   "))),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_an_adjustment_without_a_category_is_refused():
    payload = adjust()
    del payload["adjustments"][0]["category"]
    client, _ = converse(
        [reply(use("adjust_players", payload)), reply(use("finalize_decision", finalize()))]
    )
    refusal = only_result(client.requests[1])
    assert refusal["is_error"] is True and "category" in refusal["content"]


def test_a_category_or_tier_outside_the_vocabulary_is_refused():
    for bad in (adjust(category="knackered"), adjust(tier=4), adjust(tier=True)):
        client, _ = converse(
            [reply(use("adjust_players", bad)), reply(use("finalize_decision", finalize()))]
        )
        assert only_result(client.requests[1])["is_error"] is True


def test_a_quote_dated_after_today_or_not_a_date_is_refused():
    for bad in ("2099-01-01", "last Tuesday", 20261006):
        client, _ = converse(
            [
                reply(use("adjust_players", adjust(quote_date=bad))),
                reply(use("finalize_decision", finalize())),
            ]
        )
        assert only_result(client.requests[1])["is_error"] is True


def test_a_return_gameweek_before_this_one_is_refused():
    client, _ = converse(
        [
            reply(use("adjust_players", adjust(return_gw=EVENT_ID - 1))),
            reply(use("finalize_decision", finalize())),
        ]
    )
    assert only_result(client.requests[1])["is_error"] is True


def test_the_decision_keeps_every_call_in_order_and_the_record_shape_unchanged():
    resolver = FakeResolver()
    _, decision = converse(
        [
            reply(use("adjust_players", adjust(7, 60.0, "a doubt (paper)", category="doubt"))),
            reply(use("resolve", {})),
            reply(
                use(
                    "adjust_players",
                    adjust(7, 0.0, "out (club)", category="injured", return_gw=EVENT_ID + 2),
                )
            ),
            reply(use("finalize_decision", finalize())),
        ],
        resolver=resolver,
    )
    assert [r["expected_minutes"] for r in decision.record] == [60.0, 0.0]
    assert decision.record[1]["category"] == "injured"
    assert decision.record[1]["return_gw"] == EVENT_ID + 2
    # The decision's own halves keep the three keys the record and the report
    # have always read.
    assert decision.adjustments == [
        {"player_id": 7, "expected_minutes": 60.0, "reason": "a doubt (paper)"}
    ]
    assert decision.unapplied == [
        {"player_id": 7, "expected_minutes": 0.0, "reason": "out (club)"}
    ]


def test_a_fallback_keeps_no_record():
    _, decision = converse([Refusal()])
    assert decision.source.startswith("solver-fallback") and decision.record == []


def test_the_tool_schema_requires_the_new_fields_strictly():
    from aigaffer.manager.tools import TOOLS

    tool = next(t for t in TOOLS if t.get("name") == "adjust_players")
    item = tool["input_schema"]["properties"]["adjustments"]["items"]
    assert item["required"] == [
        "player_id", "expected_minutes", "reason", "category", "tier",
        "quote_date", "source", "return_gw",
    ]
    assert item["properties"]["tier"]["enum"] == [1, 2, 3]
    assert item["properties"]["source"]["anyOf"] == [{"type": "string"}, {"type": "null"}]


def test_an_id_that_is_not_a_number_is_refused_rather_than_believed():
    client, _ = converse(
        [
            reply(use("finalize_decision", finalize(plan_id="1"))),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True


def test_a_player_nobody_has_heard_of_cannot_be_adjusted():
    client, _ = converse(
        [
            reply(use("adjust_players", adjust(pid=4242))),
            reply(use("finalize_decision", finalize())),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "4242" in refusal["content"]


def test_one_bad_id_in_a_batch_voids_the_whole_batch():
    resolver = FakeResolver()
    _, decision = converse(
        [
            reply(
                use(
                    "adjust_players",
                    {
                        "adjustments": [
                            {
                                "player_id": 7, "expected_minutes": 0, "reason": "out",
                                "category": "injured", "tier": 1,
                                "quote_date": "2026-08-20", "source": None,
                                "return_gw": None,
                            },
                            {
                                "player_id": 4242, "expected_minutes": 0, "reason": "who",
                                "category": "injured", "tier": 1,
                                "quote_date": "2026-08-20", "source": None,
                                "return_gw": None,
                            },
                        ]
                    },
                )
            ),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ],
        resolver,
    )

    assert resolver.calls == [{}]
    assert decision.adjustments == []


def test_expected_minutes_are_clamped_to_a_match():
    resolver = FakeResolver()
    converse(
        [
            reply(use("adjust_players", adjust(7, 120.0))),
            reply(use("adjust_players", adjust(4, -20.0))),
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ],
        resolver,
    )

    assert resolver.calls == [{7: 90.0, 4: 0.0}]


def test_a_tool_nobody_defined_is_an_error_and_not_a_crash():
    client, decision = converse(
        [
            reply(use("sack_the_manager", {})),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert only_result(client.requests[1])["is_error"] is True
    assert decision.source == "manager"


def test_a_resolve_that_fails_is_an_error_the_manager_can_work_around():
    # The solver can fail to reach a legal squad. That loses the re-solve, not
    # the conversation: the plans he was already shown are still finalizable.
    resolver = FakeResolver(error=RuntimeError("no legal squad"))
    client, decision = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=0))),
        ],
        resolver,
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "no legal squad" not in refusal["content"], "raw exception text, at that"
    assert "RuntimeError" in refusal["content"]
    assert decision.plan is SWAP


# --- several tools in one turn ---------------------------------------------


def test_every_tool_call_in_a_turn_is_answered_in_one_user_message():
    resolver = FakeResolver()
    client, _ = converse(
        [
            reply(
                use("adjust_players", adjust(7, 0.0), block_id="a1"),
                use("adjust_players", adjust(4, 30.0), block_id="a2"),
                use("resolve", {}, block_id="r1"),
            ),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ],
        resolver,
    )
    results = sent_results(client.requests[1])

    assert [result["tool_use_id"] for result in results] == ["a1", "a2", "r1"]
    assert all(result["type"] == "tool_result" for result in results)
    assert resolver.calls == [{7: 0.0, 4: 30.0}]


def test_a_failed_call_beside_a_good_one_is_flagged_and_not_dropped():
    client, _ = converse(
        [
            reply(
                use("adjust_players", adjust(4242), block_id="a1"),
                use("adjust_players", adjust(7), block_id="a2"),
            ),
            reply(use("finalize_decision", finalize())),
        ]
    )
    results = sent_results(client.requests[1])

    assert [result.get("is_error", False) for result in results] == [True, False]


def test_the_assistant_turn_goes_back_before_its_results():
    client, _ = converse(
        [
            reply(use("resolve", {})),
            reply(use("finalize_decision", finalize(plan_id=2))),
        ]
    )
    messages = client.requests[1]["messages"]

    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert messages[1]["content"][0].type == "tool_use"


# --- the stop reasons ------------------------------------------------------


def test_a_turn_that_ends_without_a_decision_is_nudged_once_and_then_asked_outright():
    client, decision = converse(
        [
            reply(text("I think plan 1 is right."), stop="end_turn"),
            reply(text("Yes, plan 1."), stop="end_turn"),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert last_text(client.requests[1]) == NUDGE
    # On the second stall FINALIZE_NOW stands in for the nudge, not beside it.
    assert last_text(client.requests[2]) == FINALIZE_NOW
    texts = [
        block["text"]
        for message in client.requests[2]["messages"]
        if message["role"] == "user"
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    assert texts.count(NUDGE) == 1
    assert texts.count(FINALIZE_NOW) == 1
    assert all("tool_choice" not in request for request in client.requests)
    assert_append_only(client)
    assert decision.source == "manager"


def test_a_nudged_conversation_never_ends_on_an_assistant_turn():
    # A trailing assistant message is a prefill, and this model rejects those.
    # A resumed pause_turn is the one exception, and it is not one of these.
    client, _ = converse(
        [
            reply(text(), stop="end_turn"),
            reply(text(), stop="end_turn"),
            reply(use("finalize_decision", finalize())),
        ]
    )

    for request in client.requests:
        assert request["messages"][-1]["role"] == "user"


def test_a_productive_turn_buys_back_the_right_to_pause():
    # The prompt tells him a turn boundary refreshes the search allowance, so
    # a bare turn-end is now sometimes the correct move. Forcing on the second
    # stall of the whole conversation would punish him for following it: the
    # escalation is for consecutive stalling, and a turn that came back with
    # real tool calls wipes the slate.
    client, decision = converse(
        [
            reply(text("Out of searches; continuing next turn."), stop="end_turn"),
            reply(use("adjust_players", adjust())),
            reply(text("Out again; one more round."), stop="end_turn"),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert last_text(client.requests[1]) == NUDGE  # first stall: a word
    assert last_text(client.requests[3]) == NUDGE  # a stall after real work: a word again
    assert all(FINALIZE_NOW not in str(r["messages"]) for r in client.requests)
    assert decision.source == "manager"


def test_a_truncated_turn_is_dropped_and_the_decision_asked_for():
    client, decision = converse(
        [
            reply(use("adjust_players", adjust()), stop="max_tokens"),
            reply(use("finalize_decision", finalize())),
        ]
    )

    # The truncated turn is dropped, so the second request is the first with
    # FINALIZE_NOW appended and nothing else changed.
    first = client.requests[0]["messages"]
    assert client.requests[1]["messages"][: len(first) - 1] == first[:-1]
    # The briefing is the last message here, so the words ride in it as a
    # second block rather than as a user message of their own.
    assert len(client.requests[1]["messages"]) == len(first) == 1
    assert len(client.requests[1]["messages"][0]["content"]) == 2
    assert last_text(client.requests[1]) == FINALIZE_NOW
    assert all("tool_choice" not in request for request in client.requests)
    assert_append_only(client)
    assert decision.source == "manager"


def test_asking_for_the_decision_is_one_turn_of_insistence_and_not_a_mode():
    # A truncated turn costs him that turn, not the rest of his research: the
    # decision is asked for once, and if it comes back illegal the
    # conversation carries on without asking again.
    client, decision = converse(
        [
            reply(use("adjust_players", adjust()), stop="max_tokens"),
            reply(use("finalize_decision", finalize(plan_id=99))),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert last_text(client.requests[1]) == FINALIZE_NOW
    assert last_text(client.requests[2]) != FINALIZE_NOW
    assert str(client.requests[2]["messages"]).count(FINALIZE_NOW) == 1
    assert all("tool_choice" not in request for request in client.requests)
    assert_append_only(client)
    assert decision.source == "manager"


def test_a_paused_turn_is_resumed_rather_than_started_again():
    # The server stopped part-way through a turn it means to finish, so what it
    # has already produced goes into the conversation and the next request
    # continues from there. Sending the same request again would buy the same
    # pause and run its searches twice.
    paused = reply(text("Half a search in."), searched(), stop="pause_turn")
    client, decision = converse(
        [paused, reply(use("finalize_decision", finalize()))]
    )
    messages = client.requests[1]["messages"]

    assert messages[0] == {"role": "user", "content": CACHED_BRIEFING}
    assert messages[-1] == {"role": "assistant", "content": paused.content}
    assert decision.source == "manager"


def test_the_work_a_paused_turn_had_already_done_is_counted():
    client, decision = converse(
        [
            reply(searched(), stop="pause_turn"),
            reply(searched(), use("finalize_decision", finalize())),
        ]
    )

    assert len(client.requests) == 2
    assert decision.searches == 2


def test_a_turn_that_will_not_come_back_is_given_up_on():
    client, decision = converse([reply(text(), stop="pause_turn")] * 6)

    assert len(client.requests) == 4, "one turn and three resumptions"
    assert len(client.requests[3]["messages"]) == 4, "the briefing and three pauses"
    assert decision.source == "solver-fallback: pause_turn"


def test_a_refusal_falls_back_without_reading_the_response():
    _, decision = converse([Refusal()])

    assert decision.source == "solver-fallback: refusal"


# --- the cap ---------------------------------------------------------------


def test_the_twelfth_turn_is_asked_to_finalize():
    client, decision = converse([reply(use("adjust_players", adjust()))] * 12)

    assert len(client.requests) == 12
    assert FINALIZE_NOW not in str(client.requests[10]["messages"])
    assert last_text(client.requests[11]) == FINALIZE_NOW
    # Turn 11's tool results are intact and FINALIZE_NOW is one more block at
    # the end of that same user message, after the tool_result blocks.
    after = client.requests[11]["messages"][-1]
    assert after["role"] == "user"
    assert client.requests[11]["messages"][-2]["role"] == "assistant"
    results = [b for b in after["content"] if b.get("type") == "tool_result"]
    assert results and all(b["type"] == "tool_result" for b in after["content"][:-1])
    assert after["content"][-1]["text"] == FINALIZE_NOW
    assert len(client.requests[11]["messages"]) == len(client.requests[10]["messages"]) + 2
    assert all("tool_choice" not in request for request in client.requests)
    assert_append_only(client)
    assert decision.source.startswith("solver-fallback:")


def test_consecutive_truncations_ask_for_the_decision_only_once():
    # Each truncated turn is dropped, so the conversation is the same one asked
    # again: the words must not pile up on it.
    client, decision = converse(
        [
            reply(use("adjust_players", adjust()), stop="max_tokens"),
            reply(use("adjust_players", adjust()), stop="max_tokens"),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert str(client.requests[2]["messages"]).count(FINALIZE_NOW) == 1
    assert client.requests[2]["messages"][-1]["role"] == "user"
    assert_append_only(client)
    assert decision.source == "manager"


def test_the_last_turn_is_not_asked_twice_after_an_escalated_stall():
    # Stalls on turns 10 and 11: the second appends FINALIZE_NOW, and the last
    # turn must not stack a second copy straight after it.
    script = [reply(use("adjust_players", adjust()))] * 9
    script += [reply(text("Hmm."), stop="end_turn")] * 2
    script.append(reply(use("finalize_decision", finalize())))
    client, decision = converse(script)

    assert str(client.requests[11]["messages"]).count(FINALIZE_NOW) == 1
    assert last_text(client.requests[11]) == FINALIZE_NOW
    assert decision.source == "manager"


def test_a_decision_on_the_last_turn_still_counts():
    script = [reply(use("adjust_players", adjust()))] * 11
    script.append(reply(use("finalize_decision", finalize())))
    _, decision = converse(script)

    assert decision.source == "manager"
    assert decision.plan is ROLL


# --- the clock -------------------------------------------------------------


class Clock:
    """A monotonic clock the test winds by hand.

    Each call reads the next figure and the last one repeats for ever, so a
    test says what the loop should think the time is and stops worrying about
    how many times it asks.
    """

    def __init__(self, *readings: float) -> None:
        self.readings = list(readings)

    def __call__(self) -> float:
        return self.readings.pop(0) if len(self.readings) > 1 else self.readings[0]


def test_a_conversation_that_runs_out_of_time_gives_the_week_back(monkeypatch):
    # Twelve turns of an agent that searches the web between them can outlast
    # the deadline it is being asked about. The budget is checked before every
    # request, so the turn that would have blown it is never paid for.
    spent = agent.TIME_BUDGET_SECONDS + 1
    monkeypatch.setattr(agent, "monotonic", Clock(0.0, 1.0, spent))
    client, decision = converse(
        [
            reply(use("adjust_players", adjust())),
            reply(use("finalize_decision", finalize())),
        ]
    )

    assert decision.source == "solver-fallback: out of time"
    assert len(client.requests) == 1, "the second turn was never asked for"
    assert decision.plan is SOLVE0.choice


def test_a_paused_turn_is_not_resumed_past_the_budget(monkeypatch):
    # A turn can be resumed three times, and each resumption is a fresh request
    # against the same clock. Left unchecked, one paused turn could run on for
    # minutes after the budget it was started inside had gone.
    spent = agent.TIME_BUDGET_SECONDS + 1
    monkeypatch.setattr(agent, "monotonic", Clock(0.0, 1.0, spent))
    client, decision = converse([reply(text(), stop="pause_turn")] * 4)

    assert len(client.requests) == 1, "the resumption was never asked for"
    assert decision.source == "solver-fallback: out of time"


def test_a_conversation_inside_the_budget_is_never_interrupted(monkeypatch):
    monkeypatch.setattr(agent, "monotonic", Clock(0.0, agent.TIME_BUDGET_SECONDS))
    _, decision = converse([reply(use("finalize_decision", finalize()))])

    assert decision.source == "manager"


def test_the_budget_is_the_twelve_minutes_the_workflow_can_spare():
    assert agent.TIME_BUDGET_SECONDS == 720


# --- the fallback ----------------------------------------------------------


def rate_limited() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )


def test_a_rate_limit_hands_the_week_back_to_the_solver():
    _, decision = converse([rate_limited()])

    assert decision.source == "solver-fallback: RateLimitError"
    assert decision.plan is SOLVE0.choice
    assert decision.lineup is SOLVE0.lineup
    assert decision.captain == LINEUP0.captain and decision.vice == LINEUP0.vice
    assert decision.chip == "none"
    assert decision.adjustments == []
    assert decision.rationale


def test_a_broken_connection_falls_back_too():
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    _, decision = converse([anthropic.APIConnectionError(request=request)])

    assert decision.source == "solver-fallback: APIConnectionError"


def test_a_fetch_that_came_back_wrong_cannot_take_the_report_with_it():
    # The loop reads the fetch to build itself — clubs off the bootstrap,
    # positions off the board — before it sends anything. That reading has to be
    # inside the net too, or a malformed fetch raises past every handler and the
    # week's report dies with it.
    client = FakeClient([])
    malformed = replace(INPUTS, bootstrap=None)
    decision = run_manager(
        client, CFG, malformed, SOLVE0, XP, BRIEFING, FakeResolver()
    )

    assert decision.source == "solver-fallback: unexpected AttributeError"
    assert decision.plan is SOLVE0.choice
    assert decision.lineup is SOLVE0.lineup
    assert decision.searches == 0
    assert client.requests == [], "it never got as far as asking"


def test_a_failure_nobody_planned_for_is_caught_and_named_as_one():
    _, decision = converse([ValueError("the loop has a bug")])

    assert decision.source == "solver-fallback: unexpected ValueError"


def test_a_fallback_still_reports_the_searches_it_paid_for():
    _, decision = converse([reply(text(), searched()), rate_limited()])

    assert decision.searches == 1
    assert decision.adjustments == [], "adjustments the fallback did not use"


def test_the_fallback_never_lets_the_exception_text_out():
    _, decision = converse([rate_limited()])

    assert "slow down" not in decision.source
    assert "slow down" not in decision.rationale


def test_the_decision_is_the_shape_the_report_and_the_store_expect():
    _, decision = converse([reply(use("finalize_decision", finalize()))])

    assert isinstance(decision, ManagerDecision)
    assert decision.lineup.captain == decision.captain
    assert decision.lineup.vice == decision.vice
    assert set(decision.lineup.xi) | set(decision.lineup.bench) == set(ROLL.squad)


def test_the_prompt_prices_timing_but_never_the_transfer():
    # The price watch advises WHEN, never WHAT: a rising price can bring a
    # decided move forward a night, and must never talk him into one.
    assert "cheaper tonight than at the deadline" in SYSTEM_PROMPT
    assert "never let a price talk you into" in SYSTEM_PROMPT


def test_the_nudge_offers_the_refreshed_searches_before_the_exit():
    # Live GW3 scout: the gaffer paused to refresh its search allowance and
    # the nudge talked it into finalizing instead — "my allowance ran out
    # and it did not refresh", it wrote, wrongly. The nudge must offer the
    # road it paused for before it demands the destination.
    assert "search allowance, it is fresh now" in NUDGE
    assert "finalize_decision" in NUDGE, "the demand stays; it comes second"


# --- the chip field, and what leaked into it -------------------------------

# What every live finalize since 28 Aug 2026 put in chip_justification when
# no chip was played: the model reached for its own close-parameter token at
# the empty value, the strict grammar substituted a near-miss inside the JSON
# string, and the rest of the call spilled in after it — the whole rationale,
# a second time, behind a garbled tag.
LEAK = '</antmlःparameter>\n<parameter name="rationale">'
BENCH_BOOST_CASE = (
    "Play the bench boost this week: the EV panel has it at +14.1, the four on"
    " the bench all start in a double gameweek, and no later week in the"
    " window comes within five points of that. What is lost is the option"
    " value of holding it past the wildcard, which the panel prices at nothing."
)


def test_the_schema_offers_null_for_the_chip_nobody_played():
    tool = next(t for t in TOOLS if t.get("name") == "finalize_decision")
    field = tool["input_schema"]["properties"]["chip_justification"]

    assert {"type": "null"} in field["anyOf"]
    assert {"type": "string"} in field["anyOf"]
    assert "null" in field["description"]


def test_no_chip_takes_null_for_its_justification():
    client, decision = converse(
        [reply(use("finalize_decision", finalize(chip="none", justification=None)))]
    )

    assert decision.chip == "none"
    assert decision.chip_justification == ""
    assert len(client.requests) == 1


def test_what_leaked_into_an_unplayed_chips_field_is_dropped():
    client, decision = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="none", justification=LEAK + RATIONALE),
                )
            )
        ]
    )

    assert decision.chip_justification == ""
    assert decision.rationale == RATIONALE
    assert len(client.requests) == 1, "no chip, no argument to check, no turn spent"


def test_what_was_dropped_is_said_on_stdout(capsys):
    # Null is on offer now; a value that still arrives for an unplayed chip is
    # worth one line in the log, so a leak that carries on is seen.
    converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="none", justification=LEAK + RATIONALE),
                )
            )
        ]
    )

    assert CHIP_FIELD_DROPPED in capsys.readouterr().out


def test_a_chip_argued_behind_leaked_markup_is_refused():
    # The belts measure the argument's length and look for the chip's name;
    # a rationale that spilled in passes both without arguing for anything.
    client, decision = converse(
        [
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="bench_boost", justification=LEAK + BENCH_BOOST_CASE),
                )
            ),
            reply(
                use(
                    "finalize_decision",
                    finalize(chip="bench_boost", justification=BENCH_BOOST_CASE),
                )
            ),
        ]
    )
    refusal = only_result(client.requests[1])

    assert refusal["is_error"] is True
    assert "plain prose" in refusal["content"]
    assert decision.chip == "bench_boost"
    assert decision.chip_justification == BENCH_BOOST_CASE


def test_the_system_prompt_carries_the_chip_playbook():
    from aigaffer.manager.playbook import CHIP_PLAYBOOK

    assert CHIP_PLAYBOOK in SYSTEM_PROMPT
    assert "played once a season" not in SYSTEM_PROMPT
    assert "expires after GW19" in SYSTEM_PROMPT
