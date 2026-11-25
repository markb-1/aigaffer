"""The earlier call is kept unless something is new, through the loop itself.

The universe is the loop's (tests/test_manager_agent.py). The earlier run made
``SWAP`` — Gale (7) out, Quinn (17) in, Hume captain, Moss vice — and the plan
that would undo it is the board's ``ROLL`` (id 1), 3.0 net behind on that
universe's own numbers; :data:`CLOSE` narrows that to 0.6 behind.
"""

from dataclasses import replace
from datetime import UTC, date, datetime

import anthropic
import httpx
import pytest

from aigaffer.continuity import Continuity, Earlier
from aigaffer.manager import agent
from aigaffer.manager.agent import INSIST_FLOOR_SECONDS, MAX_TURNS, TIME_BUDGET_SECONDS
from aigaffer.news_ledger import Entry
from aigaffer.solver.multiweek import PlannedPath
from tests import test_manager_agent as loop
from tests.test_manager_agent import (
    FRESH, GOOD_CHIP, INPUTS, ROLL, SOLVE0, SQUAD, SWAP, FakeResolver, adjust, converse, finalize, only_result, reply,
    sent_results, text, use,
)

TODAY = date(2026, 8, 21)
THEN = datetime(2026, 8, 20, 18, 39, tzinfo=UTC)

# The roll 0.6 behind the swap, as a plan inside the noise is.
CLOSE = replace(ROLL, objective=SWAP.objective - 0.6)
BOARD = replace(SOLVE0, plans=[SWAP, CLOSE])


@pytest.fixture(autouse=True)
def close_board(monkeypatch):
    monkeypatch.setattr(loop, "SOLVE0", BOARD)


def entry(pid: int, **changes) -> Entry:
    return Entry(
        **{
            "player_id": pid, "gw": 2, "category": "nailed", "expected_minutes": 80.0, "tier": 2,
            "quote_date": "2026-08-19", "source": "paper", "note": "", "return_gw": None,
            "checked_at": "2026-08-20T18:39:00+00:00", "run": "early", "fpl": {}, **changes,
        }
    )


def told(**changes) -> Earlier:
    return Earlier(
        **{
            "mode": "early", "ts": THEN, "squad": frozenset(SWAP.squad), "squad_before": frozenset(SQUAD),
            "transfers_in": (17,), "transfers_out": (7,), "captain": 8, "vice": 13, "chip": "none",
            "reason": "he plays", "by_gaffer": True, **changes,
        }
    )


def held_to(earlier: Earlier | None = None, **changes) -> Continuity:
    return Continuity(
        **{
            "earlier": earlier or told(), "since": {}, "used": dict.fromkeys(range(1, 19), 80.0),
            "known": {7: entry(7), 17: entry(17)}, "chip_moved": False, **changes,
        }
    )


def decide(*turns, context=None, **kwargs):
    return converse(
        [turn if isinstance(turn, (loop.Response, Exception)) else reply(*turn) for turn in turns],
        continuity=context or held_to(), today=TODAY, **kwargs,
    )


def done(**kwargs):
    return [use("finalize_decision", finalize(**kwargs))]


def both(news: dict, **kwargs):
    return [use("adjust_players", news, "tu_1"), use("finalize_decision", finalize(**kwargs), "tu_2")]


KEEP = {"plan_id": 0}
ROLLS = {"plan_id": 1, "why": "his minutes are unclear"}


# --- kept, or not asked --------------------------------------------------------


def test_keeping_the_call_asks_for_nothing():
    _, decision = decide(done(**KEEP))
    assert decision.source == "manager" and decision.change_reason is None


def test_a_reason_for_a_call_that_is_kept_is_dropped():
    _, decision = decide(done(**KEEP, why="stray"))
    assert decision.change_reason is None


def test_with_no_earlier_call_nothing_is_asked():
    _, decision = converse([reply(*done(plan_id=1))])
    assert decision.plan is CLOSE and decision.change_reason is None


def test_a_vice_alone_is_not_a_change_to_answer_for():
    _, decision = decide(done(plan_id=0, vice=9))
    assert decision.vice == 9 and decision.change_reason is None


# --- a change has to say why ---------------------------------------------------


@pytest.mark.parametrize("why", [None, "", "  ", "</parameter>x", "..."])
def test_a_change_with_no_reason_is_sent_back(why):
    client, decision = decide(done(plan_id=1, why=why), done(**KEEP))
    refused = only_result(client.requests[1])
    assert refused["is_error"] and "change_reason" in refused["content"] and "plan 0" in refused["content"]
    assert decision.plan is SWAP


def test_a_change_inside_the_noise_with_nothing_new_is_sent_back():
    client, decision = decide(done(**ROLLS), done(**KEEP))
    refused = only_result(client.requests[1])["content"]
    assert "-0.6 net against plan 0" in refused and "change_player_id" in refused
    assert decision.plan is SWAP


def test_a_change_past_the_noise_needs_the_reason_alone(monkeypatch):
    monkeypatch.setattr(loop, "SOLVE0", replace(SOLVE0, plans=[SWAP, replace(ROLL, objective=SWAP.objective + 1.5)]))
    client, decision = decide(done(**ROLLS))
    assert len(client.requests) == 1 and decision.change_reason == "his minutes are unclear"


def test_a_change_just_inside_it_does_not(monkeypatch):
    monkeypatch.setattr(loop, "SOLVE0", replace(SOLVE0, plans=[SWAP, replace(ROLL, objective=SWAP.objective + 1.4)]))
    client, _ = decide(done(**ROLLS), done(**KEEP))
    assert len(client.requests) == 2


def test_the_solvers_own_earlier_week_needs_the_reason_alone():
    client, decision = decide(done(**ROLLS), context=held_to(told(by_gaffer=False)))
    assert len(client.requests) == 1 and decision.plan is CLOSE


def test_a_call_that_is_not_on_the_board_cannot_be_kept():
    client, decision = decide(done(**ROLLS), context=held_to(told(squad=frozenset(range(30, 45)))))
    assert len(client.requests) == 1 and decision.plan is CLOSE


# --- new evidence -----------------------------------------------------------------


def accepted(news=None, about=7, context=None) -> bool:
    turn = done(**ROLLS, about=about) if news is None else both(news, **ROLLS, about=about)
    client, decision = decide(turn, done(**KEEP), context=context)
    assert (len(client.requests) == 1) is (decision.plan is CLOSE)
    return decision.plan is CLOSE


@pytest.mark.parametrize("news", [
    adjust(7, 0.0, category="injured", quote_date="2026-08-21"),
    adjust(7, 0.0, category="injured", quote_date="2026-08-10"),
    adjust(7, 75.0, category="nailed", quote_date="2026-08-19", tier=2),
    adjust(7, 80.0, category="rotation", quote_date="2026-08-19", tier=2),
    adjust(7, 80.0, category="nailed", quote_date="2026-08-19", tier=1),
    adjust(7, 80.0, category="nailed", quote_date="2026-08-21", tier=2),
    adjust(7, 80.0, category="nailed", quote_date="2026-08-19", tier=2, return_gw=5),
])
def test_news_the_earlier_run_was_not_costed_on_is_enough(news):
    assert accepted(news)


def test_an_injury_to_a_man_neither_call_moves_is_enough():
    assert accepted(adjust(3, 0.0, category="injured"), about=3)


def test_a_record_the_earlier_run_never_re_solved_on_is_still_news():
    context = held_to(known={7: entry(7, category="injured", expected_minutes=0.0)})
    carried = adjust(7, 0.0, category="injured", quote_date="2026-08-19", tier=2)
    assert accepted(carried, context=context)


@pytest.mark.parametrize("news, about", [
    (None, None),
    (None, 7),
    (adjust(7, 78.0, category="nailed", quote_date="2026-08-19", tier=2), 7),
    (adjust(7, 80.0, category="nailed", quote_date="2026-08-19", tier=3), 7),
    (adjust(7, 0.0, category="injured"), None),
    (adjust(7, 0.0, category="injured"), 17),
    (adjust(7, 0.0, category="injured"), True),
    (adjust(3, 80.0, category="nailed", quote_date="2026-08-21"), 3),
])
def test_the_same_news_again_is_not_new(news, about):
    assert not accepted(news, about=about)


def test_a_first_word_on_a_man_in_the_change_counts_if_it_was_said_since():
    context = held_to(known={})
    old = adjust(17, 80.0, category="nailed", quote_date="2026-08-18")
    new = adjust(17, 80.0, category="nailed", quote_date="2026-08-20")
    assert not accepted(old, about=17, context=context)
    assert accepted(new, about=17, context=context)


@pytest.mark.parametrize("facts, about, enough", [
    ({7: [("news", "out")]}, 7, True),
    ({3: [("news", "out")]}, 3, True),
    ({3: [("fixtures", "moved")]}, 3, True),
    # Rolling keeps Gale (7) and gives up Quinn (17): only prices that argue
    # for that are evidence for it.
    ({7: [("dearer", "up")]}, 7, True),
    ({17: [("cheaper", "down")]}, 17, True),
    ({17: [("dearer", "up")]}, 17, False),
    ({7: [("cheaper", "down")]}, 7, False),
    ({3: [("dearer", "up")]}, 3, False),
    ({7: [("news", "out")]}, 17, False),
])
def test_what_the_api_changed_is_seen_without_being_told(facts, about, enough):
    assert accepted(about=about, context=held_to(since=facts)) is enough


def test_not_knowing_what_changed_is_never_a_reason_to_refuse(capsys):
    assert accepted(about=7, context=held_to(since=None))
    assert not accepted(about=None, context=held_to(since=None))
    assert "could not be checked" not in capsys.readouterr().out


def test_minutes_nobody_remembers_count_as_news():
    carried = adjust(7, 80.0, category="nailed", quote_date="2026-08-19", tier=2)
    assert accepted(carried, context=held_to(used={}))


# --- gaps ----------------------------------------------------------------------------


def test_the_gap_is_measured_inside_the_chosen_plans_own_solve():
    # 0.5 on FRESH, the earlier call in the re-solve; 2.5 on SWAP, the same
    # squad in the first solve, which is another projection's number.
    other = replace(loop.FRESH_HIT, objective=FRESH.objective + 0.5)
    resolver = FakeResolver(solved=replace(loop.SOLVE1, plans=[FRESH, other]))
    client, decision = decide(
        [use("resolve", {})], done(plan_id=3, why="two are out"), done(plan_id=2), resolver=resolver
    )
    assert "+0.5 net against plan 2" in only_result(client.requests[2])["content"]
    assert decision.plan is FRESH


def test_a_solve_without_the_earlier_call_has_no_gap_to_measure():
    # A point behind SWAP on the first solve's numbers, which is no gap at all.
    other = replace(loop.FRESH_HIT, objective=SWAP.objective - 1.0)
    resolver = FakeResolver(solved=replace(loop.SOLVE1, plans=[other], choice=other))
    client, decision = decide([use("resolve", {})], done(plan_id=2, why="two are out"), resolver=resolver)
    assert len(client.requests) == 2 and decision.plan is other


def path(chip: str) -> PlannedPath:
    return PlannedPath(moves=[], objective=0.0, weekly_xp={}, week1_chip=chip)


def test_a_plan_that_plays_a_chip_is_not_measured_against_one_that_does_not(monkeypatch):
    boosted = replace(CLOSE, path=path("bench_boost"))
    monkeypatch.setattr(loop, "SOLVE0", replace(SOLVE0, plans=[SWAP, boosted]))
    client, _ = decide(done(**ROLLS))
    assert len(client.requests) == 1


# --- chips -------------------------------------------------------------------------------


BOOSTED = told(chip="bench_boost")


def test_a_chip_newly_played_is_argued_for_in_its_own_field():
    client, decision = decide(done(plan_id=0, chip="triple_captain", justification=GOOD_CHIP, why="the fixture is soft"))
    assert len(client.requests) == 1 and decision.change_reason == "the fixture is soft"


def test_a_chip_the_gaffer_added_is_still_found_on_the_board():
    # The plan carries no chip of its own; he played one beside it. Rolling
    # instead drops the chip and the move, and both have to be answered for.
    client, _ = decide(done(**ROLLS), done(**ROLLS), done(**ROLLS), context=held_to(BOOSTED))
    said = only_result(client.requests[1])["content"]
    assert "net against plan 0" in said and "played bench_boost" in said


def test_dropping_a_chip_needs_something_new():
    client, decision = decide(
        done(plan_id=0, why="not now"), both(adjust(3, 0.0, category="injured"), plan_id=0, why="he is out", about=3),
        context=held_to(BOOSTED),
    )
    assert len(client.requests) == 2 and decision.change_reason == "he is out"


@pytest.mark.parametrize("changes", [{"chip_moved": True}, {"chip_kept": False}])
def test_a_chip_that_moved_or_cannot_be_played_may_be_dropped(changes):
    client, _ = decide(done(plan_id=0, why="not now"), context=held_to(BOOSTED, **changes))
    assert len(client.requests) == 1


# --- the armband ---------------------------------------------------------------------------


def test_an_armband_inside_the_noise_needs_something_new():
    # Moss is 1.0 behind Hume: a worse captain on the numbers, and no news.
    client, decision = decide(done(plan_id=0, captain=13, vice=8, why="a hunch"), done(**KEEP))
    assert "-1.0 xP against 8" in only_result(client.requests[1])["content"]
    assert decision.captain == 8


def test_an_armband_past_the_noise_needs_the_reason_alone():
    client, decision = decide(
        done(plan_id=0, captain=8, vice=9, why="better fixture"), context=held_to(told(captain=13, vice=8))
    )
    assert len(client.requests) == 1 and decision.captain == 8


def test_a_captain_who_is_not_in_the_eleven_cannot_be_kept():
    # Reid (18) is projected above Innes (9), and is in nobody's eleven here.
    client, _ = decide(done(plan_id=0, captain=9, why="he is gone"), context=held_to(told(captain=18)))
    assert len(client.requests) == 1


def test_an_armband_that_moves_with_the_squad_is_the_squads_to_answer_for(monkeypatch):
    monkeypatch.setattr(loop, "SOLVE0", replace(SOLVE0, plans=[SWAP, replace(ROLL, objective=SWAP.objective + 2.0)]))
    client, decision = decide(done(plan_id=1, captain=13, vice=8, why="a new plan"))
    assert len(client.requests) == 1 and decision.captain == 13


def test_news_about_the_old_captain_moves_the_armband():
    client, decision = decide(
        both(adjust(8, 0.0, category="injured"), plan_id=0, captain=13, vice=9, why="Hume is out", about=8)
    )
    assert len(client.requests) == 1 and decision.captain == 13


# --- what the owner entered ------------------------------------------------------------------


class Entered:
    def __init__(self, chip):
        self.chip = chip


@pytest.mark.parametrize("chip, earlier, plan", [
    ("free_hit", told(squad=frozenset(range(30, 45)), chip="free_hit"), 1),
    ("wildcard", told(chip="wildcard"), 0),
])
def test_a_chip_he_entered_is_the_weeks_chip(chip, earlier, plan):
    inputs = replace(INPUTS, executed=Entered(chip))
    client, decision = decide(done(plan_id=plan), context=held_to(earlier), inputs=inputs)
    assert len(client.requests) == 1 and decision.source == "manager"


# --- never the week's price --------------------------------------------------------------------


def test_he_is_sent_back_twice_and_no_more(capsys):
    client, decision = decide(done(**ROLLS), done(**ROLLS), done(**ROLLS))
    assert len(client.requests) == 3
    assert decision.plan is CLOSE and decision.source == "manager"
    assert decision.change_reason == "his minutes are unclear"
    assert "accepted (2 refusals)" in capsys.readouterr().out


def test_he_is_never_sent_back_on_the_last_turn():
    talk = [reply(text(), stop="end_turn") for _ in range(MAX_TURNS - 1)]
    client, decision = decide(*talk, done(**ROLLS))
    assert len(client.requests) == MAX_TURNS and decision.plan is CLOSE
    # Accepted as it was given, not held and handed back: the reason is his.
    assert decision.change_reason == "his minutes are unclear"


@pytest.mark.parametrize("left, sent_back", [(INSIST_FLOOR_SECONDS - 1, False), (INSIST_FLOOR_SECONDS + 1, True)])
def test_he_is_never_sent_back_short_of_time(monkeypatch, left, sent_back):
    clock = iter([0.0, 0.0] + [float(TIME_BUDGET_SECONDS - left)] * 20)
    monkeypatch.setattr(agent, "monotonic", lambda: next(clock))
    client, _ = decide(done(**ROLLS), done(**KEEP))
    assert (len(client.requests) == 2) is sent_back


def timeout() -> Exception:
    return anthropic.APITimeoutError(request=httpx.Request("POST", "https://x"))


def limited() -> Exception:
    request = httpx.Request("POST", "https://x")
    return anthropic.RateLimitError("no", response=httpx.Response(429, request=request), body=None)


@pytest.mark.parametrize("after", [
    [timeout()],
    [limited()],
    [ValueError("a bug")],
    [reply(stop="refusal")],
    [reply(text(), stop="pause_turn") for _ in range(4)],
    [reply(text(), stop="end_turn") for _ in range(MAX_TURNS - 1)],
])
def test_a_decision_sent_back_survives_whatever_stops_the_conversation(after, capsys):
    _, decision = decide([loop.searched(), *done(**ROLLS)], *after)
    assert decision.source == "manager" and decision.plan is CLOSE
    assert decision.change_reason is None and decision.searches == 1
    assert "sent back for its reason stands" in capsys.readouterr().out


def test_it_survives_the_clock(monkeypatch):
    clock = iter([0.0] * 3 + [float(TIME_BUDGET_SECONDS + 1)] * 20)
    monkeypatch.setattr(agent, "monotonic", lambda: next(clock))
    _, decision = decide(done(**ROLLS))
    assert decision.source == "manager" and decision.plan is CLOSE


def test_the_latest_one_sent_back_is_the_one_that_stands():
    _, decision = decide(done(**ROLLS), done(plan_id=0, captain=13, vice=8, why="a hunch"), timeout())
    assert decision.plan is SWAP and decision.captain == 13


def test_the_news_recorded_after_it_is_kept_with_it():
    news = adjust(3, 0.0, category="injured")
    _, decision = decide(done(**ROLLS), [use("adjust_players", news)], timeout())
    assert decision.plan is CLOSE
    assert [record["player_id"] for record in decision.record] == [3]


def test_an_illegal_decision_is_never_held():
    _, decision = decide(done(plan_id=1, captain=99), timeout())
    assert decision.source.startswith("solver-fallback")


def test_a_check_that_breaks_accepts_the_decision(monkeypatch, capsys):
    def broken(*args, **kwargs):
        raise KeyError("secret")

    monkeypatch.setattr(agent, "check_call", broken)
    client, decision = decide(done(**ROLLS))
    assert len(client.requests) == 1 and decision.plan is CLOSE
    out = capsys.readouterr().out
    assert "KeyError" in out and "secret" not in out


def test_the_adjustment_beside_a_refused_decision_is_still_answered():
    client, _ = decide(both(adjust(7, 78.0, category="nailed", quote_date="2026-08-19", tier=2), **ROLLS), done(**KEEP))
    results = sent_results(client.requests[1])
    assert [r.get("is_error", False) for r in results] == [False, True]


def test_a_spent_search_allowance_is_said_beside_the_refusal_in_one_result():
    refused_and_capped = loop.reply(
        *[loop.found() for _ in range(8)], loop.missed("max_uses_exceeded"), *done(**ROLLS)
    )
    client, decision = decide(refused_and_capped, done(**KEEP))
    [result] = sent_results(client.requests[1])
    assert result["is_error"] and "nothing new behind it" in result["content"]
    assert agent.SPENT_NOTE in result["content"]
    assert decision.plan is SWAP


def test_a_capped_turns_decision_survives_the_failure_after_it():
    capped = loop.reply(loop.missed("max_uses_exceeded"), *done(**ROLLS))
    _, decision = decide(capped, timeout())
    assert decision.source == "manager" and decision.plan is CLOSE


# --- a refusal among the loop's own words -----------------------------------------


def strictly(*turns):
    """The loop against the client that refuses what the API refuses."""
    client = loop.StrictClient([t if isinstance(t, loop.Response) else reply(*t) for t in turns])
    decision = agent.run_manager(
        client, loop.CFG, INPUTS, BOARD, loop.XP, loop.BRIEFING, FakeResolver(),
        today=TODAY, continuity=held_to(),
    )
    loop.assert_roles_alternate(client)
    return client, decision


def cut_off() -> loop.Response:
    return reply(text("Half a"), stop="max_tokens")


@pytest.mark.parametrize("before, between", [([loop.worked()] * 10, []), ([], [cut_off()])])
def test_a_refusal_carries_the_ask_for_a_decision_while_a_search_waits(before, between):
    # The ask cannot stand as a block of its own then: it is a 400.
    client, decision = strictly(*before, [loop.waiting(), *done(**ROLLS)], *between, done(**KEEP))
    [result] = sent_results(client.requests[-1])
    assert result["type"] == "tool_result" and result["is_error"] is True
    assert "nothing new behind it" in result["content"]
    assert result["content"].endswith("\n\n" + agent.FINALIZE_NOW)
    assert decision.source == "manager" and decision.plan is SWAP


def test_a_refusal_carries_the_search_note_and_then_the_ask():
    client, decision = strictly(
        [*[loop.found()] * 8, loop.waiting(), *done(**ROLLS)], cut_off(), done(**KEEP)
    )
    [result] = sent_results(client.requests[-1])
    said = result["content"]
    assert result["is_error"] is True
    assert said.index("nothing new behind it") < said.index(agent.SPENT_NOTE) < said.index(agent.FINALIZE_NOW)
    assert decision.source == "manager" and decision.plan is SWAP


def test_a_refused_decision_survives_the_ask_that_follows_it():
    client, decision = strictly(
        *[loop.worked()] * 10, [loop.waiting(), *done(**ROLLS)], [text("Thinking.")]
    )
    assert len(client.requests) == MAX_TURNS
    assert decision.source == "manager" and decision.plan is CLOSE and decision.change_reason is None
