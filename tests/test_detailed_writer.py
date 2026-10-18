"""The detailed report's writer: the gaffer's reasons, against a scripted client.

The snapshot is hand-built and small, in the shape
:func:`aigaffer.detailed.snapshot.build_snapshot` writes. The window is
gameweeks 8 and 9, and the players are:

==  =========  ===  ====  =================  ===========================  ==========
id  name       pos  club  full name          fixtures                     xP (8, 9)
==  =========  ===  ====  =================  ===========================  ==========
1   Saka       MID  ARS   Arsenal            GW8 FUL(a)  GW9 LEE(h)       4.12, 7.30
2   Salah      MID  LIV   Liverpool          GW8 WOL(a)  GW9 blank        5.00, 0.00
3   Haaland    FWD  MCI   Man City           GW8 LEE(h)  GW9 LEE(h)       6.00, 3.00
4   Justin     DEF  FUL   Fulham             GW8 ARS(h)  GW9 blank        3.00, 0.00
5   Xhaka      MID  SUN   Sunderland         GW8 WOL(a)  GW9 LEE(h)       2.00, 4.00
6   Raya       GKP  ARS   Arsenal            GW8 FUL(a)  GW9 LEE(h)       3.00, 3.00
7   Cucurella  DEF  CHE   Chelsea            GW8 LEE(a)  GW9 WOL(h)+BOU(a) 2.00, 4.00
8   Isak       FWD  LIV   Liverpool          GW8 WOL(a)  GW9 blank        7.00, 0.00
9   Gordon     MID  NEW   Newcastle          GW8 MCI(h)  GW9 ARS(a)       3.50, 3.50
==  =========  ===  ====  =================  ===========================  ==========

The chosen plan sells Justin for Cucurella, so the one transfer reason may
name any opponent of either man in the window: ARS (Justin's GW8), LEE (the
GW8 of Cucurella), WOL and BOU (his GW9 double). ARS is the only one of those
with a full name on the board, from Saka's and Raya's rows ("Arsenal"); LEE,
WOL and BOU have no player of their own here, so only their short names count.
Liverpool is nobody's opponent in this move, which is what makes it the wrong
answer below.
"""

import pytest

from aigaffer.config import Config
from aigaffer.manager.detailed_writer import (
    ANSWER,
    ReasonsUnavailable,
    SYSTEM_PROMPT,
    reasons_briefing,
    validate_reasons,
    write_reasons,
)
from aigaffer.manager.tools import ToolError
from tests.test_manager_agent import FakeClient, reply, text, use

CFG = Config(team_id=42, anthropic_api_key="sk-test", manager_model="claude-opus-5-5")


def row(name, pos, club, full, pr, f8, f9, x8, x9, *, xm=90.0, nw=""):
    return {
        "n": name, "p": pos, "c": club, "cf": full, "pr": pr, "sp": None,
        "st": "a", "nw": nw, "xm": xm,
        "f": {8: f8, 9: f9}, "x": {8: x8, 9: x9}, "t": round(x8 + x9, 2),
    }


PLAYERS = {
    1: row("Saka", "MID", "ARS", "Arsenal", 101, [["FUL", "a"]], [["LEE", "h"]], 4.12, 7.30),
    2: row("Salah", "MID", "LIV", "Liverpool", 129, [["WOL", "a"]], [], 5.0, 0.0),
    3: row("Haaland", "FWD", "MCI", "Man City", 140, [["LEE", "h"]], [["LEE", "h"]], 6.0, 3.0),
    4: row("Justin", "DEF", "FUL", "Fulham", 46, [["ARS", "h"]], [], 3.0, 0.0),
    5: row("Xhaka", "MID", "SUN", "Sunderland", 50, [["WOL", "a"]], [["LEE", "h"]], 2.0, 4.0),
    6: row("Raya", "GKP", "ARS", "Arsenal", 56, [["FUL", "a"]], [["LEE", "h"]], 3.0, 3.0),
    7: row("Cucurella", "DEF", "CHE", "Chelsea", 61,
           [["LEE", "a"]], [["WOL", "h"], ["BOU", "a"]], 2.0, 4.0),
    8: row("Isak", "FWD", "LIV", "Liverpool", 160, [["WOL", "a"]], [], 7.0, 0.0),
    9: row("Gordon", "MID", "NEW", "Newcastle", 75, [["MCI", "h"]], [["ARS", "a"]], 3.5, 3.5,
           xm=45.0),
}


def plan(ins, outs, *, hits=0, xp=20.0, net=19.0, resolved=False):
    return {
        "in": ins, "out": outs, "hits": hits, "xp": xp, "net": net, "path": [],
        "ft_kept": 1, "ft_credit": 0.0, "transfer_cost": 0.0,
        "bench_charge": 0.0, "resolved": resolved,
    }


def snap(**over):
    base = {
        "gw": 8, "mode": "deadline", "ts": "2026-10-17T10:00Z", "events": [8, 9],
        "bank": 5, "ft": 1,
        "players": PLAYERS,
        "squad": [1, 2, 3, 4, 5, 6],
        "xi": [1, 2, 3, 7, 6],
        "bench": [5],
        "captain": 1, "vice": 2,
        "plans": {0: plan([7], [4]), 1: plan([], [], xp=18.0, net=18.0)},
        "chosen": 0,
        "decision": {
            "rationale": "Cucurella's double beats Justin's blank.",
            "adjustments": [{"player_id": 2, "expected_minutes": 70.0, "reason": "a knock (club)"}],
            "chip": "none", "chip_justification": "",
            "source": "manager",
        },
        "chip_evs": {"bench_boost": 6.1},
        "calendar": ["Bench boost: GW9, the double"],
        "freehit": None,
        "skipped": [
            # Isak for Haaland: 7.00 - 9.00 = -2.00, and 160 - 140 = +2.0m.
            {"id": 8, "pos": "FWD", "replaces": 3, "gap": -2.0, "price_diff": 20,
             "tag": "unaffordable"},
            # Gordon for Xhaka: 7.00 - 6.00 = +1.00; 45 minutes.
            {"id": 9, "pos": "MID", "replaces": 5, "gap": 1.0, "price_diff": 25,
             "tag": "minutes risk"},
        ],
    }
    base.update(over)
    return base


TRANSFER = (
    "Cucurella has WOL (h) and BOU (a) in GW9 for 4.00 points, where Justin"
    " blanks after Arsenal (h)."
)
GOOD = {
    "transfers": [{"out": 4, "in": 7, "reason": TRANSFER}],
    "captain": "Saka, on total xP over the window: 11.42 against Haaland's 9.00, with LEE (h) in GW9.",
    "chips": "No chip: the bench boost adds 6.10 now, and GW9's double is worth more.",
    "plan": "The one move beats rolling by 1.00 net: 19.00 against 18.00.",
    "skipped": [
        {"id": 8, "reason": "Isak blanks in GW9 and costs 2.0m more than Haaland."},
        {"id": 9, "reason": "Gordon is down for 45 minutes."},
    ],
}


def answer(**over):
    return {**GOOD, **over}


def transfer(reason, out=4, came=7):
    return {"transfers": [{"out": out, "in": came, "reason": reason}]}


# --- the opponent check -------------------------------------------------------


def test_a_transfer_reason_must_name_an_opponent():
    # No opponent at all: a reason about the men and not the games.
    with pytest.raises(ToolError, match="opponent"):
        validate_reasons(answer(**transfer("Cucurella is simply the better pick by 3.00 points.")), snap())
    # Liverpool is a club on the board, but neither man plays them this window.
    with pytest.raises(ToolError, match="opponent"):
        validate_reasons(answer(**transfer("Cucurella beats Justin by 3.00 before Liverpool.")), snap())
    with pytest.raises(ToolError, match="opponent"):
        validate_reasons(answer(**transfer("Cucurella beats Justin by 3.00 before LIV.")), snap())

    # Justin's GW8 opponent, by full name with the venue, and by short name.
    full = validate_reasons(answer(**transfer("Justin has only Arsenal (h) and then a blank: 3.00.")), snap())
    assert full["transfers"][0]["reason"] == "Justin has only Arsenal (h) and then a blank: 3.00."
    validate_reasons(answer(**transfer("Justin has only ARS and then a blank: 3.00.")), snap())
    # The man in's opponents count as well — BOU is one half of his GW9 double.
    validate_reasons(answer(**transfer("Cucurella's double includes BOU: 4.00 in GW9.")), snap())
    # The report's own away spelling, lower case with the venue in brackets.
    validate_reasons(answer(**transfer("Cucurella goes wol(h)+bou(a) for 4.00 in GW9.")), snap())


def test_a_short_name_is_a_whole_word_in_capitals():
    # "Bout" holds BOU's letters and "arsenal's"-style prose is fine, but a
    # short name has to stand as itself: lower case without a venue is just
    # an English word ("new", "sun"), and part of a word is no club at all.
    with pytest.raises(ToolError, match="opponent"):
        validate_reasons(answer(**transfer("Cucurella wins this bout by 3.00.")), snap())
    with pytest.raises(ToolError, match="opponent"):
        validate_reasons(answer(**transfer("Cucurella beats Justin by 3.00 against wol.")), snap())


def test_a_transfer_reason_quotes_a_number():
    with pytest.raises(ToolError, match="number"):
        validate_reasons(answer(**transfer("Cucurella has the double against WOL and BOU.")), snap())


def test_a_move_neither_man_has_a_game_in_needs_no_opponent():
    # A window of blanks for both men (it cannot happen in a real season, but
    # the check must not demand a name that does not exist): out Justin, in a
    # man with no fixtures either. With no opponent to name, none is required.
    blank = {**PLAYERS[4], "f": {8: [], 9: []}}
    nobody = {**PLAYERS[7], "f": {8: [], 9: []}}
    board = snap(players={**PLAYERS, 4: blank, 7: nobody})
    validate_reasons(answer(**transfer("Neither man plays; Cucurella is 3.00 cheaper to hold.")), board)


# --- word limits, plain words, and the gaffer's own voice -----------------------


def test_word_limits_are_enforced():
    # 60 words per transfer reason, 40 per skipped line, 80 for each of the
    # captain, chips and plan. Each reason below is padded to the limit
    # exactly, then one past it.
    def padded(head, words):
        return head + " x" * (words - len(head.split()))

    head = "Cucurella has WOL for 4.00."
    assert len(head.split()) == 5
    validate_reasons(answer(**transfer(padded(head, 60))), snap())
    with pytest.raises(ToolError, match="61 words"):
        validate_reasons(answer(**transfer(padded(head, 61))), snap())

    # A skipped line past its limit is not an error but a line dropped (see
    # test_a_bad_skipped_line_costs_only_itself): at 40 Isak's is kept, at 41
    # it is gone and Gordon's stands alone.
    skip = [{"id": 8, "reason": padded("Isak", 40)}, GOOD["skipped"][1]]
    assert [i["id"] for i in validate_reasons(answer(skipped=skip), snap())["skipped"]] == [8, 9]
    skip[0] = {"id": 8, "reason": padded("Isak", 41)}
    assert [i["id"] for i in validate_reasons(answer(skipped=skip), snap())["skipped"]] == [9]

    for key in ("captain", "chips", "plan"):
        validate_reasons(answer(**{key: padded("Saka", 80)}), snap())
        with pytest.raises(ToolError, match=f"{key}.*81 words"):
            validate_reasons(answer(**{key: padded("Saka", 81)}), snap())


def test_the_projection_never_says_anything():
    # The numbers are the gaffer's: he does not hide behind a model.
    for phrase in ("The projection says Saka.", "The model says Saka.",
                   "Saka, because the solver prefers him.", "According to the model, Saka."):
        with pytest.raises(ToolError, match="your own"):
            validate_reasons(answer(captain=phrase), snap())
    with pytest.raises(ToolError, match="your own"):
        validate_reasons(answer(**transfer("The projections say WOL is worth 4.00.")), snap())


def test_reasons_are_plain_sentences_flattened_to_one_line():
    cleaned = validate_reasons(answer(plan="  The one move\nbeats rolling by 1.00.  "), snap())
    assert cleaned["plan"] == "The one move beats rolling by 1.00."
    for bad in ("## A heading", "", "   ", 7):
        with pytest.raises(ToolError, match="plan"):
            validate_reasons(answer(plan=bad), snap())
    with pytest.raises(ToolError):
        validate_reasons(answer(captain="Saka <parameter name='x'>"), snap())


# --- leniency: a skipped line, and the chips on a week with none ------------------


def test_a_bad_skipped_line_costs_only_itself():
    # Isak's line runs to 41 words, or carries markup, beside a good line for
    # Gordon. The answer is taken first time — no retry spent, the transfer,
    # captain and plan reasons kept — with Gordon's line alone.
    long = {"id": 8, "reason": "Isak" + " x" * 40}
    marked = {"id": 8, "reason": "## Isak"}
    for bad in (long, marked):
        cleaned = validate_reasons(answer(skipped=[bad, GOOD["skipped"][1]]), snap())
        assert cleaned["skipped"] == [GOOD["skipped"][1]]
        assert cleaned["transfers"][0]["reason"] == TRANSFER
    # A bad line does not use up the id: a good one after it for the same man
    # is taken.
    again = validate_reasons(answer(skipped=[long, GOOD["skipped"][0]]), snap())
    assert again["skipped"] == [GOOD["skipped"][0]]
    # A voice phrase and an empty line are dropped the same way.
    for reason in ("The model says Isak is too dear.", "   "):
        assert validate_reasons(
            answer(skipped=[{"id": 8, "reason": reason}, GOOD["skipped"][1]]), snap()
        )["skipped"] == [GOOD["skipped"][1]]

    client = FakeClient([reply(use(ANSWER, answer(skipped=[long, GOOD["skipped"][1]])))])
    reasons = write_reasons(CFG, snap(), client=client)
    assert len(client.requests) == 1
    assert reasons["skipped"] == [GOOD["skipped"][1]] and reasons["plan"] == GOOD["plan"]


def test_the_chips_reason_may_be_empty_on_a_week_with_no_chip():
    # The snapshot's chip is "none": an empty or missing chips reason is "".
    assert validate_reasons(answer(chips=""), snap())["chips"] == ""
    assert validate_reasons(answer(chips="   "), snap())["chips"] == ""
    absent = {k: v for k, v in GOOD.items() if k != "chips"}
    assert validate_reasons(absent, snap())["chips"] == ""
    # One given is still checked: plain, in bounds, his own voice.
    with pytest.raises(ToolError, match="chips"):
        validate_reasons(answer(chips="## No chip"), snap())
    # A chip played needs its reason.
    boost = snap(decision={**snap()["decision"], "chip": "bench_boost"})
    with pytest.raises(ToolError, match="chips"):
        validate_reasons(answer(chips=""), boost)
    assert validate_reasons(GOOD, boost)["chips"] == GOOD["chips"]


def test_transfers_captain_and_plan_stay_strict():
    # The leniency is the skipped lines' and the chipless chips' alone.
    for over in ({"captain": ""}, {"plan": "x " * 81}, {"transfers": []}):
        with pytest.raises(ToolError):
            validate_reasons(answer(**over), snap())


# --- ids --------------------------------------------------------------------------


def test_ids_must_be_the_snapshots():
    # The transfer ids are the chosen plan's pair (out 4, in 7), and every
    # pair gets one reason. The skipped ids are the snapshot's rows, each once,
    # but a line that breaks that is dropped rather than refused.
    for moves in (
        [{"out": 4, "in": 5, "reason": TRANSFER}],  # the wrong man in
        [{"out": 7, "in": 4, "reason": TRANSFER}],  # the pair the wrong way round
        [],  # the move left out
        [{"out": 4, "in": 7, "reason": TRANSFER}] * 2,  # one move twice
        [{"out": "4", "in": 7, "reason": TRANSFER}],  # a string, not an id
        [{"out": True, "in": 7, "reason": TRANSFER}],  # a bool is not an id either
    ):
        with pytest.raises(ToolError, match="transfers"):
            validate_reasons(answer(transfers=moves), snap())

    def kept(skipped):
        return validate_reasons(answer(skipped=skipped), snap())["skipped"]

    # Gordon left out: Isak's line alone, and Gordon's note blank in the report.
    assert kept([{"id": 8, "reason": "Isak blanks."}]) == [{"id": 8, "reason": "Isak blanks."}]
    # Saka (1) is owned, not a skipped row; "8" is a string and True a bool:
    # each dropped, the two good lines kept.
    for stray in ({"id": 1, "reason": "Saka is owned."}, {"id": "8", "reason": "x"},
                  {"id": True, "reason": "x"}, "not a line"):
        assert kept([*GOOD["skipped"], stray]) == GOOD["skipped"]
    # Isak twice: the first line stands, the second is dropped.
    twice = [GOOD["skipped"][0], {"id": 8, "reason": "Isak again."}, GOOD["skipped"][1]]
    assert kept(twice) == GOOD["skipped"]
    # Not a list at all is no lines, not a failed answer.
    assert kept("Isak and Gordon") == []

    # A week with no move takes an empty list, and the result keeps the order
    # the snapshot gives the rows in, not the order the answer did.
    roll = snap(plans={0: plan([], [])})
    cleaned = validate_reasons(answer(transfers=[], skipped=GOOD["skipped"][::-1]), roll)
    assert cleaned["transfers"] == []
    assert [item["id"] for item in cleaned["skipped"]] == [8, 9]
    with pytest.raises(ToolError, match="transfers"):
        validate_reasons(answer(), roll)


def test_a_valid_answer_is_the_shape_the_renderer_reads():
    cleaned = validate_reasons(GOOD, snap())
    assert cleaned == {
        "transfers": [{"out": 4, "in": 7, "reason": TRANSFER}],
        "captain": GOOD["captain"],
        "chips": GOOD["chips"],
        "plan": GOOD["plan"],
        "skipped": GOOD["skipped"],
    }


# --- the conversation ---------------------------------------------------------------


def test_a_valid_answer_first_time_is_one_request_and_forces_nothing():
    client = FakeClient([reply(use(ANSWER, GOOD))])
    reasons = write_reasons(CFG, snap(), client=client)

    assert reasons["captain"] == GOOD["captain"] and len(client.requests) == 1
    request = client.requests[0]
    assert "tool_choice" not in request and "thinking" not in request
    assert [tool["name"] for tool in request["tools"]] == [ANSWER]
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 16000
    assert request["model"] == "claude-opus-5-5"
    # The schema asks for integer ids, as the renderer matches them.
    schema = request["tools"][0]["input_schema"]["properties"]
    assert schema["transfers"]["items"]["properties"]["out"]["type"] == "integer"
    assert schema["skipped"]["items"]["properties"]["id"]["type"] == "integer"


def test_one_retry_then_unavailable():
    bad = answer(**transfer("Cucurella is the better pick by 3.00."))
    # Wrong, then right: the second request carries the first's error as its
    # tool_result, in words, and the answer is taken.
    client = FakeClient([reply(use(ANSWER, bad, "tu_1")), reply(use(ANSWER, GOOD, "tu_2"))])
    assert write_reasons(CFG, snap(), client=client)["plan"] == GOOD["plan"]
    result = client.requests[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1"
    assert result["is_error"] is True and "opponent" in result["content"]

    # Wrong twice: no third request, and the reasons are unavailable.
    client = FakeClient([reply(use(ANSWER, bad, "tu_1")), reply(use(ANSWER, bad, "tu_2"))])
    with pytest.raises(ReasonsUnavailable):
        write_reasons(CFG, snap(), client=client)
    assert len(client.requests) == 2


def test_an_api_error_is_unavailable():
    import anthropic
    import httpx

    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(ReasonsUnavailable, match="APIConnectionError"):
        write_reasons(CFG, snap(), client=FakeClient([error]))


def test_a_max_tokens_turn_is_dropped():
    # Cut off mid-sentence: the turn never happened, and the ask joins the
    # still-open briefing rather than make two user turns in a row.
    client = FakeClient([reply(text("Cucurella has…"), stop="max_tokens"), reply(use(ANSWER, GOOD))])
    write_reasons(CFG, snap(), client=client)
    messages = client.requests[1]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert ANSWER in messages[0]["content"][-1]["text"]


def test_talking_instead_of_answering_gets_a_plain_ask():
    client = FakeClient([reply(text("Let me think."), stop="end_turn"), reply(use(ANSWER, GOOD))])
    write_reasons(CFG, snap(), client=client)
    messages = client.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert ANSWER in messages[-1]["content"][-1]["text"]


def test_every_tool_use_gets_a_result():
    # Two calls in one failed turn: each gets its own tool_result, in one user
    # message, or the API answers the next request with a 400.
    bad = answer(captain="")
    client = FakeClient([
        reply(use("other", {}, "tu_a"), use(ANSWER, bad, "tu_b")),
        reply(use(ANSWER, GOOD, "tu_c")),
    ])
    write_reasons(CFG, snap(), client=client)
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user"
    assert [r["tool_use_id"] for r in last["content"]] == ["tu_a", "tu_b"]
    assert all(r["is_error"] is True for r in last["content"])
    assert ANSWER in last["content"][0]["content"]


# --- what he reads ---------------------------------------------------------------------


def test_the_briefing_carries_the_moves_the_men_and_his_own_words():
    briefing = reasons_briefing(snap())

    # The move, by the ids the answer must use, and the gap both ways:
    # Cucurella 6.00 - Justin 3.00 = +3.00 weighted and plain (t is the sum here).
    assert "out 4 Justin → in 7 Cucurella" in briefing
    assert "+3.00 weighted (+3.00 plain)" in briefing
    # Each man's games, with the opponent's full name where the board has one.
    assert "GW8 ARS (Arsenal) h 3.00" in briefing
    assert "GW9 WOL h + BOU a 4.00" in briefing
    assert "GW9 no game 0.00" in briefing
    # The skipped rows by id, with their tags; the plans; the chips.
    assert "id 8 Isak (FWD) instead of Haaland: -2.00 weighted, +2.0m, unaffordable" in briefing
    assert "id 9 Gordon (MID) instead of Xhaka: +1.00 weighted, +2.5m, minutes risk" in briefing
    assert "Plan 0 (chosen): in Cucurella; out Justin; 0 hits; xP 20.00; net 19.00" in briefing
    assert "Plan 1: in none; out none; 0 hits; xP 18.00; net 18.00" in briefing
    assert "Bench Boost: +6.10" in briefing and "Bench boost: GW9, the double" in briefing
    # The armbands and the captain's rivals, by plain total over the window:
    # Saka 11.42 captain; the XI's other attackers are Haaland 9.00 and Salah
    # 5.00 (the vice).
    assert "Captain: Saka (11.42 over the window); vice: Salah" in briefing
    assert "Haaland (9.00)" in briefing
    # His own words and his minutes calls.
    assert "Cucurella's double beats Justin's blank." in briefing
    assert "Salah: 70 expected minutes (a knock (club))" in briefing


def test_a_week_with_no_move_says_so_in_the_briefing():
    briefing = reasons_briefing(snap(plans={0: plan([], [])}))
    assert "No transfers this week: give transfers as an empty list." in briefing


def test_the_prompt_carries_the_rules():
    assert "never write \"the projection says\" or \"the model says\"" in SYSTEM_PROMPT.lower()
    assert "60 words" in SYSTEM_PROMPT and "40 words" in SYSTEM_PROMPT and "80 words" in SYSTEM_PROMPT
    assert "do not change" in SYSTEM_PROMPT.lower()
    assert "what the armband was chosen on" in SYSTEM_PROMPT
    assert ANSWER in SYSTEM_PROMPT
