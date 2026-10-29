"""Tests for the Week plan: the detailed plan as five texts on a phone.

Every snapshot here is hand-built and small, on the board of
tests/test_detailed_render.py: a window of gameweeks 8 and 9 and these men.

==  ========  ===  =====  =============================  =============  ====
id  name      pos  price  fixtures                       xP (GW8, GW9)  mins
==  ========  ===  =====  =============================  =============  ====
1   Saka      MID  10.1   GW8 LEE(h)  GW9 LEE(h)+wol(a)  4.12, 7.30     90
2   Salah     MID  12.9   GW8 wol(a)  GW9 blank          5.00, 0.00     90
3   Haaland   FWD  14.0   GW8 LEE(h)  GW9 LEE(h)         6.00, 3.00     90
4   Justin    DEF   4.6   GW8 LEE(h)  GW9 blank          3.00, 0.00     30, news "Knock"
5   Xhaka     MID   5.0   GW8 wol(a)  GW9 LEE(h)         2.00, 4.00     90
6   Raya      GKP   5.6   GW8 LEE(h)  GW9 LEE(h)         3.00, 3.00     90
7   Cucurella DEF   6.1   GW8 wol(a)  GW9 LEE(h)         2.00, 4.00     90
8   Outsider  FWD   5.0   not in the squad               0.00, 0.00     0, news "Out"
==  ========  ===  =====  =============================  =============  ====

The squad is Saka, Salah, Haaland, Justin and Raya, all five in the XI, with
Xhaka on the bench. The chosen plan swaps Justin for Cucurella. Window totals:
Saka 11.42, Salah 5.00, Haaland 9.00, Justin 3.00, Xhaka 6.00, Raya 6.00,
Cucurella 6.00. The run is a scout run stamped Wed 28 Oct 2026 23:35 UTC for a
deadline of Sat 31 Oct 2026 11:00 UTC, and the clocks went back on 25 Oct, so
Irish time is UTC.
"""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aigaffer import weekplan
from aigaffer.detailed import render as detailed
from aigaffer.detailed.skipped import BENCH
from aigaffer.detailed.snapshot import load_snapshots, write_snapshot
from aigaffer.executed import Executed
from aigaffer.report.telegram import send_message
from aigaffer.weekplan import (
    ENTERED,
    NO_PLAN,
    SOLVER_NOTES,
    UNDERWAY,
    render_week_plan,
    week_plan,
)

NOW = datetime(2026, 10, 29, 20, 0, tzinfo=UTC)  # after the run, before the deadline
REPO = Path(__file__).parent.parent


def row(name, pos, pr, f8, f9, x8, x9, *, st="a", nw="", xm=90.0, club="ARS"):
    return {
        "n": name, "p": pos, "c": club, "cf": club, "pr": pr, "sp": None,
        "st": st, "nw": nw, "xm": xm,
        "f": {8: f8, 9: f9}, "x": {8: x8, 9: x9}, "t": round(x8 + x9, 2),
    }


PLAYERS = {
    1: row("Saka", "MID", 101, [["LEE", "h"]], [["LEE", "h"], ["WOL", "a"]], 4.12, 7.30),
    2: row("Salah", "MID", 129, [["WOL", "a"]], [], 5.0, 0.0),
    3: row("Haaland", "FWD", 140, [["LEE", "h"]], [["LEE", "h"]], 6.0, 3.0),
    4: row("Justin", "DEF", 46, [["LEE", "h"]], [], 3.0, 0.0, st="d", nw="Knock", xm=30.0),
    5: row("Xhaka", "MID", 50, [["WOL", "a"]], [["LEE", "h"]], 2.0, 4.0),
    6: row("Raya", "GKP", 56, [["LEE", "h"]], [["LEE", "h"]], 3.0, 3.0),
    7: row("Cucurella", "DEF", 61, [["WOL", "a"]], [["LEE", "h"]], 2.0, 4.0),
    8: row("Outsider", "FWD", 50, [], [], 0.0, 0.0, st="i", nw="Out", xm=0.0),
}

RATIONALE = (
    "WHAT I DID\nSwapped Justin for Cucurella.\n\n"
    "WHAT I LEARNED\n"
    "- Justin: limped off in training.\n"
    "- What I couldn't check: Leeds, Wolves.\n\n"
    "WHY THIS PLAN\n"
    "- Cucurella's fixtures beat Justin's."
)


def plan(ins, outs, *, hits=0, net=19.0, resolved=False, path=None):
    return {
        "in": ins, "out": outs, "hits": hits, "xp": net + 1, "net": net,
        "path": path or [], "ft_kept": 1, "ft_credit": 0.0, "transfer_cost": 0.0,
        "bench_charge": 0.0, "resolved": resolved,
    }


def snap(mode="scout", ts="Wed 28 Oct 2026 23:35 UTC", **over):
    base = {
        "gw": 8, "mode": mode, "ts": ts, "deadline": "Sat 31 Oct 2026 11:00 UTC",
        "events": [8, 9], "bank": 37, "ft": 1,
        "players": copy.deepcopy(PLAYERS),
        "squad": [1, 2, 3, 4, 6], "xi": [1, 2, 3, 4, 6], "bench": [5],
        "captain": 1, "vice": 2,
        "plans": {0: plan([7], [4]), 1: plan([], [], net=18.0)},
        "chosen": 0,
        "decision": {
            "rationale": RATIONALE, "adjustments": [], "chip": "none",
            "chip_justification": "", "source": "manager",
        },
        "chip_evs": {"bench_boost": 6.1, "wildcard": -3.0},
        "calendar": ["Bench Boost (GW1–19): saved for GW9, worth +6.1 there"],
        "freehit": None,
        "skipped": [
            {"id": 3, "pos": "FWD", "replaces": 4, "gap": 6.0,
             "price_diff": 94, "tag": "unaffordable"},
        ],
    }
    base.update(over)
    return base


def recorded(at, *, gw=8):
    return Executed(
        gw=gw, mode="deadline", recorded_at=at, transfers_in=[7], transfers_out=[4],
        squad_after=[1, 2, 3, 6, 7], chip="none", captain=1, vice=2, buy_prices={},
        sell_prices={}, bank_after=37, ft_after=0, ft_before=1, verdicts=[],
    )


def texts(*snapshots, now=NOW, executed=None):
    return render_week_plan(list(snapshots), now, executed)


# --- The shape: five texts, in order ----------------------------------------


def test_five_texts_each_numbered_and_in_order():
    out = texts(snap())

    assert len(out) == 5
    assert [text.splitlines()[0].split(" · ")[0] for text in out] == [
        "1/5", "2/5", "3/5", "4/5", "5/5",
    ]
    assert "week plan" in out[0] and "Team news" in out[1] and "Why this plan" in out[2]
    assert "The road ahead" in out[3] and "Upcoming gameweeks" in out[4]


def test_no_snapshot_is_refused_not_guessed_at():
    with pytest.raises(ValueError):
        render_week_plan([], NOW, None)


def test_the_header_names_the_run_and_the_deadline_in_irish_time():
    first = texts(snap())[0].splitlines()[0]

    # Clocks went back on 25 Oct, so 23:35 UTC is 23:35 and 11:00 is 11:00.
    assert first == (
        "1/5 · 📋 GW8 week plan · as of Wed scout run 23:35"
        " · deadline Sat 31 Oct 11:00 (Irish time)"
    )


def test_summer_time_moves_the_day_as_well_as_the_hour():
    # Tue 13 Oct 23:35 UTC is 00:35 on Wednesday in Dublin (UTC+1), and a
    # deadline of Sat 17 Oct 10:30 UTC is 11:30 there.
    summer = snap(ts="Tue 13 Oct 2026 23:35 UTC", deadline="Sat 17 Oct 2026 10:30 UTC")

    first = texts(summer, now=datetime(2026, 10, 14, 12, 0, tzinfo=UTC))[0].splitlines()[0]

    assert "as of Wed scout run 00:35" in first
    assert "deadline Sat 17 Oct 11:30 (Irish time)" in first


# --- Text 1: what to do -----------------------------------------------------


def test_do_this_is_printed_verbatim_and_no_fallback_card_appears():
    card = "## ⚡ TRIPLE CAPTAIN THIS WEEK\n\nDo this:\n1. SELL Justin (£4.6m)\n\nCaptain Saka"

    first = texts(snap(do_this=card))[0]

    assert card in first
    assert "Action card (the gameweek report has the full checklist)" not in first


def test_without_do_this_a_plain_card_is_built_from_the_snapshot():
    first = texts(snap())[0]

    # Justin £4.6m out, Cucurella £6.1m in, one free transfer, no hit; the XI
    # is a keeper, one defender, Saka then Salah (11.42 over 5.00) and
    # Haaland: 1-2-1; Xhaka on the bench.
    assert (
        "Action card (the gameweek report has the full checklist)\n"
        "Transfers: out Justin (£4.6m), in Cucurella (£6.1m) — 1 FT, no hit\n"
        "Captain Saka · Vice Salah\n"
        "XI (1-2-1):\n"
        "GKP: Raya\n"
        "DEF: Justin\n"
        "MID: Saka, Salah\n"
        "FWD: Haaland\n"
        "Bench: Xhaka"
    ) in first
    assert "Chip:" not in first, "no chip, no chip line"


def test_a_summary_line_closes_the_text():
    # One transfer on one free transfer: 1 used, 0 left over; bank 37 tenths.
    assert texts(snap())[0].splitlines()[-1] == "1 FT used, 0 carried over · bank £3.7m · no hit"


def test_a_hit_is_named_in_points_and_a_banked_transfer_counted():
    # Two moves on two free transfers and one hit: 2 - 1 = 1 FT used...
    # hits are paid for beyond the free ones, so with ft 2, one hit and two
    # moves the free ones used are 2 - 1 = 1 and 1 carried over.
    two = snap(plans={0: plan([7, 5], [4, 2], hits=1), 1: plan([], [], net=18.0)}, ft=2)

    assert texts(two)[0].splitlines()[-1] == "1 FT used, 1 carried over · bank £3.7m · -4 pts in hits"


def test_a_chip_week_leaves_the_free_transfers_alone():
    wild = snap(decision={**snap()["decision"], "chip": "wildcard"})

    summary = texts(wild)[0].splitlines()[-1]

    assert summary == "Wildcard: free transfers untouched · bank £3.7m · no hit"
    assert "Chip: Wildcard" in texts(wild)[0]


def test_changes_are_listed_when_something_changed():
    earlier = snap(mode="early", captain=3)

    first = texts(earlier, snap())[0]

    assert "Changed since the early run:\n• Captain: Saka (was: Haaland)" in first


def test_changes_are_left_out_when_nothing_changed_and_on_a_first_look():
    assert "Changed since" not in texts(snap(mode="early"), snap())[0]
    assert "First look" not in texts(snap())[0]


def test_more_than_six_changes_are_cut_with_a_count():
    # Seven men's minutes differ from the earlier run (80 then, 90 now) and the
    # gaffer overruled all seven this time: seven lines, six shown, "+1 more".
    earlier = snap(mode="early")
    for player in earlier["players"].values():
        player["xm"] = 80.0
    now = snap()
    for player in now["players"].values():
        player["xm"] = 90.0
    now["decision"]["adjustments"] = [
        {"player_id": pid, "expected_minutes": 90.0, "reason": "fit"} for pid in range(1, 8)
    ]

    first = texts(earlier, now)[0]

    shown = [line for line in first.splitlines() if "xMins 80 → 90" in line]
    assert len(shown) == 6
    assert "+1 more" in first


def test_entered_moves_newer_than_the_run_are_said():
    after_run = recorded("2026-10-29T08:00:00+00:00")

    first = texts(snap(), executed=after_run)[0]

    assert ENTERED in first
    assert first.index(ENTERED) < first.index("Action card"), "said before the card it qualifies"


def test_entered_moves_older_than_the_run_or_for_another_week_are_not_said():
    before_run = recorded("2026-10-28T20:00:00+00:00")
    other_week = recorded("2026-10-29T08:00:00+00:00", gw=9)

    assert ENTERED not in texts(snap(), executed=before_run)[0]
    assert ENTERED not in texts(snap(), executed=other_week)[0]


def test_a_passed_deadline_says_the_gameweek_is_underway():
    after = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)

    assert UNDERWAY.format(gw=8) in texts(snap(), now=after)[0]
    assert UNDERWAY.format(gw=8) not in texts(snap(), now=NOW)[0]


def test_a_snapshot_without_a_deadline_says_unknown_and_never_raises():
    undated = snap()
    del undated["deadline"]

    first = texts(undated, now=datetime(2030, 1, 1, tzinfo=UTC))[0]

    assert "deadline unknown" in first.splitlines()[0]
    assert "has passed" not in first, "no deadline, no claim about it"


def test_a_timestamp_in_another_format_is_not_compared_and_does_not_raise():
    odd = snap(ts="2026-10-17T10:00Z")

    first = texts(odd, executed=recorded("2026-10-29T08:00:00+00:00"))[0]

    assert first.splitlines()[0].startswith("1/5 · 📋 GW8 week plan · as of the scout run ·")
    assert ENTERED not in first


def test_a_long_do_this_is_kept_whole_and_still_sends_in_chunks_under_the_limit():
    long_card = "\n".join(f"{i}. SELL somebody number {i} (£5.0m) and BUY somebody else" for i in range(1, 120))
    assert len(long_card) > 4000

    first = texts(snap(do_this=long_card))[0]
    assert long_card in first

    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    send_message("t", "c", first, http=http)

    assert len(bodies) >= 2
    assert all(len(body["text"]) <= 4096 for body in bodies)
    assert "119. SELL somebody number 119" in bodies[-1]["text"]


# --- Text 2: team news ------------------------------------------------------


def test_what_he_learned_is_verbatim_and_his_other_blocks_stay_out():
    second = texts(snap())[1]

    assert "What the gaffer learned:\n- Justin: limped off in training." in second
    assert "WHAT I DID" not in second and "Swapped Justin" not in second
    assert "Cucurella's fixtures beat" not in second, "that is text 3's"


def test_the_unchecked_clubs_line_becomes_a_caution():
    second = texts(snap())[1]

    assert "⚠️ Not checked: Leeds, Wolves." in second
    assert "- What I couldn't check" not in second


def test_a_man_with_news_or_few_minutes_is_flagged_with_the_gaffers_reason_first():
    news_only = texts(snap())[1]
    overruled = snap()
    overruled["decision"]["adjustments"] = [
        {"player_id": 4, "expected_minutes": 30.0, "reason": "Limped off in training"}
    ]

    # Justin: 30 min < 65 and FPL news "Knock".
    assert "Justin (DEF) 30 min · Knock" in news_only
    assert "Justin (DEF) 30 min · Limped off in training" in texts(overruled)[1]


def test_a_flagged_man_outside_the_squad_and_the_targets_is_not_listed():
    # Outsider has news and 0 minutes but plays no part in the plan.
    assert "Outsider" not in texts(snap())[1]


def test_a_planned_target_with_a_worry_is_flagged():
    worried = snap()
    worried["players"][7]["xm"] = 40.0
    worried["players"][7]["nw"] = "Late fitness test"

    assert "Cucurella (DEF) 40 min · Late fitness test" in texts(worried)[1]


def test_a_quiet_squad_says_nothing_was_flagged():
    quiet = snap()
    quiet["players"][4].update(st="a", nw="", xm=90.0)

    assert "Nothing flagged" in texts(quiet)[1]


def test_a_solver_only_run_has_no_research_to_show():
    solver = snap(decision={
        "rationale": "", "adjustments": [], "chip": "none", "chip_justification": "",
        "source": "solver",
    })

    out = texts(solver)

    assert SOLVER_NOTES in out[1]
    assert detailed.NO_GAFFER in out[2], "text 3 says whose pick it is"


# --- Reasons end on a sentence -----------------------------------------------


def test_a_short_reason_is_left_alone():
    assert weekplan._short("Limped off in training.") == "Limped off in training."
    assert weekplan._short("  two   spaces\nand a newline ") == "two spaces and a newline"
    assert weekplan._short(None) == ""


def test_whole_sentences_are_kept_while_they_fit():
    # 120 + 1 + 100 = 221 characters: two sentences within 240 stay whole, and
    # the third (which would take it to 332) is dropped with its full stop.
    first = "A" * 119 + "."
    second = "B" * 99 + "."
    third = "C" * 110 + "."

    assert weekplan._short(f"{first} {second} {third}") == f"{first} {second}"


def test_a_long_first_sentence_is_cut_at_a_word_with_an_ellipsis():
    words = " ".join(["justify"] * 60)  # 479 characters, no stop

    out = weekplan._short(words + ". Then more.")

    assert out.endswith("justify…")
    assert len(out) <= weekplan.MAX_REASON
    assert set(out.rstrip("…").split()) == {"justify"}


def test_no_cut_ever_lands_inside_a_word():
    # Words of every length from 3 to 12, so the limit falls inside a word at
    # some offset whatever the starting point.
    for start in range(10):
        words = [("w" * (3 + (i + start) % 10)) + str(i) for i in range(80)]
        out = weekplan._short(" ".join(words))
        assert out.endswith("…")
        assert len(out) <= weekplan.MAX_REASON
        assert out[:-1].split() == words[: len(out[:-1].split())], out


# --- Text 3: why this plan --------------------------------------------------


def test_why_this_plan_is_the_gaffers_bullets_verbatim():
    third = texts(snap())[2]

    assert "- Cucurella's fixtures beat Justin's." in third
    assert "limped off" not in third


def test_runner_up_and_rolling_are_measured_against_the_chosen_plan():
    # Chosen nets 19.0; rolling (plan 1) 18.0 is 1.0 behind; plan 2 buys Xhaka
    # for a hit and nets 17.4, 1.6 behind.
    board = snap(plans={
        0: plan([7], [4]),
        1: plan([], [], net=18.0),
        2: plan([7, 5], [4, 2], hits=1, net=17.4),
    })

    assert "Next best: +Xhaka for a hit, 1.6 behind · Rolling: 1.0 behind" in texts(board)[2]


def test_a_shortlist_plan_on_older_minutes_is_not_the_runner_up():
    # Plan 3 was costed before the gaffer's re-solve (resolved False) and nets
    # 25.0 against the chosen re-solved plan's 19.0: different footing, so it
    # is not "ahead of the chosen plan" and not the next best.
    board = snap(plans={
        0: plan([7], [4], resolved=True),
        1: plan([], [], net=18.0, resolved=True),
        3: plan([2], [4], net=25.0, resolved=False),
    })

    third = texts(board)[2]

    assert "Rolling: 1.0 behind" in third
    assert "Next best" not in third
    assert "25.0" not in third and "ahead" not in third


def test_when_the_chosen_plan_is_rolling_only_the_next_best_is_shown():
    board = snap(plans={0: plan([], [], net=19.0), 1: plan([7], [4], net=18.5)})

    third = texts(board)[2]

    assert "Next best: +Cucurella, 0.5 behind" in third
    assert "Rolling" not in third


def test_a_rival_that_buys_the_same_men_but_sells_another_names_the_sale():
    # Chosen sells Justin (4) for Cucurella (7), nets 19.0. Plan 2 buys the
    # same man but sells Salah (2) instead, nets 17.5: 1.5 behind. It buys
    # nothing the chosen plan does not, so the line names what it sells that
    # the chosen plan keeps: -Salah.
    board = snap(plans={0: plan([7], [4]), 2: plan([7], [2], net=17.5)})

    assert "Next best: -Salah, 1.5 behind" in texts(board)[2]


def test_a_rival_with_the_same_moves_says_only_its_road_ahead_differs():
    # Plan 2 buys Cucurella for Justin exactly as the chosen plan does and nets
    # 18.2, 0.8 behind: it differs only in its later weeks. It must not name
    # Cucurella, whom the chosen plan already buys.
    board = snap(plans={0: plan([7], [4]), 2: plan([7], [4], net=18.2)})

    third = texts(board)[2]

    assert "Next best: same moves, a different road ahead, 0.8 behind" in third
    assert "Next best: +Cucurella" not in third


def test_a_rival_that_buys_other_men_names_only_the_extra_buys():
    # Chosen buys Cucurella (7). Plan 2 buys Cucurella and Xhaka (5) for a hit
    # and nets 17.4: only Xhaka is what it buys that the chosen plan does not.
    board = snap(plans={0: plan([7], [4]), 2: plan([7, 5], [4, 2], hits=1, net=17.4)})

    assert "Next best: +Xhaka for a hit, 1.6 behind" in texts(board)[2]


def test_the_armbands_carry_the_window_and_this_weeks_xp():
    third = texts(snap())[2]

    # Saka 11.42 over the window, 4.12 this week; Salah 5.00 and 5.00; the one
    # rival is Haaland, 9.00 and 6.00.
    assert "Captain Saka — 11.4 over the window, 4.1 this week" in third
    assert "Vice Salah — 5.0 over the window, 5.0 this week" in third
    assert "Rivals:\n  Haaland — 9.0 over the window, 6.0 this week" in third


def test_the_skipped_transfers_come_with_the_reason_given_in_the_report():
    plain = texts(snap())[2]
    words = snap(reasons={"skipped": [{"id": 3, "reason": "He would sit on the bench."}]})

    assert "• Haaland over Justin (+6.0): unaffordable" in plain
    assert "• Haaland over Justin (+6.0): He would sit on the bench." in texts(words)[2]


def test_only_the_three_highest_gain_skipped_transfers_are_shown():
    many = snap(skipped=[
        {"id": 3, "pos": "FWD", "replaces": 4, "gap": gap, "price_diff": 0, "tag": f"tag{gap}"}
        for gap in (1.0, 5.0, 2.0, 4.0, 3.0)
    ])

    third = texts(many)[2]

    assert [f"tag{gap}" in third for gap in (5.0, 4.0, 3.0, 2.0, 1.0)] == [
        True, True, True, False, False,
    ]


def test_a_transfer_skipped_only_for_the_bench_gives_way_to_one_the_plan_beat():
    # The benched forward has the biggest gap, but the plan never weighed him;
    # the defender it weighed and beat is the one worth explaining.
    board = snap(skipped=[
        {"id": 3, "pos": "FWD", "replaces": 4, "gap": 17.0, "price_diff": 0, "tag": BENCH},
        {"id": 3, "pos": "DEF", "replaces": 4, "gap": 8.0, "price_diff": 0, "tag": "beaten"},
    ])

    third = texts(board)[2]

    assert third.index("(+8.0)") < third.index("(+17.0)")


def test_triggers_are_printed_when_reasons_carry_them_and_not_otherwise():
    asked = snap(reasons={"triggers": ["If Saka is ruled out on Friday: Saka → Xhaka now"]})

    third = texts(asked)[2]

    assert "What would change the plan:\n• If Saka is ruled out on Friday: Saka → Xhaka now" in third
    assert "What would change the plan" not in texts(snap())[2]


# --- Text 4: the road ahead and chips ---------------------------------------


def test_the_road_ahead_uses_the_reports_own_lines():
    road = [{"gw": 9, "out": [2], "in": [5], "chip": "none", "hits": 0}]
    board = snap(plans={0: plan([7], [4], path=road), 1: plan([], [], net=18.0)})

    fourth = texts(board)[3]

    assert "Current plan, re-planned every run:\nGW9: out Salah (£12.9m), in Xhaka (£5.0m) — 1 FT, no hit" in fourth


def test_a_path_move_from_an_old_snapshot_without_hits_still_prints():
    road = [{"gw": 9, "out": [2], "in": [5], "chip": "none"}]
    board = snap(plans={0: plan([7], [4], path=road), 1: plan([], [], net=18.0)})

    assert "GW9: out Salah (£12.9m), in Xhaka (£5.0m)\n" in texts(board)[3] + "\n"


def test_no_later_moves_say_so():
    assert "No moves planned after this week." in texts(snap())[3]


def test_the_chips_are_the_calendar_and_this_weeks_values():
    fourth = texts(snap())[3]

    assert "No chip this week." in fourth
    assert "Bench Boost (GW1–19): saved for GW9, worth +6.1 there" in fourth
    assert "Worth this week: Bench Boost +6.1 · Wildcard -3.0 (over the window)" in fourth
    played = snap(decision={**snap()["decision"], "chip": "triple_captain"})
    assert "Chip this week: Triple Captain" in texts(played)[3]


# --- Text 5: upcoming gameweeks ---------------------------------------------


def test_the_squads_week_by_week_total_is_the_elevens():
    # GW8: 4.12 + 5.00 + 6.00 + 3.00 + 3.00 = 21.12; GW9: 7.30 + 0 + 3.00 + 0 + 3.00 = 13.30.
    assert "XI projected, before the armband: GW8 21.1 · GW9 13.3" in texts(snap())[4]


def test_each_man_has_a_compact_strip_home_upper_away_lower_blank_dash_double_plus():
    fifth = texts(snap())[4]

    # Order: keeper, defender, midfielders by window (Saka 11.42 then Salah
    # 5.00), forward; then the targets (Cucurella).
    assert (
        "Raya LEE LEE — 6.0\n"
        "Justin LEE — — 3.0\n"
        "Saka LEE LEE+wol — 11.4\n"
        "Salah wol — — 5.0\n"
        "Haaland LEE LEE — 9.0\n"
        "\n"
        "Targets:\n"
        "Cucurella wol LEE — 6.0"
    ) in fifth
    assert "(h)" not in fifth, "the report's wide cell is not reused"
    assert "swing" not in fifth.lower()


def test_the_later_weeks_are_called_softer_only_when_the_window_is_long():
    assert "softer" not in texts(snap())[4]
    long_window = snap(events=[8, 9, 10, 11, 12, 13])
    for player in long_window["players"].values():
        player["f"].update({gw: [] for gw in (10, 11, 12, 13)})
        player["x"].update({gw: 0.0 for gw in (10, 11, 12, 13)})

    assert "GW12–GW13 are softer" in texts(long_window)[4]


# --- A free-hit week --------------------------------------------------------


def test_a_free_hit_week_shows_the_temporary_team_and_leaves_the_transfers_alone():
    # The standing squad is not the team that plays: Xhaka and Cucurella
    # replace Salah and Justin for the week; Cucurella is on the bench.
    free_hit = snap(
        decision={**snap()["decision"], "chip": "free_hit"},
        freehit={"squad": [1, 3, 5, 6, 7], "xi": [1, 3, 5, 6]},
        xi=[1, 3, 5, 6],
        plans={0: plan([], []), 1: plan([], [], net=18.0)},
    )

    out = texts(free_hit)

    assert "Chip: Free Hit" in out[0]
    assert "Bench: Cucurella" in out[0]
    assert "Transfers: none — keep the free transfer." in out[0]
    assert out[0].splitlines()[-1] == "Free Hit: free transfers untouched · bank £3.7m · no hit"
    assert "Xhaka" in out[4] and "Salah" not in out[4]


# --- The committed snapshots -------------------------------------------------


@pytest.mark.parametrize("gw", [8, 9])
def test_the_committed_snapshots_render_five_texts_under_telegrams_limit(gw):
    snapshots = load_snapshots(REPO / "state", gw)
    assert snapshots, "the repository carries this week's snapshots"

    out = render_week_plan(snapshots, NOW, None)

    assert len(out) == 5
    assert all(0 < len(text) < 4000 for text in out)


def test_the_committed_gw9_plan_matches_the_gaffers_numbers():
    out = render_week_plan(load_snapshots(REPO / "state", 9), NOW, None)

    assert out[0].splitlines()[0] == (
        "1/5 · 📋 GW9 week plan · as of Wed scout run 23:35"
        " · deadline Sat 31 Oct 11:00 (Irish time)"
    )
    assert "Next best: +Konsa for a hit, 1.6 behind · Rolling: 1.9 behind" in out[2]
    assert "Captain Haaland — 32.9 over the window" in out[2]
    assert "GW10: out Thomas (£4.0m), in Konsa (£4.7m)" in out[3]


# --- The detailed renderer's public helpers ---------------------------------


def test_the_helpers_the_week_plan_borrows_are_public_and_unchanged():
    board = snap()

    assert detailed.name(board, 1) == "Saka" and detailed.name(board, None) == "none"
    assert detailed.window(board, 1) == 11.42
    assert detailed.moves(board) == ([4], [7])
    assert detailed.chip(board) == "none"
    assert detailed.priced(board, [4, 7]) == "Justin (£4.6m), Cucurella (£6.1m)"
    assert detailed.planned(board, {"out": [4], "in": [7], "hits": 0}) == (
        "out Justin (£4.6m), in Cucurella (£6.1m) — 1 FT, no hit"
    )
    assert not hasattr(detailed, "_name"), "the private spellings are gone, not duplicated"


# --- The handler ------------------------------------------------------------


@pytest.fixture
def state(tmp_path):
    state_dir = tmp_path / "state"
    write_snapshot(state_dir, 8, "scout", snap())
    return state_dir


def cfg_for(state_dir):
    return SimpleNamespace(state_dir=state_dir)


def test_the_handler_sends_all_five_in_order_and_returns_none(state, tmp_path):
    sent = []

    result = week_plan(cfg_for(state), tmp_path / "state.lock", NOW, sent.append)

    assert result is None
    assert len(sent) == 5
    assert [text.split(" · ")[0] for text in sent] == ["1/5", "2/5", "3/5", "4/5", "5/5"]
    assert sent == render_week_plan(load_snapshots(state, 8), NOW, None)


def test_the_handler_reads_the_owners_recorded_week(state, tmp_path):
    folder = state / "executed"
    folder.mkdir()
    (folder / "gw8.json").write_text(json.dumps(recorded("2026-10-29T08:00:00+00:00").to_json()))
    sent = []

    week_plan(cfg_for(state), tmp_path / "state.lock", NOW, sent.append)

    assert ENTERED in sent[0]


def test_with_no_snapshot_one_plain_line_is_the_whole_answer(tmp_path):
    sent = []

    week_plan(cfg_for(tmp_path / "state"), tmp_path / "state.lock", NOW, sent.append)

    assert sent == [NO_PLAN]


def test_the_files_are_read_under_the_lock_and_the_texts_rendered_outside_it(state, tmp_path, monkeypatch):
    events = []

    class Spy:
        def __init__(self, path, on_wait=None):
            pass

        def __enter__(self):
            events.append("lock")

        def __exit__(self, *exc):
            events.append("unlock")

    real_load, real_render = weekplan.load_snapshots, weekplan.render_week_plan
    monkeypatch.setattr(weekplan, "state_lock", Spy)
    monkeypatch.setattr(weekplan, "load_snapshots", lambda *a: events.append("read") or real_load(*a))
    monkeypatch.setattr(weekplan, "render_week_plan", lambda *a: events.append("render") or real_render(*a))

    week_plan(cfg_for(state), tmp_path / "state.lock", NOW, lambda text: None)

    assert events == ["lock", "read", "unlock", "render"]


def test_a_render_that_fails_sends_nothing_so_a_retry_cannot_double_send(state, tmp_path, monkeypatch):
    sent = []

    def broken(snapshots, now, executed):
        raise KeyError("a snapshot missing a key")

    monkeypatch.setattr(weekplan, "render_week_plan", broken)

    with pytest.raises(KeyError):
        week_plan(cfg_for(state), tmp_path / "state.lock", NOW, sent.append)

    assert sent == []


def test_all_five_are_rendered_before_the_first_is_sent(state, tmp_path, monkeypatch):
    # A snapshot whose fifth text cannot be made (a window with no events)
    # must not leave the first four sent.
    broken = snap(events=[])
    write_snapshot(state, 8, "scout", broken)
    sent = []

    with pytest.raises(IndexError):
        week_plan(cfg_for(state), tmp_path / "state.lock", NOW, sent.append)

    assert sent == []
