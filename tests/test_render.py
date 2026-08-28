"""Tests for the weekly report.

The renderer reads names, clubs, prices, positions and projected totals and
nothing else, so the universe here is a flat one: twenty-one players spread
over the three fixture clubs, priced and projected so that every ordering the
report makes can be read straight off the table. Ids 1-15 are the squad as it
stands; 16-21 are the field it could sign from.

==  ======  ===  ===  ======  ====  ============================================
id  name    club pos  price   xP    role
==  ======  ===  ===  ======  ====  ============================================
1   Alvez   ASH  GKP  £5.5m   20.0  starts
2   Byrne   BRW  GKP  £4.5m    8.0  benched, and still first on the bench
3   Costa   ASH  DEF  £6.0m   30.0  starts
4   Dodd    CRV  DEF  £4.0m   26.0  starts
5   Egan    ASH  DEF  £5.0m   34.0  starts
6   Fenn    BRW  DEF  £4.5m   12.0  benched
7   Gale    CRV  DEF  £4.0m   10.0  sold by the recommended plan
8   Hume    ASH  MID  £12.5m  48.0  starts, and captains — best xP in the game
9   Innes   BRW  MID  £9.0m   36.0  starts
10  Jonker  CRV  MID  £7.5m   40.0  starts
11  Kerr    ASH  MID  £7.0m   35.0  starts
12  Lang    BRW  MID  £5.5m   14.0  benched
13  Moss    ASH  FWD  £10.5m  42.0  starts, and is vice-captain
14  Nunes   BRW  FWD  £8.0m   31.0  starts
15  Oduya   CRV  FWD  £6.0m   37.0  starts
16  Pike    ASH  MID  £8.0m   41.5  watchlist
17  Quinn   BRW  DEF  £5.5m   33.0  watchlist
18  Reid    CRV  FWD  £9.5m   44.0  bought by the recommended plan
19  Salas   ASH  MID  £6.5m   38.0  watchlist
20  Tovey   BRW  GKP  £5.0m   22.0  watchlist
21  Ubaldi  CRV  DEF  £4.0m    9.0  the worst player on the board
==  ======  ===  ===  ======  ====  ============================================

The eleven is 3-4-3 and the four candidate plans are canned: only their
transfer lists and their three numbers are ever read, so the squad each one
leaves behind is set for the watchlist's benefit and no further.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from aigaffer.data.models import Bootstrap, Event, Player
from aigaffer.manager.agent import ManagerDecision
from aigaffer.model.xp import PlayerProjection
from aigaffer.report.render import (
    FRESH_SOLVE,
    GAFFER_VERDICT,
    HUMAN_JUDGES,
    NEWS_MOVED,
    NO_FULL_REPORT,
    REMINDER_UNCHANGED,
    render_digest,
    render_reminder,
    render_reminder_digest,
    render_report,
)
from aigaffer.solver.lineup import ChipEvs, Lineup
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan
from tests.fixtures import BOOTSTRAP_JSON

GKP, DEF, MID, FWD = 1, 2, 3, 4
ASH, BRW, CRV = 1, 2, 3

UNIVERSE = [
    # id, name, club, position, price, projected points over the horizon
    (1, "Alvez", ASH, GKP, 55, 20.0),
    (2, "Byrne", BRW, GKP, 45, 8.0),
    (3, "Costa", ASH, DEF, 60, 30.0),
    (4, "Dodd", CRV, DEF, 40, 26.0),
    (5, "Egan", ASH, DEF, 50, 34.0),
    (6, "Fenn", BRW, DEF, 45, 12.0),
    (7, "Gale", CRV, DEF, 40, 10.0),
    (8, "Hume", ASH, MID, 125, 48.0),
    (9, "Innes", BRW, MID, 90, 36.0),
    (10, "Jonker", CRV, MID, 75, 40.0),
    (11, "Kerr", ASH, MID, 70, 35.0),
    (12, "Lang", BRW, MID, 55, 14.0),
    (13, "Moss", ASH, FWD, 105, 42.0),
    (14, "Nunes", BRW, FWD, 80, 31.0),
    (15, "Oduya", CRV, FWD, 60, 37.0),
    (16, "Pike", ASH, MID, 80, 41.5),
    (17, "Quinn", BRW, DEF, 55, 33.0),
    (18, "Reid", CRV, FWD, 95, 44.0),
    (19, "Salas", ASH, MID, 65, 38.0),
    (20, "Tovey", BRW, GKP, 50, 22.0),
    (21, "Ubaldi", CRV, DEF, 40, 9.0),
]

SQUAD = list(range(1, 16))


def element(pid: int, name: str, club: int, position: int, price: int) -> Player:
    """A bootstrap element; the stats the renderer never looks at are zeroed."""
    return Player(
        id=pid,
        web_name=name,
        team=club,
        element_type=position,
        now_cost=price,
        status="a",
        minutes=0,
        starts=0,
        total_points=0,
        bonus=0,
        saves=0,
    )


FIXTURE = Bootstrap.model_validate(BOOTSTRAP_JSON)
EVENT = FIXTURE.next_event()
BOOTSTRAP = Bootstrap(
    events=FIXTURE.events,
    teams=FIXTURE.teams,
    elements=[element(*row[:5]) for row in UNIVERSE],
)

# The renderer reads ``total`` for every number it prints. ``per_gw`` is here
# for its length alone: it is how long the horizon is, which is what the chip
# panel labels the wildcard's number with.
HORIZON = 6
XP = {
    pid: PlayerProjection(
        player_id=pid,
        per_gw={gw: round(points / HORIZON, 1) for gw in range(2, 2 + HORIZON)},
        total=points,
    )
    for pid, _, _, _, _, points in UNIVERSE
}

LINEUP = Lineup(
    xi=[1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
    captain=8,
    vice=13,
    bench=[2, 12, 6, 7],
)

CHIPS = ChipEvs(bench_boost=3.2, triple_captain=8.4, free_hit=-1.5, wildcard=12.0)

ROLL = Plan(
    squad=SQUAD,
    xi=LINEUP.xi,
    transfers_in=[],
    transfers_out=[],
    hits=0,
    xp_total=252.0,
    objective=252.0,
)
ONE = Plan(
    squad=[pid for pid in SQUAD if pid != 7] + [18],
    xi=LINEUP.xi,
    transfers_in=[18],
    transfers_out=[7],
    hits=0,
    xp_total=255.5,
    objective=255.5,
)
TWO = Plan(
    squad=[pid for pid in SQUAD if pid not in (6, 7)] + [16, 18],
    xi=LINEUP.xi,
    transfers_in=[16, 18],
    transfers_out=[6, 7],
    hits=1,
    xp_total=259.0,
    objective=255.0,
)
DRAFT = Plan(
    squad=SQUAD,
    xi=LINEUP.xi,
    transfers_in=SQUAD,
    transfers_out=[],
    hits=0,
    xp_total=252.0,
    objective=252.0,
)
PLANS = [ONE, TWO, ROLL]


def with_path(plan: Plan, moves: list[PlannedMove]) -> Plan:
    """``plan`` as the window hands it over: the same opening gameweek, with
    the rest of the story hanging off it."""
    return replace(
        plan,
        path=PlannedPath(moves=moves, objective=plan.objective, weekly_xp={}),
    )


# Two more gameweeks of the recommended plan: a free move, then a pair that
# outruns the bank of free transfers by one and pays four points for it.
FIRST = PlannedMove(event=3, transfers_in=[17], transfers_out=[6], hits=0)
SECOND = PlannedMove(event=4, transfers_in=[16, 20], transfers_out=[2, 12], hits=1)
AHEAD = with_path(ONE, [FIRST, SECOND])

ADVISORY = "Advisory — re-planned every run; only this week's moves are ever made."
SINGLE_WEEK = "Single-week engine (multi-week solve unavailable this run)."
WINDOW_UNITS = (
    "xP and net are the whole window, decayed and with the armband in;"
    " transfers and hits shown are this week's unless the row says otherwise."
)


# The gaffer's own week, as the manager loop hands it over. His plan and his
# eleven are already the ones the report is being rendered with — the pipeline
# swaps them in before it calls this — so what the section adds is the half of
# the decision that is words: why, what he overruled, and who decided.
RATIONALE = (
    "Gale is out for a month and the solver did not know it. Reid comes in,"
    " and I have left the armbands alone."
)
ADJUSTMENTS = [
    {"player_id": 7, "expected_minutes": 0.0, "reason": "hamstring (BBC, Friday)"},
    {"player_id": 11, "expected_minutes": 60.0, "reason": "rested midweek"},
]
GOOD_CHIP = (
    "The bench boost is worth +3.2 this week, the whole bench has a home"
    " fixture, and what we give up is a double gameweek eight months away."
)


def gaffer(
    source: str = "manager",
    chip: str = "none",
    justification: str = "",
    adjustments: list[dict] | None = None,
    searches: int = 3,
    unapplied: list[dict] | None = None,
) -> ManagerDecision:
    """One manager decision, as the report is handed one."""
    return ManagerDecision(
        plan=ONE,
        lineup=LINEUP,
        captain=LINEUP.captain,
        vice=LINEUP.vice,
        chip=chip,
        chip_justification=justification,
        rationale=RATIONALE,
        adjustments=ADJUSTMENTS if adjustments is None else adjustments,
        searches=searches,
        source=source,
        unapplied=unapplied or [],
    )


def report(
    choice: Plan = ONE,
    event: Event = EVENT,
    view: ManagerDecision | None = None,
    engine_expected: bool = False,
    plans: list[Plan] | None = None,
    lineup: Lineup = LINEUP,
    free_transfers: int | None = 1,
    selling_prices: dict[int, int] | None = None,
) -> str:
    """The report as Task 12 will ask for it.

    ``plans`` is the board, and it defaults to the canned single-week one: a
    run off the window hands over a board whose every row carries a path, and
    the rows are measured differently when it does.

    ``free_transfers`` is the bank the action block reads when the week rolls,
    and defaults to the one the pipeline's ordinary scout run holds; ``lineup``
    is the eleven the block and the team sheet are drawn from.
    """
    return render_report(
        "scout",
        event,
        PLANS if plans is None else plans,
        choice,
        lineup,
        CHIPS,
        BOOTSTRAP,
        XP,
        view,
        engine_expected=engine_expected,
        free_transfers=free_transfers,
        selling_prices=selling_prices,
    )


def section(text: str, heading: str) -> list[str]:
    """The non-blank lines under ``## heading``, down to the next heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"## {heading}"))
    rest = lines[start + 1 :]
    end = next((i for i, line in enumerate(rest) if line.startswith("## ")), len(rest))
    return [line for line in rest[:end] if line]


def bullets(text: str, heading: str) -> list[str]:
    """The list items under ``## heading``."""
    return [line for line in section(text, heading) if line.startswith("- ")]


def test_the_report_reads_top_to_bottom_in_one_pass():
    headings = [line for line in report().splitlines() if line.startswith("#")]

    assert headings == [
        "# AI Gaffer — GW2 scout",
        "## Do this",
        "## Recommendation",
        "## Starting XI (3-4-3)",
        "## Candidate plans",
        "## Chip EV",
        "## Watchlist",
    ]


def test_the_report_ends_with_a_newline():
    # It is committed to the repo as a markdown file, not only messaged.
    assert report().endswith("xP\n")


def test_the_deadline_is_stated_in_utc():
    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in report()


def test_a_deadline_in_another_timezone_is_converted():
    # The same instant, served from a summer-time clock two hours ahead.
    berlin = EVENT.model_copy(
        update={
            "deadline_time": datetime(
                2025, 8, 22, 19, 30, tzinfo=timezone(timedelta(hours=2))
            )
        }
    )

    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in report(event=berlin)


def test_a_deadline_with_no_timezone_is_read_as_utc():
    # Not the runner's local clock, which is whatever the machine says it is.
    naive = EVENT.model_copy(update={"deadline_time": datetime(2025, 8, 22, 17, 30)})

    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in report(event=naive)


def test_the_recommendation_names_who_is_out_and_who_is_in():
    assert section(report(), "Recommendation") == [
        "1 transfer, no hit.",
        "- Out: Gale (DEF, CRV, £4.0m)",
        "- In: Reid (FWD, CRV, £9.5m)",
    ]


def test_a_plan_that_takes_a_hit_says_what_it_costs():
    assert section(report(choice=TWO), "Recommendation") == [
        "2 transfers, -4 pts in hits.",
        "- Out: Fenn (DEF, BRW, £4.5m), Gale (DEF, CRV, £4.0m)",
        "- In: Pike (MID, ASH, £8.0m), Reid (FWD, CRV, £9.5m)",
    ]


def test_an_opening_draft_buys_without_selling_anybody():
    # Fifteen signings and nobody to sell. The bullet the sales would have gone
    # in is left out rather than printed empty: "- Out: " with nothing after it
    # is how a report that lost half its own list reads, and this is the first
    # report of a season.
    drafted = report(choice=DRAFT)
    lines = section(drafted, "Recommendation")

    assert lines[0] == "15 transfers, no hit."
    assert lines[1].startswith("- In: ")
    assert lines[1].count("£") == 15
    assert len(lines) == 2
    assert "- Out:" not in drafted


def test_standing_still_says_roll_the_transfer():
    rolled = report(choice=ROLL)

    assert section(rolled, "Recommendation") == ["Roll the transfer."]
    assert "- Out:" not in rolled
    assert "- In:" not in rolled


def test_the_eleven_is_laid_out_by_formation_with_the_armbands_marked():
    # Best projected first in each row, and the vice-captain is a forward, so
    # the markers follow the players rather than the rows.
    assert bullets(report(), "Starting XI")[:4] == [
        "- GKP: Alvez",
        "- DEF: Egan, Costa, Dodd",
        "- MID: Hume (C), Jonker, Innes, Kerr",
        "- FWD: Moss (V), Oduya, Nunes",
    ]


def test_the_bench_is_numbered_in_substitution_order():
    # Byrne is the worst player in the squad and still first on the bench: a
    # keeper can only come on for a keeper.
    assert bullets(report(), "Starting XI")[4] == (
        "- Bench: 1. Byrne (GKP), 2. Lang (MID), 3. Fenn (DEF), 4. Gale (DEF)"
    )


def test_every_candidate_plan_gets_a_row_and_the_choice_is_flagged():
    assert bullets(report(), "Candidate plans") == [
        "- 1 transfer | 0 hits | 255.5 xP | 255.5 net  <- recommended",
        "- 2 transfers | 1 hit | 259.0 xP | 255.0 net",
        "- 0 transfers | 0 hits | 252.0 xP | 252.0 net",
    ]


def test_the_gaffers_pick_is_flagged_on_the_row_that_holds_his_squad():
    # He finalized a plan the solver had already reached, but a re-solve minted
    # a fresh object for it with fresh numbers on it. Flagging by identity
    # would flag nothing and quietly break the section's one invariant.
    same = replace(ONE, xp_total=256.9, objective=256.9)
    resolved = report(choice=same, view=replace(gaffer(), plan=same))
    rows = bullets(resolved, "Candidate plans")

    assert sum("recommended" in row for row in rows) == 1
    assert rows[0].endswith("<- recommended")
    assert "re-solved" not in resolved


def test_a_pick_the_shortlist_never_had_says_where_it_came_from():
    # The plan he chose was solved on his own minutes and is not among the
    # four below. Saying nothing would leave a list with no recommendation on
    # it and a recommendation above with no list behind it.
    fresh = Plan(
        squad=[pid for pid in SQUAD if pid != 6] + [17],
        xi=LINEUP.xi,
        transfers_in=[17],
        transfers_out=[6],
        hits=0,
        xp_total=257.0,
        objective=257.0,
    )

    elsewhere = report(choice=fresh, view=replace(gaffer(), plan=fresh))

    assert not any("recommended" in row for row in bullets(elsewhere, "Candidate plans"))
    assert section(elsewhere, "Candidate plans")[-1] == (
        "The gaffer re-solved after his adjustments;"
        " his pick above is not on this list."
    )


def test_a_report_with_no_manager_never_explains_a_re_solve():
    assert "re-solved" not in report()


# --- the do-this action block ----------------------------------------------
#
# The bot advises and never executes, so the moves are made by hand on a phone
# against a deadline. The block is a checklist at the very top: what to do, in
# the order the FPL app takes it, and nothing a reader has to parse the rest of
# the report to act on.


def test_the_action_block_leads_the_report():
    # First thing under the title, before the recommendation it summarizes.
    headings = [line for line in report().splitlines() if line.startswith("#")]

    assert headings[:2] == ["# AI Gaffer — GW2 scout", "## Do this"]


def test_the_action_block_is_a_terminal_checklist():
    # A deadline, the swap, and the armbands — each line imperative and whole,
    # the position-club-price tag spaced so it reads as one label at arm's
    # length. Reid stays benched, so there is no lineup line to set.
    assert section(report(), "Do this") == [
        "⏰ Make these by Fri 22 Aug 2025 17:30 UTC — GW2",
        "SELL Gale (DEF CRV £4.0m) → BUY Reid (FWD CRV £9.5m)",
        "CAPTAIN Hume · VICE Moss",
    ]


def test_each_transfer_is_its_own_sell_then_buy_line():
    swaps = [line for line in section(report(choice=TWO), "Do this") if "→" in line]

    assert swaps == [
        "SELL Fenn (DEF BRW £4.5m) → BUY Pike (MID ASH £8.0m)",
        "SELL Gale (DEF CRV £4.0m) → BUY Reid (FWD CRV £9.5m)",
    ]


def test_a_sale_that_raises_less_than_the_listed_price_says_so():
    # Gale lists at £4.0m and the ledger says his sale raises £3.6m: the
    # number the owner sees in the app when he confirms, printed inside the
    # SELL tag so the checklist and the phone agree to the pound.
    block = section(report(selling_prices={7: 36}), "Do this")

    assert (
        "SELL Gale (DEF CRV £4.0m, sells £3.6m) → BUY Reid (FWD CRV £9.5m)"
        in block
    )


def test_a_sale_at_the_listed_price_keeps_the_plain_tag():
    # Selling at the listed price is the ordinary case and not worth a word:
    # a tag that always said "sells" would bury the week it mattered.
    block = section(report(selling_prices={7: 40}), "Do this")

    assert "SELL Gale (DEF CRV £4.0m) → BUY Reid (FWD CRV £9.5m)" in block


def test_a_buy_never_grows_a_sells_tag():
    # Reid is bought, and buys pay the listed price by definition — a selling
    # price for him in the dict (he could be a squad man on another plan's
    # board) must not leak into the BUY side of the line.
    block = section(report(selling_prices={7: 40, 18: 80}), "Do this")

    assert "BUY Reid (FWD CRV £9.5m)" in str(block)
    assert "Reid (FWD CRV £9.5m, sells" not in str(block)


def test_a_week_that_rolls_says_so_and_banks_its_free_transfer():
    rolled = section(report(choice=ROLL), "Do this")

    assert "No transfers — roll (bank 1 free transfer)." in rolled
    assert not any(line.startswith("SELL") for line in rolled)


def test_a_pair_of_banked_transfers_is_pluralized():
    assert "roll (bank 2 free transfers)." in report(choice=ROLL, free_transfers=2)


def test_a_chip_the_gaffer_plays_is_a_line_to_act_on():
    played = section(report(view=gaffer(chip="bench_boost", justification=GOOD_CHIP)), "Do this")

    assert "PLAY Bench Boost" in played


def test_no_chip_prints_no_chip_line():
    # Omitted entirely, not printed as "no chip": a checklist has no line for
    # the thing you are not doing.
    assert not any(line.startswith("PLAY") for line in section(report(view=gaffer()), "Do this"))


def test_a_signing_that_must_start_asks_for_the_lineup():
    # Reid is bought and goes straight into the eleven, where the FPL app would
    # have benched him: the block says to fix the lineup and leaves the full XI
    # to the section below.
    starting = replace(LINEUP, xi=[1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 18], bench=[2, 12, 6, 15])
    block = section(report(lineup=starting), "Do this")

    assert "Set lineup: 3-4-3" in block
    # Still a checklist, not the report: the eleven itself is not duplicated.
    assert not any("Hume" in line and "Jonker" in line for line in block)


def test_a_signing_left_on_the_bench_needs_no_lineup_line():
    assert not any(line.startswith("Set lineup") for line in section(report(), "Do this"))


def test_a_draft_has_no_action_block():
    # The whole squad is the action; the recommendation below is the fifteen to
    # buy, and there is no scannable move to lift out of it.
    assert "## Do this" not in report(choice=DRAFT)


def test_the_block_follows_the_gaffers_own_plan():
    # He re-solved on his own minutes and finalized a plan the shortlist never
    # had. The checklist tells the owner to make that move, not the solver's
    # original.
    fresh = replace(
        ONE,
        squad=[pid for pid in SQUAD if pid != 6] + [17],
        transfers_in=[17],
        transfers_out=[6],
    )
    block = section(report(choice=fresh, view=replace(gaffer(), plan=fresh)), "Do this")

    assert "SELL Fenn (DEF BRW £4.5m) → BUY Quinn (DEF BRW £5.5m)" in block
    assert not any("Gale" in line for line in block)


# --- what the numbers on a row are measured over ---------------------------
#
# A row off the window prints this gameweek's transfers beside two totals for
# the whole window, and the window's totals have every gameweek's hits already
# taken off them. A hit the plan pays in GW4 is therefore inside the net and
# nowhere else on the line, and the row does not add up until it says so.


def test_a_row_whose_path_pays_hits_says_when_they_fall():
    board = [AHEAD, TWO, ROLL]

    assert bullets(report(plans=board, choice=AHEAD), "Candidate plans")[0] == (
        "- 1 transfer | 0 hits now, 1 over the window"
        " | 255.5 xP | 255.5 net  <- recommended"
    )


def test_a_row_counts_this_weeks_hits_in_the_window_total():
    # Two hits over the window, one of them taken now: the first number is a
    # part of the second and not a rival to it.
    later = with_path(TWO, [SECOND])

    assert bullets(report(plans=[later], choice=later), "Candidate plans")[0] == (
        "- 2 transfers | 1 hit now, 2 over the window"
        " | 259.0 xP | 255.0 net  <- recommended"
    )


def test_a_path_that_pays_no_hits_leaves_the_cell_as_it_was():
    # The two numbers would be the same number, and a row that prints it twice
    # reads as a row with something to explain.
    free = with_path(ONE, [FIRST])
    rows = bullets(report(plans=[free, TWO, ROLL], choice=free), "Candidate plans")

    assert rows[0].startswith("- 1 transfer | 0 hits | 255.5 xP")


def test_a_board_off_the_window_says_what_its_totals_cover():
    lines = section(report(plans=[AHEAD, TWO, ROLL], choice=AHEAD), "Candidate plans")

    assert lines[0] == WINDOW_UNITS


def test_a_single_week_board_is_measured_as_it_always_was():
    # No window behind these rows, so there is no window to caption: the
    # section is the one this printed before the planner existed.
    plain = report()

    assert WINDOW_UNITS not in plain
    assert "over the window" not in plain


# --- the road ahead --------------------------------------------------------


def test_the_road_ahead_names_every_move_the_window_intends():
    # A gameweek at a time: who goes, who comes in, what each of them costs,
    # and what the move spends. The last line is the whole point of the
    # section — none of it is entered but the week above it.
    assert section(report(choice=AHEAD), "The road ahead") == [
        "- GW3: out Fenn (£4.5m), in Quinn (£5.5m) — 1 FT, no hit",
        "- GW4: out Byrne (£4.5m), Lang (£5.5m),"
        " in Pike (£8.0m), Tovey (£5.0m) — 1 FT, -4 pts in hits",
        ADVISORY,
    ]


def test_the_road_ahead_shows_planned_chip_weeks():
    # A chip the window means to play rides on its gameweek's line, and a
    # gameweek that only plays a chip — moving nobody — is the chip alone.
    boost = PlannedMove(
        event=5, transfers_in=[], transfers_out=[], hits=0, chip="bench_boost"
    )
    swap = PlannedMove(
        event=6, transfers_in=[17], transfers_out=[6], hits=0, chip="wildcard"
    )
    road = section(report(choice=with_path(ONE, [boost, swap])), "The road ahead")

    assert "- GW5: Bench Boost" in road
    assert any(
        line.startswith("- GW6: out Fenn") and line.endswith("— Wildcard")
        for line in road
    )


def test_the_road_ahead_follows_the_shortlist_it_came_off():
    headings = [
        line for line in report(choice=AHEAD).splitlines() if line.startswith("#")
    ]

    assert headings == [
        "# AI Gaffer — GW2 scout",
        "## Do this",
        "## Recommendation",
        "## Starting XI (3-4-3)",
        "## Candidate plans",
        "## The road ahead",
        "## Chip EV",
        "## Watchlist",
    ]


def test_a_plan_off_the_single_week_solver_has_no_road_ahead():
    # There is no window behind it, so there is nothing to print and nothing
    # to caveat: the report is the one it was before the window existed.
    assert "## The road ahead" not in report()
    assert ADVISORY not in report()


def test_a_window_that_plans_nothing_further_prints_no_road_either():
    # A path is the list of things a plan means to do, and an empty one means
    # it means to do nothing — which is a heading over an empty section.
    assert "## The road ahead" not in report(choice=with_path(ONE, []))


def test_the_road_ahead_is_the_gaffers_when_the_week_is_his():
    # His plan is a plan: if he finalized one off a re-solve, its path is the
    # one the report prints, because that is the week being recommended.
    view = replace(gaffer(), plan=AHEAD)

    assert section(report(choice=AHEAD, view=view), "The road ahead")[0] == (
        "- GW3: out Fenn (£4.5m), in Quinn (£5.5m) — 1 FT, no hit"
    )


def test_a_week_the_window_could_not_answer_says_which_engine_did():
    # The multi-week engine was configured on and came back with nothing, so
    # the shortlist below was drawn up by the other one. A reader comparing
    # this week's report with last week's is owed the reason it got shorter.
    assert section(report(engine_expected=True), "Candidate plans")[-1] == SINGLE_WEEK


def test_the_engine_is_never_mentioned_when_the_window_answered():
    assert SINGLE_WEEK not in report(choice=AHEAD, engine_expected=True)


def test_a_planner_turned_off_on_purpose_explains_nothing():
    # AIGAFFER_PLANNER=single is a configuration, not a degradation, and a
    # report that apologised for it every week would be crying wolf.
    assert SINGLE_WEEK not in report()


def test_the_chip_panel_signs_every_number():
    assert bullets(report(), "Chip EV") == [
        "- Bench boost: +3.2",
        "- Triple captain: +8.4",
        "- Free hit: -1.5",
        "- Wildcard: +12.0 xP over 6 GWs (horizon)",
    ]


def test_the_wildcard_row_says_it_is_not_a_gameweek_number():
    # It is priced as the difference between two decayed horizon totals, and
    # the three above it are next gameweek's. Four figures under one heading,
    # one of them measuring something else, is a chip played on a comparison
    # nobody made.
    panel = section(report(), "Chip EV")

    assert "except the wildcard" in panel[0]
    assert "horizon" in panel[-1] and "horizon" not in " ".join(panel[1:-1])


# --- a chip the solver itself plans ----------------------------------------
#
# With no manager the chip is the solver's own: the recommended plan's path
# carries the chip it plays this week, and the checklist, the panel and (for a
# free hit) the team sheet all read it from there. The pre-chip behaviour is the
# same object with a week1_chip of "none", which is every single-week plan.


def planning(chip: str, **path_fields) -> Plan:
    """The recommended plan with a chip on its opening gameweek — a window plan
    that plays ``chip`` this week, the way the solver hands it over."""
    return replace(
        ONE,
        path=PlannedPath(
            moves=[], objective=ONE.objective, weekly_xp={}, week1_chip=chip,
            **path_fields,
        ),
    )


def test_a_chip_the_solver_plans_is_a_line_to_act_on_with_no_manager():
    # Ruling 4: when the recommended plan plays a chip this week, the checklist
    # says so even with nobody in the manager's chair.
    played = section(report(choice=planning("bench_boost")), "Do this")

    assert "PLAY Bench Boost" in played


def test_the_chip_panel_notes_a_planned_chip_as_planned():
    # Priced like the others until the plan plays it; then the panel says it is
    # planned, because the reader is being asked to act on it and not only weigh
    # it. The four numbers are unchanged — the note rides below them.
    panel = section(report(choice=planning("triple_captain")), "Chip EV")

    assert "- Triple captain: +8.4" in panel
    assert any(line.startswith("Planned this gameweek: Triple Captain") for line in panel)


def test_a_plan_that_plans_no_chip_notes_nothing():
    # week1_chip "none" is the pre-chip world, and the panel is the one it was.
    assert not any(
        line.startswith("Planned this gameweek") for line in section(report(), "Chip EV")
    )


# --- a free hit fields a temporary team (the critical T3 carry) -------------
#
# On a free-hit week the plan's own squad is the STANDING team, which reverts;
# the eleven the owner fields is the temporary one the solver priced, surfaced
# on the path. The report fields that eleven — in the team sheet and checklist,
# clearly labelled — not the standing squad.

FREE_HIT_SQUAD = [20, 3, 4, 5, 17, 21, 8, 9, 10, 11, 16, 13, 14, 18, 2]
FREE_HIT_XI = [20, 3, 4, 5, 8, 9, 10, 11, 13, 14, 18]
FREE_HIT_LINEUP = Lineup(xi=FREE_HIT_XI, captain=8, vice=13, bench=[2, 21, 16, 17])
FREE_HIT_LABEL = "Free Hit XI (this week only)"


def free_hit_report() -> str:
    # A roll that plays a free hit: the standing squad rolls (ROLL), the path
    # carries the temporary fifteen, and the lineup handed in is that team's.
    choice = replace(
        ROLL,
        path=PlannedPath(
            moves=[], objective=ROLL.objective, weekly_xp={},
            week1_chip="free_hit",
            week1_freehit_squad=FREE_HIT_SQUAD,
            week1_freehit_xi=FREE_HIT_XI,
        ),
    )
    return report(choice=choice, lineup=FREE_HIT_LINEUP)


def test_the_starting_xi_is_the_free_hit_team_labelled_as_temporary():
    report_text = free_hit_report()
    xi = section(report_text, "Starting XI")

    # Labelled as the one-week team it is, on the heading itself.
    heading = next(line for line in report_text.splitlines() if "Starting XI" in line)
    assert heading == f"## Starting XI (3-4-3) — {FREE_HIT_LABEL}"
    # The free-hit keeper is on the sheet; the standing keeper is not — this is a
    # different eleven, not the standing one.
    assert any("Tovey" in row for row in xi)
    assert not any("Alvez" in row for row in xi)


def test_the_do_this_block_builds_the_free_hit_team_and_says_it_reverts():
    block = section(free_hit_report(), "Do this")

    assert "PLAY Free Hit" in block
    built = next(line for line in block if line.startswith(FREE_HIT_LABEL))
    assert "Tovey" in built and "Reid" in built
    assert "reverts next week" in built
    # It is a temporary build, not a swap on the standing squad: no roll line and
    # no sell/buy line belongs here.
    assert not any(line.startswith("No transfers") for line in block)
    assert not any("SELL" in line for line in block)


def test_a_free_hit_with_no_temp_squad_fields_the_standing_eleven():
    # A degenerate case — the effective chip is a free hit but no squad was put
    # behind it — falls back to the standing eleven rather than mislabelling one.
    choice = replace(
        ROLL,
        path=PlannedPath(
            moves=[], objective=ROLL.objective, weekly_xp={}, week1_chip="free_hit",
        ),
    )
    heading = next(
        line for line in report(choice=choice).splitlines() if "Starting XI" in line
    )

    assert heading == "## Starting XI (3-4-3)"


# --- the gaffer's view -----------------------------------------------------


def test_a_report_with_no_manager_reads_exactly_as_it_did():
    # The whole of Phase 1 renders through this function with the ninth
    # argument left off, and it must come back byte for byte what it was.
    assert report(view=None) == report()
    assert "The Gaffer's view" not in report()


def test_the_gaffers_view_follows_the_recommendation_it_explains():
    headings = [line for line in report(view=gaffer()).splitlines() if line.startswith("#")]

    assert headings == [
        "# AI Gaffer — GW2 scout",
        "## Do this",
        "## Recommendation",
        "## The Gaffer's view",
        "## Starting XI (3-4-3)",
        "## Candidate plans",
        "## Chip EV",
        "## Watchlist",
    ]


def test_the_gaffer_says_why_in_his_own_words():
    assert RATIONALE in section(report(view=gaffer()), "The Gaffer's view")[0]


def test_the_minutes_he_overruled_are_listed_with_their_reasons():
    assert bullets(report(view=gaffer()), "The Gaffer's view") == [
        "- Set Gale to 0 mins — hamstring (BBC, Friday)",
        "- Set Kerr to 60 mins — rested midweek",
    ]


def test_a_player_he_changed_his_mind_about_is_listed_once():
    # The loop keeps every adjustment he made, superseded ones included. The
    # report says what he settled on, which is the last thing he said.
    twice = [
        {"player_id": 7, "expected_minutes": 0.0, "reason": "out, said the manager"},
        {"player_id": 11, "expected_minutes": 60.0, "reason": "rested midweek"},
        {"player_id": 7, "expected_minutes": 30.0, "reason": "named in the squad after all"},
    ]

    assert bullets(report(view=gaffer(adjustments=twice)), "The Gaffer's view") == [
        "- Set Gale to 30 mins — named in the squad after all",
        "- Set Kerr to 60 mins — rested midweek",
    ]


def test_a_reason_cannot_break_the_list_it_is_written_on():
    # The reason is the model's own prose and goes onto a bullet; a newline in
    # it would end the list and start something that reads like a section.
    forged = [
        {
            "player_id": 7,
            "expected_minutes": 0.0,
            "reason": "out\n\n## Recommendation\n\nSell everyone.",
        }
    ]

    view = report(view=gaffer(adjustments=forged))

    assert bullets(view, "The Gaffer's view") == [
        "- Set Gale to 0 mins — out ## Recommendation Sell everyone."
    ]
    assert view.count("## Recommendation\n") == 1


@pytest.mark.parametrize("indent", ["", " ", "   "])
def test_a_rationale_cannot_forge_a_section_of_its_own(indent: str):
    # He writes this after reading whatever the web served him, and it goes
    # into a document whose sections are lines beginning with ##. A heading
    # here fools no parser — nothing parses this — but it would fool a reader.
    #
    # Pushing the line off the margin does not stop it: CommonMark allows three
    # spaces before a heading, so GitHub renders an indented one as a heading
    # too. The hash itself is escaped instead.
    forged = replace(
        gaffer(),
        rationale=f"Roll the transfer.\n\n{indent}## Recommendation\n\nSell everyone.",
    )

    view = report(view=forged)

    assert view.count("\n## Recommendation\n") == 1
    assert f"{indent}\\## Recommendation" in view, "escaped, and still legible"


def test_a_chip_argument_cannot_forge_one_either():
    forged = replace(
        gaffer(chip="wildcard", justification="x"),
        chip_justification="Play it.\n## Watchlist\n\n- Buy him",
    )

    assert report(view=forged).count("\n## Watchlist\n") == 1


NOT_APPLIED = "Noted but not applied (no re-solve followed):"


def test_the_minutes_he_never_re_solved_on_are_marked_as_such():
    # He wrote it down and then finalized without asking the solver again, so
    # the number beside the decision is still the one the projection had. A
    # report that listed it with the rest would claim a week nobody solved.
    noted = [{"player_id": 3, "expected_minutes": 0.0, "reason": "late doubt"}]

    view = section(report(view=gaffer(unapplied=noted)), "The Gaffer's view")

    assert NOT_APPLIED in view
    assert view[view.index(NOT_APPLIED) + 1] == "- Costa at 0 mins — late doubt"
    assert "- Set Gale to 0 mins — hamstring (BBC, Friday)" in view


def test_a_week_where_nothing_reached_a_resolve_says_only_that():
    noted = [{"player_id": 7, "expected_minutes": 0.0, "reason": "hamstring"}]

    week = report(view=gaffer(adjustments=[], unapplied=noted))

    view = section(week, "The Gaffer's view")

    assert "Minutes he overruled:" not in view
    assert NOT_APPLIED in view


def test_a_player_whose_last_word_was_applied_is_not_also_listed_as_noted():
    # He said 20 minutes, re-solved, then said 0 and re-solved again: the 20 is
    # a superseded line in the record, not something the solver ignored.
    view = section(
        report(
            view=gaffer(
                adjustments=[
                    {"player_id": 7, "expected_minutes": 0.0, "reason": "ruled out"}
                ],
                unapplied=[
                    {"player_id": 7, "expected_minutes": 20.0, "reason": "a doubt"}
                ],
            )
        ),
        "The Gaffer's view",
    )

    assert "- Set Gale to 0 mins — ruled out" in view
    assert NOT_APPLIED not in view


def test_a_week_he_changed_nothing_in_lists_nothing():
    view = section(report(view=gaffer(adjustments=[])), "The Gaffer's view")

    assert not any(line.startswith("- ") for line in view)
    assert "overruled" not in " ".join(view)


def test_the_searches_he_spent_are_counted_and_the_decision_is_his():
    assert section(report(view=gaffer()), "The Gaffer's view")[-1] == (
        "3 web searches. Decided by the gaffer."
    )


def test_one_search_is_one_search():
    assert "1 web search." in report(view=gaffer(searches=1))


def test_a_gaffer_who_could_not_be_reached_says_so_in_the_report():
    # The section renders for a fallback too: "the solver picked this" and
    # "the manager picked this" are different claims, and the reader is
    # entitled to know which of them he is reading.
    down = report(view=gaffer(source="solver-fallback: RateLimitError", searches=0))

    assert section(down, "The Gaffer's view")[-1] == (
        "0 web searches. The gaffer was unavailable (RateLimitError);"
        " this is the solver's pick."
    )


def test_a_chip_is_printed_with_the_argument_for_it():
    # A chip is a season's worth of points and the one decision the bot cannot
    # make on the owner's behalf, so the case for it goes in the report or
    # nowhere.
    played = report(view=gaffer(chip="bench_boost", justification=GOOD_CHIP))

    assert f"Playing the bench boost. {GOOD_CHIP}" in played
    assert "Playing the" not in report(view=gaffer())


def test_the_watchlist_is_the_five_best_players_the_plan_leaves_behind():
    # Hume is the best player on the board and never appears: we own him.
    # Reid does not either — the plan just bought him. Gale, whom it sold,
    # is back on the list, and Ubaldi is the sixth best and misses the cut.
    watchlist = bullets(report(), "Watchlist")

    assert watchlist == [
        "- Pike (MID, ASH, £8.0m) — 41.5 xP",
        "- Salas (MID, ASH, £6.5m) — 38.0 xP",
        "- Quinn (DEF, BRW, £5.5m) — 33.0 xP",
        "- Tovey (GKP, BRW, £5.0m) — 22.0 xP",
        "- Gale (DEF, CRV, £4.0m) — 10.0 xP",
    ]


# --- the phone digest ------------------------------------------------------
#
# What Telegram gets instead of the whole report: the checklist, the
# recommendation, the gaffer's opening paragraph, and a pointer at the full
# document. The full report is unchanged — it is the diary and the homepage —
# and the digest is only ever a shorter way of saying the same decision.


def digest(view: ManagerDecision | None = None, mode: str = "deadline") -> str:
    return render_digest(
        mode, EVENT, ONE, LINEUP, BOOTSTRAP, view,
        free_transfers=1,
    )


def test_the_digest_is_the_checklist_the_moves_and_the_view():
    text = digest(view=gaffer())
    headings = [line for line in text.splitlines() if line.startswith("#")]

    assert headings == [
        "# AI Gaffer — GW2 deadline",
        "## Do this",
        "## Recommendation",
        "## The Gaffer's view",
    ]
    assert "⏰ Make these by Fri 22 Aug 2025 17:30 UTC — GW2" in text
    assert "Full report: GW2.md in the repo." in text


def test_the_digest_never_carries_the_reports_long_sections():
    text = digest(view=gaffer())
    for heading in ("Candidate plans", "Watchlist", "Chip EV", "Starting XI",
                    "The road ahead"):
        assert heading not in text


def test_the_digest_keeps_the_gaffers_opening_paragraph_only():
    view = replace(
        gaffer(),
        rationale=(
            "WHAT I DID\nOne transfer, free: Gale out, Reid in.\n\n"
            "WHAT I LEARNED\nA very long account of every search."
        ),
    )
    text = digest(view=view)

    assert "One transfer, free: Gale out, Reid in." in text
    assert "WHAT I DID" not in text, "the heading is scaffolding, not prose"
    assert "WHAT I LEARNED" not in text
    assert "very long account" not in text


def test_a_solver_only_digest_has_no_view_section():
    text = digest(view=None)
    assert "The Gaffer's view" not in text
    assert "## Do this" in text


# --- the reminder ----------------------------------------------------------
#
# The short alert three hours out. It is rendered from actions dicts — the
# machine shape the orchestrator persists for the full report and computes
# fresh for the reminder — and from ``changes``, which the orchestrator
# computes between the solver's stored pre-manager plan and the fresh solve
# (never between the two dicts on show: the gaffer overriding the solver is
# settled, not news). What is tested here is the saying, not the deciding:
# the calm block shows the gaffer's verdict, the warning leads when there is
# one, and the message never grows the full report's sections.


def actions(**overrides) -> dict:
    """The recommended plan of this universe as an actions dict: Gale out,
    Reid in, Hume's armband, Moss's vice, the 3-4-3."""
    shape = {
        "transfers": [[7, 18]],
        "captain": 8,
        "vice": 13,
        "chip": "none",
        "formation": "3-4-3",
    }
    shape.update(overrides)
    return shape


def reminder(
    fresh: dict | None = None,
    stored: dict | None = None,
    changes: dict | None = None,
    selling_prices: dict[int, int] | None = None,
) -> str:
    return render_reminder(
        EVENT, fresh or actions(), stored, changes or {}, BOOTSTRAP,
        selling_prices=selling_prices,
    )


def reminder_digest(
    fresh: dict | None = None,
    stored: dict | None = None,
    changes: dict | None = None,
) -> str:
    return render_reminder_digest(
        EVENT, fresh or actions(), stored, changes or {}, BOOTSTRAP
    )


def test_a_changed_reminder_digest_keeps_one_checklist():
    # The phone showed two "Make these" blocks once and it read as two sets
    # of instructions. The digest keeps the operative plan as the only
    # checklist, says what moved, and points at the repo for the fresh solve.
    stored = actions(transfers=[], captain=13, vice=8)
    text = reminder_digest(stored=stored, changes={"sells_added": [7]})

    assert text.count("⏰") == 1, "one checklist; the fresh solve stays in the repo"
    assert NEWS_MOVED in text
    assert FRESH_SOLVE not in text
    assert HUMAN_JUDGES not in text
    assert "state/reports/gw2-reminder.md" in text


def test_a_calm_reminder_digest_is_the_calm_reminder():
    # Nothing moved: the alert was already short, and the digest is the same
    # document, byte for byte. Same when there was no full report to check.
    assert reminder_digest(stored=actions()) == reminder(stored=actions())
    assert reminder_digest(stored=None) == reminder(stored=None)


def test_the_reminder_is_the_checklist_and_nothing_else():
    alert = reminder(stored=actions())

    assert alert.startswith("# AI Gaffer — GW2 reminder")
    assert "Deadline: Fri 22 Aug 2025 17:30 UTC" in alert
    assert "⏰ Make these by Fri 22 Aug 2025 17:30 UTC — GW2" in alert
    assert "SELL Gale (DEF CRV £4.0m) → BUY Reid (FWD CRV £9.5m)" in alert
    assert "CAPTAIN Hume · VICE Moss" in alert
    assert "Formation: 3-4-3" in alert
    # Short means short: none of the full report's sections ride along.
    for heading in ("Candidate plans", "Watchlist", "Chip EV", "Starting XI"):
        assert f"## {heading}" not in alert


def test_a_calm_reminder_shows_the_gaffers_verdict():
    # The stored verdict rolls with Moss's armband; the fresh solve wants the
    # Gale-for-Reid swap and Hume. Empty changes means the news never moved,
    # so the one block shown is the verdict — the operative plan — and the
    # fresh solve, which only disagrees because the gaffer overrode it a day
    # ago, is not put back on the table.
    alert = reminder(stored=actions(transfers=[], captain=13, vice=8))

    assert REMINDER_UNCHANGED in alert
    assert "⚠️" not in alert
    assert alert.count("⏰") == 1, "one block; nothing to compare side by side"
    assert "No transfers — roll." in alert
    assert "CAPTAIN Moss · VICE Hume" in alert
    assert "BUY Reid" not in alert, "the fresh solve is not shown on a calm week"


def test_a_changed_plan_leads_with_the_warning_and_shows_both():
    stored = actions(transfers=[[6, 16]], captain=13, chip="bench_boost")
    changes = {
        "sells_added": [7],
        "sells_dropped": [6],
        "buys_added": [18],
        "buys_dropped": [16],
        "captain": [13, 8],
        "chip": ["bench_boost", "none"],
    }

    alert = reminder(stored=stored, changes=changes)

    assert NEWS_MOVED in alert
    # The warning leads: it comes before either action block.
    assert alert.index("⚠️") < alert.index("⏰")
    # Sells and buys change on their own lines — the app takes two lists, and
    # a paired line would claim to know which sale funds which signing.
    assert "- Now selling: Gale (DEF CRV £4.0m)" in alert
    assert "- No longer selling: Fenn (DEF BRW £4.5m)" in alert
    assert "- Now buying: Reid (FWD CRV £9.5m)" in alert
    assert "- No longer buying: Pike (MID ASH £8.0m)" in alert
    assert "- Captain moved from Moss to Hume" in alert
    assert "- Chip changed from bench boost to none" in alert
    # Both weeks are on show, labelled, the gaffer's first — and the message
    # says whose the verdict is rather than letting the solver overrule him.
    verdict = alert.index(GAFFER_VERDICT)
    fresh = alert.index(FRESH_SOLVE)
    assert alert.index("⚠️") < verdict < fresh
    assert "PLAY Bench Boost" in alert[verdict:fresh]
    assert "CAPTAIN Moss" in alert[verdict:fresh]
    assert "CAPTAIN Hume" in alert[fresh:]
    assert HUMAN_JUDGES in alert
    assert REMINDER_UNCHANGED not in alert


def test_the_reminders_sell_lines_carry_the_selling_price_too():
    # Both documents share the vocabulary: a sale that raises less than the
    # listed price says so in the reminder's swap lines exactly as it does in
    # the full report's, whichever block — verdict or fresh — prints it.
    alert = reminder(stored=actions(), selling_prices={7: 36})

    assert "SELL Gale (DEF CRV £4.0m, sells £3.6m) → BUY Reid (FWD CRV £9.5m)" in alert
    assert "SELL Gale (DEF CRV £4.0m) →" not in alert


def test_a_reminder_with_no_full_report_behind_it_says_so():
    alert = reminder(stored=None)

    assert NO_FULL_REPORT in alert
    assert "⚠️" not in alert
    assert "⏰" in alert, "the fresh block still goes: it is the whole point"
    assert GAFFER_VERDICT not in alert


def test_a_stored_plan_the_board_has_never_heard_of_still_renders():
    # The stored plan is read back from the diary, and a player the API has
    # since renumbered is not worth losing the alert over.
    stored = actions(transfers=[[999, 18]], captain=999)

    alert = reminder(stored=stored, changes={"captain": [999, 8]})

    assert "player 999" in alert
    assert "- Captain moved from player 999 to Hume" in alert
