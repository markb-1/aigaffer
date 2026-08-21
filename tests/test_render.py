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

from datetime import datetime, timedelta, timezone

from aigaffer.data.models import Bootstrap, Event, Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.report.render import render_report
from aigaffer.solver.lineup import ChipEvs, Lineup
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

# The renderer reads ``total`` alone, so ``per_gw`` is left empty.
XP = {
    pid: PlayerProjection(player_id=pid, per_gw={}, total=points)
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
PLANS = [ONE, TWO, ROLL]


def report(choice: Plan = ONE, event: Event = EVENT) -> str:
    """The report as Task 12 will ask for it."""
    return render_report("scout", event, PLANS, choice, LINEUP, CHIPS, BOOTSTRAP, XP)


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


def test_the_chip_panel_signs_every_number():
    assert bullets(report(), "Chip EV") == [
        "- Bench boost: +3.2",
        "- Triple captain: +8.4",
        "- Free hit: -1.5",
        "- Wildcard: +12.0",
    ]


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
