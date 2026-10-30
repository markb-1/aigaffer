"""The shared opponent lookup: who each club plays, and at which end.

It moved out of the detailed snapshot so the manager's briefing can read the
same schedule. The snapshot's own tests (tests/test_detailed_snapshot.py)
still pin its output; these pin the function on its own.
"""

from aigaffer.data.models import Bootstrap, Fixture
from aigaffer.data.schedule import games_text, opponents, phone_games
from tests.fixtures import BOOTSTRAP_JSON

# Clubs 1, 2, 3 are ASH, BRW, CRV.
CLUBS = {team.id: team for team in Bootstrap.model_validate(BOOTSTRAP_JSON).teams}


def test_the_home_side_reads_h_and_the_away_side_reads_a():
    # Fixture: club 1 (ASH) at home to club 2 (BRW) in GW5.
    found = opponents([Fixture(id=1, event=5, team_h=1, team_a=2)], CLUBS, {5})

    assert found[(5, 1)] == [("BRW", "h")]
    assert found[(5, 2)] == [("ASH", "a")]


def test_a_double_reads_in_kickoff_order_and_a_stray_fixture_is_dropped():
    fixtures = [
        # Played second although its id is lower: kick-off orders, not id.
        Fixture(id=1, event=5, team_h=1, team_a=3, kickoff_time="2025-09-20T15:00:00Z"),
        Fixture(id=2, event=5, team_h=2, team_a=1, kickoff_time="2025-09-17T19:00:00Z"),
        Fixture(id=3, event=None, team_h=1, team_a=2),
        Fixture(id=4, event=9, team_h=1, team_a=2),
    ]

    found = opponents(fixtures, CLUBS, {5})

    # ASH: away at BRW first, then home to CRV. Unscheduled and out-of-window
    # fixtures are nobody's game.
    assert found[(5, 1)] == [("BRW", "a"), ("CRV", "h")]
    assert (9, 1) not in found


def test_games_text_is_the_house_cell_without_the_xp():
    assert games_text([("LEE", "h")]) == "LEE(h)"
    assert games_text([("LEE", "h"), ("WOL", "a")]) == "LEE(h)+wol(a)"
    assert games_text([]) == "—"


def test_phone_games_home_upper_away_lower():
    assert phone_games([("BHA", "h")]) == "BHA"
    # A snapshot holds JSON lists, not tuples.
    assert phone_games([["LEE", "a"]]) == "lee"


def test_phone_games_double_and_blank():
    assert phone_games([("BHA", "h"), ("ARS", "a")]) == "BHA+ars"
    assert phone_games([]) == "—"
