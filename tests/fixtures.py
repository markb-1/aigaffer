"""Shared offline test fixtures: a tiny, self-consistent FPL universe.

Every test in the suite talks to the FPL API through ``fake_fpl_transport``
so nothing ever hits the network.

The universe:

* 2 events — GW1 (finished, current) and GW2 (next).
* 3 teams — Ashford (strongest), Brightwood (mid), Cravenside (weakest),
  each with distinct home/away attack and defence strengths.
* 8 players covering all four positions, with distinct prices and stats:

  ==  ============  ====  ===  ====  ======================================
  id  web_name      team  pos  cost  notes
  ==  ============  ====  ===  ====  ======================================
  1   Alvez         1     GK   55    nailed keeper
  2   Byrne         2     GK   45    rotated keeper, higher saves per 90
  3   Costa         1     DEF  60    attacking, high defensive contribution
  4   Dodd          3     DEF  40    doubtful, 75% chance of playing
  5   Ferrer        1     MID  125   premium attacker
  6   Grant         2     MID  75    mid-price all-rounder
  7   Haas          2     FWD  105   premium striker
  8   Ito           3     FWD  50    injured, no chance of playing set
  ==  ============  ====  ===  ====  ======================================

Field names, types and extra keys mirror a real bootstrap payload (checked
against the live endpoint): ``expected_goals`` and ``expected_assists`` are
season totals arriving as strings, ``saves`` and ``defensive_contribution``
season totals as integers. Every one of them is consistent with the player's
``minutes`` — Ferrer's 5.81 xG over 950 minutes is the 0.55 a game he is
meant to be — because the model divides them itself and a fixture that
disagreed with itself would pin the wrong number.

Eight players cannot make a legal fifteen, so the end-to-end pipeline test
runs on a second universe — the ``PIPELINE_*`` payloads — which is this one
plus thirteen more players and three more clubs. It is kept separate rather
than folded in because the unit tests above are pinned to the small universe:
its league averages, its player count, its three clubs.
"""

from typing import Any

import httpx


def fake_fpl_transport(
    routes: dict[str, Any], statuses: dict[str, int] | None = None
) -> httpx.MockTransport:
    """Serve ``routes`` (URL path -> JSON payload); 404 for anything else.

    ``statuses`` maps a path to the status code it answers with instead,
    whatever ``routes`` holds for it: how a test asks for the 429 or the 503
    the live API hands out under load.
    """
    failing = statuses or {}

    def handler(request: httpx.Request) -> httpx.Response:
        status = failing.get(request.url.path)
        if status is not None:
            return httpx.Response(status, json={"detail": "no"})
        payload = routes.get(request.url.path)
        if payload is None:
            return httpx.Response(404, json={"detail": "not found"})
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


EVENTS_JSON = [
    {
        "id": 1,
        "name": "Gameweek 1",
        "deadline_time": "2025-08-15T17:30:00Z",
        "finished": True,
        "is_previous": False,
        "is_current": True,
        "is_next": False,
        "average_entry_score": 57,
    },
    {
        "id": 2,
        "name": "Gameweek 2",
        "deadline_time": "2025-08-22T17:30:00Z",
        "finished": False,
        "is_previous": False,
        "is_current": False,
        "is_next": True,
        "average_entry_score": 0,
    },
]

TEAMS_JSON = [
    {
        "id": 1,
        "name": "Ashford",
        "short_name": "ASH",
        "strength": None,
        "strength_overall_home": 5,
        "strength_overall_away": 5,
        "strength_attack_home": 1300,
        "strength_attack_away": 1250,
        "strength_defence_home": 1280,
        "strength_defence_away": 1240,
    },
    {
        "id": 2,
        "name": "Brightwood",
        "short_name": "BRW",
        "strength": None,
        "strength_overall_home": 3,
        "strength_overall_away": 3,
        "strength_attack_home": 1150,
        "strength_attack_away": 1100,
        "strength_defence_home": 1120,
        "strength_defence_away": 1080,
    },
    {
        "id": 3,
        "name": "Cravenside",
        "short_name": "CRV",
        "strength": None,
        "strength_overall_home": 2,
        "strength_overall_away": 2,
        "strength_attack_home": 1000,
        "strength_attack_away": 980,
        "strength_defence_home": 1020,
        "strength_defence_away": 960,
    },
]

ELEMENTS_JSON = [
    {
        "id": 1,
        "web_name": "Alvez",
        "first_name": "Ander",
        "team": 1,
        "element_type": 1,
        "now_cost": 55,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 900,
        "starts": 10,
        "total_points": 45,
        "bonus": 5,
        "saves": 30,
        "expected_goals": "0.00",
        "expected_assists": "0.10",
        "defensive_contribution": 0,
    },
    {
        "id": 2,
        "web_name": "Byrne",
        "first_name": "Barry",
        "team": 2,
        "element_type": 1,
        "now_cost": 45,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 450,
        "starts": 5,
        "total_points": 20,
        "bonus": 1,
        "saves": 22,
        "expected_goals": "0.00",
        "expected_assists": "0.00",
        "defensive_contribution": 0,
    },
    {
        "id": 3,
        "web_name": "Costa",
        "first_name": "Caio",
        "team": 1,
        "element_type": 2,
        "now_cost": 60,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 880,
        "starts": 10,
        "total_points": 52,
        "bonus": 6,
        "saves": 0,
        "expected_goals": "1.17",
        "expected_assists": "1.76",
        "defensive_contribution": 30,
    },
    {
        "id": 4,
        "web_name": "Dodd",
        "first_name": "Dan",
        "team": 3,
        "element_type": 2,
        "now_cost": 40,
        "status": "d",
        "chance_of_playing_next_round": 75,
        "minutes": 700,
        "starts": 8,
        "total_points": 30,
        "bonus": 2,
        "saves": 0,
        "expected_goals": "0.39",
        "expected_assists": "0.62",
        "defensive_contribution": 24,
    },
    {
        "id": 5,
        "web_name": "Ferrer",
        "first_name": "Felix",
        "team": 1,
        "element_type": 3,
        "now_cost": 125,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 950,
        "starts": 11,
        "total_points": 90,
        "bonus": 12,
        "saves": 0,
        "expected_goals": "5.81",
        "expected_assists": "4.22",
        "defensive_contribution": 8,
    },
    {
        "id": 6,
        "web_name": "Grant",
        "first_name": "Gus",
        "team": 2,
        "element_type": 3,
        "now_cost": 75,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 600,
        "starts": 7,
        "total_points": 40,
        "bonus": 3,
        "saves": 0,
        "expected_goals": "1.67",
        "expected_assists": "2.00",
        "defensive_contribution": 16,
    },
    {
        "id": 7,
        "web_name": "Haas",
        "first_name": "Hugo",
        "team": 2,
        "element_type": 4,
        "now_cost": 105,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 870,
        "starts": 10,
        "total_points": 78,
        "bonus": 9,
        "saves": 0,
        "expected_goals": "5.99",
        "expected_assists": "2.13",
        "defensive_contribution": 4,
    },
    {
        "id": 8,
        "web_name": "Ito",
        "first_name": "Iku",
        "team": 3,
        "element_type": 4,
        "now_cost": 50,
        "status": "i",
        "chance_of_playing_next_round": None,
        "minutes": 200,
        "starts": 2,
        "total_points": 10,
        "bonus": 0,
        "saves": 0,
        "expected_goals": "0.67",
        "expected_assists": "0.22",
        "defensive_contribution": 2,
    },
]

BOOTSTRAP_JSON = {
    "events": EVENTS_JSON,
    "teams": TEAMS_JSON,
    "elements": ELEMENTS_JSON,
    # Keys the models do not declare — present to prove they are ignored.
    "total_players": 11_000_000,
    "element_types": [{"id": 1, "singular_name_short": "GKP"}],
    "phases": [{"id": 1, "name": "Overall"}],
}

# GW1: Ashford v Cravenside. GW2: Ashford v Brightwood and Brightwood v
# Cravenside, so Brightwood has a double and Ashford/Cravenside a single.
# Fixture 4 is postponed (``event`` is null).
#
# ``finished_provisional`` is the flag the live payload raises at full time,
# against ``finished``, which waits for the round's data check hours later.
# Every fixture here is one or the other, never provisional-only: the tests
# that turn on the gap between them build their own payloads.
FIXTURES_JSON = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 3, "finished": True,
     "finished_provisional": True},
    {"id": 2, "event": 2, "team_h": 1, "team_a": 2, "finished": False,
     "finished_provisional": False},
    {"id": 3, "event": 2, "team_h": 2, "team_a": 3, "finished": False,
     "finished_provisional": False},
    {"id": 4, "event": None, "team_h": 3, "team_a": 1, "finished": False,
     "finished_provisional": False},
]

# The whole 8-player universe as one squad, captained by Ferrer (5) with
# Haas (7) as vice.
PICKS_JSON = {
    "active_chip": None,
    "automatic_subs": [],
    "entry_history": {
        "event": 1,
        "points": 57,
        "bank": 13,
        "value": 1004,
        "event_transfers": 1,
        "event_transfers_cost": 0,
    },
    "picks": [
        {"element": 1, "position": 1, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 3, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 4, "position": 3, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 5, "position": 4, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
        {"element": 6, "position": 5, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 7, "position": 6, "multiplier": 1, "is_captain": False, "is_vice_captain": True},
        {"element": 2, "position": 12, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
        {"element": 8, "position": 13, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
    ],
}

TRANSFERS_JSON = [
    {"element_in": 5, "element_in_cost": 125, "element_out": 8, "element_out_cost": 50, "event": 1},
]

HISTORY_JSON = {
    "current": [{"event": 1, "points": 57, "total_points": 57, "rank": 1_000_000}],
    "past": [],
    "chips": [{"name": "wildcard", "time": "2025-08-15T10:00:00Z", "event": 1}],
}

# ``history_past`` carries a full season's totals, one row a season, oldest
# first — checked against the live endpoint on 2026-08-22. 3230 minutes over a
# 38-gameweek season is a prior of 85.0, which is below the 90 the round-1 row
# above already proves, so every player in this universe is projected off his
# history exactly as he was before the prior existed.
ELEMENT_SUMMARY_JSON = {
    "fixtures": [{"id": 2, "event": 2, "is_home": True}],
    "history": [
        {"element": 5, "fixture": 1, "round": 1, "minutes": 90, "total_points": 12, "bonus": 3},
    ],
    "history_past": [
        {
            "season_name": "2024/25",
            "element_code": 154561,
            "start_cost": 120,
            "end_cost": 125,
            "total_points": 210,
            "minutes": 3230,
            "starts": 36,
            "goals_scored": 18,
            "assists": 11,
            "bonus": 28,
            "bps": 720,
            "expected_goals": "16.40",
            "expected_assists": "9.10",
        }
    ],
}


# ---------------------------------------------------------------------------
# The pipeline universe: the eight players above plus enough of a league to
# field a legal fifteen — 21 players over 6 clubs, two keepers, five
# defenders, five midfielders and three forwards to a squad, at most three
# from any one club.
# ---------------------------------------------------------------------------


def _element(
    pid: int,
    name: str,
    team: int,
    element_type: int,
    now_cost: int,
    *,
    minutes: int,
    starts: int,
    total_points: int,
    bonus: int,
    xg: float = 0.0,
    xa: float = 0.0,
    saves: int = 0,
    defcon: int = 0,
) -> dict:
    """An available bootstrap element, with every key the real payload has.

    The eight players above are written out longhand because the tests that
    read them read one field at a time; these thirteen exist to be counted,
    priced and picked, so they are built from a table instead. ``xg`` and
    ``xa`` are season totals over ``minutes``, and go out as the strings the
    live payload sends.
    """
    return {
        "id": pid,
        "web_name": name,
        "first_name": name,
        "team": team,
        "element_type": element_type,
        "now_cost": now_cost,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": minutes,
        "starts": starts,
        "total_points": total_points,
        "bonus": bonus,
        "saves": saves,
        "expected_goals": f"{xg:.2f}",
        "expected_assists": f"{xa:.2f}",
        "defensive_contribution": defcon,
    }


PIPELINE_TEAMS_JSON = TEAMS_JSON + [
    {
        "id": 4,
        "name": "Dunmore",
        "short_name": "DUN",
        "strength": None,
        "strength_overall_home": 4,
        "strength_overall_away": 4,
        "strength_attack_home": 1200,
        "strength_attack_away": 1160,
        "strength_defence_home": 1180,
        "strength_defence_away": 1140,
    },
    {
        "id": 5,
        "name": "Eastvale",
        "short_name": "EAS",
        "strength": None,
        "strength_overall_home": 3,
        "strength_overall_away": 3,
        "strength_attack_home": 1120,
        "strength_attack_away": 1080,
        "strength_defence_home": 1100,
        "strength_defence_away": 1060,
    },
    {
        "id": 6,
        "name": "Fairhaven",
        "short_name": "FAI",
        "strength": None,
        "strength_overall_home": 2,
        "strength_overall_away": 2,
        "strength_attack_home": 1040,
        "strength_attack_away": 1010,
        "strength_defence_home": 1060,
        "strength_defence_away": 1000,
    },
]

# Cravenside gains three fit players (its two above are a doubt and an injury),
# and the three new clubs bring three or four each.
PIPELINE_ELEMENTS_JSON = ELEMENTS_JSON + [
    _element(9, "Jarvis", 3, 1, 50, minutes=810, starts=9, total_points=38, bonus=3,
             saves=26),
    _element(10, "Kelly", 3, 2, 45, minutes=810, starts=9, total_points=34, bonus=2,
             xg=0.36, xa=0.81, defcon=26),
    _element(11, "Lozano", 3, 3, 70, minutes=760, starts=9, total_points=44, bonus=4,
             xg=1.86, xa=2.20, defcon=14),
    _element(12, "Meier", 4, 2, 55, minutes=900, starts=10, total_points=41, bonus=4,
             xg=0.90, xa=1.10, defcon=33),
    _element(13, "Novak", 4, 2, 40, minutes=700, starts=8, total_points=26, bonus=1,
             xg=0.23, xa=0.39, defcon=22),
    _element(14, "Ozturk", 4, 3, 85, minutes=880, starts=10, total_points=58, bonus=7,
             xg=3.42, xa=2.74, defcon=10),
    _element(15, "Pryce", 4, 4, 65, minutes=640, starts=7, total_points=35, bonus=3,
             xg=2.70, xa=0.85, defcon=3),
    _element(16, "Quill", 5, 2, 50, minutes=850, starts=10, total_points=37, bonus=3,
             xg=0.57, xa=0.94, defcon=28),
    _element(17, "Reyes", 5, 3, 95, minutes=900, starts=10, total_points=66, bonus=8,
             xg=4.50, xa=3.30, defcon=9),
    _element(18, "Sarr", 5, 4, 60, minutes=590, starts=6, total_points=31, bonus=2,
             xg=2.23, xa=0.98, defcon=2),
    _element(19, "Tandy", 6, 2, 42, minutes=780, starts=9, total_points=29, bonus=2,
             xg=0.17, xa=0.52, defcon=24),
    _element(20, "Voss", 6, 3, 65, minutes=700, starts=8, total_points=39, bonus=3,
             xg=1.56, xa=1.87, defcon=12),
    _element(21, "Wynne", 6, 4, 55, minutes=520, starts=6, total_points=27, bonus=2,
             xg=1.73, xa=0.58, defcon=2),
]

PIPELINE_BOOTSTRAP_JSON = {
    "events": EVENTS_JSON,
    "teams": PIPELINE_TEAMS_JSON,
    "elements": PIPELINE_ELEMENTS_JSON,
}

# Every club plays once in GW1, GW2 and GW3, so no side is projected at
# nothing for want of a fixture. Fixture 10 is postponed (``event`` is null).
PIPELINE_FIXTURES_JSON = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 3, "finished": True,
     "finished_provisional": True},
    {"id": 2, "event": 1, "team_h": 2, "team_a": 4, "finished": True,
     "finished_provisional": True},
    {"id": 3, "event": 1, "team_h": 5, "team_a": 6, "finished": True,
     "finished_provisional": True},
    {"id": 4, "event": 2, "team_h": 1, "team_a": 2, "finished": False,
     "finished_provisional": False},
    {"id": 5, "event": 2, "team_h": 3, "team_a": 5, "finished": False,
     "finished_provisional": False},
    {"id": 6, "event": 2, "team_h": 4, "team_a": 6, "finished": False,
     "finished_provisional": False},
    {"id": 7, "event": 3, "team_h": 2, "team_a": 1, "finished": False,
     "finished_provisional": False},
    {"id": 8, "event": 3, "team_h": 5, "team_a": 3, "finished": False,
     "finished_provisional": False},
    {"id": 9, "event": 3, "team_h": 6, "team_a": 4, "finished": False,
     "finished_provisional": False},
    {"id": 10, "event": None, "team_h": 3, "team_a": 6, "finished": False,
     "finished_provisional": False},
]

# A legal fifteen worth £97.2m with £2.8m in the bank: 4-4-2, Ferrer captain
# and Haas vice, the doubtful Dodd and the injured Ito on the bench.
PICKS_15_JSON = {
    "active_chip": None,
    "automatic_subs": [],
    "entry_history": {
        "event": 1,
        "points": 61,
        "bank": 28,
        "value": 1000,
        "event_transfers": 1,
        "event_transfers_cost": 0,
    },
    "picks": [
        {"element": 1, "position": 1, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 3, "position": 2, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 13, "position": 3, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 16, "position": 4, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 19, "position": 5, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 5, "position": 6, "multiplier": 2, "is_captain": True, "is_vice_captain": False},
        {"element": 6, "position": 7, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 11, "position": 8, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 14, "position": 9, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 7, "position": 10, "multiplier": 1, "is_captain": False, "is_vice_captain": True},
        {"element": 15, "position": 11, "multiplier": 1, "is_captain": False, "is_vice_captain": False},
        {"element": 2, "position": 12, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
        {"element": 4, "position": 13, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
        {"element": 20, "position": 14, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
        {"element": 8, "position": 15, "multiplier": 0, "is_captain": False, "is_vice_captain": False},
    ],
}
