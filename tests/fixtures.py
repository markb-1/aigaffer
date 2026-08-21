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
against the live endpoint), including the per-90 stats arriving as JSON
numbers and ``defensive_contribution`` as a season-total integer.
"""

from typing import Any

import httpx


def fake_fpl_transport(routes: dict[str, Any]) -> httpx.MockTransport:
    """Serve ``routes`` (URL path -> JSON payload); 404 for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
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
        "expected_goals_per_90": 0.0,
        "expected_assists_per_90": 0.01,
        "saves_per_90": 3.0,
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
        "expected_goals_per_90": 0.0,
        "expected_assists_per_90": 0.0,
        "saves_per_90": 4.4,
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
        "expected_goals_per_90": 0.12,
        "expected_assists_per_90": 0.18,
        "saves_per_90": 0.0,
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
        "expected_goals_per_90": 0.05,
        "expected_assists_per_90": 0.08,
        "saves_per_90": 0.0,
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
        "expected_goals_per_90": 0.55,
        "expected_assists_per_90": 0.4,
        "saves_per_90": 0.0,
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
        "expected_goals_per_90": 0.25,
        "expected_assists_per_90": 0.3,
        "saves_per_90": 0.0,
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
        "expected_goals_per_90": 0.62,
        "expected_assists_per_90": 0.22,
        "saves_per_90": 0.0,
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
        "expected_goals_per_90": 0.3,
        "expected_assists_per_90": 0.1,
        "saves_per_90": 0.0,
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
FIXTURES_JSON = [
    {"id": 1, "event": 1, "team_h": 1, "team_a": 3, "finished": True},
    {"id": 2, "event": 2, "team_h": 1, "team_a": 2, "finished": False},
    {"id": 3, "event": 2, "team_h": 2, "team_a": 3, "finished": False},
    {"id": 4, "event": None, "team_h": 3, "team_a": 1, "finished": False},
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

ELEMENT_SUMMARY_JSON = {
    "fixtures": [{"id": 2, "event": 2, "is_home": True}],
    "history": [
        {"element": 5, "fixture": 1, "round": 1, "minutes": 90, "total_points": 12, "bonus": 3},
    ],
    "history_past": [{"season_name": "2024/25", "total_points": 210}],
}
