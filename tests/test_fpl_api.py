import copy

import httpx
import pytest

from aigaffer.data.fpl_api import FplClient
from tests.fixtures import (
    BOOTSTRAP_JSON,
    ELEMENT_SUMMARY_JSON,
    FIXTURES_JSON,
    HISTORY_JSON,
    PICKS_JSON,
    TRANSFERS_JSON,
    fake_fpl_transport,
)


def make_client(routes):
    transport = fake_fpl_transport(routes)
    return FplClient(http=httpx.Client(transport=transport))


def test_bootstrap_parses_events():
    c = make_client({"/api/bootstrap-static/": BOOTSTRAP_JSON})
    bs = c.bootstrap()
    assert [e.id for e in bs.events] == [1, 2]
    assert bs.next_event().id == 2
    assert bs.current_event().id == 1
    assert bs.current_event().finished is True
    assert bs.next_event().deadline_time.isoformat() == "2025-08-22T17:30:00+00:00"


def test_bootstrap_parses_teams():
    c = make_client({"/api/bootstrap-static/": BOOTSTRAP_JSON})
    ashford = c.bootstrap().teams[0]
    assert (ashford.id, ashford.name, ashford.short_name) == (1, "Ashford", "ASH")
    assert ashford.strength_attack_home == 1300
    assert ashford.strength_attack_away == 1250
    assert ashford.strength_defence_home == 1280
    assert ashford.strength_defence_away == 1240


def test_bootstrap_parses_players():
    c = make_client({"/api/bootstrap-static/": BOOTSTRAP_JSON})
    players = {p.id: p for p in c.bootstrap().elements}
    assert len(players) == 8
    assert {p.element_type for p in players.values()} == {1, 2, 3, 4}

    keeper = players[1]
    assert keeper.web_name == "Alvez"
    assert keeper.team == 1
    assert keeper.now_cost == 55
    assert keeper.status == "a"
    assert keeper.chance_of_playing_next_round is None
    assert (keeper.minutes, keeper.starts, keeper.total_points) == (900, 10, 45)
    assert (keeper.bonus, keeper.saves) == (5, 30)
    assert keeper.saves_per_90 == pytest.approx(3.0)

    mid = players[5]
    assert mid.expected_goals_per_90 == pytest.approx(0.55)
    assert mid.expected_assists_per_90 == pytest.approx(0.40)
    assert mid.defensive_contribution == pytest.approx(8.0)

    doubtful = players[4]
    assert doubtful.status == "d"
    assert doubtful.chance_of_playing_next_round == 75


def test_bootstrap_ignores_unknown_keys():
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    payload["brand_new_top_level_key"] = {"anything": 1}
    payload["elements"][0]["brand_new_element_key"] = "ignore me"
    c = make_client({"/api/bootstrap-static/": payload})
    assert c.bootstrap().elements[0].web_name == "Alvez"


def test_decimal_stats_sent_as_strings_are_coerced():
    # FPL serves several decimal stats as strings (e.g. expected_goals);
    # the per-90 family is numeric today but must not be relied on.
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    element = next(e for e in payload["elements"] if e["id"] == 5)
    element["expected_goals_per_90"] = "0.55"
    element["defensive_contribution"] = "8"
    c = make_client({"/api/bootstrap-static/": payload})
    mid = next(p for p in c.bootstrap().elements if p.id == 5)
    assert mid.expected_goals_per_90 == pytest.approx(0.55)
    assert mid.defensive_contribution == pytest.approx(8.0)


def test_missing_optional_stats_default_to_zero():
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    element = next(e for e in payload["elements"] if e["id"] == 1)
    for key in (
        "expected_goals_per_90",
        "expected_assists_per_90",
        "saves_per_90",
        "defensive_contribution",
    ):
        del element[key]
    c = make_client({"/api/bootstrap-static/": payload})
    keeper = next(p for p in c.bootstrap().elements if p.id == 1)
    assert keeper.expected_goals_per_90 == 0.0
    assert keeper.expected_assists_per_90 == 0.0
    assert keeper.saves_per_90 == 0.0
    assert keeper.defensive_contribution == 0.0


def test_event_helpers_return_none_when_unflagged():
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    for event in payload["events"]:
        event["is_next"] = False
        event["is_current"] = False
    c = make_client({"/api/bootstrap-static/": payload})
    bs = c.bootstrap()
    assert bs.next_event() is None
    assert bs.current_event() is None


def test_fixtures_parses_including_unscheduled():
    c = make_client({"/api/fixtures/": FIXTURES_JSON})
    fixtures = c.fixtures()
    assert [(f.id, f.event, f.team_h, f.team_a) for f in fixtures] == [
        (1, 1, 1, 3),
        (2, 2, 1, 2),
        (3, 2, 2, 3),
        (4, None, 3, 1),
    ]


def test_picks_maps_bank():
    c = make_client({"/api/entry/99/event/1/picks/": PICKS_JSON})
    squad = c.picks(99, 1)
    assert squad.bank == PICKS_JSON["entry_history"]["bank"]
    assert squad.event == 1
    assert len(squad.player_ids) == len(PICKS_JSON["picks"])
    assert squad.player_ids == [1, 3, 4, 5, 6, 7, 2, 8]


def test_picks_parses_captaincy():
    c = make_client({"/api/entry/99/event/1/picks/": PICKS_JSON})
    picks = {p.element: p for p in c.picks(99, 1).picks}
    assert picks[5].is_captain is True
    assert picks[5].is_vice_captain is False
    assert picks[5].position == 4
    assert picks[7].is_vice_captain is True
    assert picks[1].is_captain is False


def test_transfers_returns_raw_dicts():
    c = make_client({"/api/entry/99/transfers/": TRANSFERS_JSON})
    transfers = c.transfers(99)
    assert transfers == TRANSFERS_JSON
    assert transfers[0]["event"] == 1


def test_chips_used_reads_chips_key():
    c = make_client({"/api/entry/99/history/": HISTORY_JSON})
    chips = c.chips_used(99)
    assert [(chip["name"], chip["event"]) for chip in chips] == [("wildcard", 1)]


def test_chips_used_when_none_played():
    payload = copy.deepcopy(HISTORY_JSON)
    del payload["chips"]
    c = make_client({"/api/entry/99/history/": payload})
    assert c.chips_used(99) == []


def test_element_history_parses():
    c = make_client({"/api/element-summary/5/": ELEMENT_SUMMARY_JSON})
    history = c.element_history(5)
    assert len(history) == 1
    assert (history[0].round, history[0].minutes, history[0].total_points) == (1, 90, 12)


def test_requests_hit_the_real_fpl_base_url():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=BOOTSTRAP_JSON)

    c = FplClient(http=httpx.Client(transport=httpx.MockTransport(handler)))
    c.bootstrap()
    assert seen == ["https://fantasy.premierleague.com/api/bootstrap-static/"]


def test_http_error_raises():
    c = make_client({})
    with pytest.raises(httpx.HTTPStatusError):
        c.bootstrap()


def test_http_error_raises_for_picks():
    c = make_client({})
    with pytest.raises(httpx.HTTPStatusError):
        c.picks(99, 1)
