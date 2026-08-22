import copy

import httpx
import pytest

from aigaffer.data.fpl_api import (
    RETRY_BACKOFF,
    USER_AGENT,
    FplClient,
    default_http_client,
)
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

    mid = players[5]
    assert mid.expected_goals == pytest.approx(5.81)
    assert mid.expected_assists == pytest.approx(4.22)
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
    # FPL serves the expected-goals family as strings, which is how the
    # fixtures carry them; the rest are numbers today but must not be relied
    # on to stay that way. Either arrives as a float on the model.
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    element = next(e for e in payload["elements"] if e["id"] == 5)
    assert isinstance(element["expected_goals"], str)
    element["expected_assists"] = 4.22  # a number where a string was
    element["defensive_contribution"] = "8"  # and a string where a number was
    c = make_client({"/api/bootstrap-static/": payload})
    mid = next(p for p in c.bootstrap().elements if p.id == 5)
    assert mid.expected_goals == pytest.approx(5.81)
    assert mid.expected_assists == pytest.approx(4.22)
    assert mid.defensive_contribution == pytest.approx(8.0)


def test_missing_optional_stats_default_to_zero():
    payload = copy.deepcopy(BOOTSTRAP_JSON)
    element = next(e for e in payload["elements"] if e["id"] == 1)
    for key in ("expected_goals", "expected_assists", "defensive_contribution"):
        del element[key]
    c = make_client({"/api/bootstrap-static/": payload})
    keeper = next(p for p in c.bootstrap().elements if p.id == 1)
    assert keeper.expected_goals == 0.0
    assert keeper.expected_assists == 0.0
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


def test_fixtures_carry_whether_the_match_has_been_played():
    # Both flags, because the API sets them hours apart: ``finished_provisional``
    # goes up at full time and ``finished`` only once the round's data is
    # checked. Fixture 1 is the played one in this universe.
    c = make_client({"/api/fixtures/": FIXTURES_JSON})
    fixtures = {f.id: f for f in c.fixtures()}
    assert fixtures[1].finished is True
    assert fixtures[1].played is True
    assert fixtures[2].finished is False
    assert fixtures[2].played is False


def test_a_fixture_finished_only_provisionally_counts_as_played():
    # The live shape on 2026-08-22: a match kicked off the previous evening,
    # ninety minutes on the clock and its minutes already in element-summary,
    # still ``finished: false`` seventeen hours later.
    c = make_client(
        {
            "/api/fixtures/": [
                {
                    "id": 1,
                    "event": 1,
                    "team_h": 1,
                    "team_a": 3,
                    "finished": False,
                    "finished_provisional": True,
                    "started": True,
                    "minutes": 90,
                }
            ]
        }
    )
    assert c.fixtures()[0].played is True


def test_a_fixture_payload_with_neither_flag_has_not_been_played():
    # Both default to False: an older payload, or a field the API renames, must
    # never read as a match that has happened.
    c = make_client({"/api/fixtures/": [{"id": 1, "event": 1, "team_h": 1, "team_a": 3}]})
    assert c.fixtures()[0].played is False


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
    # The fixture the row belongs to, which is what the fetch's filter reads.
    assert history[0].fixture == 1


def test_a_history_row_with_no_fixture_id_parses_as_zero():
    payload = copy.deepcopy(ELEMENT_SUMMARY_JSON)
    del payload["history"][0]["fixture"]
    c = make_client({"/api/element-summary/5/": payload})
    assert c.element_history(5)[0].fixture == 0


def test_element_summary_returns_this_season_and_the_ones_before_it():
    # One request, both halves of it: the season so far and the seasons behind
    # it. The pipeline needs both and the API serves them together.
    c = make_client({"/api/element-summary/5/": ELEMENT_SUMMARY_JSON})

    history, past = c.element_summary(5)

    assert [(h.round, h.minutes) for h in history] == [(1, 90)]
    assert [(s.season_name, s.minutes) for s in past] == [("2024/25", 3230)]


def test_element_summary_parses_a_full_past_season_row():
    # The live shape, checked against element-summary on 2026-08-22: five past
    # seasons, oldest first, each a full season's totals. Only two fields are
    # declared; the rest must be ignored rather than break the parse.
    live = {
        "fixtures": [],
        "history": [],
        "history_past": [
            {
                "season_name": "2025/26",
                "element_code": 154561,
                "start_cost": 55,
                "end_cost": 62,
                "total_points": 162,
                "minutes": 3330,
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 19,
                "goals_conceded": 26,
                "saves": 60,
                "bonus": 11,
                "bps": 633,
                "influence": "541.6",
                "creativity": "33.5",
                "threat": "0.0",
                "ict_index": "57.5",
                "starts": 37,
                "expected_goals": "0.00",
                "expected_goals_conceded": "27.56",
            }
        ],
    }
    c = make_client({"/api/element-summary/1/": live})

    _, past = c.element_summary(1)

    assert len(past) == 1
    assert past[0].season_name == "2025/26"
    assert past[0].minutes == 3330


def test_element_summary_of_a_player_with_no_past_seasons():
    c = make_client({"/api/element-summary/5/": {"history": [], "fixtures": []}})
    assert c.element_summary(5) == ([], [])


def test_element_history_is_the_first_half_of_the_summary():
    # The backtest asks for the history alone; it must be the same list, off
    # the same one request.
    c = make_client({"/api/element-summary/5/": ELEMENT_SUMMARY_JSON})
    assert c.element_history(5) == c.element_summary(5)[0]


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


# --- politeness and patience -----------------------------------------------


def test_the_default_client_says_who_is_calling():
    # The FPL API is public, unauthenticated and someone else's; a scheduled
    # bot hitting it should at least be identifiable.
    assert default_http_client().headers["User-Agent"] == USER_AGENT


def test_a_client_built_without_one_gets_the_default():
    assert FplClient()._http.headers["User-Agent"] == USER_AGENT


def flaky_client(statuses: list[int]) -> tuple[FplClient, list, list[float]]:
    """A client whose responses follow ``statuses``, then 200 forever.

    Returns it with the list of requests it made and the sleeps it took, so a
    test can pin how many tries it had and how long it waited between them.
    """
    codes = list(statuses)
    seen: list[httpx.Request] = []
    slept: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        code = codes.pop(0) if codes else 200
        return httpx.Response(code, json=BOOTSTRAP_JSON)

    client = FplClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=slept.append,
    )
    return client, seen, slept


def test_a_rate_limit_is_retried_after_a_wait():
    client, seen, slept = flaky_client([429])

    assert client.bootstrap().next_event().id == 2
    assert len(seen) == 2
    assert slept == [RETRY_BACKOFF[0]]


def test_server_errors_are_retried_with_a_growing_backoff():
    client, seen, slept = flaky_client([503, 500])

    assert client.bootstrap().next_event().id == 2
    assert len(seen) == 3
    assert slept == list(RETRY_BACKOFF)


def test_retries_are_bounded_and_the_last_failure_raises():
    client, seen, slept = flaky_client([429, 429, 429, 429])

    with pytest.raises(httpx.HTTPStatusError):
        client.bootstrap()
    assert len(seen) == len(RETRY_BACKOFF) + 1
    assert slept == list(RETRY_BACKOFF)


def test_a_404_is_an_answer_and_is_not_retried():
    # The picks endpoint 404s for a gameweek a manager did not play, and the
    # orchestrator reads that as "no squad". Retrying it wastes six seconds
    # to arrive at the same place.
    client, seen, slept = flaky_client([404])

    with pytest.raises(httpx.HTTPStatusError):
        client.bootstrap()
    assert len(seen) == 1
    assert slept == []
