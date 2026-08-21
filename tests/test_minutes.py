from aigaffer.data.models import GwHistory, Player
from aigaffer.model.minutes import availability, expected_minutes


def player(**overrides) -> Player:
    fields = {
        "id": 1,
        "web_name": "Test",
        "team": 1,
        "element_type": 3,
        "now_cost": 50,
        "status": "a",
        "chance_of_playing_next_round": None,
        "minutes": 900,
        "starts": 10,
        "total_points": 50,
        "bonus": 5,
        "saves": 0,
    }
    return Player(**{**fields, **overrides})


def history(*minutes: int) -> list[GwHistory]:
    return [
        GwHistory(round=gw, minutes=mins, total_points=0)
        for gw, mins in enumerate(minutes, start=1)
    ]


def test_available_player_is_fully_available():
    assert availability(player(status="a")) == 1.0


def test_chance_of_playing_wins_over_status():
    assert availability(player(status="d", chance_of_playing_next_round=75)) == 0.75


def test_zero_chance_of_playing_is_unavailable():
    assert availability(player(status="d", chance_of_playing_next_round=0)) == 0.0


def test_unavailable_statuses_without_a_chance_are_zero():
    for status in ("i", "s", "u", "n"):
        assert availability(player(status=status)) == 0.0, status


def test_doubtful_without_a_chance_is_treated_as_available():
    # 'd' is not one of the ruled-out statuses, so with no percentage set we
    # have nothing better to go on than "he plays".
    assert availability(player(status="d")) == 1.0


def test_expected_minutes_is_the_mean_of_recent_starts():
    assert expected_minutes(history(90, 90, 60, 90, 90), player()) == 84.0


def test_expected_minutes_scales_by_availability():
    flagged = player(status="d", chance_of_playing_next_round=75)
    assert expected_minutes(history(90, 90, 90, 90, 90), flagged) == 67.5


def test_injured_player_expects_no_minutes():
    injured = player(status="i", chance_of_playing_next_round=None)
    assert expected_minutes(history(90, 90, 90, 90, 90), injured) == 0.0


def test_only_the_last_five_gameweeks_count():
    # The first three (0-minute) gameweeks are ancient history and ignored.
    assert expected_minutes(history(0, 0, 0, 90, 80, 70, 60, 50), player()) == 70.0


def test_short_history_uses_what_is_there():
    assert expected_minutes(history(90, 60), player()) == 75.0


def test_no_history_falls_back_to_a_starter_estimate():
    assert expected_minutes([], player(starts=3)) == 75.0


def test_no_history_falls_back_to_a_bench_estimate():
    assert expected_minutes([], player(starts=0)) == 20.0


def test_no_history_fallback_scales_by_availability():
    flagged = player(starts=3, status="d", chance_of_playing_next_round=50)
    assert expected_minutes([], flagged) == 37.5
