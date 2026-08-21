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
    # Mean 75.0, and the starter fallback is 75.0 too, so the early-season
    # blend below cannot be read off this one either way.
    assert expected_minutes(history(90, 60), player(starts=2)) == 75.0


def test_a_short_history_above_the_fallback_is_kept():
    # Mean 85.0 beats the 75.0 a starter falls back on, so the blend takes the
    # history: the fallback is a floor under thin evidence, not a ceiling on it.
    assert expected_minutes(history(90, 80), player(starts=2)) == 85.0


# --- the early-season blend ------------------------------------------------
#
# GW1 of 2026-27: the World Cup ended eleven days before the season did not
# wait for it, the rested internationals sat out the opening weekend, and one
# gameweek of 0 minutes was the whole history. Every one of them projected zero.
#
# Every ``starts`` below is one the history under it could actually have
# produced. It is a season-to-date count, so a player with two gameweeks
# behind him has started at most two of them, and a test that hands the model
# ten starts with one gameweek of history is testing a payload the API cannot
# serve — which is a test that proves the blend against a case that never
# arrives. The case that does arrive is the last one in this block.


def test_one_rested_gameweek_does_not_zero_a_starter():
    # GW2, and the shape the fix is for: he started the opener and was rested
    # for the second, so the record is one start and a mean of 45 minutes. The
    # floor holds him at what a starter is worth rather than halving him on the
    # strength of one Saturday.
    assert expected_minutes(history(0, 90), player(starts=1)) == 75.0


def test_one_full_gameweek_beats_the_starter_fallback():
    assert expected_minutes(history(90), player(starts=1)) == 90.0


def test_two_gameweeks_still_take_the_starter_floor():
    # Both started, the second of them abandoned at 20 minutes: mean 55.0
    # against a fallback of 75, and the fallback is what a role is worth.
    assert expected_minutes(history(90, 20), player(starts=2)) == 75.0


def test_by_the_third_gameweek_the_history_speaks_for_itself():
    # Mean 30.0, and no floor under it: three weeks is evidence of a role.
    assert expected_minutes(history(0, 0, 90), player(starts=1)) == 30.0


def test_the_blend_floor_for_a_player_who_has_never_started():
    assert expected_minutes(history(0), player(starts=0)) == 20.0


def test_the_blend_cannot_raise_a_player_who_is_not_playing():
    # Availability scales the blend like anything else: a floor under the
    # minutes he would play is not a claim that he will play at all.
    injured = player(starts=1, status="i", chance_of_playing_next_round=0)
    assert expected_minutes(history(0, 90), injured) == 0.0


def test_the_blend_scales_by_availability_like_the_rest():
    doubtful = player(starts=1, status="d", chance_of_playing_next_round=50)
    assert expected_minutes(history(0, 90), doubtful) == 37.5


def test_a_premium_rested_in_gameweek_one_is_understated_and_stays_so():
    # The half of the GW1 case the blend cannot reach, written down because it
    # is a limitation rather than a bug. A £14.5m striker rested for the
    # opening weekend has one gameweek of 0 minutes *and* no starts to go with
    # it — the two fields are both season-to-date and both say nothing — so the
    # floor under him is the bench estimate, and the model believes 20 minutes
    # of a player who will play ninety.
    #
    # There is nothing in the payload to be right with: last season is not in
    # it, and reading a role off a price or off total_points would be a guess
    # wearing the clothes of a measurement. The mitigation is the gaffer, who
    # reads the team news and sets the minutes by hand — the week a79b01d was
    # written out of. From GW2 the blend takes over: one start on the record is
    # all it needs, which is the test at the top of this block.
    premium = player(now_cost=145, starts=0, minutes=0, total_points=0)

    assert expected_minutes(history(0), premium) == 20.0


def test_no_history_falls_back_to_a_starter_estimate():
    assert expected_minutes([], player(starts=3)) == 75.0


def test_no_history_falls_back_to_a_bench_estimate():
    assert expected_minutes([], player(starts=0)) == 20.0


def test_no_history_fallback_scales_by_availability():
    flagged = player(starts=3, status="d", chance_of_playing_next_round=50)
    assert expected_minutes([], flagged) == 37.5
