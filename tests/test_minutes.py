import pytest

from aigaffer.data.models import GwHistory, PastSeason, Player
from aigaffer.model.minutes import availability, expected_minutes, season_prior


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
# Every history below is gameweeks that have been *played*. That is the
# caller's promise, kept in ``aigaffer.orchestrator._played``, and it is the
# half of the GW1 2026-27 case this module does not handle: the zeroes that
# wrote off every premium in the gaffer's first live week were rows the API
# had added for matches nobody had kicked off yet, not gameweeks anybody sat
# out. Filtered out at the fetch, they never reach this model at all.
#
# What is left for the blend is a thin history of real gameweeks: a rotated
# August, where a mean over one or two Saturdays is not evidence of a role.
#
# Every ``starts`` below is one the history under it could actually have
# produced. It is a season-to-date count, so a player with two gameweeks
# behind him has started at most two of them, and a test that hands the model
# ten starts with one gameweek of history is testing a payload the API cannot
# serve — which is a test that proves the blend against a case that never
# arrives. The case that does arrive is the last one in this block.


def test_one_rested_gameweek_does_not_zero_a_starter():
    # GW3, and the shape the blend is for: he sat out the opener and started
    # the second, so the record is one start and a mean of 45 minutes. The
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


def test_a_premium_on_the_opening_weekend_with_no_past_is_understated():
    # What the opening weekend costs a player with no Premier League behind
    # him. Once the rows for unplayed matches are cut, a £14.5m striker in GW1
    # has no history at all *and* no starts to go with it — season-to-date
    # fields say nothing about a season that has not started — so the floor
    # under him is the bench estimate, and the model believes 20 minutes of a
    # player who will play ninety.
    #
    # This is now the case of the brand-new signing and the promoted club only:
    # a returning player is read off last season instead, which is the block
    # below. There is still nothing in the payload to be right with for these
    # ones, and the mitigation is still the gaffer, who reads the team news and
    # sets the minutes by hand. From GW2 there is a played gameweek to read and
    # the blend takes over.
    premium = player(now_cost=145, starts=0, minutes=0, total_points=0)

    assert expected_minutes([], premium) == 20.0
    assert expected_minutes([], premium, prior=None) == 20.0


def test_no_history_falls_back_to_a_starter_estimate():
    assert expected_minutes([], player(starts=3)) == 75.0


def test_no_history_falls_back_to_a_bench_estimate():
    assert expected_minutes([], player(starts=0)) == 20.0


def test_no_history_fallback_scales_by_availability():
    flagged = player(starts=3, status="d", chance_of_playing_next_round=50)
    assert expected_minutes([], flagged) == 37.5


# --- last season -----------------------------------------------------------
#
# The floor under a thin history used to be a guess: 75 minutes for a player
# with a start to his name, 20 for everyone else. For a player who was in the
# division last season it does not have to be. His minutes are in the payload
# — ``history_past``, one row a season — and a per-gameweek average of them is
# a measurement of the role he had, which beats both halves of the guess.
#
# 3200 minutes over a 38-gameweek season is 84.2 a gameweek, and that is the
# number these tests are written against.

PRIOR = 3200 / 38  # 84.21…, a first-choice player's last season


def test_a_returning_premium_is_read_off_last_season():
    # GW1: no history and no starts, which used to mean the bench estimate.
    # Last season says he played all but a few minutes of every gameweek.
    premium = player(now_cost=145, starts=0, minutes=0, total_points=0)

    assert expected_minutes([], premium, prior=PRIOR) == pytest.approx(84.21, abs=0.01)


def test_a_prior_lifts_a_premium_rested_on_the_opening_weekend():
    # The live shape of GW1 2026-27 once the phantom rows are gone: one played
    # gameweek he was rested for. Mean 0, starts 0, and the old floor was the
    # substitute's twenty. Last season is the floor now.
    premium = player(now_cost=145, starts=0, minutes=0, total_points=0)

    assert expected_minutes(history(0), premium) == 20.0
    assert expected_minutes(history(0), premium, prior=PRIOR) == pytest.approx(
        84.21, abs=0.01
    )


def test_a_prior_replaces_the_starts_guess_rather_than_stacking_on_it():
    # Both directions. A first-choice player last season is worth more than the
    # 75 a start this season would have floored him at; a fringe one is worth
    # less than either guess, and the measurement wins there too.
    fringe = player(starts=0)

    assert expected_minutes([], player(starts=1), prior=PRIOR) == pytest.approx(84.21, abs=0.01)
    assert expected_minutes([], fringe, prior=12.0) == 12.0


def test_a_short_history_above_the_prior_is_kept():
    # The prior is a floor under thin evidence, not a ceiling on it: two full
    # gameweeks beat a season of rotation.
    assert expected_minutes(history(90, 90), player(starts=2), prior=30.0) == 90.0


def test_by_the_third_gameweek_the_prior_is_ignored_like_the_starts_guess():
    # Three gameweeks of this season is evidence of this season's role, and
    # nothing older is allowed under it.
    assert expected_minutes(history(0, 0, 90), player(starts=1), prior=PRIOR) == 30.0


def test_a_prior_scales_by_availability_like_everything_else():
    injured = player(starts=0, status="i", chance_of_playing_next_round=None)
    doubtful = player(starts=0, status="d", chance_of_playing_next_round=50)

    assert expected_minutes([], injured, prior=PRIOR) == 0.0
    assert expected_minutes([], doubtful, prior=PRIOR) == pytest.approx(42.1, abs=0.01)


def past(*seasons: tuple[str, int]) -> list[PastSeason]:
    return [PastSeason(season_name=name, minutes=mins) for name, mins in seasons]


def test_a_prior_is_last_seasons_minutes_over_a_season():
    assert season_prior(past(("2025/26", 3200))) == pytest.approx(84.21, abs=0.01)


def test_the_prior_reads_the_most_recent_season_on_the_record():
    # The API serves them oldest first, and a player's record can run years
    # back. Only the last one is his role now.
    seasons = past(("2023/24", 500), ("2024/25", 1900), ("2025/26", 3200))

    assert season_prior(seasons) == pytest.approx(84.21, abs=0.01)


def test_a_player_with_no_past_season_has_no_prior():
    # A new signing from abroad, or a promoted club's player: no Premier League
    # behind him, so nothing to read and the starts guess stands.
    assert season_prior([]) is None


def test_a_prior_is_clamped_to_a_match():
    # 38 times 90 is 3420 and nothing larger is a real season, but the model
    # never returns more than a match whatever the payload says.
    assert season_prior(past(("2025/26", 3420))) == 90.0
    assert season_prior(past(("2025/26", 9999))) == 90.0
    assert season_prior(past(("2025/26", -10))) == 0.0


def test_a_season_nobody_played_is_a_prior_of_nothing():
    # Registered and never used. It is thin evidence and it is still evidence,
    # and a player the model projects at nothing is a player it will not buy —
    # which is the right way round to be wrong about him.
    assert season_prior(past(("2025/26", 0))) == 0.0
