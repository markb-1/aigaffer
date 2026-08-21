"""Tests for the expected-points model.

Two teams make the league arithmetic easy to do by hand: every strength
column averages 1200, so a factor is just the opponent's column over 1200.
Lions at home to Bears therefore gives ``att_factor`` 1200/1000 = 1.2 and
``lam`` 1.4 x 1320/1200 = 1.54, while Bears away at Lions gives 1200/1500 =
0.8 and 1.4 x 1020/1200 = 1.19. Every expected value below is written out
term by term from those numbers.
"""

import math

import pytest

from aigaffer.data.models import Bootstrap, Fixture, Player, Team
from aigaffer.model.xp import (
    ASSIST_PTS,
    CS_PTS,
    GOAL_PTS,
    LEAGUE_AVG_GOALS,
    LeagueAverages,
    PlayerProjection,
    appearance_points,
    assist_points,
    bonus_points,
    clean_sheet_points,
    conceded_points,
    decayed_total,
    defcon_points,
    fixture_factors,
    fixture_points,
    goal_points,
    league_averages,
    p60,
    played,
    project_all,
    save_points,
    strength_ratio,
)
from tests.fixtures import BOOTSTRAP_JSON, FIXTURES_JSON

LIONS = Team(
    id=1,
    name="Lions",
    short_name="LIO",
    strength_attack_home=1020,
    strength_attack_away=1080,
    strength_defence_home=1500,
    strength_defence_away=1400,
    strength_overall_home=4,
    strength_overall_away=4,
)
BEARS = Team(
    id=2,
    name="Bears",
    short_name="BEA",
    strength_attack_home=1380,
    strength_attack_away=1320,
    strength_defence_home=900,
    strength_defence_away=1000,
    strength_overall_home=2,
    strength_overall_away=2,
)

# Means of the two teams above: 1200 in every strength column, 3 overall.
HOME_AVERAGES = LeagueAverages(attack=1200.0, defence=1200.0, overall=3.0)
AWAY_AVERAGES = LeagueAverages(attack=1200.0, defence=1200.0, overall=3.0)

# Lions host Bears in gameweek 2.
HOME_FIXTURE = Fixture(id=1, event=2, team_h=1, team_a=2)


def player(**overrides) -> Player:
    """A Lions midfielder: 0.5 xG90, 0.3 xA90, 9 bonus and 18 defcon in 900."""
    fields = {
        "id": 1,
        "web_name": "Test",
        "team": 1,
        "element_type": 3,
        "now_cost": 50,
        "status": "a",
        "minutes": 900,
        "starts": 10,
        "total_points": 50,
        "bonus": 9,
        "saves": 0,
        "expected_goals_per_90": 0.5,
        "expected_assists_per_90": 0.3,
        "saves_per_90": 0.0,
        "defensive_contribution": 18,
    }
    return Player(**{**fields, **overrides})


def bootstrap(*players: Player) -> Bootstrap:
    return Bootstrap(events=[], teams=[LIONS, BEARS], elements=list(players))


def project(
    subject: Player, fixtures: list[Fixture], minutes: float = 90.0, **kwargs
) -> PlayerProjection:
    """Project one player from gameweek 2, which is when the fixtures are."""
    projections = project_all(
        bootstrap(subject), fixtures, {subject.id: minutes}, start_event=2, **kwargs
    )
    return projections[subject.id]


def approx(value: float):
    return pytest.approx(value, abs=1e-6)


# --- minutes ---------------------------------------------------------------


def test_an_hour_makes_an_appearance_certain():
    assert played(30.0) == 0.5
    assert played(60.0) == 1.0
    assert played(90.0) == 1.0


def test_lasting_sixty_minutes_is_only_certain_for_a_full_game():
    assert p60(45.0) == 0.5
    assert p60(90.0) == 1.0


def test_appearance_pays_once_for_playing_and_once_for_the_hour():
    assert appearance_points(90.0) == 2.0
    assert appearance_points(45.0) == 1.25  # played 0.75 + p60 0.5


# --- scoring components ----------------------------------------------------


def test_goals_are_paid_at_the_position_rate():
    mid = player(element_type=3)
    assert goal_points(mid, 90.0, 1.2) == approx(0.5 * 1.0 * 1.2 * GOAL_PTS[3])
    forward = player(element_type=4)
    assert goal_points(forward, 90.0, 1.2) == approx(0.5 * 1.0 * 1.2 * GOAL_PTS[4])


def test_goals_scale_with_minutes_and_the_attack_factor():
    assert goal_points(player(), 45.0, 0.8) == approx(0.5 * 0.5 * 0.8 * 5)


def test_assists_pay_the_same_for_every_position():
    expected = approx(0.3 * 1.2 * ASSIST_PTS)
    assert assist_points(player(element_type=2), 90.0, 1.2) == expected
    assert assist_points(player(element_type=4), 90.0, 1.2) == expected


def test_clean_sheets_decay_with_the_goals_the_opponent_is_expected_to_score():
    defender = player(element_type=2)
    assert clean_sheet_points(defender, 72.0, 1.19) == approx(math.exp(-1.19) * 4 * 0.8)
    assert clean_sheet_points(player(element_type=3), 72.0, 1.19) == approx(
        math.exp(-1.19) * 1 * 0.8
    )


def test_forwards_get_nothing_for_a_clean_sheet():
    assert CS_PTS[4] == 0
    assert clean_sheet_points(player(element_type=4), 90.0, 1.19) == 0.0


def test_only_keepers_and_defenders_are_docked_for_goals_conceded():
    expected = approx(-(1.19 / 2) * 0.8)
    assert conceded_points(player(element_type=1), 72.0, 1.19) == expected
    assert conceded_points(player(element_type=2), 72.0, 1.19) == expected
    assert conceded_points(player(element_type=3), 72.0, 1.19) == 0.0
    assert conceded_points(player(element_type=4), 72.0, 1.19) == 0.0


def test_only_keepers_score_for_saves():
    keeper = player(element_type=1, saves_per_90=3.0)
    assert save_points(keeper, 90.0) == approx(1.0)
    assert save_points(keeper, 45.0) == approx(0.5)
    assert save_points(player(element_type=2, saves_per_90=3.0), 90.0) == 0.0


def test_bonus_projects_the_season_rate_per_ninety():
    # 9 bonus points in 900 minutes is 0.9 a game.
    assert bonus_points(player(), 90.0) == approx(0.9)
    assert bonus_points(player(), 45.0) == approx(0.45)


def test_bonus_is_capped_at_one_point_a_game():
    # 30 bonus in 900 minutes projects to 3.0 a game, which never happens.
    assert bonus_points(player(bonus=30), 90.0) == approx(1.0)


def test_a_player_with_no_minutes_played_scores_no_bonus_or_defcon():
    fresh = player(minutes=0, bonus=0, defensive_contribution=0)
    assert bonus_points(fresh, 90.0) == 0.0
    assert defcon_points(fresh, 90.0) == 0.0


def test_defcon_projects_the_season_rate_per_ninety():
    # 36 defcon points in 720 minutes is 4.5 a game.
    busy = player(minutes=720, defensive_contribution=36)
    assert defcon_points(busy, 72.0) == approx(4.5 * 0.8)


# --- opponent strength -----------------------------------------------------


def test_league_averages_are_the_mean_of_every_team():
    assert league_averages([LIONS, BEARS], at_home=True) == HOME_AVERAGES
    assert league_averages([LIONS, BEARS], at_home=False) == AWAY_AVERAGES


def test_league_averages_use_the_columns_for_the_venue():
    # The shared universe's home and away columns differ, unlike the two
    # teams above: attack at home is 1300/1150/1000, away 1250/1100/980.
    teams = Bootstrap(**BOOTSTRAP_JSON).teams
    assert league_averages(teams, at_home=True) == LeagueAverages(
        attack=1150.0, defence=3420 / 3, overall=10 / 3
    )
    assert league_averages(teams, at_home=False) == LeagueAverages(
        attack=1110.0, defence=3280 / 3, overall=10 / 3
    )


def test_strength_ratio_compares_the_opponent_with_the_league():
    assert strength_ratio(1320, 1200.0, 4, 3.0) == approx(1.1)


def test_strength_ratio_falls_back_to_overall_when_the_column_is_zero():
    assert strength_ratio(0, 0.0, 4, 3.0) == approx(4 / 3)


def test_strength_ratio_falls_back_when_only_the_league_average_is_zero():
    assert strength_ratio(1320, 0.0, 4, 3.0) == approx(4 / 3)


def test_strength_ratio_is_neutral_when_every_column_is_zero():
    assert strength_ratio(0, 0.0, 0, 0.0) == 1.0


def test_factors_for_a_home_player_use_the_opponents_away_strengths():
    att_factor, lam = fixture_factors(BEARS, opponent_at_home=False, averages=AWAY_AVERAGES)
    assert att_factor == approx(1.2)  # 1200 / 1000
    assert lam == approx(1.54)  # 1.4 * 1320 / 1200


def test_factors_for_an_away_player_use_the_opponents_home_strengths():
    att_factor, lam = fixture_factors(LIONS, opponent_at_home=True, averages=HOME_AVERAGES)
    assert att_factor == approx(0.8)  # 1200 / 1500
    assert lam == approx(1.19)  # 1.4 * 1020 / 1200


def test_factors_are_clamped_at_the_extremes():
    monster = Team(
        id=3,
        name="Monsters",
        short_name="MON",
        strength_attack_home=5000,
        strength_attack_away=5000,
        strength_defence_home=5000,
        strength_defence_away=5000,
        strength_overall_home=5,
        strength_overall_away=5,
    )
    att_factor, lam = fixture_factors(monster, opponent_at_home=True, averages=HOME_AVERAGES)
    assert att_factor == approx(0.7)
    assert lam == approx(LEAGUE_AVG_GOALS * 1.3)


def test_preseason_teams_fall_back_to_their_overall_strength():
    # The live API serves 0 in every positional column before the season.
    blank = Team(
        id=3,
        name="Blanks",
        short_name="BLA",
        strength_attack_home=0,
        strength_attack_away=0,
        strength_defence_home=0,
        strength_defence_away=0,
        strength_overall_home=4,
        strength_overall_away=4,
    )
    averages = LeagueAverages(attack=0.0, defence=0.0, overall=3.0)
    att_factor, lam = fixture_factors(blank, opponent_at_home=True, averages=averages)
    assert att_factor == approx(3 / 4)
    assert lam == approx(LEAGUE_AVG_GOALS * 1.3)  # 4/3 clamped to 1.3


def test_teams_with_no_strengths_at_all_get_neutral_factors():
    nobody = Team(
        id=3,
        name="Nobodies",
        short_name="NOB",
        strength_attack_home=0,
        strength_attack_away=0,
        strength_defence_home=0,
        strength_defence_away=0,
        strength_overall_home=0,
        strength_overall_away=0,
    )
    averages = LeagueAverages(attack=0.0, defence=0.0, overall=0.0)
    assert fixture_factors(nobody, opponent_at_home=True, averages=averages) == (
        1.0,
        LEAGUE_AVG_GOALS,
    )


# --- one fixture -----------------------------------------------------------


def test_fixture_points_sum_every_component():
    keeper = player(
        element_type=1,
        expected_goals_per_90=0.0,
        expected_assists_per_90=0.0,
        saves_per_90=3.0,
        bonus=3,
        defensive_contribution=0,
    )
    expected = (
        2.0  # appearance
        + math.exp(-1.54) * 4  # clean sheet
        - 0.77  # conceded: 1.54 / 2
        + 1.0  # saves: 3.0 / 3
        + 0.3  # bonus: 3 in 900 minutes
    )
    assert fixture_points(keeper, 90.0, att_factor=1.2, lam=1.54) == approx(expected)


# --- projections -----------------------------------------------------------


def test_projects_a_midfielder_at_home():
    mid = player(id=10, team=1, element_type=3)
    projection = project(mid, [HOME_FIXTURE], horizon=1)
    expected = (
        2.0  # appearance
        + 0.5 * 1.2 * GOAL_PTS[3]  # goals: 3.0
        + 0.3 * 1.2 * ASSIST_PTS  # assists: 1.08
        + math.exp(-1.54) * CS_PTS[3]  # clean sheet
        + 0.9  # bonus: 9 in 900 minutes
        + 1.8  # defcon: 18 in 900 minutes
    )
    assert projection.per_gw[2] == approx(expected)
    assert projection.player_id == 10


def test_projects_a_defender_away():
    defender = player(
        id=20,
        team=2,
        element_type=2,
        expected_goals_per_90=0.1,
        expected_assists_per_90=0.2,
        minutes=720,
        bonus=4,
        defensive_contribution=36,
    )
    projection = project(defender, [HOME_FIXTURE], minutes=72.0, horizon=1)
    expected = (
        1.8  # appearance: played 1.0 + p60 0.8
        + 0.1 * 0.8 * 0.8 * GOAL_PTS[2]  # goals: 0.384
        + 0.2 * 0.8 * 0.8 * ASSIST_PTS  # assists: 0.384
        + math.exp(-1.19) * CS_PTS[2] * 0.8  # clean sheet
        - (1.19 / 2) * 0.8  # conceded: -0.476
        + 0.5 * 0.8  # bonus: 4 in 720 minutes
        + 4.5 * 0.8  # defcon: 36 in 720 minutes
    )
    assert projection.per_gw[2] == approx(expected)


def test_a_blank_gameweek_scores_nothing():
    projection = project(player(id=10), [HOME_FIXTURE], horizon=3)
    assert projection.per_gw[3] == 0.0
    assert projection.per_gw[4] == 0.0


def test_a_double_gameweek_counts_both_fixtures():
    mid = player(id=10)
    replay = Fixture(id=2, event=2, team_h=1, team_a=2)
    single = project(mid, [HOME_FIXTURE], horizon=1)
    double = project(mid, [HOME_FIXTURE, replay], horizon=1)
    assert double.per_gw[2] == approx(2 * single.per_gw[2])


def test_unscheduled_fixtures_are_ignored():
    postponed = Fixture(id=9, event=None, team_h=1, team_a=2)
    projection = project(player(id=10), [postponed], horizon=2)
    assert projection.per_gw == {2: 0.0, 3: 0.0}
    assert projection.total == 0.0


def test_a_player_expected_to_play_no_minutes_projects_zero():
    projection = project(player(id=10), [HOME_FIXTURE], minutes=0.0, horizon=2)
    assert projection == PlayerProjection(player_id=10, per_gw={2: 0.0, 3: 0.0}, total=0.0)


def test_a_player_missing_from_xmins_projects_zero():
    projections = project_all(bootstrap(player(id=10)), [HOME_FIXTURE], {}, start_event=2)
    assert projections[10].total == 0.0


def test_the_horizon_bounds_the_gameweeks_projected():
    projections = project_all(bootstrap(player(id=10)), [], {10: 90.0}, start_event=3)
    assert sorted(projections[10].per_gw) == [3, 4, 5, 6, 7, 8]


def test_later_gameweeks_are_discounted_in_the_total():
    assert decayed_total({2: 4.0, 3: 4.0}, 0.85) == approx(4.0 + 3.4)


def test_the_decay_follows_gameweek_order_not_insertion_order():
    assert decayed_total({3: 4.0, 2: 8.0}, 0.5) == approx(8.0 + 2.0)


def test_the_projection_total_is_the_decayed_sum():
    next_week = Fixture(id=2, event=3, team_h=1, team_a=2)
    projection = project(player(id=10), [HOME_FIXTURE, next_week], horizon=2, decay=0.85)
    per_gw = projection.per_gw
    assert projection.total == approx(per_gw[2] + 0.85 * per_gw[3])


def test_every_player_in_the_bootstrap_is_projected():
    squad = [player(id=n, team=1 + n % 2) for n in range(1, 6)]
    projections = project_all(bootstrap(*squad), [HOME_FIXTURE], {}, start_event=2)
    assert sorted(projections) == [1, 2, 3, 4, 5]


def test_the_shared_fixture_universe_projects_brightwoods_double():
    # Brightwood (team 2) play twice in GW2: away at Ashford and at home to
    # Cravenside. Grant (6) is theirs, so his gameweek is the two added up.
    universe = Bootstrap(**BOOTSTRAP_JSON)
    schedule = [Fixture(**fixture) for fixture in FIXTURES_JSON]
    xmins = {element.id: 90.0 for element in universe.elements}
    grant = 6

    def gw2(fixtures: list[Fixture]) -> float:
        projections = project_all(universe, fixtures, xmins, start_event=2, horizon=1)
        return projections[grant].per_gw[2]

    away = [f for f in schedule if f.id == 2]
    home = [f for f in schedule if f.id == 3]
    assert gw2(schedule) == approx(gw2(away) + gw2(home))
    assert gw2(away) != gw2(home)  # the two opponents and venues differ


def test_a_fixture_is_scored_against_the_right_opponent_and_venue():
    # Brightwood host Cravenside, so Cravenside's away strengths and the
    # league's away averages are what Grant (6) is scored against.
    universe = Bootstrap(**BOOTSTRAP_JSON)
    grant = next(element for element in universe.elements if element.id == 6)
    cravenside = next(team for team in universe.teams if team.id == 3)
    att_factor, lam = fixture_factors(
        cravenside,
        opponent_at_home=False,
        averages=league_averages(universe.teams, at_home=False),
    )

    fixtures = [Fixture(id=3, event=2, team_h=2, team_a=3)]
    projections = project_all(universe, fixtures, {6: 90.0}, start_event=2, horizon=1)
    assert projections[6].per_gw[2] == approx(fixture_points(grant, 90.0, att_factor, lam))
