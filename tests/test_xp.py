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
from aigaffer.model.priors import SeasonPrior, prior_rates
from aigaffer.model.xp import (
    ASSIST_PTS,
    CLUB_CHANGE_FACTOR,
    CS_PTS,
    DEFCON90_PRIOR,
    PRIOR_SEASON_DISCOUNT,
    SAVES_PRIOR_DISCOUNT,
    GOAL_PTS,
    LEAGUE_AVG_GOALS,
    SAVES90_PRIOR,
    SHRINKAGE_NINETIES,
    XA90_PRIOR,
    XG90_PRIOR,
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
    """A Lions midfielder with a season of 900 minutes behind him — ten full
    matches, so every season total below is ten times its rate per ninety.

    5.0 xG and 3.0 xA are 0.5 and 0.3 a game; 9 bonus is 0.9 a game; 120
    defensive actions are twelve a game, which is his threshold exactly.
    """
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
        "expected_goals": 5.0,
        "expected_assists": 3.0,
        "defensive_contribution": 120,
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
    # 5.0 xG in 900 minutes is 0.5 a game — but ten full matches (10 nineties)
    # is only 0.625 of the shrinkage weight against K = 6, so the rate is
    # pulled three-eighths of the way to the positional prior: a midfielder's
    # toward 0.12, a forward's toward the higher 0.30.
    mid_rate = 0.625 * 0.5 + 0.375 * 0.12  # 0.3575
    fwd_rate = 0.625 * 0.5 + 0.375 * 0.30  # 0.425
    mid = player(element_type=3)
    assert goal_points(mid, 90.0, 1.2) == approx(mid_rate * 1.0 * 1.2 * GOAL_PTS[3])
    forward = player(element_type=4)
    assert goal_points(forward, 90.0, 1.2) == approx(fwd_rate * 1.0 * 1.2 * GOAL_PTS[4])


def test_goals_scale_with_minutes_and_the_attack_factor():
    # The shrunk rate (0.3575 for this midfielder) still scales linearly with
    # the minutes on the pitch and the attack factor.
    rate = 0.625 * 0.5 + 0.375 * 0.12  # 0.3575
    assert goal_points(player(), 45.0, 0.8) == approx(rate * 0.5 * 0.8 * 5)


def test_assists_pay_the_same_for_every_position():
    # ASSIST_PTS is one number, not a per-position table like goals and clean
    # sheets: three points to anyone. Midfielders and forwards share the same
    # expected-assist prior (0.12), so their shrunk rates match too, and the
    # equality isolates the points multiplier from the shrinkage.
    expected = approx((0.625 * 0.3 + 0.375 * 0.12) * 1.2 * ASSIST_PTS)
    assert assist_points(player(element_type=3), 90.0, 1.2) == expected
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
    # 30 saves in 900 minutes is 3 a game, and 3 saves is a point.
    keeper = player(element_type=1, saves=30)
    assert save_points(keeper, 90.0) == approx(1.0)
    assert save_points(keeper, 45.0) == approx(0.5)
    assert save_points(player(element_type=2, saves=30), 90.0) == 0.0


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


def test_a_season_total_with_no_minutes_behind_it_is_not_a_rate():
    # The live bootstrap serves a handful of these: a full season of
    # defensive contributions against zero minutes played. Read as a rate
    # they are worth thousands of points a game.
    stale = player(minutes=0, bonus=9, defensive_contribution=232)
    assert bonus_points(stale, 90.0) == 0.0
    assert defcon_points(stale, 90.0) == 0.0


def test_the_same_floor_governs_goals_assists_and_saves():
    # One rule, not four: the contradictory payload is worth nothing in the
    # attacking terms too, and not only in the two that were written last.
    stale = player(
        element_type=1, minutes=0, expected_goals=12.0, expected_assists=8.0, saves=40
    )
    assert goal_points(stale, 90.0, 1.2) == 0.0
    assert assist_points(stale, 90.0, 1.2) == 0.0
    assert save_points(stale, 90.0) == 0.0


def test_a_shrunk_cameo_reads_as_the_prior_not_as_a_rate():
    # Six defensive actions in a minute off the bench is a sample, not a rate.
    # _per_90 floors its own rate at six a game; then shrinkage weighs that one
    # ninetieth of a match against six pseudo-matches of the midfield prior (7
    # actions a game) and lands him at essentially the prior, ~6.998 — not the
    # 540 a naive division reads, and well short of clearing his threshold on
    # his own evidence.
    cameo = player(minutes=1, defensive_contribution=6)
    nineties = 1 / 90
    weight = nineties / (nineties + 6.0)
    rate = weight * 6.0 + (1 - weight) * 7.0  # ~6.998, the MID prior
    chance = (rate - 12 / 2) / 12  # threshold 12 for a midfielder
    assert defcon_points(cameo, 90.0) == approx(2 * chance)


def test_a_shrunk_bench_goal_is_a_fraction_of_a_full_rate():
    # One expected goal in the one minute he has played: the floor reads it as
    # one per ninety, and shrinkage then pulls that lone match of evidence
    # almost all the way back to the midfield prior of 0.12. So the cameo goal
    # is worth about six-tenths of a point, not the five a genuine goal-a-game
    # midfielder's rate would score.
    cameo = player(minutes=1, expected_goals=1.0)
    nineties = 1 / 90
    weight = nineties / (nineties + 6.0)
    rate = weight * 1.0 + (1 - weight) * 0.12
    assert goal_points(cameo, 90.0, 1.0) == approx(rate * GOAL_PTS[3])


def test_a_defender_averaging_his_threshold_is_a_coin_flip_on_it():
    # 100 defensive actions in 900 minutes is 10 a game, a defender's threshold
    # exactly. The defender prior now sits at that same bar (10), so shrinkage
    # leaves a threshold-level defender where he is — 10 — and he is a coin flip
    # on the two points. That the prior sits at the bar rather than above it is
    # the whole point: an unproven defender is a coin flip, not a near-certainty.
    defender = player(element_type=2, minutes=900, defensive_contribution=100)
    rate = 0.625 * 10.0 + 0.375 * 10.0  # 10.0, unmoved
    chance = (rate - 10 / 2) / 10
    assert defcon_points(defender, 90.0) == approx(2 * chance)  # 1.0


def test_a_midfielder_well_short_of_his_threshold_never_earns_it():
    # 5 a game against a threshold of 12 is no defensive contributor. Shrinkage
    # pulls him toward the midfield prior of 7 — still below the half-threshold
    # of 6 — so he lands at 5.75 and earns nothing: the model does not invent a
    # defensive contribution the position does not have.
    quiet = player(element_type=3, minutes=900, defensive_contribution=50)
    rate = 0.625 * 5.0 + 0.375 * 7.0  # 5.75, short of the 6 half-threshold
    assert rate < 12 / 2
    assert defcon_points(quiet, 90.0) == 0.0


def test_even_the_busiest_defender_is_not_a_certainty():
    # 30 a game shrinks to 22.5 (toward the 10 prior) and is still more than
    # twice the bar — capped below certainty all the same, because nobody does
    # it every week. Half a match on the pitch is half the chances to do it.
    monster = player(element_type=2, minutes=900, defensive_contribution=300)
    assert defcon_points(monster, 45.0) == approx(1.9 * 0.5)


def test_a_keeper_earns_nothing_for_defensive_contributions():
    # Keepers are not eligible for the points, whatever they do.
    keeper = player(element_type=1, minutes=900, defensive_contribution=900)
    assert defcon_points(keeper, 90.0) == 0.0


# --- rate shrinkage --------------------------------------------------------


def test_the_shrinkage_constants_are_the_documented_ones():
    # They are tunables the whole caution turns on; a change to any of them
    # should be a deliberate edit here, not a silent drift in the model.
    assert SHRINKAGE_NINETIES == 6.0
    assert XG90_PRIOR == {1: 0.0, 2: 0.05, 3: 0.12, 4: 0.30}
    assert XA90_PRIOR == {1: 0.0, 2: 0.05, 3: 0.12, 4: 0.12}
    # The defcon prior sits AT its threshold for a defender (10) and a notch
    # below for a midfielder (bar 12): an unproven defender is a coin flip on
    # the bar, not the near-certainty a prior above it would have made him.
    assert DEFCON90_PRIOR == {1: 0.0, 2: 10.0, 3: 7.0, 4: 3.0}
    assert SAVES90_PRIOR == {1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_shrinkage_pulls_a_one_game_outlier_defender_toward_the_prior():
    # The bug in miniature: a £4.5m defender with 1.4 xG in his one 90-minute
    # game reads, divided honestly, as a 1.4-xG/90 elite striker. Shrinkage
    # weighs that single match against six pseudo-matches of the 0.05 defender
    # prior and drags the rate down to about a quarter of a goal a game.
    outlier = player(element_type=2, minutes=90, expected_goals=1.4)
    w = 1 / (1 + SHRINKAGE_NINETIES)  # one ninety played
    rate = w * 1.4 + (1 - w) * XG90_PRIOR[2]
    assert goal_points(outlier, 90.0, 1.0) == approx(rate * GOAL_PTS[2])
    assert rate == pytest.approx(0.242857, abs=1e-5)  # down from 1.4


def test_a_full_season_of_minutes_barely_shrinks_a_genuine_rate():
    # The same 1.4-xG/90 output, earned over a full season (3000 minutes, ~33
    # nineties) rather than one game. Now his own play carries 0.847 of the
    # weight, so the rate barely leaves his own: the genuine elite keeps his
    # number where the one-game fluke above lost his.
    own = 1.4
    proven = player(element_type=4, minutes=3000, expected_goals=own * 3000 / 90)
    w = (3000 / 90) / ((3000 / 90) + SHRINKAGE_NINETIES)
    rate = w * own + (1 - w) * XG90_PRIOR[4]
    assert goal_points(proven, 90.0, 1.0) == approx(rate * GOAL_PTS[4])
    assert abs(rate - own) < abs(rate - XG90_PRIOR[4])  # own data dominates
    assert rate == pytest.approx(1.232, abs=1e-3)


def test_the_de_cuyper_haaland_ordering_flip():
    # The live GW2 bug this shrinkage exists to fix. De Cuyper (DEF, 1.4 xG) and
    # Guehi (DEF, 0.9 xG) each played one 90-minute game; so did Haaland (FWD,
    # 0.8 xG). Unshrunk, the cheap defenders' goal rates dwarf Haaland's and the
    # model captains one of them. Shrunk toward the positional priors — 0.05 for
    # a defender, 0.30 for a forward — Haaland's rate comes out on top, which is
    # the whole point of the exercise.
    de_cuyper = player(element_type=2, minutes=90, expected_goals=1.4)
    guehi = player(element_type=2, minutes=90, expected_goals=0.9)
    haaland = player(element_type=4, minutes=90, expected_goals=0.8)

    def goal_rate(p: object) -> float:
        # goal_points over a neutral fixture, with the position's goal value
        # divided back out, is exactly the shrunk goal rate.
        return goal_points(p, 90.0, 1.0) / GOAL_PTS[p.element_type]

    assert goal_rate(haaland) > goal_rate(de_cuyper) > goal_rate(guehi)
    assert goal_rate(haaland) == pytest.approx(0.371429, abs=1e-5)
    assert goal_rate(de_cuyper) == pytest.approx(0.242857, abs=1e-5)
    assert goal_rate(guehi) == pytest.approx(0.171429, abs=1e-5)


def test_no_minutes_gets_no_prior_only_zero():
    # Shrinkage anchors a thin sample to the prior, but a player with no minutes
    # has no sample to anchor — the stale payload of a season's totals against
    # zero minutes played — so he stays at zero, prior included. No minutes is
    # no evidence, and inventing a league-average rate for him would undo the
    # very caution the no-minutes rule is there for.
    fresh = player(element_type=4, minutes=0, expected_goals=5.0)
    assert goal_points(fresh, 90.0, 1.0) == 0.0


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


def test_the_attack_and_defence_averages_are_not_interchangeable():
    # Lions and Bears average 1200 in every column, which makes the arithmetic
    # above easy and hides one mistake: swap the attack mean for the defence
    # mean and nothing above moves. The shared universe has three teams whose
    # means differ, so it can tell.
    #
    #   league home attack  = (1300 + 1150 + 1000) / 3 = 1150
    #   league home defence = (1280 + 1120 + 1020) / 3 = 1140
    #
    # Cravenside at home has a 1020 defence and a 1000 attack, so:
    #   att_factor = 1 / (1020 / 1140) = 1140 / 1020 = 1.117647058823529...
    #   lam        = 1.4 * (1000 / 1150)            = 1.217391304347826...
    #
    # With the two means swapped they would read 1.127450980... and
    # 1.228070175..., and neither is clamped, so the difference survives.
    teams = Bootstrap(**BOOTSTRAP_JSON).teams
    cravenside = next(team for team in teams if team.id == 3)

    att_factor, lam = fixture_factors(
        cravenside,
        opponent_at_home=True,
        averages=league_averages(teams, at_home=True),
    )

    assert att_factor == approx(1.1176470588235294)
    assert lam == approx(1.2173913043478262)


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
        expected_goals=0.0,
        expected_assists=0.0,
        saves=30,
        bonus=3,
        defensive_contribution=0,
    )
    expected = (
        2.0  # appearance
        + math.exp(-1.54) * 4  # clean sheet
        - 0.77  # conceded: 1.54 / 2
        + 1.0  # saves: 30 in 900 minutes is 3 a game, and 3 saves is a point
        + 0.3  # bonus: 3 in 900 minutes
    )
    assert fixture_points(keeper, 90.0, att_factor=1.2, lam=1.54) == approx(expected)


# --- projections -----------------------------------------------------------


def test_projects_a_midfielder_at_home():
    mid = player(id=10, team=1, element_type=3)
    projection = project(mid, [HOME_FIXTURE], horizon=1)
    # 900 minutes is ten nineties, 0.625 of the weight on his own rates: goals
    # shrink 0.5 -> 0.3575 (toward 0.12), assists 0.3 -> 0.2325 (toward 0.12),
    # and his 12-a-game defensive rate falls to 10.125 (toward 7), a 0.34375
    # chance of the two points.
    goal_rate = 0.625 * 0.5 + 0.375 * 0.12  # 0.3575
    assist_rate = 0.625 * 0.3 + 0.375 * 0.12  # 0.2325
    defcon_rate = 0.625 * 12.0 + 0.375 * 7.0  # 10.125
    expected = (
        2.0  # appearance
        + goal_rate * 1.2 * GOAL_PTS[3]  # goals: 2.145
        + assist_rate * 1.2 * ASSIST_PTS  # assists: 0.837
        + math.exp(-1.54) * CS_PTS[3]  # clean sheet
        + 0.9  # bonus: 9 in 900 minutes, unshrunk
        + 2 * ((defcon_rate - 12 / 2) / 12)  # defcon: 0.34375 chance of 2 points
    )
    assert projection.per_gw[2] == approx(expected)
    assert projection.player_id == 10


def test_projects_a_defender_away():
    # 720 minutes is eight nineties, weight 8/14 on his own rates: 0.8 xG is
    # 0.1 a game shrunk to 0.0786 (toward the 0.05 defender prior), 1.6 xA is
    # 0.2 shrunk to 0.1357, and 80 actions is 10 a game — his threshold, and now
    # the defender prior too, so shrinkage leaves it at 10: a 0.5 chance of the
    # two points.
    defender = player(
        id=20,
        team=2,
        element_type=2,
        expected_goals=0.8,
        expected_assists=1.6,
        minutes=720,
        bonus=4,
        defensive_contribution=80,
    )
    projection = project(defender, [HOME_FIXTURE], minutes=72.0, horizon=1)
    w = 8 / 14
    goal_rate = w * 0.1 + (1 - w) * 0.05
    assist_rate = w * 0.2 + (1 - w) * 0.05
    defcon_rate = w * 10.0 + (1 - w) * 10.0  # 10.0, unmoved
    expected = (
        1.8  # appearance: played 1.0 + p60 0.8
        + goal_rate * 0.8 * 0.8 * GOAL_PTS[2]  # minutes 72/90 = 0.8, att 0.8
        + assist_rate * 0.8 * 0.8 * ASSIST_PTS
        + math.exp(-1.19) * CS_PTS[2] * 0.8  # clean sheet
        - (1.19 / 2) * 0.8  # conceded: -0.476
        + 0.5 * 0.8  # bonus: 4 in 720 minutes, unshrunk
        + 2 * ((defcon_rate - 10 / 2) / 10) * 0.8  # defcon over 72 minutes
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
    assert projection == PlayerProjection(
        player_id=10,
        per_gw={2: 0.0, 3: 0.0},
        total=0.0,
        attacking_per_gw={2: 0.0, 3: 0.0},
    )


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


# --- minute overrides: the coming gameweek only ------------------------
#
# An override is what somebody who has read the team news says about the
# gameweek that news is about — the next one — and not a verdict on the whole
# window the planner plans over. Three weeks of Lions v Bears, alternating
# venue, so every gameweek in the window has something in it to change.

THREE_WEEKS = [
    HOME_FIXTURE,
    Fixture(id=2, event=3, team_h=2, team_a=1),
    Fixture(id=3, event=4, team_h=1, team_a=2),
]


def overridden(
    subject: Player, minutes: float, coming: float, **kwargs
) -> PlayerProjection:
    """``subject`` on ``minutes`` a week, overridden to ``coming`` in GW2."""
    projections = project_all(
        bootstrap(subject),
        THREE_WEEKS,
        {subject.id: minutes},
        start_event=2,
        horizon=3,
        overrides={subject.id: coming},
        **kwargs,
    )
    return projections[subject.id]


def test_an_override_is_for_the_coming_gameweek_only():
    mid = player(id=10)
    base = project(mid, THREE_WEEKS, minutes=90.0, horizon=3)
    as_a_sub = project(mid, THREE_WEEKS, minutes=30.0, horizon=3)

    projection = overridden(mid, minutes=90.0, coming=30.0)

    # The coming week is the substitute's, fixture by fixture...
    assert projection.per_gw[2] == as_a_sub.per_gw[2]
    assert projection.attacking_per_gw[2] == as_a_sub.attacking_per_gw[2]
    # ...and every week after it is exactly what it was with no override.
    for gw in (3, 4):
        assert projection.per_gw[gw] == base.per_gw[gw]
        assert projection.attacking_per_gw[gw] == base.attacking_per_gw[gw]
    # The total the solver maximises is the decayed sum of the new weeks, not
    # the old total with something taken off it.
    assert projection.total == decayed_total(projection.per_gw, 0.85)


def test_a_player_ruled_out_this_week_is_back_the_week_after():
    mid = player(id=10)
    base = project(mid, THREE_WEEKS, minutes=90.0, horizon=3)

    projection = overridden(mid, minutes=90.0, coming=0.0)

    assert projection.per_gw[2] == 0.0
    assert projection.attacking_per_gw[2] == 0.0
    assert projection.per_gw[3] == base.per_gw[3] > 0
    assert projection.per_gw[4] == base.per_gw[4] > 0
    # GW2 is worth nothing, so the total is GW3 and GW4 at their usual discount.
    assert projection.total == approx(0.85 * base.per_gw[3] + 0.85**2 * base.per_gw[4])


def test_half_the_minutes_is_not_half_the_points():
    # The override has to be projected, not applied as a ratio to the
    # projection it replaces: appearance points are not linear in minutes.
    # Forty-five minutes is a certain appearance (45/60 = 0.75 of one, capped
    # at an hour) and half a chance of the hour — 1.25 points against the
    # 1.0 that halving a ninety-minute week's 2.0 would give. Everything else
    # a midfielder scores is a rate times minutes and does halve, so the
    # whole of the gap is the appearance's 0.25.
    mid = player(id=10)
    full = project(mid, THREE_WEEKS, minutes=90.0, horizon=3)

    projection = overridden(mid, minutes=90.0, coming=45.0)

    assert appearance_points(45.0) == approx(1.25)
    assert projection.per_gw[2] - full.per_gw[2] / 2 == approx(0.25)
    # The attacking slice is a rate times minutes, so that one does halve.
    assert projection.attacking_per_gw[2] == approx(full.attacking_per_gw[2] / 2)


def test_an_override_reaches_a_player_the_minutes_model_never_saw():
    # A player with no xmins entry projects at zero — and an override for him
    # is the minutes of the coming week and nothing more: the weeks after it
    # go back to the zero he had.
    mid = player(id=10)
    projections = project_all(
        bootstrap(mid), THREE_WEEKS, {}, start_event=2, horizon=3,
        overrides={10: 90.0},
    )
    played_out = project(mid, THREE_WEEKS, minutes=90.0, horizon=3)

    assert projections[10].per_gw == {2: played_out.per_gw[2], 3: 0.0, 4: 0.0}


def test_an_override_of_the_minutes_he_already_had_changes_nothing():
    mid = player(id=10)
    assert overridden(mid, minutes=72.0, coming=72.0) == project(
        mid, THREE_WEEKS, minutes=72.0, horizon=3
    )


@pytest.mark.parametrize("none", [None, {}])
def test_no_overrides_is_the_projection_without_them_to_the_byte(none):
    # Every caller that predates overrides — the backtest, these tests — gets
    # exactly what it got before, and so does a manager who adjusted nobody.
    squad = bootstrap(player(id=10), player(id=11, team=2, element_type=2))
    xmins = {10: 90.0, 11: 60.0}

    without = project_all(squad, THREE_WEEKS, xmins, start_event=2, horizon=3)
    explicit = project_all(
        squad, THREE_WEEKS, xmins, start_event=2, horizon=3, overrides=none
    )

    assert explicit == without


def test_an_override_touches_nobody_else():
    squad = bootstrap(player(id=10), player(id=11, team=2, element_type=2))
    xmins = {10: 90.0, 11: 60.0}

    before = project_all(squad, THREE_WEEKS, xmins, start_event=2, horizon=3)
    after = project_all(
        squad, THREE_WEEKS, xmins, start_event=2, horizon=3, overrides={10: 0.0}
    )

    assert after[11] == before[11]


# --- the prior season, pooled -----------------------------------------

LAST_SEASON = SeasonPrior(
    code=1, element_type=4, team_code=43, minutes=2953,
    expected_goals=25.5, expected_assists=5.0, saves=0,
    defensive_contribution=30.0,
)


def test_no_prior_is_todays_model_to_the_byte():
    # The closed form on current-season evidence alone — what the model
    # computed before this change existed. prior=None (a newcomer, or any
    # hand-built player with code 0) must reproduce it exactly; the full
    # suite passing untouched is the broader form of this same claim.
    p = player(element_type=4, expected_goals=1.4, minutes=90)
    expected_rate = (1.4 + 6 * 0.30) / (1 + 6)
    assert goal_points(p, 90, 1.0, prior=None) == pytest.approx(expected_rate * 4)


def test_last_season_pools_as_discounted_evidence():
    # Haaland's real shape: no minutes yet, a monster season behind him.
    # delta=1/3 makes 2953 prior minutes into 984 effective ones, and the
    # closed form (total + K*prior)/(nineties + K) gives 0.608 xG/90 —
    # double the positional floor of 0.30 he opens at today.
    p = player(element_type=4, expected_goals=0.0, minutes=0, team_code=43)
    pts = goal_points(p, 90, 1.0, prior=LAST_SEASON)
    d = PRIOR_SEASON_DISCOUNT
    expected_rate = (d * 25.5 + 6 * 0.30) / (d * 2953 / 90 + 6)
    assert pts == pytest.approx(expected_rate * 4)  # FWD goals pay 4


def test_a_summer_move_halves_the_trust():
    stayed = player(element_type=4, expected_goals=0.0, minutes=0, team_code=43)
    moved = player(element_type=4, expected_goals=0.0, minutes=0, team_code=7)
    assert goal_points(moved, 90, 1.0, prior=LAST_SEASON) < goal_points(
        stayed, 90, 1.0, prior=LAST_SEASON
    )
    d = PRIOR_SEASON_DISCOUNT * CLUB_CHANGE_FACTOR
    expected_rate = (d * 25.5 + 6 * 0.30) / (d * 2953 / 90 + 6)
    assert goal_points(moved, 90, 1.0, prior=LAST_SEASON) == pytest.approx(
        expected_rate * 4
    )


def test_saves_trust_last_season_less():
    # Saves/90 is mostly a team stat, so its discount is 1/4, not 1/3.
    keeper_past = SeasonPrior(
        code=2, element_type=1, team_code=43, minutes=2700,
        expected_goals=0.0, expected_assists=0.0, saves=100,
        defensive_contribution=0.0,
    )
    keeper = player(element_type=1, saves=0, minutes=0, team_code=43)
    d = SAVES_PRIOR_DISCOUNT
    expected_rate = (d * 100 + 6 * 3.0) / (d * 2700 / 90 + 6)
    assert save_points(keeper, 90, prior=keeper_past) == pytest.approx(
        expected_rate / 3
    )


def test_a_reclassified_player_keeps_his_history():
    # The evidence is per-90 facts and travels with the man; the positional
    # anchor is the new classification's. A forward's season behind a
    # midfielder pools the same totals against the midfield prior of 0.12.
    p = player(element_type=3, expected_goals=0.0, minutes=0, team_code=43)
    d = PRIOR_SEASON_DISCOUNT
    expected_rate = (d * 25.5 + 6 * 0.12) / (d * 2953 / 90 + 6)
    assert goal_points(p, 90, 1.0, prior=LAST_SEASON) == pytest.approx(
        expected_rate * 5
    )  # MID goals pay 5


def test_haaland_projects_elite_before_a_ball_is_kicked():
    # The pin this whole upgrade exists for, against the real vendored row.
    real = prior_rates()[223094]
    p = player(
        element_type=4, expected_goals=0.0, minutes=0,
        code=223094, team_code=real.team_code,
    )
    rate = goal_points(p, 90, 1.0, prior=real) / 4
    assert rate == pytest.approx(0.61, abs=0.01)
    assert rate > 2 * 0.30, "double the positional floor he opens at today"


def test_a_stale_total_against_no_minutes_stays_dead():
    # The live incident _per_90's guard was written for: a bootstrap can
    # serve a season of totals against zero minutes played. Pooling must
    # not resurrect it — the prior's evidence carries the player past the
    # guard, never the self-contradicting current total.
    stale = player(element_type=4, expected_goals=8.0, minutes=0, team_code=43)
    honest = player(element_type=4, expected_goals=0.0, minutes=0, team_code=43)
    assert goal_points(stale, 90, 1.0, prior=LAST_SEASON) == goal_points(
        honest, 90, 1.0, prior=LAST_SEASON
    )


def test_the_stingy_defence_projects_the_cleaner_sheet():
    # The seam between project_all and a fitted TeamStrengths, pinned so
    # the team/opponent argument order can never silently swap: two
    # identical defenders meet, and only the one whose OWN team has the
    # stingy fitted defence gets the cleaner sheet.
    from math import log

    from aigaffer.model.strength import TeamStrengths

    stingy = Team(
        id=1, name="Stingy", short_name="STI", code=101,
        strength_attack_home=1000, strength_attack_away=1000,
        strength_defence_home=1000, strength_defence_away=1000,
        strength_overall_home=1000, strength_overall_away=1000,
    )
    leaky = Team(
        id=2, name="Leaky", short_name="LEA", code=102,
        strength_attack_home=1000, strength_attack_away=1000,
        strength_defence_home=1000, strength_defence_away=1000,
        strength_overall_home=1000, strength_overall_away=1000,
    )
    fit = TeamStrengths(
        mu=log(1.4), home=0.0,
        attack={101: 0.0, 102: 0.0},
        defence={101: -0.5, 102: 0.5},
    )
    home_def = player(id=1, team=1, element_type=2)
    away_def = player(id=2, team=2, element_type=2)
    fixture = Fixture(id=1, event=2, team_h=1, team_a=2)
    universe = Bootstrap(events=[], teams=[stingy, leaky], elements=[home_def, away_def])
    projections = project_all(
        universe, [fixture], {1: 90.0, 2: 90.0}, start_event=2, strengths=fit
    )
    assert projections[1].per_gw[2] > projections[2].per_gw[2]
