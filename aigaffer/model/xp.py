"""Expected points, gameweek by gameweek.

Each fixture is scored on its own: the opponent's strength sets how likely a
goal or a clean sheet is, the player's expected minutes set how much of that
he is around for, and his season so far says what he does with the time. A
gameweek is the sum over his club's fixtures in it, so a blank scores nothing
and a double scores twice without any special case.

Every rate the model uses — goals, assists, saves, bonus, defensive
contributions — is a season total divided by season minutes by the same
function, :func:`_per_90`, and so obeys the same small-sample rule: a player
with no minutes has no rate at all, and a rate is never taken over less than
a full match. That rule is the whole of the model's caution about thin
evidence, and it only holds if nothing goes around it. The API publishes its
own per-90 columns and they are not used: they are already divided, by a
denominator we cannot see, and a substitute with one appearance behind him is
exactly the player they flatter.

Points further out are worth less to a decision made today — the squad, the
prices and the fixtures will all have moved by then — so ``total`` discounts
each gameweek by a decay factor before summing.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import exp

from aigaffer.data.models import Bootstrap, Fixture, Player, Team

GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
ASSIST_PTS = 3
LEAGUE_AVG_GOALS = 1.4

# Strength ratios are bounded: the columns are noisy, and no fixture is ever
# so good or so bad that it triples a player's output.
MIN_FACTOR = 0.7
MAX_FACTOR = 1.3

GOALKEEPER = 1
DEFENDER = 2
MIDFIELDER = 3
FORWARD = 4

# The fewest minutes a per-90 rate may be taken over: a full match.
MIN_RATE_MINUTES = 90

# Defensive contributions pay two points once in a match, to a defender who
# reaches ten defensive actions or to anyone further forward who reaches
# twelve. Keepers are not eligible, so they have no threshold at all.
DEFCON_POINTS = 2
DEFCON_THRESHOLDS = {DEFENDER: 10, MIDFIELDER: 12, FORWARD: 12}
MAX_DEFCON_CHANCE = 0.95


@dataclass
class PlayerProjection:
    player_id: int
    per_gw: dict[int, float]
    total: float


@dataclass(frozen=True)
class LeagueAverages:
    """Mean team strength across the league, for one venue."""

    attack: float
    defence: float
    overall: float


def league_averages(teams: list[Team], at_home: bool) -> LeagueAverages:
    """League means of the strength columns for teams playing ``at_home``."""
    strengths = [_venue_strengths(team, at_home) for team in teams]
    return LeagueAverages(
        attack=_mean(attack for attack, _, _ in strengths),
        defence=_mean(defence for _, defence, _ in strengths),
        overall=_mean(overall for _, _, overall in strengths),
    )


def strength_ratio(
    strength: int, average: float, fallback: int, fallback_average: float
) -> float:
    """How strong the opponent is as a multiple of the league average.

    Out of season the API serves 0 in every positional strength column, so we
    fall back on the coarser overall strength, and on a neutral 1.0 when even
    that is missing. A missing league average is treated the same way: the
    ratio is meaningless without one.
    """
    for value, mean in ((strength, average), (fallback, fallback_average)):
        if value > 0 and mean > 0:
            return value / mean
    return 1.0


def fixture_factors(
    opponent: Team, opponent_at_home: bool, averages: LeagueAverages
) -> tuple[float, float]:
    """``(att_factor, lam)`` for a fixture against ``opponent``.

    ``att_factor`` scales the player's attacking returns and ``lam`` is the
    goals his team is expected to concede. ``averages`` must be the league
    means for the venue the opponent is playing at.
    """
    attack, defence, overall = _venue_strengths(opponent, opponent_at_home)
    # A defence stronger than average (ratio > 1) suppresses attacking returns.
    defence_ratio = strength_ratio(defence, averages.defence, overall, averages.overall)
    attack_ratio = strength_ratio(attack, averages.attack, overall, averages.overall)
    return _clamp(1 / defence_ratio), LEAGUE_AVG_GOALS * _clamp(attack_ratio)


def played(minutes: float) -> float:
    """Chance of appearing at all; an hour of expected minutes is certainty."""
    return min(1.0, minutes / 60)


def p60(minutes: float) -> float:
    """Chance of lasting 60 minutes, which is what clean sheets pay on."""
    return min(1.0, minutes / 90)


def appearance_points(minutes: float) -> float:
    """One point for playing, one more for reaching the hour."""
    return played(minutes) + p60(minutes)


def goal_points(player: Player, minutes: float, att_factor: float) -> float:
    """The season's expected goals as a rate per ninety, scored on."""
    rate = _per_90(player.expected_goals, player.minutes)
    goals = rate * (minutes / 90) * att_factor
    return goals * GOAL_PTS[player.element_type]


def assist_points(player: Player, minutes: float, att_factor: float) -> float:
    rate = _per_90(player.expected_assists, player.minutes)
    assists = rate * (minutes / 90) * att_factor
    return assists * ASSIST_PTS


def clean_sheet_points(player: Player, minutes: float, lam: float) -> float:
    """Poisson chance the opponent fails to score, paid on 60 minutes."""
    return exp(-lam) * CS_PTS[player.element_type] * p60(minutes)


def conceded_points(player: Player, minutes: float, lam: float) -> float:
    """Keepers and defenders lose a point for every two goals conceded."""
    if player.element_type not in (GOALKEEPER, DEFENDER):
        return 0.0
    return -(lam / 2) * p60(minutes)


def save_points(player: Player, minutes: float) -> float:
    """A point per three saves, keepers only."""
    if player.element_type != GOALKEEPER:
        return 0.0
    return (_per_90(player.saves, player.minutes) / 3) * (minutes / 90)


def bonus_points(player: Player, minutes: float) -> float:
    """The player's season bonus rate per 90, capped at a point a game."""
    return min(1.0, _per_90(player.bonus, player.minutes)) * (minutes / 90)


def defcon_points(player: Player, minutes: float) -> float:
    """Expected defensive-contribution points from one fixture.

    ``defensive_contribution`` counts defensive *actions* over the season,
    not points — a busy midfielder makes fourteen a game. The points are a
    threshold: two of them, once in a match, to a defender who reaches ten
    actions or to anyone further forward who reaches twelve. So the season
    count becomes a rate per ninety, and the rate becomes the chance of
    clearing the bar on the day.

    That chance is a straight line rather than a distribution, pinned at the
    one point worth being right about — a player who averages the threshold
    clears it about half the time — and capped below certainty, because
    nobody does it every week. It is a proxy, and a deliberately crude one:
    what it has to get right is that this is worth at most two points a
    match, which reading the rate as points did not.
    """
    threshold = DEFCON_THRESHOLDS.get(player.element_type)
    if threshold is None:
        return 0.0
    rate = _per_90(player.defensive_contribution, player.minutes)
    chance = min(MAX_DEFCON_CHANCE, max(0.0, (rate - threshold / 2) / threshold))
    return DEFCON_POINTS * chance * (minutes / 90)


def fixture_points(
    player: Player, minutes: float, att_factor: float, lam: float
) -> float:
    """Expected points from one fixture against a known opponent."""
    return (
        appearance_points(minutes)
        + goal_points(player, minutes, att_factor)
        + assist_points(player, minutes, att_factor)
        + clean_sheet_points(player, minutes, lam)
        + conceded_points(player, minutes, lam)
        + save_points(player, minutes)
        + bonus_points(player, minutes)
        + defcon_points(player, minutes)
    )


def decayed_total(per_gw: dict[int, float], decay: float) -> float:
    """Sum of ``per_gw`` in gameweek order, discounting each step by ``decay``."""
    return sum(decay**step * per_gw[gw] for step, gw in enumerate(sorted(per_gw)))


def project_all(
    bootstrap: Bootstrap,
    fixtures: list[Fixture],
    xmins: dict[int, float],
    start_event: int,
    horizon: int = 6,
    decay: float = 0.85,
) -> dict[int, PlayerProjection]:
    """Project every bootstrap player over ``horizon`` gameweeks from
    ``start_event``, keyed by player id.

    A player with no entry in ``xmins`` is treated as expecting no minutes,
    which projects to zero rather than to a guess.
    """
    teams = {team.id: team for team in bootstrap.teams}
    home_averages = league_averages(bootstrap.teams, at_home=True)
    away_averages = league_averages(bootstrap.teams, at_home=False)
    schedule = _schedule(fixtures)
    gameweeks = range(start_event, start_event + horizon)

    projections = {}
    for player in bootstrap.elements:
        minutes = xmins.get(player.id, 0.0)
        per_gw = {}
        for gw in gameweeks:
            points = 0.0
            for opponent_id, opponent_at_home in schedule.get((gw, player.team), ()):
                opponent = teams[opponent_id]
                averages = home_averages if opponent_at_home else away_averages
                att_factor, lam = fixture_factors(opponent, opponent_at_home, averages)
                points += fixture_points(player, minutes, att_factor, lam)
            per_gw[gw] = points
        projections[player.id] = PlayerProjection(
            player_id=player.id, per_gw=per_gw, total=decayed_total(per_gw, decay)
        )
    return projections


def _schedule(fixtures: list[Fixture]) -> dict[tuple[int, int], list[tuple[int, bool]]]:
    """(gameweek, team id) -> the opponents it faces and where they play.

    Fixtures without a gameweek are postponements waiting to be rescheduled;
    they are worth nothing until the API says when they are.
    """
    schedule: dict[tuple[int, int], list[tuple[int, bool]]] = defaultdict(list)
    for fixture in fixtures:
        if fixture.event is None:
            continue
        schedule[(fixture.event, fixture.team_h)].append((fixture.team_a, False))
        schedule[(fixture.event, fixture.team_a)].append((fixture.team_h, True))
    return dict(schedule)


def _venue_strengths(team: Team, at_home: bool) -> tuple[int, int, int]:
    """The team's (attack, defence, overall) strengths for one venue."""
    if at_home:
        return (
            team.strength_attack_home,
            team.strength_defence_home,
            team.strength_overall_home,
        )
    return (
        team.strength_attack_away,
        team.strength_defence_away,
        team.strength_overall_away,
    )


def _mean(values: Iterable[int]) -> float:
    strengths = list(values)
    return sum(strengths) / len(strengths) if strengths else 0.0


def _per_90(total: float, minutes: int) -> float:
    """A season total as a rate per ninety minutes, or 0.0 without one.

    Every rate in this module comes through here, so this is the one place
    the model decides what thin evidence is worth. Two things the live
    bootstrap does and the fixtures never showed. A player can carry a season
    of defensive contributions against zero minutes played: the payload
    contradicts itself, and reading his whole season as one game is how a
    £4.5m substitute comes to project four thousand points and get himself
    drafted. And a player with a minute or two behind him has a sample, not a
    rate. So no minutes is no evidence, and a rate is never taken over less
    than a full match.
    """
    if minutes <= 0:
        return 0.0
    return (total / max(minutes, MIN_RATE_MINUTES)) * 90


def _clamp(factor: float) -> float:
    return min(MAX_FACTOR, max(MIN_FACTOR, factor))
