"""Expected points, gameweek by gameweek.

Each fixture is scored on its own: the opponent's strength sets how likely a
goal or a clean sheet is, the player's expected minutes set how much of that
he is around for, and his season so far says what he does with the time. A
gameweek is the sum over his club's fixtures in it, so a blank scores nothing
and a double scores twice without any special case.

Every rate the model uses — goals, assists, saves, bonus, defensive
contributions — starts as a season total divided by season minutes by the same
function,
:func:`_per_90`, and so obeys the same small-sample rule: a player with no
minutes has no rate at all, and a rate is never taken over less than a full
match. That rule is the whole of the model's first line of caution about thin
evidence, and it only holds if nothing goes around it. The API publishes its
own per-90 columns and they are not used: they are already divided, by a
denominator we cannot see, and a substitute with one appearance behind him is
exactly the player they flatter.

The floor stops a cameo becoming a goal a minute, but it does nothing about
the other half of the small-sample problem: one full match of an
elite-looking number annualises into an elite rate. A £4.5m defender with 1.4
expected goals in his opener reads, honestly divided, as a 1.4-xG/90 striker,
and the model would captain him. So after the division each rate is *shrunk*
toward a rough league-average for the position (:func:`_shrunk_rate`), by an
amount that fades as the player's own minutes pile up: at the start his rate
is mostly the positional prior, and by a couple of seasons it is almost
entirely his own. That is the model's second line of caution, and it is what
keeps one loud gameweek from speaking for a whole season.

Points further out are worth less to a decision made today — the squad, the
prices and the fixtures will all have moved by then — so ``total`` discounts
each gameweek by a decay factor before summing.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
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

# How hard a per-90 rate is shrunk toward its positional prior, in "pseudo
# matches": the prior is weighted as if it were this many nineties of the
# player's own play. A player with exactly this many real nineties behind him
# is a fifty-fifty blend of his own rate and the prior; below it the prior
# leads, above it his own data does. Six is roughly two-thirds of a season —
# enough that one loud gameweek cannot annualise into an elite rate, little
# enough that a genuine elite has taken the rate back over well before
# Christmas. A tunable, and the one number this whole caution turns on.
SHRINKAGE_NINETIES = 6.0

# Rough league-average per-90 output by position (1 GK, 2 DEF, 3 MID, 4 FWD):
# what a player we know nothing about is assumed to do, and what a thin sample
# is pulled toward. Deliberately coarse — their job is to be a sane anchor for
# an unproven player, not a second model — and tunable. Goals and assists are
# expected-goal and expected-assist rates; saves are a keeper's own.
#
# The defensive-contribution prior is in *actions* per 90 and feeds the
# threshold model in :func:`defcon_points`, not points, so where it sits
# relative to the threshold is the whole of its effect. It must sit AT the
# threshold, not above it: a prior above the bar shrinks every thin-sample
# defender toward a near-certain +2 (a 14-actions prior against a 10 threshold
# reads as a 0.9 hit chance), which over-values every cheap defender the moment
# the season starts — the very distortion shrinkage exists to remove, moved one
# term over. Pinned at the threshold instead (DEF 10 = its bar), the unproven
# defender is a coin flip on it, which is what "we do not know yet" should mean.
# Midfield sits a notch below its higher bar for the same reason: a typical
# midfielder is not a defensive-points contributor, and the prior should not
# pretend he is.
XG90_PRIOR = {GOALKEEPER: 0.0, DEFENDER: 0.05, MIDFIELDER: 0.12, FORWARD: 0.30}
XA90_PRIOR = {GOALKEEPER: 0.0, DEFENDER: 0.05, MIDFIELDER: 0.12, FORWARD: 0.12}
DEFCON90_PRIOR = {GOALKEEPER: 0.0, DEFENDER: 10.0, MIDFIELDER: 7.0, FORWARD: 3.0}
SAVES90_PRIOR = {GOALKEEPER: 3.0, DEFENDER: 0.0, MIDFIELDER: 0.0, FORWARD: 0.0}

# Defensive contributions pay two points once in a match, to a defender who
# reaches ten defensive actions or to anyone further forward who reaches
# twelve. Keepers are not eligible, so they have no threshold at all.
DEFCON_POINTS = 2
DEFCON_THRESHOLDS = {DEFENDER: 10, MIDFIELDER: 12, FORWARD: 12}
MAX_DEFCON_CHANCE = 0.95


@dataclass
class PlayerProjection:
    """A player's expected points, gameweek by gameweek, and the decayed total.

    ``attacking_per_gw`` is the goals-and-assists slice of each gameweek — the
    ceiling half of the same numbers, carried alongside the whole so that the
    captaincy can be chosen on it without a second pass over the fixtures (see
    :func:`aigaffer.solver.lineup.pick_lineup`). It defaults to empty, because a
    projection built by hand for a test is not making a claim about a player's
    ceiling, and a caller that has none falls back on the total.
    """

    player_id: int
    per_gw: dict[int, float]
    total: float
    attacking_per_gw: dict[int, float] = field(default_factory=dict)


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
    """The season's expected goals as a rate per ninety, shrunk and scored on."""
    rate = _shrunk_rate(
        player.expected_goals, player.minutes, XG90_PRIOR[player.element_type]
    )
    goals = rate * (minutes / 90) * att_factor
    return goals * GOAL_PTS[player.element_type]


def assist_points(player: Player, minutes: float, att_factor: float) -> float:
    rate = _shrunk_rate(
        player.expected_assists, player.minutes, XA90_PRIOR[player.element_type]
    )
    assists = rate * (minutes / 90) * att_factor
    return assists * ASSIST_PTS


def attacking_points(player: Player, minutes: float, att_factor: float) -> float:
    """Expected points from goals and assists in one fixture — the ceiling.

    The high-variance half of a player's return, the part that doubles into a
    haul, as against the floor of appearance, clean sheets, saves, bonus and
    defensive contributions, none of which a captain's armband multiplies into
    anything worth having. It is the sum of the two attacking components and
    adds no formula of its own; the captaincy is chosen on it
    (:func:`aigaffer.solver.lineup.pick_lineup`) so the armband goes to a genuine
    goal threat rather than to a cheap player whose *total* a steady floor and a
    kind fixture have padded past one.
    """
    return goal_points(player, minutes, att_factor) + assist_points(
        player, minutes, att_factor
    )


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
    rate = _shrunk_rate(player.saves, player.minutes, SAVES90_PRIOR[player.element_type])
    return (rate / 3) * (minutes / 90)


def bonus_points(player: Player, minutes: float) -> float:
    """The player's season bonus rate per 90, capped at a point a game.

    Not shrunk, unlike the goal, assist, save and defcon rates. Bonus is
    already a soft, self-limiting heuristic — a season count capped at one
    point a match — rather than an unbounded rate a single loud gameweek can
    annualise into something absurd, so it has neither the problem shrinkage
    fixes nor an obvious positional prior to shrink toward. Left as it was.
    """
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
    rate = _shrunk_rate(
        player.defensive_contribution,
        player.minutes,
        DEFCON90_PRIOR[player.element_type],
    )
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


def projected_events(projections: dict[int, PlayerProjection]) -> list[int]:
    """Which gameweeks these projections cover, in order.

    The window a multi-week planner plans over is exactly the window there are
    numbers for, and this is how a caller asks what that is rather than
    recomputing it from a horizon and a starting gameweek. The two would agree
    today and are one edit apart from disagreeing, and the way they would
    disagree is silent: a planner given a gameweek nobody projected plans it on
    zeros, which reads as a fixture-free week rather than as a bug.

    A union rather than any one player's, because a projection missing a
    gameweek is a fact about that player and not about the schedule.
    """
    return sorted(
        {gw for projection in projections.values() for gw in projection.per_gw}
    )


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
        attacking_per_gw = {}
        for gw in gameweeks:
            points = 0.0
            attacking = 0.0
            for opponent_id, opponent_at_home in schedule.get((gw, player.team), ()):
                opponent = teams[opponent_id]
                averages = home_averages if opponent_at_home else away_averages
                att_factor, lam = fixture_factors(opponent, opponent_at_home, averages)
                points += fixture_points(player, minutes, att_factor, lam)
                attacking += attacking_points(player, minutes, att_factor)
            per_gw[gw] = points
            attacking_per_gw[gw] = attacking
        projections[player.id] = PlayerProjection(
            player_id=player.id,
            per_gw=per_gw,
            total=decayed_total(per_gw, decay),
            attacking_per_gw=attacking_per_gw,
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


def _shrunk_rate(total: float, minutes: int, prior: float) -> float:
    """A per-90 rate pulled toward ``prior`` by the thinness of its evidence.

    Empirical-Bayes shrinkage. Treat the positional ``prior`` as if it were
    :data:`SHRINKAGE_NINETIES` (``K``) nineties of prior observation, and the
    player's own play as the ``minutes / 90`` real nineties he has actually
    put in. The posterior rate is the weighted mean of the two::

        weight  = nineties / (nineties + K)
        shrunk  = weight * own_rate + (1 - weight) * prior

    so the own rate leads once he has more than ``K`` nineties behind him and
    the prior leads before that. One gameweek (``nineties`` ~ 1) is about a
    seventh his own and six-sevenths the prior; three seasons (~100 nineties)
    is all but entirely his own. This is the whole of the model's defence
    against one loud match annualising into an elite rate.

    Two deliberate asymmetries, both in the name of not trusting thin
    evidence too far:

    * ``own_rate`` keeps :func:`_per_90`'s full-match floor, so a one-minute
      cameo with a goal in it enters the blend as a 1.0-xG/90 own rate and not
      the 90 the honest division would give. The floor caps *how loud* the
      sample can be.

    * the shrinkage ``weight``, though, is taken over the *real* nineties
      (``minutes / 90``), floor and all left out — a one-minute cameo is one
      ninetieth of a match of evidence and is weighted as such, near zero, not
      as the full match the floor would otherwise smuggle in. The floor must
      not be allowed to hide the small sample from the weight.

    (When a player has a full match or more the two coincide: the floor stops
    binding at ninety minutes, and ``shrunk`` is then exactly
    ``(total + K * prior) / (nineties + K)``.)

    A player with no minutes at all is the one case the prior does *not* reach:
    :func:`_per_90` already reads no minutes as no evidence — the stale,
    self-contradicting payload of a full season's totals against zero minutes
    played — and inventing a league-average rate for him would undo exactly the
    caution that guards against. No minutes is no rate, prior included.
    """
    own_rate = _per_90(total, minutes)
    if minutes <= 0:
        return own_rate  # 0.0: no minutes is no evidence, not even the prior
    nineties = minutes / 90
    weight = nineties / (nineties + SHRINKAGE_NINETIES)
    return weight * own_rate + (1 - weight) * prior


def _clamp(factor: float) -> float:
    return min(MAX_FACTOR, max(MIN_FACTOR, factor))
