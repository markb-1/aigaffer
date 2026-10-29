"""Attacking form: what his last few matches say against his season.

The projection's rates are a season's, shrunk toward a prior, and that is
right for a six-week plan — but it is deaf to a striker who has had nothing
for a month, or one in the run of his season. A triple captain is one match,
and that one match is where a run matters. So this reads his last
:data:`~aigaffer.model.minutes.FORM_GAMEWEEKS` gameweeks of expected goals
and assists per ninety against the rate the projection already uses, and
hands back a multiplier for the chip (:meth:`PlayerProjection.tc_points`).

Expected goals, not goals: a run of scuffed tap-ins and a run of missed
sitters are the same form, and only xG can tell them apart from luck. And
the same caution as every rate here: the ratio is shrunk toward neutral by
how many nineties it rests on, and clamped, so two loud cameos cannot move
an armband.
"""

from aigaffer.data.models import GwHistory, Player
from aigaffer.model.minutes import FORM_GAMEWEEKS
from aigaffer.model.priors import SeasonPrior
from aigaffer.model.xp import (
    MIN_RATE_MINUTES,
    PRIOR_SEASON_DISCOUNT,
    XA90_PRIOR,
    XG90_PRIOR,
    _pooled,
    _shrunk_rate,
)

# The form ratio is weighed as if neutral were this many nineties of
# evidence: five full matches is five-eighths his recent rate, three-eighths
# no opinion.
FORM_SHRINK_NINETIES = 3.0
# No run is worth more than this either way to a single match.
MIN_FORM = 0.8
MAX_FORM = 1.25


def attacking_form(
    player: Player, history: list[GwHistory], prior: SeasonPrior | None = None
) -> float:
    """His recent xG+xA per ninety as a multiple of his season rate, shrunk
    toward 1.0 and clamped; 1.0 where there is nothing to read."""
    rounds = sorted({row.round for row in history})[-FORM_GAMEWEEKS:]
    recent = [row for row in history if row.round in rounds]
    minutes = sum(row.minutes for row in recent)
    # A payload that never carried the expected numbers is no evidence, not
    # a month of nothing: read as 0.0 it would mark every player cold.
    carried = any("expected_goals" in row.model_fields_set for row in recent)
    if minutes < MIN_RATE_MINUTES or not carried:
        return 1.0
    involvement = sum(row.expected_goals + row.expected_assists for row in recent)
    recent_rate = involvement / minutes * 90

    goals, evidence = _pooled(
        player.expected_goals, player.minutes, player, prior,
        prior.expected_goals if prior else 0.0, PRIOR_SEASON_DISCOUNT,
    )
    assists, assist_evidence = _pooled(
        player.expected_assists, player.minutes, player, prior,
        prior.expected_assists if prior else 0.0, PRIOR_SEASON_DISCOUNT,
    )
    season_rate = _shrunk_rate(goals, evidence, XG90_PRIOR[player.element_type]) + _shrunk_rate(
        assists, assist_evidence, XA90_PRIOR[player.element_type]
    )
    if season_rate <= 0:
        return 1.0

    nineties = minutes / 90
    weight = nineties / (nineties + FORM_SHRINK_NINETIES)
    form = 1.0 + weight * (recent_rate / season_rate - 1.0)
    return min(MAX_FORM, max(MIN_FORM, form))
