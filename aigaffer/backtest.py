"""Does the model rank a gameweek better than chance?

One finished gameweek at a time: rebuild expected minutes from the history
*before* it, project that single gameweek, and set the projection against
what each player actually scored. Two numbers come back — the rank
correlation over the whole field, and how many of the model's top twenty
actually returned — because a bot that advises transfers is only ever asked
to put players in order, and those are the two ways of asking whether it can.

The population is the pipeline's own shortlist — the players it would
actually consider, by :func:`~aigaffer.orchestrator.history_pool` — narrowed
to those with a history entry for the gameweek. A player with no fixture that
week has no score to correlate with, and the rest of the six hundred are
projected at zero minutes and would only pad the field with ties.

**This is not a true backtest, and its numbers are not season simulations.**
Three things here know how the gameweek turned out, and all three flatter the
model:

* **The rates.** The per-90 rates, the prices and the team strengths come from
  the bootstrap as it stands today, which includes the gameweek being scored
  and every one since. The model is being asked to rank a week it has already
  seen. What leaks is bounded — a season rate moves little for one week of it
  — but it leaks the model's way.
* **Who is in the field.** The shortlist admits only players available *today*,
  so anyone since injured or gone is never graded. The population is the
  survivors, and a projection is never checked against the weeks it got wrong
  about a man who has since broken down.
* **Which survivors.** The shortlist then cuts by season-to-date total points —
  today's total, including everything scored *after* the gameweek being graded.
  The field is skewed towards players who went on to do well, which is the
  ordering the model is being congratulated for predicting.

Only the minutes model is held honest, because it is the one input that turns
over fast enough for a peek to flatter it outright. So the level of rho is
optimistic and should not be read as a forecast of live performance: what the
harness is good for is the sign, the trend from week to week, and catching a
model that has stopped ranking better than chance.
"""

from dataclasses import dataclass
from itertools import groupby
from math import sqrt

from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, Fixture, GwHistory
from aigaffer.model.minutes import expected_minutes
from aigaffer.model.xp import project_all
from aigaffer.orchestrator import history_pool

# The top of the model's ranking, and what counts as a returning score there.
TOP_N = 20
HIT_POINTS = 5

# A sanity figure, not a measurement: three decimals is more than it can bear.
PLACES = 3


@dataclass(frozen=True)
class PlayerOutcome:
    """What the model said about a player, and what he went and scored."""

    player_id: int
    projected: float
    actual: int


def finished_gameweeks(bootstrap: Bootstrap) -> list[int]:
    """The gameweeks that have been played, in order. Empty pre-season."""
    return sorted(event.id for event in bootstrap.events if event.finished)


def backtest_gw(
    client: FplClient, bootstrap: Bootstrap, fixtures: list[Fixture], gw: int
) -> dict:
    """Score the model's ranking of one **finished** gameweek.

    Returns ``{"gw", "n", "spearman", "top20_hit_rate"}``: the population that
    could be scored, the rank correlation of projected against actual points
    over it, and the share of the model's top :data:`TOP_N` who took
    :data:`HIT_POINTS` or more off it. When fewer than :data:`TOP_N` players
    qualify the hit rate is taken over the whole field, which ``n`` reports.

    Whether ``gw`` has actually finished is the caller's business: an
    unfinished one has no scores to join against and comes back empty.
    """
    players = {player.id: player for player in bootstrap.elements}
    histories = {pid: client.element_history(pid) for pid in history_pool(players, [])}
    xmins = {
        pid: expected_minutes(_before(history, gw), players[pid])
        for pid, history in histories.items()
    }
    projections = project_all(bootstrap, fixtures, xmins, gw, horizon=1)

    outcomes = [
        PlayerOutcome(pid, projections[pid].per_gw[gw], sum(scores))
        for pid, history in histories.items()
        if (scores := _scored(history, gw))
    ]
    rho = _spearman(
        [outcome.projected for outcome in outcomes],
        [outcome.actual for outcome in outcomes],
    )
    return {
        "gw": gw,
        "n": len(outcomes),
        "spearman": round(rho, PLACES),
        "top20_hit_rate": round(_hit_rate(outcomes), PLACES),
    }


def _before(history: list[GwHistory], gw: int) -> list[GwHistory]:
    """The season up to but not including ``gw``.

    The whole harness turns on this line. The minutes model reads the last
    five gameweeks, so a history that still has ``gw`` in it hands the model
    the very thing it is being asked to predict.
    """
    return [entry for entry in history if entry.round < gw]


def _scored(history: list[GwHistory], gw: int) -> list[int]:
    """What he actually took off ``gw``, one entry per fixture: a double
    gameweek pays twice, and an empty list means he had no fixture at all."""
    return [entry.total_points for entry in history if entry.round == gw]


def _hit_rate(outcomes: list[PlayerOutcome]) -> float:
    """The share of the model's best :data:`TOP_N` who returned.

    Ties in the projection are broken by player id so the same field always
    produces the same twenty.
    """
    if not outcomes:
        return 0.0
    ranked = sorted(
        outcomes, key=lambda outcome: (-outcome.projected, outcome.player_id)
    )
    top = ranked[:TOP_N]
    return sum(outcome.actual >= HIT_POINTS for outcome in top) / len(top)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation of two equal-length series, from -1.0 to 1.0.

    Spearman is Pearson on the ranks, and tied values share the average of
    the places they cover — the standard midrank convention, which is what
    makes a field of blanking defenders one tie rather than an arbitrary
    order. Fewer than two points, or a series that never varies, has no
    correlation to measure: both return 0.0 rather than a NaN.
    """
    if len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    """Places from 1, tied values sharing the average of theirs."""
    ranks = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda i: values[i])
    place = 1
    for _, group in groupby(order, key=lambda i: values[i]):
        tied = list(group)
        midrank = place + (len(tied) - 1) / 2
        for i in tied:
            ranks[i] = midrank
        place += len(tied)
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Correlation of two equal-length series; 0.0 if either never varies."""
    dxs, dys = _deviations(xs), _deviations(ys)
    spread = sqrt(sum(d * d for d in dxs) * sum(d * d for d in dys))
    if spread == 0:
        return 0.0
    return sum(dx * dy for dx, dy in zip(dxs, dys, strict=True)) / spread


def _deviations(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    return [value - mean for value in values]
