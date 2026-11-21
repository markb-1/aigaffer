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

* **The rates.** The per-90 rates and the prices come from the bootstrap as
  it stands today, which includes the gameweek being scored and every one
  since. The model is being asked to rank a week it has already seen. What
  leaks is bounded — a season rate moves little for one week of it — but it
  leaks the model's way. (The fitted team strengths are the exception: they
  are rebuilt from results before the scored round, dated at its deadline,
  and do not leak.)
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

**The team-strength fit is graded on its own** (:func:`strength_eval`), on
matches rather than players, because there it can be done honestly. A
rolling origin: every matchday is forecast from a fit on the results dated
strictly before it, so a day's matches never see each other, and a lag
moves the fit back further to ask the question production asks of a fit
made at the deadline and read weeks later. The score is the Poisson
log-likelihood of the goals that went in, with the Brier and ranked
probability scores on home, draw and away, and the Brier score on the clean
sheets, which is what keepers and defenders are paid on. Every setting is
set against the current one by a paired bootstrap that resamples whole
matchdays, and again whole weeks, because a weekend's matches share a fit
and their errors with it.
"""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import groupby
from math import ceil, exp, floor, lgamma, log, sqrt
from random import Random

from aigaffer.data.fpl_api import FplClient
from aigaffer.data.models import Bootstrap, Fixture, GwHistory
from aigaffer.model.minutes import expected_minutes
from aigaffer.model.strength import (
    MatchResult,
    TeamStrengths,
    build_team_strengths,
    fit_team_strengths,
)
from aigaffer.model.xp import project_all
from aigaffer.orchestrator import history_pool

# The top of the model's ranking, and what counts as a returning score there.
TOP_N = 20
HIT_POINTS = 5

# A sanity figure, not a measurement: three decimals is more than it can bear.
PLACES = 3

# The strength grid: half-lives of about one, two, three, four, six, nine,
# twelve and eighteen months, in days, crossed with the pseudo-match anchor.
HALF_LIVES = (30, 61, 91, 122, 183, 274, 365, 548)
PSEUDO_GRID = (3, 6, 10)

# The first matches of a season, by count: twelve rounds' worth, which is
# where how much last season counts matters most.
EARLY_MATCHES = 120

# How far back a fit is moved to stand in for production's, made at the
# deadline and read for up to six gameweeks: about four.
LAG_DAYS = 28

# Goals per side on the scoreline grid. The mass past fifteen is nothing at
# any rate a league produces; what is there is renormalised away.
GOAL_CAP = 15

# The paired bootstrap: draws, and a fixed seed so a report reads the same
# every time it is run on the same results.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 2026


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
    strengths = strengths_before(bootstrap, fixtures, gw)
    projections = project_all(
        bootstrap, fixtures, xmins, gw, horizon=1, strengths=strengths
    )

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


def strengths_before(
    bootstrap: Bootstrap, fixtures: list[Fixture], gw: int
) -> TeamStrengths | None:
    """The team strengths as they could have been fitted at ``gw``'s deadline.

    The strength fit lives under the same no-peeking rule as the minutes
    model: only fixtures from rounds before ``gw``, dated at that gameweek's
    own deadline so the decay reads the calendar as it stood. The fit
    failing, or a codeless bootstrap, is the editorial columns — the same net
    the live run stands on. Public because the replay
    (:mod:`aigaffer.replay`) rebuilds every past deadline's board the same way.
    """
    event = next((e for e in bootstrap.events if e.id == gw), None)
    # A gameweek the bootstrap has never heard of has no deadline to date
    # the decay at; the fit stands aside rather than smuggle in today's
    # calendar and make the same backtest score two things on two days.
    if event is None:
        return None
    return build_team_strengths(
        bootstrap,
        [f for f in fixtures if f.event is not None and f.event < gw],
        event.deadline_time.date(),
    )


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


# --- the team-strength fit, graded on matches ----------------------------------


@dataclass(frozen=True)
class StrengthSetting:
    """One cell of the grid: the decay as a half-life, and the anchor."""

    half_life_days: int
    pseudo_matches: float

    @property
    def xi(self) -> float:
        return log(2) / self.half_life_days

    def label(self) -> str:
        return f"{self.half_life_days}d p{self.pseudo_matches:g}"


# The fit as it ships: a half-life of a year (XI_PER_DAY is ln2/365 to two
# significant figures) and six pseudo-matches.
CURRENT = StrengthSetting(365, 6)


@dataclass(frozen=True)
class MatchScore:
    """One match forecast and scored. ``promoted`` is ``"tilted"`` when a
    side had never been seen before the season — the fit anchors it to the
    promoted record — ``"untilted"`` when a side missed last season but was
    seen before it, so the fit treats it as known, and empty otherwise."""

    date: str
    season: int
    home: int
    away: int
    early: bool
    promoted: str
    lam_home: float
    lam_away: float
    ll: float
    brier: float
    rps: float
    cs: float


def season_of(day: str) -> int:
    """The season a ``YYYY-MM-DD`` falls in, turning in July with the ids."""
    year, month = int(day[:4]), int(day[5:7])
    return year if month >= 7 else year - 1


def score_match(
    lam_home: float, lam_away: float, home_goals: int, away_goals: int
) -> tuple[float, float, float, float]:
    """``(log-likelihood, Brier, RPS, clean-sheet Brier)`` for one match.

    The log-likelihood is of both teams' goals under independent Poissons:
    higher is better. Brier and RPS are on home, draw and away from the
    same Poissons' scoreline grid, RPS taking the outcomes in that order so
    a draw is a nearer miss than the wrong winner. The clean-sheet Brier is
    the mean of the two sides', each forecast at ``exp(-λ)`` of the other.
    Lower is better for all three.
    """
    ll = _log_poisson(home_goals, lam_home) + _log_poisson(away_goals, lam_away)
    home_pmf, away_pmf = _poisson_pmf(lam_home), _poisson_pmf(lam_away)
    p_home = p_draw = 0.0
    for i, ph in enumerate(home_pmf):
        for j, pa in enumerate(away_pmf):
            if i > j:
                p_home += ph * pa
            elif i == j:
                p_draw += ph * pa
    total = sum(home_pmf) * sum(away_pmf)
    p_home, p_draw = p_home / total, p_draw / total
    p_away = 1.0 - p_home - p_draw
    o_home = float(home_goals > away_goals)
    o_draw = float(home_goals == away_goals)
    o_away = float(home_goals < away_goals)
    brier = (p_home - o_home) ** 2 + (p_draw - o_draw) ** 2 + (p_away - o_away) ** 2
    rps = ((p_home - o_home) ** 2 + (p_home + p_draw - o_home - o_draw) ** 2) / 2
    cs = (
        (exp(-lam_away) - float(away_goals == 0)) ** 2
        + (exp(-lam_home) - float(home_goals == 0)) ** 2
    ) / 2
    return ll, brier, rps, cs


def rolling_origin(
    results: list[MatchResult],
    setting: StrengthSetting,
    *,
    lag_days: int = 0,
    teams: dict[int, set[int]] | None = None,
) -> list[MatchScore]:
    """Forecast and score every match of every season that has one before it.

    Each matchday ``d`` is forecast from a fit on the results dated before
    ``d - lag_days``, dated at that day — with no lag, strictly before the
    day itself, so a day's matches never see each other. The fit is given
    the scored season's teams, as production gives it the live twenty, so a
    promoted side has its tilted anchor on its first day rather than no
    rating. ``teams`` overrides that set per season (the live one is the
    bootstrap's); otherwise it is the codes in the season's own rows.

    Results are matched by club ``code``, so seasons join across the July
    id reset; the caller's live rows must already carry codes
    (:func:`~aigaffer.model.strength.live_results`).
    """
    ordered = sorted(results, key=lambda m: m.date)
    dates = [m.date for m in ordered]
    by_season: dict[int, list[MatchResult]] = {}
    for m in ordered:
        by_season.setdefault(season_of(m.date), []).append(m)
    seasons = sorted(by_season)
    season_teams = {
        season: {c for m in rows for c in (m.home_code, m.away_code)}
        for season, rows in by_season.items()
    }
    season_teams.update(teams or {})

    scores = []
    for season in seasons[1:]:
        rows = by_season[season]
        last = {c for m in by_season.get(season - 1, []) for c in (m.home_code, m.away_code)}
        before = {
            c
            for earlier in seasons
            if earlier < season
            for m in by_season[earlier]
            for c in (m.home_code, m.away_code)
        }
        early = _early_days(rows)
        for day, group in groupby(rows, key=lambda m: m.date):
            cutoff = date.fromisoformat(day) - timedelta(days=lag_days)
            history = ordered[: bisect_left(dates, cutoff.isoformat())]
            fit = fit_team_strengths(
                history,
                season_teams[season],
                cutoff,
                xi_per_day=setting.xi,
                pseudo_matches=setting.pseudo_matches,
            )
            for m in group:
                lam_home = exp(
                    fit.mu + fit.attack[m.home_code] + fit.defence[m.away_code] + fit.home
                )
                lam_away = exp(fit.mu + fit.attack[m.away_code] + fit.defence[m.home_code])
                sides = (m.home_code, m.away_code)
                if any(c not in before for c in sides):
                    promoted = "tilted"
                elif any(c not in last for c in sides):
                    promoted = "untilted"
                else:
                    promoted = ""
                scores.append(
                    MatchScore(
                        day, season, m.home_code, m.away_code, day in early, promoted,
                        lam_home, lam_away,
                        *score_match(lam_home, lam_away, m.home_goals, m.away_goals),
                    )
                )
    return scores


def summarise(scores: list[MatchScore]) -> dict[str, dict]:
    """Mean scores per match, overall and by slice; empty slices left out."""
    slices = {
        "all": scores,
        "early": [s for s in scores if s.early],
        "late": [s for s in scores if not s.early],
        **{
            str(season): [s for s in scores if s.season == season]
            for season in sorted({s.season for s in scores})
        },
        "promoted tilted": [s for s in scores if s.promoted == "tilted"],
        "promoted untilted": [s for s in scores if s.promoted == "untilted"],
        "established": [s for s in scores if not s.promoted],
    }
    return {
        name: {
            "n": len(chosen),
            "ll": _mean([s.ll for s in chosen]),
            "brier": _mean([s.brier for s in chosen]),
            "rps": _mean([s.rps for s in chosen]),
            "cs": _mean([s.cs for s in chosen]),
        }
        for name, chosen in slices.items()
        if chosen
    }


def paired_bootstrap(
    a: list[float],
    b: list[float],
    blocks: list,
    draws: int | None = None,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """``(mean a - b, 2.5th, 97.5th percentile)`` per match, by whole blocks.

    ``a`` and ``b`` score the same matches in the same order and ``blocks``
    says which matchday (or week) each one belongs to. A draw takes as many
    blocks as there are, with replacement, and reads the edge as the ratio of
    summed differences to matches drawn, so a heavy matchday weighs what its
    matches weigh. Seeded: the same inputs give the same interval.
    """
    draws = BOOTSTRAP_DRAWS if draws is None else draws
    groups: dict = {}
    for diff, block in zip((x - y for x, y in zip(a, b, strict=True)), blocks, strict=True):
        groups.setdefault(block, []).append(diff)
    sums = [(sum(g), len(g)) for g in groups.values()]
    observed = sum(total for total, _ in sums) / sum(n for _, n in sums)
    rng = Random(seed)
    edges = []
    for _ in range(draws):
        picked = [sums[rng.randrange(len(sums))] for _ in sums]
        edges.append(sum(t for t, _ in picked) / sum(n for _, n in picked))
    edges.sort()
    lo = edges[floor(0.025 * (draws - 1))]
    hi = edges[ceil(0.975 * (draws - 1))]
    return observed, lo, hi


def strength_eval(
    results: list[MatchResult], teams: dict[int, set[int]] | None = None
) -> dict:
    """The whole grid, the best cell, and the best set against the current.

    ``compare`` holds what the decision reads: the log-likelihood edge of
    the best cell over :data:`CURRENT` with intervals by matchday and by ISO
    week, the clean-sheet Brier edge the same two ways (negative is better),
    the same log-likelihood edge with every fit lagged :data:`LAG_DAYS`,
    and the best cell against itself at six pseudo-matches, which says
    whether the anchor is worth moving at all. Each edge is
    ``(point, low, high)``.
    """
    grid, scored = [], {}
    cells = [StrengthSetting(h, p) for h in HALF_LIVES for p in PSEUDO_GRID]
    for cell in dict.fromkeys([*cells, CURRENT]):
        scored[cell] = rolling_origin(results, cell, teams=teams)
        if cell in cells:
            grid.append(
                {
                    "half_life": cell.half_life_days,
                    "pseudo": cell.pseudo_matches,
                    "setting": cell,
                    "slices": summarise(scored[cell]),
                }
            )
    best = max(grid, key=lambda row: row["slices"]["all"]["ll"])["setting"]
    b, c = scored[best], scored[CURRENT]
    days = [s.date for s in b]
    weeks = [date.fromisoformat(s.date).isocalendar()[:2] for s in b]

    def edge(metric, left, right, blocks):
        return paired_bootstrap(
            [getattr(s, metric) for s in left], [getattr(s, metric) for s in right], blocks
        )

    lagged_best = rolling_origin(results, best, lag_days=LAG_DAYS, teams=teams)
    lagged_current = rolling_origin(results, CURRENT, lag_days=LAG_DAYS, teams=teams)
    anchored = StrengthSetting(best.half_life_days, CURRENT.pseudo_matches)
    if anchored not in scored:
        scored[anchored] = rolling_origin(results, anchored, teams=teams)
    return {
        "grid": grid,
        "best": best,
        "current": CURRENT,
        "days": len(set(days)),
        "compare": {
            "ll_matchday": edge("ll", b, c, days),
            "ll_week": edge("ll", b, c, weeks),
            "cs_matchday": edge("cs", b, c, days),
            "cs_week": edge("cs", b, c, weeks),
            "ll_lag": edge("ll", lagged_best, lagged_current, days),
            "lag_slices": (summarise(lagged_best), summarise(lagged_current)),
            "pseudo": edge("ll", b, scored[anchored], days),
            "slices": (summarise(b), summarise(c)),
        },
    }


def render_strength(evaluation: dict) -> str:
    """The evaluation as a fixed-width table for a terminal."""
    grid = evaluation["grid"]
    slices = [
        name for name in grid[0]["slices"]
        if name not in ("late",)
    ]
    header = f"{'half-life':<20}" + "".join(f"{name[:12]:>13}" for name in slices)
    lines = [
        f"Team-strength fit, rolling origin over {evaluation['days']} matchdays."
        " Mean log-likelihood per match (higher is better):",
        header,
    ]
    for row in grid:
        cell = row["setting"]
        label = cell.label() + (" (current)" if cell == evaluation["current"] else "")
        lines.append(
            f"{label:<20}"
            + "".join(
                f"{row['slices'][name]['ll']:>13.4f}" if name in row["slices"] else f"{'':>13}"
                for name in slices
            )
        )
    lines += ["", f"{'cell':<20}{'n':>6}{'brier':>9}{'rps':>9}{'cs brier':>10}"]
    for row in grid:
        whole = row["slices"]["all"]
        lines.append(
            f"{row['setting'].label():<20}{whole['n']:>6}{whole['brier']:>9.4f}"
            f"{whole['rps']:>9.4f}{whole['cs']:>10.4f}"
        )
    best, current, compare = evaluation["best"], evaluation["current"], evaluation["compare"]

    def interval(name):
        point, lo, hi = compare[name]
        return f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]"

    best_slices, current_slices = compare["slices"]
    lag_best, lag_current = compare["lag_slices"]
    lines += [
        "",
        f"Best {best.label()} against current {current.label()} (best minus current, 95%):",
        f"  log-likelihood, by matchday   {interval('ll_matchday')}",
        f"  log-likelihood, by week       {interval('ll_week')}",
        f"  clean-sheet Brier, by matchday {interval('cs_matchday')}",
        f"  clean-sheet Brier, by week    {interval('cs_week')}",
        f"  log-likelihood, {LAG_DAYS}-day lag   {interval('ll_lag')}"
        f"  (best {lag_best['all']['ll']:.4f}, current {lag_current['all']['ll']:.4f})",
        f"  {best.half_life_days}d p{best.pseudo_matches:g} against p{current.pseudo_matches:g}"
        f"   {interval('pseudo')}",
        "  by slice, log-likelihood:",
    ]
    for name, whole in best_slices.items():
        if name in current_slices:
            lines.append(
                f"    {name:<18}{whole['ll'] - current_slices[name]['ll']:+.4f}"
                f"  (n={whole['n']})"
            )
    return "\n".join(lines)


def _early_days(rows: list[MatchResult]) -> set[str]:
    """The season's opening days, the first :data:`EARLY_MATCHES` matches by
    count. A day the line falls inside goes wherever most of its matches lie:
    it is early when its middle match is."""
    early, played = set(), 0
    for day, count in sorted(Counter(m.date for m in rows).items()):
        if played + count / 2 < EARLY_MATCHES:
            early.add(day)
        played += count
    return early


def _poisson_pmf(lam: float) -> list[float]:
    pmf = [exp(-lam)]
    for k in range(1, GOAL_CAP + 1):
        pmf.append(pmf[-1] * lam / k)
    return pmf


def _log_poisson(k: int, lam: float) -> float:
    return k * log(lam) - lam - lgamma(k + 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
