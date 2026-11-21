"""Tests for the rolling-origin evaluation of the team-strength fit.

Invented leagues throughout: codes in the hundreds, scores made up. What is
pinned is the no-peek rule (nothing from the scored day, or from inside the
lag, reaches the fit), the scoring rules on hand-computed numbers, the slices
the report is cut by, and the bootstrap's arithmetic.
"""

from datetime import date, timedelta

import pytest

from aigaffer import __main__ as cli
from aigaffer.backtest import (
    CURRENT,
    StrengthSetting,
    paired_bootstrap,
    rolling_origin,
    score_match,
    season_of,
    strength_eval,
)
from aigaffer.model.strength import XI_PER_DAY, MatchResult

TEAMS = (101, 102, 103, 104)


def season_rows(year: int, teams=TEAMS, start: str = "-08-16") -> list[MatchResult]:
    """A double round robin, one match a day from mid-August, varied scores."""
    day = date.fromisoformat(f"{year}{start}")
    rows = []
    for i, h in enumerate(teams):
        for j, a in enumerate(teams):
            if h != a:
                rows.append(MatchResult(day.isoformat(), h, a, (i + 2 * j) % 4, (j + i) % 3))
                day += timedelta(days=1)
    return rows


# --- the setting ------------------------------------------------------------


def test_the_current_setting_is_the_constants():
    assert CURRENT.half_life_days == 365
    assert CURRENT.pseudo_matches == 6
    assert CURRENT.xi == pytest.approx(XI_PER_DAY, abs=5e-6)


def test_the_season_turns_in_july():
    assert season_of("2025-08-16") == 2025
    assert season_of("2026-05-24") == 2025
    assert season_of("2026-07-01") == 2026


# --- the scoring rules --------------------------------------------------------


def test_a_goalless_draw_between_two_one_goal_sides():
    # λ = 1 each way, 0-0. Log-likelihood is log(e^-1)·2 = -2. The draw's
    # probability is e^-2·I0(2) = 0.308508; each win takes half the rest.
    ll, brier, rps, cs = score_match(1.0, 1.0, 0, 0)
    p_draw = 0.3085083
    p_win = (1 - p_draw) / 2
    assert ll == pytest.approx(-2.0)
    assert brier == pytest.approx(2 * p_win**2 + (1 - p_draw) ** 2, abs=1e-6)
    assert rps == pytest.approx(0.5 * (p_win**2 + p_win**2), abs=1e-6)
    # Both clean sheets kept, each forecast at e^-1.
    assert cs == pytest.approx((1 - 0.3678794) ** 2, abs=1e-6)


def test_a_certain_home_win_scores_near_perfect():
    ll, brier, rps, cs = score_match(5.0, 1e-6, 3, 0)
    assert brier == pytest.approx(0.0, abs=1e-3)
    assert rps == pytest.approx(0.0, abs=1e-3)
    # The away clean sheet was not kept and was never on; the home one
    # was kept and was all but certain.
    assert cs == pytest.approx(0.0, abs=1e-3)
    assert ll < 0


def test_the_ranked_score_punishes_a_near_miss_less_than_a_far_one():
    # Forecasting a home win and seeing a draw is a smaller miss than
    # seeing an away win: the RPS knows the outcomes are ordered.
    _, _, near, _ = score_match(2.5, 0.5, 1, 1)
    _, _, far, _ = score_match(2.5, 0.5, 0, 1)
    assert near < far


# --- the rolling origin --------------------------------------------------------


def two_seasons():
    return season_rows(2025) + season_rows(2026)


def test_only_seasons_with_history_are_scored():
    scores = rolling_origin(two_seasons(), CURRENT)
    assert {s.season for s in scores} == {2026}
    assert len(scores) == 12


def test_nothing_from_the_scored_day_reaches_the_fit():
    rows = two_seasons()
    scored_day = rows[-3].date
    # A second match on that day, a thrashing: the first match's forecast
    # must not move, because the fit is made before either kicks off.
    planted = rows + [MatchResult(scored_day, 103, 104, 9, 0)]
    before = [s for s in rolling_origin(rows, CURRENT) if s.date == scored_day]
    after = [s for s in rolling_origin(planted, CURRENT) if s.date == scored_day]
    assert (before[0].lam_home, before[0].lam_away) == (after[0].lam_home, after[0].lam_away)
    assert len(after) == 2


def test_the_lag_keeps_recent_results_out():
    rows = two_seasons()
    last = rows[-1].date
    # A thrashing ten days before the last match: it moves the forecast
    # with no lag, and is invisible at a 28-day lag.
    day = (date.fromisoformat(last) - timedelta(days=10)).isoformat()
    planted = [r for r in rows if r.date != day] + [MatchResult(day, 101, 102, 9, 0)]

    def lam(results, lag):
        scores = rolling_origin(results, CURRENT, lag_days=lag)
        final = next(s for s in scores if s.date == last)
        return final.lam_home, final.lam_away

    assert lam(rows, 0) != lam(planted, 0)
    assert lam(rows, 28) == lam(planted, 28)


def test_a_promoted_side_scores_on_its_first_day_under_the_tilt(monkeypatch):
    # 105 is new in 2026 and plays on the season's first day: the fit is
    # given the season's teams, so it has a rating — the tilted anchor —
    # rather than no rating and a KeyError.
    from aigaffer.model import strength

    old = season_rows(2025)
    new = season_rows(2026, teams=(105, 101, 102, 103))
    first = rolling_origin(old + new, CURRENT)[0]
    assert first.date == new[0].date
    assert (first.home, first.promoted) == (105, "new")
    # The tilt is what set that forecast: untilted, 105 would score more.
    monkeypatch.setattr(strength, "PROMOTED_ATTACK", 1.0)
    untilted = rolling_origin(old + new, CURRENT)[0]
    assert first.lam_home < untilted.lam_home


def test_a_match_between_established_sides_is_not_in_the_promoted_slice():
    old = season_rows(2025)
    new = season_rows(2026, teams=(105, 101, 102, 103))
    scores = rolling_origin(old + new, CURRENT)
    assert {s.promoted for s in scores if 105 not in (s.home, s.away)} == {""}


def test_a_side_back_after_a_season_away_is_returning_not_new():
    # 104 played in 2024, sat out 2025, and is back in 2026: production
    # has seen it, so it gets no tilt — and the slice must say so.
    rows = (
        season_rows(2024)
        + season_rows(2025, teams=(101, 102, 103, 105))
        + season_rows(2026)
    )
    labels = {s.promoted for s in rolling_origin(rows, CURRENT) if s.season == 2026 and 104 in (s.home, s.away)}
    assert labels == {"returning"}


def test_early_season_is_the_first_matches_by_count():
    from aigaffer import backtest

    rows = season_rows(2025) + season_rows(2026)
    original = backtest.EARLY_MATCHES
    try:
        backtest.EARLY_MATCHES = 5
        scores = rolling_origin(rows, CURRENT)
    finally:
        backtest.EARLY_MATCHES = original
    assert [s.early for s in scores] == [True] * 5 + [False] * 7


def test_a_matchday_straddling_the_line_goes_where_most_of_it_lies():
    from aigaffer import backtest

    day = "2026-08-20"
    rows = season_rows(2025) + [
        MatchResult("2026-08-16", 101, 102, 1, 0),
        MatchResult("2026-08-17", 103, 104, 1, 0),
        # Three on one day, with the line after the first of them: the day
        # is mostly past it, so it is not early.
        MatchResult(day, 102, 101, 1, 1),
        MatchResult(day, 104, 103, 0, 2),
        MatchResult(day, 101, 103, 2, 2),
    ]
    original = backtest.EARLY_MATCHES
    try:
        backtest.EARLY_MATCHES = 3
        scores = rolling_origin(rows, CURRENT)
    finally:
        backtest.EARLY_MATCHES = original
    assert [s.early for s in scores] == [True, True, False, False, False]


# --- the bootstrap --------------------------------------------------------------


def test_identical_forecasts_differ_by_nothing():
    a = [-2.0, -3.0, -1.5, -2.5]
    assert paired_bootstrap(a, list(a), ["d1", "d1", "d2", "d3"]) == (0.0, 0.0, 0.0)


def test_a_uniform_edge_is_recovered_exactly():
    a = [-2.0, -3.0, -1.5, -2.5]
    b = [x - 0.1 for x in a]
    diff, lo, hi = paired_bootstrap(a, b, ["d1", "d1", "d2", "d3"])
    assert diff == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_the_bootstrap_is_the_same_every_time():
    a = [-2.0, -3.0, -1.5, -2.5, -1.0, -2.2]
    b = [-2.1, -2.8, -1.7, -2.4, -1.3, -2.0]
    blocks = ["d1", "d2", "d3", "d4", "d5", "d6"]
    first = paired_bootstrap(a, b, blocks)
    assert first == paired_bootstrap(a, b, blocks)
    diff, lo, hi = first
    assert lo <= diff <= hi
    assert lo < hi


def test_resampling_whole_blocks_keeps_a_block_together():
    # One block holds all the edge: with two blocks the resampled edge is
    # only ever 0, half or all of it, never anything between.
    a = [1.0, 1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    diff, lo, hi = paired_bootstrap(a, b, ["w1", "w1", "w2", "w2"])
    assert diff == pytest.approx(0.5)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


# --- the grid and the command ---------------------------------------------------


def small_grid(monkeypatch):
    from aigaffer import backtest

    monkeypatch.setattr(backtest, "HALF_LIVES", (91, 365))
    monkeypatch.setattr(backtest, "PSEUDO_GRID", (3, 6))
    monkeypatch.setattr(backtest, "BOOTSTRAP_DRAWS", 50)


def test_the_evaluation_covers_the_grid_and_names_the_current_cell(monkeypatch):
    small_grid(monkeypatch)
    evaluation = strength_eval(two_seasons())
    cells = {(row["half_life"], row["pseudo"]) for row in evaluation["grid"]}
    assert cells == {(91, 3), (91, 6), (365, 3), (365, 6)}
    assert evaluation["current"] == StrengthSetting(365, 6)
    assert evaluation["best"] in {StrengthSetting(h, p) for h, p in cells}
    best_row = max(evaluation["grid"], key=lambda row: row["slices"]["all"]["ll"])
    assert (best_row["half_life"], best_row["pseudo"]) == (
        evaluation["best"].half_life_days,
        evaluation["best"].pseudo_matches,
    )
    for name in ("ll_matchday", "ll_week", "cs_matchday", "cs_week", "ll_lag", "pseudo"):
        assert name in evaluation["compare"]


def test_the_command_prints_the_grid(monkeypatch, capsys):
    from tests.test_backtest import backtest_routes, serve

    small_grid(monkeypatch)
    monkeypatch.setattr(cli, "load_results", lambda: two_seasons())
    serve(monkeypatch, backtest_routes())

    assert cli.main(["backtest", "--strength"]) == 0

    printed = capsys.readouterr().out
    assert "half-life" in printed
    assert "365d p6 (current)" in printed
