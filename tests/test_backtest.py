"""Tests for the backtest harness: the rank statistics, and the join.

There is no scipy to check the statistics against, so ``_spearman`` is pinned
on hand-computed vectors — including a tied one, which is where a rank
correlation is easiest to get wrong.

The harness itself runs over the ``PIPELINE_*`` universe from
:mod:`tests.fixtures` with fabricated per-player histories, so what the
assertions are about is the join: who gets counted, what the actual score is
compared against, and that nothing from the gameweek being scored — or from
any gameweek after it — reaches the minutes model.
"""

import copy

import httpx
import pytest

from aigaffer import __main__ as cli
from aigaffer.backtest import _spearman, backtest_gw, finished_gameweeks
from aigaffer.data.fpl_api import FplClient
from tests.fixtures import (
    PIPELINE_BOOTSTRAP_JSON,
    PIPELINE_ELEMENTS_JSON,
    PIPELINE_FIXTURES_JSON,
    fake_fpl_transport,
)

# The gameweek under test, with GW1 behind it and GW3 ahead of it.
GW = 2

EVERY_PLAYER = [element["id"] for element in PIPELINE_ELEMENTS_JSON]

# The universe is 21 players; the doubtful Dodd and the injured Ito are not on
# the pipeline's shortlist, so they are not on the backtest's either.
POPULATION = 19


# --- the rank statistic ----------------------------------------------------


def test_a_perfect_ranking_correlates_at_one():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0


def test_a_reversed_ranking_correlates_at_minus_one():
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0


def test_only_the_order_counts():
    # Spearman is Pearson on the ranks: the shape of the values is irrelevant.
    assert _spearman([1, 2, 3, 4], [1, 4, 9, 100]) == 1.0


def test_ties_share_the_average_of_their_places():
    # Ranks are [1, 2.5, 2.5, 4] against [1, 2, 3, 4]: 4.5 / sqrt(4.5 * 5).
    assert _spearman([1, 2, 2, 3], [10, 20, 30, 40]) == pytest.approx(0.94868329805)


def test_too_few_points_correlate_at_nothing():
    assert _spearman([], []) == 0.0
    assert _spearman([1], [2]) == 0.0


def test_a_series_that_never_varies_correlates_at_nothing():
    assert _spearman([1, 1, 1], [1, 2, 3]) == 0.0
    assert _spearman([1, 2, 3], [7, 7, 7]) == 0.0


# --- the harness -----------------------------------------------------------


def summary(*rounds: tuple[int, int, int]) -> dict:
    """An element-summary payload from ``(round, minutes, points)`` triples."""
    return {
        "fixtures": [],
        "history": [
            {
                "element": 1,
                "fixture": rnd,
                "round": rnd,
                "minutes": minutes,
                "total_points": points,
                "bonus": 0,
            }
            for rnd, minutes, points in rounds
        ],
        "history_past": [],
    }


def season(pid: int) -> dict:
    """Two gameweeks played, scoring unlike the rest of the field.

    The scores have to differ from player to player or there is no order to
    correlate with — and some of them have to clear five, or the hit rate is
    zero whatever the model says.
    """
    return summary((1, 90, pid % 5), (GW, 90, (pid * 3) % 9))


def backtest_routes(
    histories: dict[int, dict] | None = None, bootstrap: dict = PIPELINE_BOOTSTRAP_JSON
) -> dict:
    """The endpoints a backtest touches; ``histories`` overrides by player id."""
    summaries = {
        element["id"]: season(element["id"]) for element in PIPELINE_ELEMENTS_JSON
    }
    summaries.update(histories or {})
    return {
        "/api/bootstrap-static/": bootstrap,
        "/api/fixtures/": PIPELINE_FIXTURES_JSON,
        **{
            f"/api/element-summary/{pid}/": payload
            for pid, payload in summaries.items()
        },
    }


def make_client(routes: dict) -> FplClient:
    return FplClient(http=httpx.Client(transport=fake_fpl_transport(routes)))


def run(routes: dict, gw: int = GW) -> dict:
    client = make_client(routes)
    return backtest_gw(client, client.bootstrap(), client.fixtures(), gw)


def preseason_bootstrap() -> dict:
    """The universe before a ball is kicked: GW1 next, nothing finished."""
    payload = copy.deepcopy(PIPELINE_BOOTSTRAP_JSON)
    payload["events"][0].update(is_current=False, is_next=True, finished=False)
    payload["events"][1].update(is_next=False)
    return payload


def test_the_backtest_scores_the_gameweek():
    result = run(backtest_routes())

    assert set(result) == {"gw", "n", "spearman", "top20_hit_rate"}
    assert result["gw"] == GW
    assert result["n"] == POPULATION
    assert -1.0 <= result["spearman"] <= 1.0
    assert 0.0 <= result["top20_hit_rate"] <= 1.0


def test_a_player_who_did_not_play_that_gameweek_is_not_counted():
    # Jarvis has a season, but none of it is the gameweek being scored: there
    # is nothing to compare his projection with.
    routes = backtest_routes({9: summary((1, 90, 3), (3, 90, 7))})

    assert run(routes)["n"] == POPULATION - 1


def test_a_gameweek_nobody_played_scores_nothing():
    routes = backtest_routes({pid: summary((1, 90, 4)) for pid in EVERY_PLAYER})

    assert run(routes) == {"gw": GW, "n": 0, "spearman": 0.0, "top20_hit_rate": 0.0}


def test_a_double_gameweek_is_scored_over_both_fixtures():
    # Everyone takes three points off the gameweek, which is not a return.
    # Ferrer plays twice and takes three off each, which is.
    histories = {pid: summary((1, 90, 2), (GW, 90, 3)) for pid in EVERY_PLAYER}
    histories[5] = summary((1, 90, 2), (GW, 90, 3), (GW, 90, 3))

    result = run(backtest_routes(histories))

    assert result["n"] == POPULATION
    assert result["top20_hit_rate"] == pytest.approx(1 / POPULATION, abs=0.001)


def test_the_gameweek_being_scored_never_reaches_the_minutes_model():
    # Same actual scores, wildly different minutes in the gameweek itself and
    # in the one after it. A model that peeked would rank the field
    # differently; this one has not been told, so the numbers must not move.
    peekable = {
        pid: summary(
            (1, 90, pid % 5),
            (GW, 90 * (pid % 2), (pid * 3) % 9),
            (3, 90 * ((pid + 1) % 2), 9),
        )
        for pid in EVERY_PLAYER
    }

    assert run(backtest_routes(peekable)) == run(backtest_routes())


def test_the_hit_rate_is_the_share_of_the_top_who_returned():
    returned = {pid: summary((1, 90, 2), (GW, 90, 6)) for pid in EVERY_PLAYER}
    blanked = {pid: summary((1, 90, 2), (GW, 90, 4)) for pid in EVERY_PLAYER}

    assert run(backtest_routes(returned))["top20_hit_rate"] == 1.0
    assert run(backtest_routes(blanked))["top20_hit_rate"] == 0.0


def test_the_finished_gameweeks_are_the_ones_to_backtest():
    client = make_client(backtest_routes())
    preseason = make_client(backtest_routes(bootstrap=preseason_bootstrap()))

    assert finished_gameweeks(client.bootstrap()) == [1]
    assert finished_gameweeks(preseason.bootstrap()) == []


# --- the command line ------------------------------------------------------


def serve(monkeypatch, routes: dict) -> None:
    monkeypatch.setattr(cli, "FplClient", lambda: make_client(routes))


def test_the_cli_backtests_the_gameweek_it_is_given(monkeypatch, capsys, tmp_path):
    # No team, no state: a backtest is about the model, not about a manager.
    monkeypatch.delenv("FPL_TEAM_ID", raising=False)
    monkeypatch.setenv("AIGAFFER_STATE_DIR", str(tmp_path))
    serve(monkeypatch, backtest_routes())

    assert cli.main(["backtest", "--gw", "1"]) == 0

    printed = capsys.readouterr().out
    assert "'gw': 1" in printed
    assert "'spearman'" in printed
    assert list(tmp_path.iterdir()) == []


def test_the_cli_backtests_the_last_finished_gameweek_by_default(monkeypatch, capsys):
    serve(monkeypatch, backtest_routes())

    assert cli.main(["backtest"]) == 0
    assert "'gw': 1" in capsys.readouterr().out


def test_a_gameweek_still_being_played_is_refused(monkeypatch, capsys):
    serve(monkeypatch, backtest_routes())

    assert cli.main(["backtest", "--gw", "2"]) == 1
    assert "GW2 has not finished" in capsys.readouterr().out


def test_a_season_with_nothing_finished_is_said_so(monkeypatch, capsys):
    serve(monkeypatch, backtest_routes(bootstrap=preseason_bootstrap()))

    assert cli.main(["backtest"]) == 1
    assert "no gameweek has finished" in capsys.readouterr().out


def test_the_gameweek_being_scored_never_reaches_the_strength_fit(monkeypatch):
    # The fit gets only fixtures from rounds before the one being scored,
    # dated at that gameweek's own deadline — the same no-peeking rule the
    # minutes model lives under, enforced at the seam.
    from aigaffer import backtest as bt

    seen = {}

    def spy(bootstrap, fixtures, as_of, vendored=None):
        seen["events"] = {f.event for f in fixtures}
        seen["as_of"] = as_of
        return None

    monkeypatch.setattr(bt, "build_team_strengths", spy)
    run(backtest_routes())

    assert seen, "the backtest asked the fit"
    assert all(event is not None and event < GW for event in seen["events"])
