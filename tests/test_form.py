"""Tests for attacking form.

The subject is test_xp's midfielder: 5.0 xG and 3.0 xA over 900 minutes,
which shrunk six nineties toward the midfield prior of 0.12 each is
0.625 x 0.5 + 0.375 x 0.12 = 0.3575 xG and 0.625 x 0.3 + 0.375 x 0.12 =
0.2325 xA a ninety: a season rate of 0.59.
"""

import pytest

from aigaffer.data.models import GwHistory
from aigaffer.model.form import FORM_SHRINK_NINETIES, MAX_FORM, MIN_FORM, attacking_form
from tests.test_xp import player

SEASON_RATE = 0.59


def run(per_match: float, matches: int = 5, minutes: int = 90) -> list[GwHistory]:
    """``matches`` gameweeks of ``minutes`` each, all xG, ``per_match`` a game."""
    return [
        GwHistory(
            round=r, minutes=minutes, total_points=0,
            expected_goals=per_match, expected_assists=0.0,
        )
        for r in range(1, matches + 1)
    ]


def test_a_run_like_his_season_is_neutral():
    assert attacking_form(player(), run(SEASON_RATE)) == pytest.approx(1.0)


def test_a_modestly_hot_run_is_shrunk_toward_neutral():
    # 1.2 times his rate over five nineties: weighted 5 / (5 + 3).
    weight = 5 / (5 + FORM_SHRINK_NINETIES)
    assert attacking_form(player(), run(SEASON_RATE * 1.2)) == pytest.approx(1 + weight * 0.2)


def test_a_red_hot_run_and_a_dry_one_are_clamped():
    assert attacking_form(player(), run(SEASON_RATE * 3)) == MAX_FORM
    assert attacking_form(player(), run(0.0)) == MIN_FORM


def test_only_the_last_five_gameweeks_count():
    # Five dry weeks long ago, then five on his season rate: neutral.
    old = run(0.0)
    recent = [row.model_copy(update={"round": row.round + 5}) for row in run(SEASON_RATE)]
    assert attacking_form(player(), old + recent) == pytest.approx(1.0)


def test_less_than_a_match_is_no_evidence():
    assert attacking_form(player(), run(0.0, matches=1, minutes=45)) == 1.0


def test_a_payload_without_expected_numbers_is_no_evidence():
    rows = [GwHistory(round=r, minutes=90, total_points=2) for r in range(1, 6)]
    assert attacking_form(player(), rows) == 1.0


def test_the_api_serves_the_expected_numbers_as_strings():
    row = GwHistory.model_validate(
        {"round": 1, "minutes": 90, "total_points": 2,
         "expected_goals": "0.45", "expected_assists": "0.10"}
    )
    assert (row.expected_goals, row.expected_assists) == (0.45, 0.10)
