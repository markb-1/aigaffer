"""The team-strength fit, on leagues where the answer is known."""

from datetime import date

from aigaffer.model.strength import (
    MatchResult,
    fit_team_strengths,
    load_results,
)

TODAY = date(2026, 9, 1)


def league(*results: tuple[int, int, int, int], when: str = "2026-08-01"):
    """Matches as (home, away, hg, ag), all on one recent day."""
    return [MatchResult(when, h, a, hg, ag) for h, a, hg, ag in results]


def round_robin(scores: dict, when: str = "2026-08-01"):
    return [MatchResult(when, h, a, hg, ag) for (h, a), (hg, ag) in scores.items()]


def test_a_dominant_team_fits_a_dominant_rating():
    # Team 1 wins every match 3-0, home and away, against 2 and 3, who
    # draw each other 1-1. The fit must call 1 the best attack and the
    # best defence on the board.
    matches = league(
        (1, 2, 3, 0), (2, 1, 0, 3), (1, 3, 3, 0), (3, 1, 0, 3),
        (2, 3, 1, 1), (3, 2, 1, 1),
    )
    fit = fit_team_strengths(matches, {1, 2, 3}, TODAY)
    _, lam_vs_1 = fit.factors(2, 1, opponent_at_home=False)
    _, lam_vs_2 = fit.factors(1, 2, opponent_at_home=False)
    assert lam_vs_1 > lam_vs_2, "facing team 1 concedes more than facing team 2"


def test_the_strong_defence_concedes_less_than_the_weak_one():
    # The pin this upgrade exists for: two teams face the SAME opponent,
    # and the one with the stingy record concedes fewer expected goals.
    # Today's model literally cannot say this.
    matches = league(
        (1, 3, 1, 0), (3, 1, 0, 1),   # team 1 keeps clean sheets
        (2, 3, 2, 3), (3, 2, 3, 2),   # team 2 leaks
    )
    fit = fit_team_strengths(matches, {1, 2, 3}, TODAY)
    _, lam_stingy = fit.factors(1, 3, opponent_at_home=True)
    _, lam_leaky = fit.factors(2, 3, opponent_at_home=True)
    assert lam_stingy < lam_leaky


def test_home_advantage_is_recovered():
    # Every team scores twice at home, once away: the fitted home boost
    # must make an opponent at home a worse prospect for a clean sheet.
    scores = {}
    for h in (1, 2, 3):
        for a in (1, 2, 3):
            if h != a:
                scores[(h, a)] = (2, 1)
    fit = fit_team_strengths(round_robin(scores), {1, 2, 3}, TODAY)
    _, lam_opp_home = fit.factors(1, 2, opponent_at_home=True)
    _, lam_opp_away = fit.factors(1, 2, opponent_at_home=False)
    assert lam_opp_home > lam_opp_away


def test_an_old_thrashing_is_outweighed_by_a_recent_one():
    # The same two teams, opposite results, two years apart: the fit
    # believes the recent match more.
    matches = [
        MatchResult("2024-09-01", 1, 2, 0, 4),
        MatchResult("2026-08-25", 1, 2, 4, 0),
    ]
    fit = fit_team_strengths(matches, {1, 2}, TODAY)
    _, lam_1_faces_2 = fit.factors(1, 2, opponent_at_home=False)
    _, lam_2_faces_1 = fit.factors(2, 1, opponent_at_home=False)
    assert lam_2_faces_1 > lam_1_faces_2, "team 1 is the recent thrasher"


def test_a_promoted_side_reads_below_average():
    # Team 4 has no matches at all. Its pseudo-anchor is tilted: it scores
    # under the average and concedes over it, instead of being flattered
    # as a league-average side.
    matches = league((1, 2, 1, 1), (2, 3, 1, 1), (3, 1, 1, 1))
    fit = fit_team_strengths(matches, {1, 2, 3, 4}, TODAY)
    _, lam_faces_promoted = fit.factors(1, 4, opponent_at_home=False)
    _, lam_faces_average = fit.factors(1, 2, opponent_at_home=False)
    assert lam_faces_promoted < lam_faces_average, "weaker attack to face"
    _, lam_promoted_concedes = fit.factors(4, 1, opponent_at_home=False)
    _, lam_average_concedes = fit.factors(2, 1, opponent_at_home=False)
    assert lam_promoted_concedes > lam_average_concedes, "leakier defence"


def test_the_vendored_results_load():
    results = load_results()
    assert len(results) == 760, "two full seasons"
    assert all(r.home_goals >= 0 and r.away_goals >= 0 for r in results)
    assert load_results() is load_results()
