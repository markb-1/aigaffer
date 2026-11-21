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


def test_the_live_season_joins_the_fit(capsys):
    # One finished live fixture on top of nothing vendored: a hand-built
    # bootstrap whose teams carry codes, and a thrashing that must show
    # up in the ratings.
    from aigaffer.data.models import Bootstrap, Event, Fixture, Team

    def team(id, code, name):
        return Team(
            id=id, name=name, short_name=name[:3].upper(), code=code,
            strength_attack_home=1, strength_attack_away=1,
            strength_defence_home=1, strength_defence_away=1,
            strength_overall_home=1, strength_overall_away=1,
        )

    from aigaffer.model.strength import build_team_strengths

    bootstrap = Bootstrap(
        events=[], teams=[team(1, 101, "Alpha"), team(2, 102, "Beta")],
        elements=[],
    )
    fixtures = [
        Fixture(
            id=1, event=1, team_h=1, team_a=2, finished=True,
            kickoff_time="2026-08-25T14:00:00Z",
            team_h_score=4, team_a_score=0,
        ),
        Fixture(id=2, event=2, team_h=2, team_a=1),  # unplayed: ignored
    ]
    fit = build_team_strengths(
        bootstrap, fixtures, TODAY, vendored=[]
    )
    assert fit is not None
    _, lam_beta_concedes = fit.factors(102, 101, opponent_at_home=False)
    _, lam_alpha_concedes = fit.factors(101, 102, opponent_at_home=False)
    assert lam_beta_concedes > lam_alpha_concedes, "the thrashing registered"


def test_a_failed_fit_says_so_and_stands_aside(capsys):
    from aigaffer.model.strength import build_team_strengths

    fit = build_team_strengths(None, None, TODAY)  # garbage in
    assert fit is None
    out = capsys.readouterr().out
    assert "team-strength fit failed" in out
    assert "FPL strength columns stand" in out


def test_a_codeless_bootstrap_stands_down_quietly(capsys):
    # Every hand-built payload predates the codes; that is the old world,
    # not a failure, and it must not spend a warning line on every test.
    from aigaffer.data.models import Bootstrap, Team
    from aigaffer.model.strength import build_team_strengths

    bare = Team(
        id=1, name="Alpha", short_name="ALP",
        strength_attack_home=1, strength_attack_away=1,
        strength_defence_home=1, strength_defence_away=1,
        strength_overall_home=1, strength_overall_away=1,
    )
    fit = build_team_strengths(
        Bootstrap(events=[], teams=[bare], elements=[]), [], TODAY
    )
    assert fit is None
    assert capsys.readouterr().out == ""


def test_the_fit_converges_before_the_cap():
    # The loop must actually reach its tolerance, not settle into a
    # step-and-recentre limit cycle that burns every sweep: the sweep
    # count rides the result so the claim is checkable.
    from aigaffer.model.strength import MAX_SWEEPS

    matches = league(
        (1, 2, 3, 0), (2, 1, 0, 3), (1, 3, 3, 0), (3, 1, 0, 3),
        (2, 3, 1, 1), (3, 2, 1, 1),
    )
    fit = fit_team_strengths(matches, {1, 2, 3}, TODAY)
    assert 0 < fit.sweeps < MAX_SWEEPS


def test_the_attack_factor_reads_the_opponents_defence_and_the_venue():
    # The half of the contract no test pinned: a leaky opponent lifts a
    # player's attacking factor above a stingy one, and his own team at
    # home lifts it above the same fixture away.
    matches = league(
        (1, 3, 1, 0), (3, 1, 0, 1),   # team 1 keeps clean sheets
        (2, 3, 2, 3), (3, 2, 3, 2),   # team 2 leaks
    )
    fit = fit_team_strengths(matches, {1, 2, 3}, TODAY)
    vs_stingy, _ = fit.factors(3, 1, opponent_at_home=False)
    vs_leaky, _ = fit.factors(3, 2, opponent_at_home=False)
    assert vs_leaky > vs_stingy

    # The venue pin needs a league where home means something: every side
    # scores twice at home and once away, and the boost must land on the
    # attacking side — the player's factor is higher when his team hosts.
    scores = {}
    for h in (1, 2, 3):
        for a in (1, 2, 3):
            if h != a:
                scores[(h, a)] = (2, 1)
    biased = fit_team_strengths(round_robin(scores), {1, 2, 3}, TODAY)
    at_home, _ = biased.factors(1, 2, opponent_at_home=False)
    away, _ = biased.factors(1, 2, opponent_at_home=True)
    assert at_home > away, "the venue boost belongs to the attacking side"


# --- the knobs the backtest turns ------------------------------------------


def two_seasons():
    """An old season where 1 beats 2, and a recent one where 2 beats 1."""
    old = [MatchResult("2025-09-01", 1, 2, 3, 0), MatchResult("2025-09-08", 2, 1, 0, 3)]
    new = [MatchResult("2026-08-20", 1, 2, 0, 2), MatchResult("2026-08-27", 2, 1, 2, 0)]
    return old + new


def test_the_constants_are_the_default_setting():
    from aigaffer.model.strength import PSEUDO_MATCHES, XI_PER_DAY

    default = fit_team_strengths(two_seasons(), {1, 2}, TODAY)
    explicit = fit_team_strengths(
        two_seasons(), {1, 2}, TODAY,
        xi_per_day=XI_PER_DAY, pseudo_matches=PSEUDO_MATCHES,
    )
    assert explicit == default


def test_the_constants_are_read_when_the_fit_runs(monkeypatch):
    # None means the module constant as it stands at call time, so a
    # patched constant reaches every caller that passes nothing.
    from aigaffer.model import strength

    default = fit_team_strengths(two_seasons(), {1, 2}, TODAY)
    monkeypatch.setattr(strength, "XI_PER_DAY", 0.02)
    assert fit_team_strengths(two_seasons(), {1, 2}, TODAY) != default


def test_a_shorter_half_life_believes_the_recent_season_more():
    slow = fit_team_strengths(two_seasons(), {1, 2}, TODAY, xi_per_day=0.0005)
    fast = fit_team_strengths(two_seasons(), {1, 2}, TODAY, xi_per_day=0.02)
    assert fast.attack[2] - fast.attack[1] > slow.attack[2] - slow.attack[1]


def test_more_pseudo_matches_pull_the_ratings_to_the_middle():
    loose = fit_team_strengths(two_seasons(), {1, 2}, TODAY, pseudo_matches=1)
    tight = fit_team_strengths(two_seasons(), {1, 2}, TODAY, pseudo_matches=20)
    assert abs(tight.attack[2]) < abs(loose.attack[2])


def test_the_live_season_joins_through_code_not_id():
    # The July id reset: id 1 is code 43 this season. A join on the id
    # would file the result under somebody else's history.
    from aigaffer.data.models import Bootstrap, Fixture, Team
    from aigaffer.model.strength import live_results

    def team(id, code, name):
        return Team(
            id=id, name=name, short_name=name[:3].upper(), code=code,
            strength_attack_home=1, strength_attack_away=1,
            strength_defence_home=1, strength_defence_away=1,
            strength_overall_home=1, strength_overall_away=1,
        )

    bootstrap = Bootstrap(
        events=[], teams=[team(1, 43, "Alpha"), team(2, 7, "Beta")], elements=[]
    )
    fixtures = [
        Fixture(
            id=1, event=1, team_h=1, team_a=2, finished=True,
            kickoff_time="2026-08-25T14:00:00Z", team_h_score=2, team_a_score=1,
        ),
        Fixture(id=2, event=2, team_h=2, team_a=1),
    ]
    assert live_results(bootstrap, fixtures) == [
        MatchResult("2026-08-25", 43, 7, 2, 1)
    ]
