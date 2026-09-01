"""Team strengths fitted on results, not read off editorial columns.

A time-decayed Poisson with home advantage — the Dixon-Coles construction
less its τ low-score correction, and deliberately so: τ redistributes
probability among the 0-0, 1-0, 0-1 and 1-1 cells of the exact-scoreline
joint, and nothing here consumes scorelines. What the projection model
asks for is marginal rates — how much a fixture inflates a player's
attacking returns, and how many goals his team should concede for the
clean-sheet odds — and on marginals τ's effect is second order. Wearing
the full name unearned would be a comment that lies.

The model, for one match with team ``i`` at home to ``j``::

    λ_home = exp(μ + att_i + def_j + home)
    λ_away = exp(μ + att_j + def_i)

with ``Σatt = Σdef = 0`` for identifiability, every match weighted by
``exp(-XI_PER_DAY · days_ago)``, and the fit a coordinate ascent whose
per-parameter step is the iterative-scaling one, ``log(observed /
expected)`` — forty-two parameters over a few hundred weighted matches,
milliseconds, no numpy.

Evidence is two vendored seasons (``aigaffer/data/results.csv``,
distilled from the public vaastav/Fantasy-Premier-League dataset —
credit where due — and keyed by the club ``code`` that survives the July
id reset) plus whatever finished fixtures the live API serves, so the
fit sharpens every week without anyone touching the file. The decay does
the fading the player priors do with their discount: a match a year old
carries about half weight, and last season is largely displaced by this
one around midwinter.

Thin evidence lands softly, the house way: every team also carries
:data:`PSEUDO_MATCHES` full-weight matches of league-average scoring
against a league-average phantom, so three loud fixtures cannot run a
rating away. A team with no vendored top-flight row — promoted — has
that anchor tilted (:data:`PROMOTED_ATTACK`, :data:`PROMOTED_DEFENCE`):
scoring under the average and conceding over it, instead of being
flattered as an average side until the table corrects the compliment.
"""

import csv
from dataclasses import dataclass
from datetime import date
from functools import cache
from importlib import resources
from math import exp, log

# The decay, per day: Dixon-Coles' calibrated region, a half-life of
# about a year. Same fade philosophy as the player priors' discount —
# August leans on last season, midwinter has displaced it.
XI_PER_DAY = 0.0019

# The soft anchor, in full-weight pseudo-matches of league-average
# scoring. Six is the shrinkage constant the player rates use, wearing
# team colours: enough that a rating needs real evidence to travel,
# little enough that half a season owns it outright.
PSEUDO_MATCHES = 6

# What a team nobody has seen in the top flight is anchored to: scoring
# fifteen percent under the league average and conceding fifteen percent
# over it. First-guess constants, tunable; the decay-weighted live season
# corrects them within a handful of gameweeks either way.
PROMOTED_ATTACK = 0.85
PROMOTED_DEFENCE = 1.15

# Convergence: iterative scaling on this problem moves in strides, so the
# cap is generous and the tolerance is stricter than anything downstream
# can see. Deterministic by construction — fixed sweeps, sorted teams.
MAX_SWEEPS = 200
TOLERANCE = 1e-9


@dataclass(frozen=True)
class MatchResult:
    """One finished match, dated for the decay. ``date`` is YYYY-MM-DD."""

    date: str
    home_code: int
    away_code: int
    home_goals: int
    away_goals: int


@cache
def load_results() -> list[MatchResult]:
    """The vendored seasons, oldest first. Cached; missing is broken."""
    path = resources.files("aigaffer.data").joinpath("results.csv")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "results.csv is vendored with the package — its absence is a"
            " broken install (packaging dropped the data file), not an"
            " off-season"
        ) from error
    return [
        MatchResult(
            date=row["date"],
            home_code=int(row["home_code"]),
            away_code=int(row["away_code"]),
            home_goals=int(row["home_goals"]),
            away_goals=int(row["away_goals"]),
        )
        for row in csv.DictReader(text.splitlines())
    ]


@dataclass(frozen=True)
class TeamStrengths:
    """The fitted ratings, asked one fixture at a time.

    ``factors`` keeps the exact contract ``fixture_factors`` has always
    served — ``(att_factor, lam)`` — so the projection model consumes a
    fit and the editorial columns through one shape. A code the fit never
    saw (zero included) gets the neutral answer rather than a crash: the
    factor of an average fixture, the league-average concession.
    """

    mu: float
    home: float
    attack: dict[int, float]
    defence: dict[int, float]

    def factors(
        self, team_code: int, opponent_code: int, opponent_at_home: bool
    ) -> tuple[float, float]:
        if team_code not in self.attack or opponent_code not in self.attack:
            return 1.0, exp(self.mu + self.home / 2)
        att_j = self.attack[opponent_code]
        def_j = self.defence[opponent_code]
        def_i = self.defence[team_code]
        # The player's own attack stays out of his factor — his per-90
        # rates were earned by this team's attack and multiplying it back
        # in would count it twice. What is left is the opponent's fitted
        # defence and the venue, centred on home/2 so that the average
        # fixture of a home-and-away season multiplies by one.
        venue = 0.0 if opponent_at_home else self.home
        att_factor = _bounded(exp(def_j + venue - self.home / 2))
        # The concession is the whole point of the fit: both teams in one
        # number, where the columns could only ever see the opponent.
        lam = exp(self.mu + att_j + def_i + (self.home if opponent_at_home else 0.0))
        return att_factor, lam


def fit_team_strengths(
    results: list[MatchResult], codes: set[int], as_of: date
) -> TeamStrengths:
    """Fit the league as it stands ``as_of``.

    ``codes`` is the teams the caller needs answers for — the live
    twenty. Teams that appear only in old results are fitted too (their
    matches carry information about everyone else's ratings) and simply
    never asked about. A code in ``codes`` with no real match anywhere is
    promoted, and anchors to the tilted pseudo-record.
    """
    weighted = [
        (m, exp(-XI_PER_DAY * max(0, (as_of - date.fromisoformat(m.date)).days)))
        for m in results
    ]
    teams = sorted(codes | {m.home_code for m, _ in weighted}
                   | {m.away_code for m, _ in weighted})
    seen = {m.home_code for m, _ in weighted} | {m.away_code for m, _ in weighted}

    # League-average goals per team per match, weighted, for the pseudo
    # anchor. A fit with no matches at all still needs a scale; 1.4 is
    # the league's long-run figure and only ever used in that vacuum.
    total_w = sum(w for _, w in weighted)
    avg = (
        sum(w * (m.home_goals + m.away_goals) / 2 for m, w in weighted) / total_w
        if total_w > 0
        else 1.4
    )

    # The anchor, per team: scored and conceded per pseudo-match against
    # a phantom of exactly average attack and defence, no venue.
    anchor = {
        code: (avg * PROMOTED_ATTACK, avg * PROMOTED_DEFENCE)
        if code not in seen
        else (avg, avg)
        for code in teams
    }

    mu = log(avg) if avg > 0 else 0.0
    home = 0.0
    attack = {code: 0.0 for code in teams}
    defence = {code: 0.0 for code in teams}

    for _ in range(MAX_SWEEPS):
        biggest = 0.0

        # μ and home first: scale before shape.
        scored = total_expected = 0.0
        home_scored = home_expected = 0.0
        for m, w in weighted:
            lam_h = exp(mu + attack[m.home_code] + defence[m.away_code] + home)
            lam_a = exp(mu + attack[m.away_code] + defence[m.home_code])
            scored += w * (m.home_goals + m.away_goals)
            total_expected += w * (lam_h + lam_a)
            home_scored += w * m.home_goals
            home_expected += w * lam_h
        for code in teams:
            for_goals, against_goals = anchor[code]
            scored += PSEUDO_MATCHES * (for_goals + against_goals)
            total_expected += PSEUDO_MATCHES * (
                exp(mu + attack[code]) + exp(mu + defence[code])
            )
        if total_expected > 0 and scored > 0:
            step = log(scored / total_expected)
            mu += step
            biggest = max(biggest, abs(step))
        if home_expected > 0 and home_scored > 0:
            step = log(home_scored / home_expected)
            home += step
            biggest = max(biggest, abs(step))

        # Then every attack and defence, by iterative scaling.
        for code in teams:
            s = e = c = ce = 0.0
            for m, w in weighted:
                if m.home_code == code:
                    s += w * m.home_goals
                    e += w * exp(mu + attack[code] + defence[m.away_code] + home)
                    c += w * m.away_goals
                    ce += w * exp(mu + attack[m.away_code] + defence[code])
                elif m.away_code == code:
                    s += w * m.away_goals
                    e += w * exp(mu + attack[code] + defence[m.home_code])
                    c += w * m.home_goals
                    ce += w * exp(mu + attack[m.home_code] + defence[code] + home)
            for_goals, against_goals = anchor[code]
            s += PSEUDO_MATCHES * for_goals
            e += PSEUDO_MATCHES * exp(mu + attack[code])
            c += PSEUDO_MATCHES * against_goals
            ce += PSEUDO_MATCHES * exp(mu + defence[code])
            if e > 0 and s > 0:
                step = log(s / e)
                attack[code] += step
                biggest = max(biggest, abs(step))
            if ce > 0 and c > 0:
                step = log(c / ce)
                defence[code] += step
                biggest = max(biggest, abs(step))

        # Re-centre so the sums stay zero, folding the means into μ so
        # that no λ moves an inch in the process.
        mean_att = sum(attack.values()) / len(teams)
        mean_def = sum(defence.values()) / len(teams)
        for code in teams:
            attack[code] -= mean_att
            defence[code] -= mean_def
        mu += mean_att + mean_def

        if biggest < TOLERANCE:
            break

    return TeamStrengths(mu=mu, home=home, attack=attack, defence=defence)


def build_team_strengths(
    bootstrap, fixtures, as_of: date, vendored: list[MatchResult] | None = None
) -> TeamStrengths | None:
    """The fit as the pipeline asks for it: vendored past, live present.

    The live season's finished fixtures join the vendored results through
    each team's permanent ``code``; a fixture whose teams carry no code —
    a hand-built test bootstrap — is skipped rather than mis-joined, and
    an unfinished or scoreless fixture never counts. The two sources
    cannot overlap: vendored rows are past seasons', live rows are this
    one's.

    Any failure — a payload shape this never saw, a date that will not
    parse, the file itself — is one line on stdout and a None, and the
    caller runs the editorial columns exactly as before this model
    existed. Never the exception's text: the house rule about what lives
    inside exceptions.

    ``vendored`` exists for the tests; the pipeline leaves it None and
    gets the packaged seasons.
    """
    try:
        results = list(load_results() if vendored is None else vendored)
        codes = {team.id: team.code for team in bootstrap.teams}
        for fixture in fixtures:
            home, away = codes.get(fixture.team_h, 0), codes.get(fixture.team_a, 0)
            if (
                not fixture.finished
                or fixture.team_h_score is None
                or fixture.team_a_score is None
                or fixture.kickoff_time is None
                or not home
                or not away
            ):
                continue
            results.append(
                MatchResult(
                    date=fixture.kickoff_time[:10],
                    home_code=home,
                    away_code=away,
                    home_goals=fixture.team_h_score,
                    away_goals=fixture.team_a_score,
                )
            )
        wanted = {code for code in codes.values() if code}
        if not wanted:
            # Not a failure and not worth a line: a bootstrap where no team
            # carries a code is a payload from before the codes were read —
            # every hand-built fixture in the tests — and the convention for
            # those is the same as Player.code's: no code, no model, quietly.
            return None
        return fit_team_strengths(results, wanted, as_of)
    except Exception as error:  # noqa: BLE001 — the fallback is the feature
        print(
            f"aigaffer: team-strength fit failed ({type(error).__name__});"
            " FPL strength columns stand"
        )
        return None


def _bounded(factor: float) -> float:
    """The projection model's own clamp, imported lazily to avoid the
    circular import: xp reads this module's fit, and this module borrows
    xp's idea of how far a fixture may swing a rate."""
    from aigaffer.model.xp import _clamp

    return _clamp(factor)
