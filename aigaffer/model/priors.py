"""Last season, distilled to what the shrinkage can pool.

The FPL API's ``history_past`` carries season totals without underlying
stats, so the model cannot learn from it that Haaland was elite. This file
reads the vendored ``aigaffer/data/prior_season.csv`` instead — distilled
from the public vaastav/Fantasy-Premier-League dataset's end-of-season
``players_raw.csv`` (credit where it is due:
github.com/vaastav/Fantasy-Premier-League) — keyed by FPL's permanent
player ``code``, which survives the July id reset. To regenerate for a
new season: fetch that season's ``players_raw.csv`` from the vaastav
repo, keep this file's header columns for every row with minutes > 0,
sort by ``code``, and write it over ``aigaffer/data/prior_season.csv``
— ten lines of csv module, no dependency.

Totals, never per-90s: rates are always computed from totals by the
model's one small-sample rule (:func:`aigaffer.model.xp._per_90`), and a
column that arrived already divided has escaped it.

A missing file raises rather than degrades: the file is vendored, so its
absence is a broken install, not an off-season.
"""

import csv
from dataclasses import dataclass
from functools import cache
from importlib import resources


@dataclass(frozen=True)
class SeasonPrior:
    """One returning player's last season, as pooled evidence."""

    code: int
    element_type: int
    team_code: int
    minutes: int
    expected_goals: float
    expected_assists: float
    saves: int
    defensive_contribution: float


@cache
def prior_rates() -> dict[int, SeasonPrior]:
    """Every player who played a minute last season, keyed by ``code``."""
    path = resources.files("aigaffer.data").joinpath("prior_season.csv")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            "prior_season.csv is vendored with the package — its absence is"
            " a broken install (packaging dropped the data file), not an"
            " off-season"
        ) from error
    priors = {}
    for row in csv.DictReader(text.splitlines()):
        prior = SeasonPrior(
            code=int(row["code"]),
            element_type=int(row["element_type"]),
            team_code=int(row["team_code"]),
            minutes=int(row["minutes"]),
            expected_goals=float(row["expected_goals"]),
            expected_assists=float(row["expected_assists"]),
            saves=int(row["saves"]),
            defensive_contribution=float(row["defensive_contribution"]),
        )
        priors[prior.code] = prior
    return priors
