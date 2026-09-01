"""The vendored prior season, as the model reads it."""

from aigaffer.model.priors import SeasonPrior, prior_rates

HAALAND = 223094  # FPL's permanent code, stable across seasons


def test_the_prior_season_loads_keyed_by_code():
    priors = prior_rates()
    haaland = priors[HAALAND]
    assert isinstance(haaland, SeasonPrior)
    assert haaland.minutes == 2953
    assert haaland.expected_goals == 25.50


def test_every_row_played_and_carries_a_sane_position():
    priors = prior_rates()
    assert len(priors) > 400, "a season has more players than this"
    assert all(p.minutes > 0 for p in priors.values())
    assert all(p.element_type in (1, 2, 3, 4) for p in priors.values())


def test_the_load_is_cached():
    assert prior_rates() is prior_rates()
