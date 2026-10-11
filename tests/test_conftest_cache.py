"""The solver cache only wraps the real solver functions, never a stub.

``conftest.solver_cache`` installs its memoising wrappers at the import sites.
A module- or session-scoped fixture may already have replaced the object at a
site with a stub (a plain function, a ``Mock``, a callable class instance); the
wrapper naming and keying assume a real function, and worse, memoising a stub
would hand one test's fake answers to the next. So the site is wrapped only
when the object there *is* the function its defining module exports.
"""

from types import SimpleNamespace

from tests.conftest import SOLVER_CACHE_ORIGINS, SOLVER_CACHE_SITES, _wrap_if_real


def _site(**names):
    return SimpleNamespace(**names)


def test_the_real_solver_function_is_wrapped():
    from aigaffer.solver import plans
    from aigaffer.solver.multiweek import optimize_path

    assert plans.optimize_path is optimize_path  # nothing has patched it
    wrapper = _wrap_if_real(plans, "optimize_path")
    assert wrapper is not None
    assert wrapper is not optimize_path
    assert wrapper.__wrapped__ is optimize_path


def test_a_stub_at_the_site_is_left_alone():
    def stub(*args, **kwargs):
        return "fake"

    stub.__module__ = "aigaffer.solver.multiweek"  # even a stub that lies
    stub.__qualname__ = "optimize_path"

    assert _wrap_if_real(_site(optimize_path=stub), "optimize_path") is None


def test_anything_that_is_not_the_origins_own_function_is_left_alone():
    class Callable:
        def __call__(self, *args, **kwargs):
            return "fake"

    assert _wrap_if_real(_site(optimize_path=Callable()), "optimize_path") is None
    assert _wrap_if_real(_site(optimize_path=lambda: 1), "optimize_path") is None


def test_every_site_name_has_a_known_origin():
    # A name added to SOLVER_CACHE_SITES without an origin would KeyError in the
    # fixture, so check the sites themselves rather than a hand-kept list.
    for names in SOLVER_CACHE_SITES.values():
        for name in names:
            assert name in SOLVER_CACHE_ORIGINS
