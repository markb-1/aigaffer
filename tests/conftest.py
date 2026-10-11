"""One rule for the whole suite: no test may reach Anthropic.

Every other network path is closed by construction — the FPL client is built
over a mock transport in the tests that use it, and the manager's loop is
handed a scripted client. The one door left open is the environment: the
manager is enabled by ``ANTHROPIC_API_KEY``, and a developer's shell has one,
so a test that builds its configuration with :meth:`Config.from_env` would
build a real client and send a real request from a test run.

That is not a hypothetical. It happened the first time the pipeline was wired
to the manager, and the failure it produced was an authentication error from
somebody's actual account rather than a test failure. So the key is taken out
of the environment for every test, and a test that wants a manager either
passes a key of its own to :class:`Config` or sets the variable itself —
either way, deliberately.
"""

import pytest


@pytest.fixture(autouse=True)
def no_anthropic_key(monkeypatch):
    """Run every test as if the machine had no Anthropic credentials."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# A session cache for repeat solves
# ---------------------------------------------------------------------------
#
# Most of the suite's wall-clock is CBC. The slow tests are full pipeline runs
# (orchestrator, manager retry, peer, early scout, what-if) over the same small
# fixture universe, so they hand the solver the *same boards* again and again:
# dozens of tests ask for the identical wildcard path, the identical free-hit
# squad, the identical one-week draft. CBC is deterministic for a given input,
# so the second answer is the first answer, bought again at several seconds.
#
# So the expensive, pure solver entry points (the multi-week path solve and the
# chip valuations) are memoised for the length of the session, under four rules that keep the cache from changing what a test checks:
#
# * The key is a hash of the function's name and the pickled arguments. The
#   arguments are the whole input, so an equal board is a hit and any change to
#   the board, the prices, the horizon or a time limit is a miss. A pickle can
#   differ for equal inputs (dict order, shared references), which only costs a
#   miss, never a wrong answer. An argument that will not pickle (a lambda, say)
#   calls straight through, uncached.
# * What is stored and returned is a deepcopy, so a test that mutates its answer
#   cannot poison the next test's.
# * It is an allow-list of modules, not a blanket. The solver's own tests
#   (test_multiweek, test_optimizer, test_plans, test_calendar, test_lineup, ...)
#   are not on it and exercise CBC every time, which is their whole job.
# * An allow-listed test that has to solve for real (one that patches CBC
#   internals to fake a time-limited stop, say, where a cached answer would hide
#   the patch) opts out with ``@pytest.mark.no_solver_cache``. Setting
#   ``AIGAFFER_NO_SOLVER_CACHE=1`` opts everything out, which is how to prove the
#   allow-listed modules still pass against the real solver. By the same token,
#   an allow-listed test that asserts on solve time (``PathSide.seconds``, the
#   what-if log's ``solve_seconds``) must carry the marker too: a cache hit
#   takes about 0 s, so the assertion would measure the cache, not the solver.
#
# The wrappers are installed per test with ``monkeypatch.setattr`` at every
# import site, because callers bind the names at import (``from ... import
# optimize_path`` in plans.py). Being per test, a test that patches the same name
# itself with a spy or a stub simply overrides the wrapper, and it is undone
# afterwards like any other patch.

import copy
import hashlib
import importlib
import os
import pickle
from functools import wraps

SOLVER_CACHE_MODULES = frozenset(
    {
        "tests.test_orchestrator",
        "tests.test_manager_retry",
        "tests.test_peer",
        "tests.test_early_scout",
        "tests.test_whatif",
    }
)

# (module that binds the name, the names it binds). Chosen by measurement, not
# by guesswork: over the allow-listed modules ``optimize_path`` took about 62 of
# the roughly 67 seconds spent in all the solver entry points (888 calls, 95
# distinct boards) and ``chip_evs`` is the next repeat (223 calls, 17 boards).
# The small ones (``optimize``, ``_free_hit_prices``, ``best_one_week_squads``
# and its helper) cost a second or two between them, so they stay uncached:
# every extra patched name is one more thing a test could trip over.
# ``chip_evs`` is a pure function of its arguments (it only calls ``optimize``
# on them) and the orchestrator is the one importer that binds the name.
SOLVER_CACHE_SITES = {
    "aigaffer.solver.plans": ("optimize_path",),
    "aigaffer.whatif": ("optimize_path",),
    "aigaffer.orchestrator": ("chip_evs",),
}

# Where each cached name is really defined. A site is wrapped only if the object
# bound there *is* that function: a module- or session-scoped fixture's stub (or
# a Mock) at the site is left alone, because memoising it would carry one test's
# fake answers into the next and its ``__module__``/``__qualname__`` cannot be
# trusted anyway.
SOLVER_CACHE_ORIGINS = {
    "optimize_path": "aigaffer.solver.multiweek",
    "chip_evs": "aigaffer.solver.lineup",
}

_SOLVER_MEMO: dict[str, object] = {}


def _memoised(func):
    """Wrap ``func`` in the session memo; one wrapper per underlying function."""
    name = f"{func.__module__}.{func.__qualname__}"

    @wraps(func)
    def cached(*args, **kwargs):
        try:
            blob = pickle.dumps((args, sorted(kwargs.items())), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return func(*args, **kwargs)
        key = hashlib.sha256(name.encode() + b"\0" + blob).hexdigest()
        if key not in _SOLVER_MEMO:
            _SOLVER_MEMO[key] = copy.deepcopy(func(*args, **kwargs))
        return copy.deepcopy(_SOLVER_MEMO[key])

    return cached


def _wrap_if_real(module, name):
    """The memoising wrapper for ``module.name``, or None if it is not the real one."""
    current = getattr(module, name)
    origin = importlib.import_module(SOLVER_CACHE_ORIGINS[name])
    if current is not getattr(origin, name):
        return None
    return _memoised(current)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_solver_cache: solve for real even in a module on the solver-cache allow-list",
    )


@pytest.fixture(autouse=True)
def solver_cache(request, monkeypatch):
    """Memoise the solver for allow-listed modules; see the section comment."""
    if os.environ.get("AIGAFFER_NO_SOLVER_CACHE"):
        return
    if request.module.__name__ not in SOLVER_CACHE_MODULES:
        return
    if request.node.get_closest_marker("no_solver_cache"):
        return
    wrappers: dict[int, object] = {}
    for module_name, names in SOLVER_CACHE_SITES.items():
        module = importlib.import_module(module_name)
        for name in names:
            original = getattr(module, name)
            if id(original) not in wrappers:
                wrappers[id(original)] = _wrap_if_real(module, name)
            wrapper = wrappers[id(original)]
            if wrapper is not None:
                monkeypatch.setattr(module, name, wrapper)
