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


class AnthropicUnreachable(Exception):
    """What the detailed writer's client raises in a test that did not script one."""


class _UnreachableClient:
    """A client for the detailed writer whose every request is refused, here."""

    def __init__(self, cfg) -> None:
        self.messages = self

    def create(self, **_):
        raise AnthropicUnreachable("tests never reach Anthropic")


@pytest.fixture(autouse=True)
def no_reasons_request(monkeypatch):
    """Close the second door: the detailed writer's own client.

    A pipeline test that wants a gaffer stubs ``run_manager`` and hands the
    config a key of its own, deliberately — and the run then asks the gaffer
    for his reasons (:mod:`aigaffer.manager.detailed_writer`) with a real
    client built on that key, which the stub never stood in for. So that
    client is replaced for every test with one that refuses each request:
    the writer fails as it would against a dead network, the run goes on
    without the reasons as it must, and a test that wants them scripts its
    own client and passes it in.

    The question handler (:mod:`aigaffer.ask`) builds the same client the
    same way, lazily, on the first question that needs the gaffer; a test
    that wires the inbox with a key and no scripted client would otherwise
    send that question for real. It gets the same refusing client, and the
    question fails into "ask again" as it would against a dead network.
    """
    monkeypatch.setattr(
        "aigaffer.manager.detailed_writer.build_review_client", _UnreachableClient
    )
    monkeypatch.setattr("aigaffer.ask.build_review_client", _UnreachableClient)


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
# chip valuations) are memoised for the length of the session, under four rules
# that keep the cache from changing what a test checks:
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
        # The settings in force are part of the input: the solver reads them at
        # call time, so a cached answer must never cross from one to another.
        from aigaffer.solver import optimizer

        try:
            blob = pickle.dumps(
                (args, sorted(kwargs.items()), optimizer.DEFAULT_SETTINGS),
                protocol=pickle.HIGHEST_PROTOCOL,
            )
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
    config.addinivalue_line(
        "markers",
        "live_settings: run under the live WINDOW settings instead of the suite's LEGACY",
    )


# The live settings as the source has them, read before the session patch below.
_LIVE_SETTINGS: list = []


@pytest.fixture(autouse=True, scope="session")
def legacy_session_settings():
    """LEGACY for the whole session, underneath the per-test patch. A module-
    or session-scoped fixture (test_orchestrator's pipeline runs) is built
    before any function-scoped one, so the per-test patch alone would leave it
    solving under the live settings while the tests beside it solve under
    LEGACY. And a test's own ``monkeypatch.undo()`` unwinds to this, not to
    the live settings."""
    from aigaffer.solver import optimizer

    _LIVE_SETTINGS[:] = [optimizer.DEFAULT_SETTINGS]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(optimizer, "DEFAULT_SETTINGS", optimizer.LEGACY)
        yield


@pytest.fixture(autouse=True)
def legacy_window_settings(request, monkeypatch):
    """Run the suite under LEGACY: every hand-computed number in it was worked
    out on the flat-bench model, and LEGACY is that model to the byte. A test
    that is about the live settings passes them explicitly or carries
    ``@pytest.mark.live_settings``, which puts the source's own back for it.
    The marker reaches this per-test patch alone, so a ``live_settings`` test
    that uses a module- or session-scoped fixture still gets that fixture
    built under LEGACY, and should build what it needs itself."""
    from aigaffer.solver import optimizer

    if request.node.get_closest_marker("live_settings"):
        monkeypatch.setattr(optimizer, "DEFAULT_SETTINGS", _LIVE_SETTINGS[0])
        return
    monkeypatch.setattr(optimizer, "DEFAULT_SETTINGS", optimizer.LEGACY)


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
