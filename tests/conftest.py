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
