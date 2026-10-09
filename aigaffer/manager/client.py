"""The Anthropic client every conversation with the gaffer is built with.

One place, because two conversations now use it — the hourly manager
(:func:`aigaffer.orchestrator._consult`) and the chip what-if's opinion
(:mod:`aigaffer.manager.chip_opinion`) — and a client built against the
clock in one and the SDK's defaults in the other would be two different
bets on how long a turn may take.

The SDK is imported inside the function and nowhere at module scope, so a run
that never asks the gaffer anything never imports it.
"""

from typing import TYPE_CHECKING

from aigaffer.config import Config

if TYPE_CHECKING:
    import anthropic

# Bounded, because the SDK is not by default: ten minutes a request and two
# retries is half an hour of one turn, inside a job that is given thirty for
# the whole run.
#
# Five minutes, not the two this was first written with. Two was chosen
# against a turn that hangs and never against a turn that works: a turn of
# this model at medium effort, running its web searches on the server before
# a single token comes back, takes minutes on purpose. Live it never once
# finished — two attempts of two minutes each, no searches, no turns, and the
# fallback every time, which is a manager who can never be reached wearing the
# clothes of a manager who was unlucky.
#
# One retry stays, and the worst case adds up rather than overlaps: the loop's
# own budget (:data:`~aigaffer.manager.agent.TIME_BUDGET_SECONDS`, twelve
# minutes) is checked before a request, a hung last request burns ten more,
# and the tools that request asked for run after it — a resolve sweeps the
# window again, two minutes at the outside. Twenty-four minutes inside the
# manager, against a job that is given thirty, which leaves the solver's own
# week time to be rendered and sent: the thing that must not be missed.
CLIENT_TIMEOUT_SECONDS = 300.0
CLIENT_RETRIES = 1


def build_client(cfg: Config) -> "anthropic.Anthropic":
    """The client, with the key from ``cfg`` and the clock above."""
    import anthropic

    return anthropic.Anthropic(
        api_key=cfg.anthropic_api_key,
        timeout=CLIENT_TIMEOUT_SECONDS,
        max_retries=CLIENT_RETRIES,
    )
