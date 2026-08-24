"""The one place that talks to the FPL API. No raw FPL URLs live elsewhere.

The endpoints are public, unauthenticated and somebody else's, and a run asks
for a couple of hundred of them in a burst. Two courtesies follow from that: a
``User-Agent`` that says who is calling, so the traffic can be identified by
the people serving it, and a short bounded retry, so a rate limit or a bad
minute costs a few seconds instead of the week's report.

Only a 429 or a 5xx is worth asking again about. A 404 is an answer — the
picks endpoint gives one for a gameweek the manager did not play — and asking
again would arrive at the same place six seconds later.
"""

import time
from collections.abc import Callable
from typing import Any

import httpx

from aigaffer.data.models import Bootstrap, Fixture, GwHistory, PastSeason, Squad

BASE_URL = "https://fantasy.premierleague.com/api"
USER_AGENT = "aigaffer/0.1 (github.com/markb-1/aigaffer)"
TIMEOUT = 30.0

# One entry per wait, so the number of tries is one more than the number of
# waits: three tries, a second and then three.
RETRY_BACKOFF = (1.0, 3.0)
RATE_LIMITED = 429


def default_http_client() -> httpx.Client:
    """The client used when the caller supplies none of its own."""
    return httpx.Client(timeout=TIMEOUT, headers={"User-Agent": USER_AGENT})


def worth_retrying(status_code: int) -> bool:
    """Is this status the API being busy rather than the API answering?"""
    return status_code == RATE_LIMITED or 500 <= status_code < 600


class FplClient:
    """Thin read-only client over the public FPL endpoints.

    Every method raises ``httpx.HTTPStatusError`` on a non-2xx response that
    survived :data:`RETRY_BACKOFF`. ``sleep`` is injectable so tests can wait
    for nothing.
    """

    def __init__(
        self,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._http = http or default_http_client()
        self._sleep = sleep

    def _get(self, path: str) -> Any:
        url = f"{BASE_URL}{path}"
        for delay in (*RETRY_BACKOFF, None):
            response = self._http.get(url)
            if delay is None or not worth_retrying(response.status_code):
                break
            self._sleep(delay)
        response.raise_for_status()
        return response.json()

    def bootstrap(self) -> Bootstrap:
        return Bootstrap.model_validate(self._get("/bootstrap-static/"))

    def fixtures(self) -> list[Fixture]:
        return [Fixture.model_validate(f) for f in self._get("/fixtures/")]

    def picks(self, team_id: int, event: int) -> Squad:
        data = self._get(f"/entry/{team_id}/event/{event}/picks/")
        return Squad(
            picks=data["picks"],
            bank=data["entry_history"]["bank"],
            event=event,
        )

    def transfers(self, team_id: int) -> list[dict]:
        """Every transfer the manager has made this season, as raw dicts."""
        return self._get(f"/entry/{team_id}/transfers/")

    def chips_used(self, team_id: int) -> list[dict]:
        """Chips already played, as ``{"name": ..., "event": ...}`` dicts."""
        return self._get(f"/entry/{team_id}/history/").get("chips", [])

    def element_summary(
        self, player_id: int
    ) -> tuple[list[GwHistory], list[PastSeason]]:
        """One player's season so far, and the seasons behind it.

        Both halves come off the same response, which is the point of asking
        for them together: a run is a request per player already, and last
        season's minutes are not worth doubling that.
        """
        data = self._get(f"/element-summary/{player_id}/")
        return (
            [GwHistory.model_validate(h) for h in data.get("history", [])],
            [PastSeason.model_validate(s) for s in data.get("history_past", [])],
        )

    def element_history(self, player_id: int) -> list[GwHistory]:
        """This season only, for the callers that have no use for the rest."""
        return self.element_summary(player_id)[0]
