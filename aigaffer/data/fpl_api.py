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

from aigaffer.data.models import (
    Bootstrap,
    EntrySummary,
    EntryWeek,
    Fixture,
    GwHistory,
    HistoryRow,
    LiveStats,
    PastSeason,
    Squad,
    Standing,
)

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
        history = data["entry_history"]
        return Squad(
            picks=data["picks"],
            bank=history["bank"],
            event=event,
            standing=Standing.model_validate(history),
        )

    def transfers(self, team_id: int) -> list[dict]:
        """Every transfer the manager has made this season, as raw dicts."""
        return self._get(f"/entry/{team_id}/transfers/")

    def chips_used(self, team_id: int) -> list[dict]:
        """Chips already played, as ``{"name": ..., "event": ...}`` dicts."""
        return self._get(f"/entry/{team_id}/history/").get("chips", [])

    def entry_summary(self, team_id: int) -> EntrySummary:
        """Points, world rank and country rank, off the one entry request."""
        data = self._get(f"/entry/{team_id}/")
        region = data.get("player_region_name")
        classic = (data.get("leagues") or {}).get("classic") or []
        # The country league is the system league named for the region; see
        # EntrySummary for why the name, and not the type alone, picks it.
        region_rank = next(
            (
                league.get("entry_rank")
                for league in classic
                if region and league.get("league_type") == "s" and league.get("name") == region
            ),
            None,
        )
        return EntrySummary(
            gw=data.get("current_event"),
            total_points=data.get("summary_overall_points"),
            overall_rank=data.get("summary_overall_rank"),
            region=region,
            region_rank=region_rank,
        )

    def live(self, event: int) -> dict[int, LiveStats]:
        """Every player's minutes and points in ``event``, by id."""
        data = self._get(f"/event/{event}/live/")
        return {
            element["id"]: LiveStats.model_validate(element.get("stats") or {})
            for element in data.get("elements", [])
        }

    def entry_week(self, team_id: int, event: int) -> EntryWeek:
        """The manager's played ``event`` as the review reads it (EntryWeek)."""
        data = self._get(f"/entry/{team_id}/event/{event}/picks/")
        history = data["entry_history"]
        return EntryWeek(
            picks=data["picks"],
            active_chip=data.get("active_chip"),
            automatic_subs=[
                (sub["element_in"], sub["element_out"])
                for sub in data.get("automatic_subs") or []
            ],
            points=history["points"],
            event_transfers_cost=history.get("event_transfers_cost") or 0,
            points_on_bench=history.get("points_on_bench") or 0,
            overall_rank=history.get("overall_rank"),
        )

    def season_history(self, team_id: int) -> list[HistoryRow]:
        """The manager's season so far, a row a gameweek."""
        data = self._get(f"/entry/{team_id}/history/")
        return [HistoryRow.model_validate(row) for row in data.get("current", [])]

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

    def classic_league(self, league_id: int, page: int) -> list[dict]:
        """One page (fifty) of a classic league's standings, as ``{"entry",
        "rank"}`` rows — the replay's way of sampling real squads by rank."""
        data = self._get(f"/leagues-classic/{league_id}/standings/?page_standings={page}")
        return [
            {"entry": row["entry"], "rank": row["rank"]}
            for row in (data.get("standings") or {}).get("results", [])
        ]
