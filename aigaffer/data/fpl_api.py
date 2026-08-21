"""The one place that talks to the FPL API. No raw FPL URLs live elsewhere."""

from typing import Any

import httpx

from aigaffer.data.models import Bootstrap, Fixture, GwHistory, Squad

BASE_URL = "https://fantasy.premierleague.com/api"


class FplClient:
    """Thin read-only client over the public FPL endpoints.

    Every method raises ``httpx.HTTPStatusError`` on a non-2xx response.
    """

    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=30.0)

    def _get(self, path: str) -> Any:
        response = self._http.get(f"{BASE_URL}{path}")
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

    def element_history(self, player_id: int) -> list[GwHistory]:
        data = self._get(f"/element-summary/{player_id}/")
        return [GwHistory.model_validate(h) for h in data.get("history", [])]
