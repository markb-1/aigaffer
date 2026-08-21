"""Pydantic models for the slice of the FPL API this bot uses.

Only the fields we actually consume are declared; the live payloads carry
many more and pydantic ignores them by default, so new upstream fields
never break parsing. Prices (``now_cost``, ``bank``) are integers in tenths
of a million, exactly as the API serves them.
"""

from datetime import datetime

from pydantic import BaseModel


class Player(BaseModel):
    """A bootstrap ``element``. ``element_type``: 1 GK, 2 DEF, 3 MID, 4 FWD.

    Every rate the model wants is taken here as a season total over season
    ``minutes``, never as one of the API's own per-90 columns: the model has
    one small-sample rule for turning a total into a rate (see
    :func:`aigaffer.model.xp._per_90`) and a column that arrived already
    divided has escaped it. The API serves the expected-goals family as
    strings; pydantic coerces them.
    """

    id: int
    web_name: str
    team: int
    element_type: int
    now_cost: int
    status: str
    chance_of_playing_next_round: int | None = None
    minutes: int
    starts: int
    total_points: int
    bonus: int
    saves: int
    expected_goals: float = 0.0
    expected_assists: float = 0.0
    defensive_contribution: float = 0.0


class Team(BaseModel):
    """The four positional strength columns are 0 out of season, so the
    coarser ``strength_overall_*`` pair is kept as a fallback for the xP
    model."""

    id: int
    name: str
    short_name: str
    strength_attack_home: int
    strength_attack_away: int
    strength_defence_home: int
    strength_defence_away: int
    strength_overall_home: int
    strength_overall_away: int


class Event(BaseModel):
    """A gameweek."""

    id: int
    deadline_time: datetime
    is_next: bool
    is_current: bool
    finished: bool


class Fixture(BaseModel):
    """``event`` is None for a fixture that has not been scheduled yet."""

    id: int
    event: int | None
    team_h: int
    team_a: int


class Bootstrap(BaseModel):
    events: list[Event]
    teams: list[Team]
    elements: list[Player]

    def next_event(self) -> Event | None:
        return next((e for e in self.events if e.is_next), None)

    def current_event(self) -> Event | None:
        return next((e for e in self.events if e.is_current), None)


class Pick(BaseModel):
    element: int
    position: int
    is_captain: bool
    is_vice_captain: bool


class Squad(BaseModel):
    """A manager's picks for one gameweek. ``bank`` is in tenths of a million."""

    picks: list[Pick]
    bank: int
    event: int

    @property
    def player_ids(self) -> list[int]:
        return [p.element for p in self.picks]


class GwHistory(BaseModel):
    """One gameweek of a player's season history."""

    round: int
    minutes: int
    total_points: int
