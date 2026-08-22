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
    """``event`` is None for a fixture that has not been scheduled yet.

    Both played-flags are declared because the API raises them hours apart and
    only the later one is called ``finished``. ``finished_provisional`` goes up
    at full time; ``finished`` waits for the round's data check, which is when
    the bonus points are confirmed. Checked live on 2026-08-22: a match that
    kicked off at 19:00 the previous evening still read ``finished: false``
    seventeen hours later, with ``finished_provisional: true``, ``minutes: 90``
    and its history rows already served. Both default to False, so a payload
    that carries neither describes a match nobody has played.
    """

    id: int
    event: int | None
    team_h: int
    team_a: int
    finished: bool = False
    finished_provisional: bool = False

    @property
    def played(self) -> bool:
        """Has this match actually been played? Full time is enough — the
        minutes are final by then, and waiting for the data check is waiting a
        day for a number that will not change."""
        return self.finished or self.finished_provisional


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
    """One gameweek of a player's season history.

    ``fixture`` is the match the row belongs to, and it is what lets a reader
    tell a gameweek that has been played from one that has only been entered
    (:func:`aigaffer.orchestrator._played`). A double gameweek is two rows with
    the same ``round`` and different fixtures, and one of them can be played
    while the other has not kicked off. It defaults to 0 — no fixture — so that
    a payload without it is judged on its round instead of silently kept.
    """

    round: int
    minutes: int
    total_points: int
    fixture: int = 0
