import os
from dataclasses import dataclass, field
from pathlib import Path

# Where in the week each report is aimed, in hours before the deadline. The
# full report lands a day out — late enough that the week has taken shape,
# early enough that there is a whole evening to read it, argue with it and act
# on it — and the reminder lands three hours out, after the press conferences,
# to say whether anything moved. Constants rather than environment variables,
# like ``horizon`` and ``decay``: an anchor is part of the design, and nothing
# should be able to shift a report by six hours from a shell.
DEADLINE_ANCHOR_HOURS = 24.0
REMINDER_ANCHOR_HOURS = 3.0

# Where the week's reporting begins: the scout window opens this many hours
# before the deadline. Beyond it the fixtures are too far off to say anything
# a person should act on. The workflow's curl-and-jq gate holds a superset of
# this with slack, so a retune here never needs a matching edit there — but
# more than two hours of widening does.
SCOUT_HORIZON_HOURS = 60.0

# Below this many finished gameweeks the season is "early", and two readers
# hang two different cautions off the one judgement. The briefing warns the
# manager that the per-90 rates under the projections are one or two matches
# of evidence; the fetch reads it the other way about — past this point a
# bootstrap where not one price has moved all season is a missing field, not
# a still market. It lives here rather than with either reader because the
# orchestrator must never need the manager package just to fetch.
EARLY_SEASON_GWS = 5


@dataclass
class Config:
    team_id: int
    telegram_token: str | None = None
    telegram_chat_id: str | None = None
    state_dir: Path = field(default_factory=lambda: Path("state"))
    horizon: int = 6
    decay: float = 0.85
    anthropic_api_key: str | None = None
    # An opt-out, not an opt-in: the manager runs whenever there is a key to run
    # it with, and __post_init__ clamps this to the key either way.
    manager_enabled: bool = True
    manager_model: str = "claude-opus-5"
    # "multi" plans the window and falls back on the single-week solver when it
    # has no answer; "single" is that solver and nothing else. An escape hatch
    # for a run that has to be quick or a week the window is misbehaving.
    planner: str = "multi"
    # Whether the planner is allowed to schedule chips. On by default: the
    # solver plans bench boost, triple captain, wildcard and free hit in the
    # window and may recommend one this week. Off makes chips advisory only —
    # priced in the panel, planned by nobody — which is Phase 2.5's behaviour
    # to the byte. Only the literal "off" disables it, the same strict
    # convention AIGAFFER_PLANNER uses: a typo should not be able to quietly
    # switch a feature off.
    chips: bool = True

    def __post_init__(self) -> None:
        # An empty key is no key: an unconfigured CI secret arrives exported and
        # empty, and the rest of the code asks `is not None`.
        self.anthropic_api_key = self.anthropic_api_key or None
        # No key, no manager. Asking for one without credentials is a run that
        # would die at the first API call, so it is a run without a manager.
        self.manager_enabled = bool(self.manager_enabled) and self.anthropic_api_key is not None

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            team_id=int(os.environ["FPL_TEAM_ID"]),
            telegram_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
            state_dir=Path(os.environ.get("AIGAFFER_STATE_DIR", "state")),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            # Only "0" turns it off; unset or anything else leaves the key to decide.
            manager_enabled=os.environ.get("AIGAFFER_MANAGER") != "0",
            manager_model=os.environ.get("AIGAFFER_MANAGER_MODEL", "claude-opus-5"),
            # AIGAFFER_MANAGER's convention: one literal value switches, and
            # everything else — a typo, a "0", an empty export — leaves the
            # default standing. Turning the better planner off by accident is
            # not a thing a misspelling should be able to do.
            planner=(
                "single" if os.environ.get("AIGAFFER_PLANNER") == "single" else "multi"
            ),
            # The same one-literal-value convention: only "off" disables the
            # chip planner, and an unset export, a typo or an empty string all
            # leave it on. Turning a feature off by accident is not a thing a
            # misspelling should be able to do.
            chips=os.environ.get("AIGAFFER_CHIPS") != "off",
        )
