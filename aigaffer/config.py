import os
from dataclasses import dataclass, field
from pathlib import Path

# Where in the week each report is aimed, in hours before the deadline. The
# full report lands eighteen hours out — for the usual Saturday-morning
# deadline that is Friday evening, after the Friday press conferences (most
# clubs speak that morning, and the league's media cut-off is early
# afternoon) and after FPL's flags have caught up with them a few hours
# later, yet with Friday night and Saturday morning left to read it, argue
# with it and act on it. A day out it landed in the middle of the pressers,
# and the reminder was the first run to see them. For a Friday-night deadline
# eighteen hours out is early Friday, after the Thursday pressers such games
# get. The reminder lands three hours out, before the deadline itself, to say
# whether anything moved since. Constants rather than environment variables,
# like ``horizon`` and ``decay``: an anchor is part of the design, and nothing
# should be able to shift a report by six hours from a shell.
DEADLINE_ANCHOR_HOURS = 18.0
REMINDER_ANCHOR_HOURS = 3.0

# Where the week's reporting begins: the scout window opens this many hours
# before the deadline. Beyond it the fixtures are too far off to say anything
# a person should act on. The workflow's curl-and-jq gate holds a superset of
# this with slack, so a retune here never needs a matching edit there — but
# more than two hours of widening does.
SCOUT_HORIZON_HOURS = 60.0

# The one report that hangs off the round just played rather than the
# deadline ahead: an early scout, sent the evening of the day after that
# round's last kickoff, at this hour UTC — a first look at the week while the
# results are fresh and the fixtures are still days off. Its window runs
# until the Thursday scout's opens, and when a round ends so close to the
# next deadline that the evening after is already inside that window, there
# is no early scout: the scout covers it. Six or seven in the evening in
# London, depending on the clocks; the runner's hourly tick lands it at :35.
EARLY_SCOUT_HOUR_UTC = 18

# How close to the end of its window a report is still worth withholding for
# a manager who did not decide. A gaffer who fails — a dead key, a rate limit,
# an install that will not import — used to send the solver's week under his
# fallback label and mark it done, which left the backup scheduler nothing to
# redo and the owner one buried line to notice. Now the tick withholds the
# report and the next one asks him again, until this many hours before the
# window closes; past that the solver's report goes out and counts, because a
# week without a report is the one failure worse than a week without a
# manager. Three hours is a few ticks of either scheduler, and for the
# deadline report still leaves six hours to act on the solver's word.
MANAGER_RETRY_FLOOR_HOURS = 3.0

# And how many ticks may withhold the same report before the waiting ends
# regardless of the clock. A dead key fails in seconds and costs nothing, but
# a manager who fails slowly — a refusal, a loop that runs out of turns or of
# time — is a full run per tick, and with two schedulers that is two an hour.
# Twelve is six hours of both or twelve of one: long enough to replace a key
# after the alert, short enough that a week he cannot decide stays cheap.
MANAGER_RETRY_LIMIT = 12

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
    manager_model: str = "claude-opus-5-5"
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
    # Whether fixtures are priced by the fitted team strengths
    # (aigaffer/model/strength.py) rather than FPL's editorial columns.
    # On by default; the fit falls back to the columns loudly on any
    # failure, and this switch exists for the week the fit itself is the
    # thing misbehaving. Only the literal "off" disables it — the house
    # convention: a typo must not quietly switch a model off.
    strength_enabled: bool = True

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
            manager_model=os.environ.get("AIGAFFER_MANAGER_MODEL", "claude-opus-5-5"),
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
            # Same convention again for the team-strength fit.
            strength_enabled=os.environ.get("AIGAFFER_STRENGTH") != "off",
        )
