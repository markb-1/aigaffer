"""The week's run, from the API to the phone.

Every module in the project answers one question; this is the one that asks
them, in order, and writes down what came back. There is no cleverness here
by design — the decisions are all made elsewhere — but there are three
judgements it has to make on its own:

* **When to run.** A report is only worth reading at three moments: two days
  out, when there is still time to plan; a day out, when the week has taken
  shape and there is still an evening to act; and three hours out, when the
  team news is in and the only question left is whether yesterday's plan
  survived it — which is the reminder, a short alert and not a report.
  :func:`decide_mode` turns the hours to the deadline into one of those or
  into nothing at all, so the cron job can fire as often as it likes and
  stand down quietly most of the time. A fourth moment hangs off the other
  end of the week: the evening after the round just played, when the
  results are in and the plans are freshest — the early scout, the same
  report as the scout, anchored on the round's last kickoff rather than the
  deadline ahead, and given way to the scout when the two would meet.
* **Whose history to fetch.** A season of history is one request per player,
  and six hundred requests is not a polite thing to do to a public API on
  every scheduled run. Only the squad and the players who could plausibly
  replace someone in it are asked for.
* **What to do when there is no squad.** Before the first deadline of a
  season there are no picks to fetch, and a manager who has just joined 404s
  on a gameweek he did not play. Either way the run degrades to drafting a
  fifteen from nothing, which is the same question with an empty squad and a
  full budget.

Nothing here raises on a failed Telegram send: by then the report is on disk
and in the store, and a message that did not arrive is not a reason to lose
it.

The run itself is three stages with a seam between them:
:func:`fetch_inputs`, which is where every request to the API happens;
:func:`build_projections`, which turns the histories it came back with into
expected minutes and expected points; and :func:`solve`, which turns those
into a shortlist, a recommendation and an eleven. The stages after the fetch
are pure functions of a :class:`PipelineInputs`, so they can be run again —
over the same fetch, with a different opinion about who is playing and for
how long — without asking the API anything twice. That is what the
``minute_overrides`` argument is for, and it is the hinge a manager reading
the team news hangs off.

A fourth stage hangs off it: :func:`_consult`, which puts the solved week to
the manager (:mod:`aigaffer.manager`) and takes back the week to actually
enter. It is optional in the strongest sense — no key, the kill switch, a
draft, a rate limit, a refusal, a conversation that reaches no decision, or a
chip we do not hold for the gameweek all end with the solver's own
recommendation and a line in the report saying so. The rule that makes the
rest of this module readable is that the decision it comes back with, whoever
made it, is the one that goes into the report, the store and the phone. There
is never a second opinion further down.

One judgement sits on top of that rule. A manager who was asked and did not
decide is a week the solver decided under his name, and while the window has
time left that week is not written down: :func:`run_pipeline` withholds it —
no diary, no run row, no phone digest — tells the phone once why, and leaves
the next tick to ask him again. Only at the floor (:data:`RETRY_FLOORS`), at
the retry limit, or on a run a person forced does the labelled week go out
and count, so that a week is never left without a report.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from aigaffer.chips import HeldChip, chip_windows, held_chips, held_for
from aigaffer.config import (
    DEADLINE_ANCHOR_HOURS,
    EARLY_SCOUT_HOUR_UTC,
    EARLY_SEASON_GWS,
    MANAGER_RETRY_FLOOR_HOURS,
    MANAGER_RETRY_LIMIT,
    REMINDER_ANCHOR_HOURS,
    SCOUT_HORIZON_HOURS,
    Config,
)
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.free_transfers import compute_free_transfers
from aigaffer.data.models import (
    Bootstrap,
    Event,
    Fixture,
    GwHistory,
    PastSeason,
    Player,
    Squad,
    Standing,
)
from aigaffer.ledger import Observation, observe
from aigaffer.model.minutes import expected_minutes, season_prior
from aigaffer.model.strength import TeamStrengths, build_team_strengths
from aigaffer.model.xp import PlayerProjection, project_all, projected_events
from aigaffer.report.render import (
    # ``agent.MANAGER``, reached without importing the manager package at
    # module scope; renamed because this module's own DECIDED is a log line.
    DECIDED as BY_MANAGER,
    NO_CHIP,
    formation,
    played_chip,
    render_digest,
    render_reminder,
    render_reminder_digest,
    render_report,
    render_withheld,
)
from aigaffer.report.telegram import send_report
from aigaffer.solver.calendar import (
    ChipCalendar,
    build_calendar,
    fallback_calendar,
    horizon_end,
)
from aigaffer.solver.lineup import (
    ChipEvs,
    Lineup,
    attacking_evs,
    chip_evs,
    pick_lineup,
)
from aigaffer.solver.multiweek import FREE_HIT, PlannedMove
from aigaffer.solver.optimizer import (
    AVAILABLE,
    CANDIDATES_PER_POSITION,
    SQUAD_SIZE,
    Plan,
    optimize,
)
from aigaffer.solver.plans import generate_plans, recommend
from aigaffer.store import Store

if TYPE_CHECKING:  # imported inside _consult and nowhere else at module scope
    from aigaffer.manager.agent import ManagerDecision

DEADLINE_MODE, SCOUT_MODE, REMINDER_MODE = "deadline", "scout", "reminder"
# The scout that runs the evening after the round just played. Its own mode,
# because the store counts reports by mode and a week gets one of each; its
# own name in the report header, because "GW4 early" says nothing.
EARLY_MODE = "early"
MODE_LABELS = {EARLY_MODE: "early scout"}
# The windows hang off the anchors in :mod:`aigaffer.config`, opening at the
# anchor and staying open until the next report's territory begins, so the
# first tick to land inside one runs as close to the anchor as the schedule
# managed — and a late tick, however late, still runs the report rather than
# standing down. They used to be ninety-minute bands closing at the anchor,
# until GitHub dropped nine straight hours of cron across one and the week's
# full report silently never went: a dropped tick now costs lateness, never
# the report. What keeps a late-open window from sending twice is the store —
# ``has_run`` in ``__main__`` stands a tick down once its report exists.
DEADLINE_WINDOW = (REMINDER_ANCHOR_HOURS, DEADLINE_ANCHOR_HOURS)
REMINDER_WINDOW = (0, REMINDER_ANCHOR_HOURS)
SCOUT_WINDOW = (DEADLINE_ANCHOR_HOURS, SCOUT_HORIZON_HOURS)

NOT_CONFIGURED = "telegram not configured: the report was kept but not sent"
ALERT_NOT_CONFIGURED = "telegram not configured: the alert had nowhere to go"

# What a run says when the history filter takes everything it was given. The
# cut in :func:`_played` is made against a payload somebody else serves, so a
# renamed flag or an id that stops matching would leave every player projected
# off his priors and nothing anywhere to say the evidence had been thrown away
# rather than never served. It is only an anomaly if rows went in: an empty
# history in August is a season that has not started, and a warning for that
# would be a warning nobody reads by September.
HISTORY_FILTER_EMPTIED = (
    "aigaffer: history filter removed every played round ({players} players had"
    " rows); projections fall back to season priors"
)

# And when the price-move column reads as never having moved. The ledger's
# seed is ``now_cost - cost_change_start``, and the model defaults the field
# to 0 so hand-built fixtures need not mention it — which means a live payload
# that dropped or renamed the column would not fail parsing but quietly seed
# every player at today's price instead of the season opener's. Weeks into a
# season not one of six hundred prices standing still is not a market, it is
# a missing field, and the line below is the one place that says so.
COST_CHANGES_MISSING = (
    "aigaffer: cost_change_start is 0 on every element this deep into the"
    " season — the field looks absent from the payload, and the purchase"
    " ledger's seeds may be degraded to now_cost"
)

# What the report says when a manager was asked for and never reached at all.
# The labelled fallbacks explain themselves in the Gaffer's view section, but a
# manager who could not even be imported leaves no decision to hang a section
# on, and the report would then be Phase 1's exactly: a fork whose install
# broke could run a season of solver-only weeks and never be told. The stdout
# line is not enough — nobody is watching it — so the one document that
# actually goes to the phone says it too.
MANAGER_UNAVAILABLE = (
    "Note: the manager is configured but was unavailable this run;"
    " this is the solver's pick."
)

# What a run says about the manager, once, on stdout. A schedule nobody is
# watching leaves the log as the only record of whether the week was decided
# by a manager who read the news or by a solver that could not.
DECIDED = "the gaffer decided"
STOOD_DOWN = "the gaffer stood down"
# And that the week was not written down over it: the next tick asks again.
WITHHELD = "the report was withheld for the next tick"
# And the early scout's own ending: written down, out of retries, and not
# sent, because the scout is hours away and will be asked properly.
SHELVED = "the early scout was kept but not sent: the scout will be asked properly"
# The other scheduler pushed this report while this run was going: two
# schedulers, one store, and a manager run long enough for the second tick
# to start before the first has finished. Nothing is sent or saved.
OVERTAKEN = (
    "the other scheduler recorded this report while this run was going"
    " — standing down"
)

# The reason a manager whose own module would not import is withheld under.
# There is no decision to label and the class name already went to stdout;
# the phone needs a key for the failure, not a second copy of the traceback.
NOT_LOADED = "not loaded"

# Below these hours to the deadline a manager who did not decide no longer
# buys the report another tick: the window is closing, and the solver's week
# goes out and counts rather than no week at all. Each is the window's near
# edge plus the floor — the last hours in which a late report is still worth
# more than a labelled one. The reminder never asks him and has no floor.
# The early scout's floor is the scout's own opening: it is retried for as
# long as the clock would still choose it, and after that the scout is hours
# away and a labelled early scout would be a second text saying less. Which
# is also why, past the retry limit, an early scout is shelved rather than
# sent — see :func:`run_pipeline`.
RETRY_FLOORS = {
    DEADLINE_MODE: REMINDER_ANCHOR_HOURS + MANAGER_RETRY_FLOOR_HOURS,
    SCOUT_MODE: DEADLINE_ANCHOR_HOURS + MANAGER_RETRY_FLOOR_HOURS,
    EARLY_MODE: SCOUT_HORIZON_HOURS,
}

# And why a decision of his was not used: he played a chip we do not hold for
# this gameweek — spent, expired, or the next set's, which is in hand from the
# start of the season and playable only from its own start. It is the one
# refusal that happens outside his own loop.
CHIP_SPENT = "chip not held for this gameweek"

# The line a run prints when the chip calendar could not be built. The calendar
# is advice to the window's bars and never the week, so the run carries on, on
# the fallback bars; this is how the log says so. A class name follows it,
# never the exception's own words.
CALENDAR_FAILED = "chip calendar failed"

# A manager with no squad has the whole board and the opening budget: fifteen
# moves from nothing, £100.0m to make them with, and no hit for any of them.
DRAFT_BANK = 1000
DRAFT_TRANSFERS = SQUAD_SIZE
DRAFT_LABEL = "initial squad draft"

NO_CHIPS = ChipEvs(bench_boost=0.0, triple_captain=0.0, free_hit=0.0, wildcard=0.0)

# Which solver answered, for the record and for the report. It is read off the
# recommendation rather than off the configuration, because asking for the
# window and getting it are two different things: a plan that came off the
# window carries a path and a plan off the single-week solver does not.
MULTI, SINGLE = "multi", "single"

# What a minute override may say. It is an absolute expectation, not a nudge,
# and a match is ninety minutes long however confidently something outside
# this module believes otherwise.
NO_MINUTES, FULL_MATCH = 0.0, 90.0


class PipelineError(RuntimeError):
    """The run cannot produce a report: no gameweek ahead, or no legal squad."""


@dataclass
class PipelineInputs:
    """Everything the API had to say, and nothing derived from it.

    One fetch is a couple of hundred requests, so it is made once and kept:
    every stage after it reads this and only this. ``squad`` and
    ``free_transfers`` are None together, and only together — a manager with
    no squad has neither. ``histories`` covers the squad and the shortlist
    (see :func:`history_pool`) and nobody else; a player missing from it is a
    player nobody is thinking of buying.

    ``histories`` holds **played matches** only, and that is a promise every
    reader of it may rely on. The API adds a history row the moment a deadline
    goes — 0 minutes, 0 points, for a match that kicks off two days later —
    and a row for a match nobody has played is not a gameweek a player sat
    out. :func:`fetch_inputs` drops them on the way in (see :func:`_played`),
    fixture by fixture rather than round by round, so no stage after the fetch
    has to know the difference and no two of them can decide it differently.

    ``prior_minutes`` is what each player averaged a gameweek in his last
    Premier League season, and it comes off the same element-summary response
    the history does. A player with no Premier League behind him — a signing
    from abroad, a promoted club's player — has no entry at all rather than a
    zero, because nothing to read and a season of not playing are different
    facts and the minutes model treats them differently.

    ``chips_used`` is the season's chip history as the API serves it, and it
    is here because two different readers need it: the free-transfer sum,
    which is what a wildcard week means for the bank, and the manager, who
    must not be offered a chip that has already been spent. It is empty for a
    manager with no squad — nothing has been played by somebody who has not
    played — and defaults to empty because a fetch that never asked has
    nothing to say about it.
    """

    bootstrap: Bootstrap
    fixtures: list[Fixture]
    event: Event
    squad: Squad | None
    free_transfers: int | None
    histories: dict[int, list[GwHistory]]
    players: dict[int, Player]
    chips_used: list[dict] = field(default_factory=list)
    prior_minutes: dict[int, float] = field(default_factory=dict)


@dataclass
class SolveResult:
    """The week's answer: what could be done, what to do, and who plays.

    ``draft_mode`` says which question was asked. A manager with no squad is
    not choosing between transfers — he is buying fifteen players — so the
    shortlist is one plan long, the chips are all worth nothing, and the
    report has to say so rather than read as this week's transfer advice.

    ``calendar`` is the chip calendar the window's bars were read off, carried
    so a reader of the solve — the record, the report, the gaffer's re-solves —
    sees the one it was judged against. None when no chip is held: the switch
    off, a draft, or a season whose chips are all gone.
    """

    plans: list[Plan]
    choice: Plan
    lineup: Lineup
    chips: ChipEvs
    draft_mode: bool
    calendar: ChipCalendar | None = None


def decide_mode(
    now: datetime, deadline: datetime, last_kickoff: datetime | None = None
) -> str | None:
    """Which report ``now`` calls for, or None for none at all.

    Three windows, nearest the deadline first, and contiguous from sixty
    hours out to the deadline itself. The last three hours hold the reminder
    — the solver checking the full report against the morning's news. From
    a day out to those three hours is the full deadline report, aimed at the
    T-24h anchor and caught up late when the schedule failed it. Beyond a
    day and up to two and a half days out is the scout report. Past sixty
    hours there is nothing worth saying yet — unless the round just played
    ended, and ``last_kickoff`` says when: from the evening of the day after
    it (:func:`early_anchor`) until the scout's window opens is the early
    scout. When that evening is already inside the scout's window there is
    no early scout; the scout is the report for it. This function answers
    for the clock alone — whether the report it names already ran is the
    store's question, asked in ``__main__``.

    The windows do not overlap as configured, but the constants are
    constants: should widening one ever make a moment ambiguous, the report
    nearest the deadline wins, because it is the one whose moment cannot be
    made up on a later tick. ``now`` and ``deadline`` must be timezone-aware;
    a naive ``last_kickoff`` is read as UTC, which is what the API serves.
    """
    hours = (deadline - now).total_seconds() / 3600
    for window, mode in (
        (REMINDER_WINDOW, REMINDER_MODE),
        (DEADLINE_WINDOW, DEADLINE_MODE),
        (SCOUT_WINDOW, SCOUT_MODE),
    ):
        if _within(hours, window):
            return mode
    if last_kickoff is not None and hours > SCOUT_HORIZON_HOURS:
        if now >= early_anchor(last_kickoff):
            return EARLY_MODE
    return None


def early_anchor(last_kickoff: datetime) -> datetime:
    """The evening of the day after ``last_kickoff``, in UTC."""
    if last_kickoff.tzinfo is None:
        last_kickoff = last_kickoff.replace(tzinfo=UTC)
    day_after = last_kickoff.astimezone(UTC) + timedelta(days=1)
    return day_after.replace(hour=EARLY_SCOUT_HOUR_UTC, minute=0, second=0, microsecond=0)


def last_kickoff(fixtures: list[Fixture], current: int, before: datetime) -> datetime | None:
    """When the current round's last match kicked off, or None for a round
    with no scheduled matches.

    Only kickoffs before ``before`` — the next deadline — count: a postponed
    match keeps its round number and moves months away, and the round it
    belonged to still ended when the rest of it did. The API serves kickoffs
    as ISO strings with a ``Z``, which is what is parsed here.
    """
    kickoffs = [
        stamp
        for fixture in fixtures
        if fixture.event == current and fixture.kickoff_time
        for stamp in [datetime.fromisoformat(fixture.kickoff_time.replace("Z", "+00:00"))]
        if stamp < before
    ]
    return max(kickoffs, default=None)


def run_pipeline(
    cfg: Config,
    client: FplClient,
    store: Store,
    mode: str,
    send: bool = True,
    save: bool = True,
    now: datetime | None = None,
    force: bool = False,
    overtaken: Callable[[int, str], bool] | None = None,
) -> str:
    """Run ``mode`` for the next gameweek and return the report.

    With ``save`` the report is written to ``state/reports/gw{n}-{mode}.md``,
    to ``GW{n}.md`` at the repo root — the copy the GitHub homepage shows,
    where the deadline run overwrites the scout's — and recorded in the
    store; with ``send`` it goes to Telegram, if a token
    and a chat are configured. A dry run turns both off and leaves nothing
    behind, which is what makes it safe to point at the live API.

    Neither flag reaches the manager. Asking him is reading and thinking, not
    sending or saving, and a dry run that skipped it would print a report
    nobody could check against the one the schedule will produce.

    A manager who was configured and could not be reached at all adds one line
    to the report before either flag is read, so that the file, the store and
    the message all say the same thing about who decided this week.

    ``overtaken``, when given, is asked once the manager has spoken and before
    anything is sent or saved: whether the other scheduler has recorded this
    report meanwhile. A yes ends the run there — the report is returned, for
    a dry run to print, and nothing else happens.

    A manager who was asked and did not decide — that accident, or any of the
    labelled fallbacks — is not written down at all while the window has more
    than its floor left and the report has been held fewer than the limit's
    ticks: the report is returned and nothing else happens, except that the
    phone is told once, per reason, with the solver's checklist. ``save`` and
    ``send`` gate that alert and the record of it as they gate everything
    else, so a dry run stays silent and leaves nothing; ``force`` — a person
    at the keyboard, or the workflow's button — skips the waiting altogether
    and takes whatever answer the run got, which is what forcing has always
    meant here.

    The reminder is the exception to almost all of that, and it branches off
    at the top: no manager, no root file, a short alert instead of a report —
    see :func:`_run_reminder`, which owns what it does keep of the flow.

    Between the fetch and the projections the purchase ledger is brought up
    to date (:func:`aigaffer.ledger.observe`): the picks are already in hand,
    and what comes back — the true selling price of every man we hold — is
    threaded through the solve, the manager's re-solves and the report's SELL
    tags. ``save`` gates the ledger's writes exactly as it gates the report's
    — and a withheld tick, which saves nothing, persists none of them either:
    a dry run prices its sales in memory and persists none of it.
    """
    if mode == REMINDER_MODE:
        return _run_reminder(
            cfg, client, store, send=send, save=save, overtaken=overtaken
        )

    inputs = fetch_inputs(cfg, client)
    # Priced now, persisted below with everything else: the ledger's one
    # reconciliation line per gameweek fires on the first run to see the
    # squad, and a withheld tick that had written the snapshot would have
    # spent it on a report nobody saved or sent.
    ledger = observe(
        store, inputs.squad, inputs.players, inputs.chips_used, persist=False
    )
    # One fit of the team strengths a run, read by every projection the run
    # makes: the week's, the calendar's and each of the gaffer's re-solves.
    strengths = _strengths(inputs, cfg)
    xmins, projections = build_projections(inputs, cfg, strengths=strengths)
    # The chip calendar, once a run and before the solve it shapes; the
    # gaffer's re-solves are judged against this same one (see _calendar).
    calendar = _calendar(
        inputs, cfg, projections, ledger.selling_prices, strengths=strengths
    )
    solved = solve(inputs, projections, cfg, ledger.selling_prices, calendar)
    gaffer = _consult(
        cfg, inputs, solved, projections, xmins, ledger.selling_prices,
        calendar=calendar, strengths=strengths,
    )

    # From here down the week is his, if there was a him: the plan he chose and
    # the eleven that goes with it, in every place the solver's own would have
    # gone. A recommendation the report prints and a decision the store keeps
    # have to be the same recommendation.
    event = inputs.event
    choice = solved.choice if gaffer is None else gaffer.plan
    lineup = solved.lineup if gaffer is None else gaffer.lineup
    # And costed on the projections the decision was made on: his, if he
    # re-solved on minutes of his own. A team sheet he picked printed beside
    # numbers he overruled is one report describing two different weeks.
    costed = projections if gaffer is None else (gaffer.projections or projections)
    # The chip this week plays, and the eleven that goes with it. On a free-hit
    # week that eleven is the temporary team the plan built, not the standing
    # squad, so the fielded lineup — and the armbands, and the record's captain
    # — come off it; ``choice`` still reports the standing squad, which reverts.
    chip = played_chip(choice, gaffer)
    lineup = _fielded_lineup(chip, choice, lineup, inputs.players, costed, event.id)
    report = render_report(
        _label(mode, drafting=solved.draft_mode),
        event,
        solved.plans,
        choice,
        lineup,
        solved.chips,
        inputs.bootstrap,
        costed,
        gaffer,
        # Asked for, which is not the same as answered. The report needs both
        # halves to tell a window that fell over from a run that wanted the
        # single-week solver; a draft is neither, since the window is never
        # asked to buy fifteen players.
        engine_expected=cfg.planner != SINGLE and not solved.draft_mode,
        # The bank the action block names when the week rolls. None on a draft,
        # which has no action block to read it.
        free_transfers=inputs.free_transfers,
        # What each sale actually raises, for the SELL tags that differ from
        # the listed price.
        selling_prices=ledger.selling_prices,
    )
    # Asked for, and not there at all. Not the same as the kill switch, no key
    # or a draft — those are choices, and the invariant is that they render
    # Phase 1's report byte for byte — and not the same as a labelled fallback
    # either, which prints a section of its own. This is the accident, and it
    # is said in the report itself so that the copy on the phone and the copy
    # in the diary both carry it.
    if gaffer is None and cfg.manager_enabled and not solved.draft_mode:
        report += f"\n{MANAGER_UNAVAILABLE}\n"
    report += _audit_line(ledger)

    # The solver's own answer, before the manager touched it, read the same
    # way the reminder will read its own solve at T-3h: the week-1 chip off
    # the path, and on a free-hit week the eleven that chip fields. The
    # reminder re-runs the solver and only the solver, so the one comparison
    # that can honestly mean "the news moved" is solver-then against
    # solver-now; diffed against the gaffer's verdict instead, every week he
    # overrode the solver would shout at T-3h about a disagreement that was
    # settled at T-24h. His verdict stays the operative plan everywhere it is
    # shown — this is only the yardstick the reminder measures the news with.
    solver_chip = played_chip(solved.choice, None)
    solver_lineup = _fielded_lineup(
        solver_chip, solved.choice, solved.lineup, inputs.players, projections,
        event.id,
    )

    decision = {
        "mode": mode,
        "event": event.id,
        "initial_draft": solved.draft_mode,
        "free_transfers": inputs.free_transfers,
        "transfers_in": choice.transfers_in,
        "transfers_out": choice.transfers_out,
        "hits": choice.hits,
        "captain": lineup.captain,
        "vice": lineup.vice,
        # The shape of the eleven, kept so the reminder can diff it: a plan
        # whose swaps and armbands held but whose eleven swapped a defender
        # for a forward is still a plan that changed on the sheet.
        "formation": formation(lineup, inputs.players),
        "xp_total": choice.xp_total,
        "objective": choice.objective,
        # The chip this week actually plays: the manager's if he decided, the
        # solver's own week-1 chip otherwise, "none" on the great many weeks
        # that play none. Recorded here so the diary reads the same as the phone.
        "chip": chip,
        "chip_evs": asdict(solved.chips),
        # What each chip in hand was being saved for, and the bars the window
        # was given — or None when no chip is held. Kept beside the panel so a
        # later reader can see why a chip went, or did not, in this week.
        "chip_calendar": calendar.record() if calendar is not None else None,
        "chip_baseline": _baseline_label(solved),
        "engine": _engine(choice),
        "solver_actions": plan_actions(
            solved.choice, solver_lineup, solver_chip, inputs.players
        ),
    }
    # The rest of the window, when there was one: ids and gameweeks, which is
    # what a later run can compare its own plan against. Names would be the
    # report's job and would age worse than the ids do.
    if choice.path is not None:
        decision["path"] = [_planned(move) for move in choice.path.moves]
    if gaffer is not None:
        # Both halves of the record go in, superseded entries and all: the ones
        # a re-solve spent, and the ones he only wrote down. This is what
        # happened, and the report is where it is read tidily.
        decision.update(
            decision_source=gaffer.source,
            rationale=gaffer.rationale,
            adjustments=gaffer.adjustments,
            unapplied=gaffer.unapplied,
            chip_justification=gaffer.chip_justification,
            searches=gaffer.searches,
        )

    # The expensive part is over; before the phone, the other scheduler's
    # word. The store in hand said nothing had run when this tick began, and
    # the minutes since are exactly when the peer's push lands. The question
    # is whether a report was recorded — a peer that only withheld is not an
    # answer, since this manager may have decided where that one did not —
    # so a manager failing on both hosts inside one window can still cost a
    # second withheld alert. That residual is small enough to keep.
    if overtaken is not None and overtaken(event.id, mode):
        print(OVERTAKEN)
        return report

    # A manager who was asked and did not decide is not a week to write down
    # while there is time to ask him again. Nothing is saved — not the diary,
    # not the store, not the homepage — so the next tick of either scheduler
    # runs the whole thing afresh, and only his answer or the floor ends the
    # waiting. The phone hears why, once per reason, with the solver's
    # checklist so the owner is never without a plan to act on meanwhile.
    # Past the floor the labelled report below goes out and counts, for every
    # report but one: late beats labelled, but labelled beats nothing. The
    # early scout is the one, and the block after this says why.
    undecided = _undecided(cfg, solved, gaffer)
    if (
        undecided is not None
        and not force
        and _worth_another_tick(mode, event, now)
        and store.withheld_count(event.id, mode) < MANAGER_RETRY_LIMIT
    ):
        print(WITHHELD)
        # The phone first and the record second: a record written before an
        # alert that never landed would keep every later tick quiet, and this
        # is the one message the whole arrangement exists to send. Nowhere to
        # send it counts as sent — an unconfigured phone stays unconfigured.
        first = not (save and store.withheld_before(event.id, mode, undecided))
        delivered = True
        if send and first:
            delivered = _deliver(
                cfg,
                render_withheld(
                    _label(mode, drafting=solved.draft_mode),
                    event, choice, lineup, inputs.bootstrap, undecided,
                    until=event.deadline_time - timedelta(hours=RETRY_FLOORS[mode]),
                    attempts=MANAGER_RETRY_LIMIT,
                    free_transfers=inputs.free_transfers,
                    selling_prices=ledger.selling_prices,
                    standing=_standing(inputs),
                ),
                unconfigured=ALERT_NOT_CONFIGURED,
            )
        if save and delivered:
            store.record_withheld(event.id, mode, undecided)
        return report

    # An early scout the manager never decided, out of retries: written down
    # so the ticks stand down, and not sent. The scout is hours away and will
    # be asked properly; a labelled early scout on the phone would be a second
    # text saying less. A person who forced it gets it regardless — they
    # asked for whatever the run had.
    if undecided is not None and mode == EARLY_MODE and not force:
        print(SHELVED)
        send = False

    if save:
        observe(store, inputs.squad, inputs.players, inputs.chips_used)
        _write_report(cfg, event.id, mode, report)
        store.save_run(event.id, mode, report, decision)
    if send:
        # The phone gets the digest — checklist, moves, the gaffer's opening
        # paragraph, a pointer at the rest. The full report just went to the
        # diary and the store, which is where the pointer sends the reader.
        digest = render_digest(
            _label(mode, drafting=solved.draft_mode),
            event, choice, lineup, inputs.bootstrap, gaffer,
            free_transfers=inputs.free_transfers,
            selling_prices=ledger.selling_prices,
            standing=_standing(inputs),
        )
        # The two lines the report carries that the renderer cannot know
        # about ride the digest too, under the same conditions: the accident
        # notice, because the phone must never read a broken fork as a
        # confident solver week, and the ledger's discrepancy note, which
        # fires at most once per gameweek and would otherwise live only in
        # a diary nobody reads on deadline day.
        if gaffer is None and cfg.manager_enabled and not solved.draft_mode:
            digest += f"\n{MANAGER_UNAVAILABLE}\n"
        digest += _audit_line(ledger)
        _deliver(cfg, digest)
    return report


def _run_reminder(
    cfg: Config,
    client: FplClient,
    store: Store,
    send: bool,
    save: bool,
    overtaken: Callable[[int, str], bool] | None = None,
) -> str:
    """Three hours out: solve again, diff against yesterday's solve, buzz once.

    The full report was decided a day ago, with the manager in the loop; what
    is left to learn between then and the deadline is the team news, and what
    is left to do about it is small. So this run is the solver alone — the
    manager is structurally never consulted here, key or no key, because
    three hours is no time for a twenty-minute conversation and the verdict
    is already his — and its whole output is a short alert: the operative
    plan, and whether the news has moved under it.

    Whether to shout is decided like against like. The deadline run's
    decision record, read back from the store
    (:meth:`~aigaffer.store.Store.decision`), keeps two plans: the gaffer's
    verdict — the operative one, the plan a person enters — and
    ``solver_actions``, the solver's own pre-manager answer. The fresh side
    here is a solver-only solve reduced to the same shape by
    :func:`plan_actions`, so the only pair whose difference can honestly mean
    "the news moved" is solver-then against solver-now
    (:func:`diff_actions`). Diffed against the verdict instead, every week
    the gaffer overrode the solver — his own captain, adjusted minutes, a
    chip — would shout at T-3h about a disagreement that was settled at
    T-24h, and a warning that cries wolf weekly is unread by October. What
    the alert *shows* keeps its authority unchanged: the gaffer's stored
    verdict, labelled as the operative plan, with the fresh solve beside it
    as information — the reminder never enters anything and never pretends
    the solver overrules him.

    A record from before ``solver_actions`` was kept has no solver-then to
    compare, and unknowable is not changed — the same rule the diff applies
    to a record from before formations were kept — so the alert stays calm
    over the verdict. When there is no record at all — the T-24h tick was
    dropped wholesale — the fresh block goes out with a line saying there
    was nothing to check it against.

    Delivery comes before the save, which is the reverse of the full report,
    and on purpose: the reminder's whole value is the buzz. Saved first, a
    failed send would mark the reminder done and no tick would retry it;
    sent first, a send that lands and a save that then fails risks one
    duplicate buzz on the next tick, which is the cheaper failure by a
    distance. So nothing — not the history file, not the store row — is kept
    until the message went or there was nowhere to send it. The full report
    keeps save-first: its diary copy has value of its own, and a duplicate
    full report is expensive.

    The alert is written to ``state/reports/gw{n}-reminder.md`` and recorded
    in the store like any run — which is what keeps an hourly schedule
    from sending it three times — but it never touches the root ``GW{n}.md``:
    that file is the polished verdict, and a checklist overwriting it would
    demote the one document the homepage shows.

    A draft week is the one shape this alert serves badly — fifteen buys are
    not swaps, and the block would say "roll" about a squad that does not
    exist — but a draft is entered off the full report, and the reminder's
    armbands and deadline line still hold.

    The purchase ledger is maintained here too — the picks were fetched
    anyway, and a transfer made between the deadline run and this one should
    be sighted three hours out, not a week later. Its writes join the
    deliver-first dance above rather than riding ``save`` directly: the
    observation is computed in memory for the message — the solve needs the
    selling prices and the alert may need the reconciliation note — and
    persisted only after the buzz went. Persisted first, a failed send would
    leave the gameweek's snapshot written, the retrying tick would find it
    and reconcile nothing, and the discrepancy line would only ever have been
    in the buzz nobody got. The trade is the dance's usual one: a send that
    lands and a save that then dies re-observes on the next tick, which
    re-learns the same prices from the same picks.
    """
    inputs = fetch_inputs(cfg, client)
    ledger = observe(
        store, inputs.squad, inputs.players, inputs.chips_used, persist=False
    )
    # A fresh calendar for a fresh solve: the deadline run's was built on the
    # board as it stood a day ago, and the yardstick here is solver-then
    # against solver-now, each on its own day's calendar. One fit, as ever.
    strengths = _strengths(inputs, cfg)
    _, projections = build_projections(inputs, cfg, strengths=strengths)
    calendar = _calendar(
        inputs, cfg, projections, ledger.selling_prices, strengths=strengths
    )
    solved = solve(inputs, projections, cfg, ledger.selling_prices, calendar)
    event = inputs.event

    # The same reading of the solve the full report would make without a
    # manager: the week-1 chip off the path, and on a free-hit week the
    # temporary eleven that chip actually fields.
    chip = played_chip(solved.choice, None)
    lineup = _fielded_lineup(
        chip, solved.choice, solved.lineup, inputs.players, projections, event.id
    )
    fresh = plan_actions(solved.choice, lineup, chip, inputs.players)
    record = store.decision(event.id, DEADLINE_MODE)
    stored = _stored_actions(record)
    # Solver-then, when the record is new enough to carry it. None is a
    # legacy record, and unknowable is not changed.
    solver_then = None if record is None else record.get("solver_actions")
    changes = {} if solver_then is None else diff_actions(solver_then, fresh)

    report = render_reminder(
        event, fresh, stored, changes, inputs.bootstrap,
        selling_prices=ledger.selling_prices,
        standing=_standing(inputs),
        free_transfers=inputs.free_transfers,
    )
    # The buzz is the digest — one checklist, what moved, a pointer — and on
    # the calm weeks it is the alert itself, byte for byte. The audit line
    # rides both: the phone is where a bank discrepancy has to be seen.
    buzz = render_reminder_digest(
        event, fresh, stored, changes, inputs.bootstrap,
        selling_prices=ledger.selling_prices,
        standing=_standing(inputs),
        free_transfers=inputs.free_transfers,
    )
    report += _audit_line(ledger)
    buzz += _audit_line(ledger)
    decision = {
        "mode": REMINDER_MODE,
        "event": event.id,
        "actions": fresh,
        "full_report_plan": stored,
        "full_report_solver_plan": solver_then,
        "changes": changes,
        "chip_calendar": calendar.record() if calendar is not None else None,
    }
    # Deliver before saving — see the docstring: a buzz that failed must
    # leave has_run false so the next tick retries, and the history file
    # goes with the store row so nothing on disk claims a reminder happened
    # that nobody felt.
    # The other scheduler's word first, as for every scheduled tick: the
    # solve is seconds, but the ticks are not, and a reminder felt twice is
    # the kind of text this run exists to keep short.
    if overtaken is not None and overtaken(event.id, REMINDER_MODE):
        print(OVERTAKEN)
        return report
    if send and not _deliver(cfg, buzz):
        return report
    if save:
        # The ledger's writes, now that the buzz went: the same observation
        # again, persisted this time. Written any earlier, the snapshot would
        # have marked the gameweek reconciled and a retried send would carry
        # no note (see the docstring).
        observe(store, inputs.squad, inputs.players, inputs.chips_used)
        # Not _write_report: the history file goes, the root verdict stays.
        path = cfg.state_dir / "reports" / f"gw{event.id}-{REMINDER_MODE}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
        store.save_run(event.id, REMINDER_MODE, report, decision)
    return report


def _audit_line(ledger: Observation) -> str:
    """The reconciliation note as the report carries it, or nothing.

    One line, appended after everything else the same way the
    manager-unavailable notice is: the ledger fires it at most once per
    gameweek (see :func:`aigaffer.ledger.observe`), and whichever run is
    first past the roll — scout, deadline or reminder — is the report that
    says it.
    """
    return f"\n{ledger.note}\n" if ledger.note else ""


def plan_actions(
    choice: Plan, lineup: Lineup, chip: str, players: dict[int, Player]
) -> dict:
    """A decided week reduced to the actions a person enters.

    The shape both sides of the reminder's comparison speak: ``transfers`` as
    ``[out, in]`` pairs for the checklist to print as swap lines, the
    armbands, the chip (``"none"`` for most weeks) and the formation. The
    pairing is presentational — both id lists arrive sorted, so which sale
    lines up with which signing means nothing, and the FPL app takes sells
    and buys as two separate lists anyway; :func:`diff_actions` reads them as
    two sets for exactly that reason. Ids and one string, no names — names
    are the renderer's job and would age worse than ids do. This is also the
    shape the deadline record keeps under ``solver_actions`` and the
    reminder's own decision record keeps under ``actions``.
    """
    return {
        "transfers": [
            [out, bought]
            for out, bought in zip(choice.transfers_out, choice.transfers_in)
        ],
        "captain": lineup.captain,
        "vice": lineup.vice,
        "chip": chip,
        "formation": formation(lineup, players),
    }


def _stored_actions(record: dict | None) -> dict | None:
    """The full report's decision record, reduced to the same actions shape.

    None in, None out: a record that does not exist is a comparison that
    cannot be made, and the reminder says so rather than inventing one. The
    reads are forgiving — ``get`` with the field absent meaning absent —
    because the record was written by whatever version of the pipeline ran a
    day ago, and a field this branch added (``formation``) is missing from
    every record before it. :func:`diff_actions` treats a missing field as
    unknowable rather than changed.

    ``chip`` is the one field a missing value maps to ``"none"`` for instead
    of to unknowable, and the asymmetry is deliberate. A chip-less record
    predates this branch by whole phases — written before the pipeline
    recorded chips at all, which is also before it could play one — so
    "none" is that record's truth, not a guess. A wrong "none" could only
    ever shout on a chip week against a record that cannot exist in
    practice, where a missing formation or solver plan belongs to weeks that
    really were played and decided; those are unknowable, this is known.
    """
    if record is None:
        return None
    return {
        "transfers": [
            [out, bought]
            for out, bought in zip(
                record.get("transfers_out") or [], record.get("transfers_in") or []
            )
        ],
        "captain": record.get("captain"),
        "vice": record.get("vice"),
        "chip": record.get("chip") or NO_CHIP,
        "formation": record.get("formation"),
    }


def diff_actions(stored: dict, fresh: dict) -> dict:
    """What moved between yesterday's solve and today's, machine-readably.

    Empty when they agree, which is the fact the reminder's tone hangs off.
    The reminder hands this the solver's stored pre-manager plan and its own
    fresh solve, so a non-empty diff means the news moved the solver off its
    own day-old answer — never that the gaffer and the solver disagree,
    which was settled at T-24h and is not news.

    Sells and buys are compared as two independent sets, never as pairs: the
    ``[out, in]`` pairing in the actions dict is presentational (both lists
    arrive sorted, so the pairs say nothing about which sale funds which
    signing), and read as pairs a plan that swapped one sale would come back
    as two dropped and two added moves naming a sale that never changed.
    The keys are ``sells_added``/``sells_dropped`` and
    ``buys_added``/``buys_dropped`` — added is what the fresh solve wants
    and the stored one did not, dropped the other way about — each a sorted
    list of player ids, which is also how the FPL app takes them. The scalar
    fields — ``captain``, ``vice``, ``chip``, ``formation`` — come back as
    ``[before, after]`` pairs, and a field that is None on either side is
    skipped: an old record that never kept the formation is a record with
    less in it, not a plan that changed shape.

    Decided here, once: the reminder's message and its decision record both
    read this dict, so they cannot disagree about whether the plan moved.
    """
    diff: dict = {}
    for side, place in (("sells", 0), ("buys", 1)):
        then = {pair[place] for pair in stored["transfers"]}
        now = {pair[place] for pair in fresh["transfers"]}
        if added := sorted(now - then):
            diff[f"{side}_added"] = added
        if dropped := sorted(then - now):
            diff[f"{side}_dropped"] = dropped
    for field_name in ("captain", "vice", "chip", "formation"):
        before, after = stored.get(field_name), fresh.get(field_name)
        if before is not None and after is not None and before != after:
            diff[field_name] = [before, after]
    return diff


def fetch_inputs(cfg: Config, client: FplClient) -> PipelineInputs:
    """Ask the API everything the run needs, once.

    This is the only stage that talks to the network, so everything after it
    can be re-run for free. The histories are the expensive part — a request
    per player — and the bootstrap and the picks are asked for first because
    they are what decides whose history is worth asking for.

    The histories are also cut down to the matches that have been played
    before they are handed on: what the API serves includes fixtures that have
    only been *entered*, and reading those as a season is the whole of the
    bug :func:`_played` exists to fix. That is why the fixtures are asked for
    ahead of the histories rather than after them — they are what the cut is
    made against.
    """
    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        raise PipelineError("the API has no gameweek ahead")
    _warn_if_cost_changes_missing(bootstrap)

    players = {player.id: player for player in bootstrap.elements}
    squad = _current_squad(client, cfg.team_id, bootstrap)
    # Asked for once and read twice: the free-transfer sum spends it here and
    # the manager's chip panel spends it later, and the entry endpoints do not
    # answer at all for a manager who has no squad.
    chips_used = [] if squad is None else client.chips_used(cfg.team_id)
    free_transfers = _free_transfers(client, cfg.team_id, squad, event, chips_used)

    # The fixtures come before the histories now, because the histories are cut
    # against them: a history row names the match it belongs to, and only
    # /fixtures/ says whether that match has been played.
    fixtures = client.fixtures()
    held = [] if squad is None else squad.player_ids
    summaries = {pid: _summary(client, pid) for pid in history_pool(players, held)}
    fetched = {pid: history for pid, (history, _) in summaries.items()}
    histories = _played(fetched, bootstrap, fixtures)
    _warn_if_emptied(fetched, histories)
    priors = {
        pid: prior
        for pid, (_, past) in summaries.items()
        if (prior := season_prior(past)) is not None
    }

    return PipelineInputs(
        bootstrap=bootstrap,
        fixtures=fixtures,
        event=event,
        squad=squad,
        free_transfers=free_transfers,
        histories=histories,
        players=players,
        chips_used=chips_used,
        prior_minutes=priors,
    )


# What ``build_projections`` is handed when nobody fitted the team strengths
# for it: fit them here. Not None, because None is an answer — the switch off,
# or a fit that failed and said so — and a caller who already has it must not
# pay for a second fit, or print a second failure line, to be told it again.
_UNFITTED = object()


def _strengths(inputs: PipelineInputs, cfg: Config) -> TeamStrengths | None:
    """The fitted team strengths, when the switch is on: the vendored seasons
    plus whatever this one has finished, refit on every run so the ratings
    sharpen weekly for free. A failed fit already said so on stdout and hands
    back None, which is the editorial columns.

    Fitted once a run and handed to every projection the run makes — the
    week's, the calendar's, each re-solve's — so all of them price fixtures
    the same way. It is cheap (a few hundredths of a second on the live
    board), but one fit a run is also one answer a run.
    """
    if not cfg.strength_enabled:
        return None
    return build_team_strengths(
        inputs.bootstrap, inputs.fixtures, datetime.now(UTC).date()
    )


def build_projections(
    inputs: PipelineInputs,
    cfg: Config,
    minute_overrides: dict[int, float] | None = None,
    *,
    horizon: int | None = None,
    strengths: "TeamStrengths | None | object" = _UNFITTED,
) -> tuple[dict[int, float], dict[int, PlayerProjection]]:
    """Expected minutes, and the expected points that follow from them.

    ``minute_overrides`` replaces what the minutes model made of a player's
    history with what the caller knows about him — the manager who has read
    that he trained alone on Friday — as expected minutes for the coming
    gameweek, not as a multiplier, and clamped to a match. For the coming
    gameweek only: the weeks after it in the window are projected on the
    model's own minutes, because Friday's team news is about Saturday and the
    multi-week planner reads every week it is given
    (:func:`aigaffer.model.xp.project_all` has the whole argument). Everyone
    else is projected exactly as he would have been.

    The minutes handed back are the coming gameweek's — the model's, with
    the overrides on top — because that is the one week anybody reads them
    for: the briefing's xMins column is the number ``adjust_players``
    overwrites, and it is the coming week's number that he overwrites. The
    weeks after it are on the model's minutes, which are the same numbers
    with no override in them.

    Each player's prior goes in beside his history: last season is what holds
    a returning player up when this one is a gameweek or two old, and a player
    with no Premier League behind him has no entry, which is the None the
    model falls back from.

    A player nobody fetched a history for has no entry here at all, which
    projects him at zero: not a player the solver will buy, which is the
    point of leaving him out.

    ``horizon`` is how many gameweeks to project, from the coming one, when
    the caller needs other than ``cfg.horizon``: the chip calendar, which
    looks past the window to the expiry of the chips it plans. ``strengths``
    is the run's fit (:func:`_strengths`), passed by a caller that already
    has it — None included, which means the editorial columns — and fitted
    here only when nobody passed one.
    """
    xmins = {
        pid: expected_minutes(
            history, inputs.players[pid], inputs.prior_minutes.get(pid)
        )
        for pid, history in inputs.histories.items()
    }
    overrides = {
        pid: min(FULL_MATCH, max(NO_MINUTES, float(minutes)))
        for pid, minutes in (minute_overrides or {}).items()
    }

    if strengths is _UNFITTED:
        strengths = _strengths(inputs, cfg)
    projections = project_all(
        inputs.bootstrap,
        inputs.fixtures,
        xmins,
        inputs.event.id,
        horizon or cfg.horizon,
        cfg.decay,
        strengths=strengths,
        overrides=overrides,
    )
    return {**xmins, **overrides}, projections


def solve(
    inputs: PipelineInputs,
    projections: dict[int, PlayerProjection],
    cfg: Config,
    selling_prices: dict[int, int] | None = None,
    calendar: ChipCalendar | None = None,
) -> SolveResult:
    """The shortlist, the plan to recommend, the eleven and the chip panel.

    Pure, and cheap enough to run more than once: given other projections it
    answers the same question about a different week, which is how an opinion
    about the team news becomes a different recommendation. That is also why
    the window the planner plans over is read off ``projections`` rather than
    computed from ``cfg``: whatever gameweeks the caller has numbers for are
    the gameweeks there is anything to plan with, and the two can then never
    come to disagree.

    ``cfg`` is read for the two things the shortlist is drawn up by and the
    projection is not: which planner to ask, and what a gameweek further out is
    worth against this one. It is also where the chip switch is read: whether
    the window may schedule a chip at all, and which are still in hand.

    ``selling_prices`` is the purchase ledger's answer for the squad we hold
    (:func:`aigaffer.ledger.observe`) and goes wherever a sale is priced: the
    shortlist's engines and the chip panel's rebuild boards. None — what a
    caller without a ledger passes, tests included — sells everyone at his
    listed price, which is the pre-ledger behaviour to the byte.

    ``calendar`` is the run's chip calendar (:func:`_calendar`), built once
    outside this function because it is not a function of these projections:
    a re-solve on the gaffer's minutes is judged against the same bars as the
    solve before it. Its bars are what each held chip must beat in each week
    of the window. None — no chip held, or a caller with no calendar — leaves
    the window its fallback bars.
    """
    held = _held(cfg, inputs)
    bars = calendar.bars if calendar is not None else None
    plans, choice = _plans(
        inputs.players,
        projections,
        inputs.squad,
        inputs.free_transfers,
        cfg,
        held,
        bars,
        selling_prices,
    )
    positions = {pid: player.element_type for pid, player in inputs.players.items()}
    gw_xp = {
        pid: projection.per_gw[inputs.event.id]
        for pid, projection in projections.items()
    }
    # The armband is chosen on the ceiling — goals and assists — not the total,
    # so a nailed defender's floor or a cheap player's kind fixture cannot win
    # it. Read off the projections the lineup is picked from, so the two agree.
    attacking = attacking_evs(projections, inputs.event.id)
    lineup = pick_lineup(choice.squad, positions, gw_xp, attacking)

    # A draft has no chips to weigh: they are played against a squad, and
    # there is not one yet.
    chips = NO_CHIPS
    if inputs.squad is not None:
        baseline = _chip_baseline(plans, choice)
        chips = chip_evs(
            baseline,
            pick_lineup(baseline.squad, positions, gw_xp, attacking),
            gw_xp,
            inputs.players,
            projections,
            inputs.squad.bank,
            inputs.event.id,
            selling_prices=selling_prices,
        )

    return SolveResult(
        plans=plans,
        choice=choice,
        lineup=lineup,
        chips=chips,
        draft_mode=inputs.squad is None,
        calendar=calendar,
    )


def _consult(
    cfg: Config,
    inputs: PipelineInputs,
    solved: SolveResult,
    projections: dict[int, PlayerProjection],
    xmins: dict[int, float],
    selling_prices: dict[int, int] | None = None,
    *,
    calendar: ChipCalendar | None = None,
    strengths: "TeamStrengths | None | object" = _UNFITTED,
) -> "ManagerDecision | None":
    """Put the week to the manager, and come back with the week to enter.

    None means there was no manager to ask: no key, the kill switch, or a
    draft — fifteen players from nothing is not a week to read the news about,
    since there is no team to change, no chip to play and no transfer to be
    talked out of. Then the pipeline is exactly the one Phase 1 shipped.

    Anything else is a decision, his or the solver's wearing his label, and
    every path through here produces one. The manager's own loop never raises
    given a solve the solver produced; the net around it is for everything
    outside the loop — building a client, building a briefing, an import of a
    dependency that is not installed — because none of that is worth the
    week's report either.

    The manager package is imported here and nowhere else. It reads this
    module's types to do its job, so the dependency runs one way at module
    scope and is closed inside a function, which also keeps the Anthropic SDK
    off the import path of every run that never asks for a manager.

    ``calendar`` and ``strengths`` are the run's own, and his re-solves reuse
    both rather than make their own: the calendar is built on the model's
    minutes, not his — his are for the coming gameweek and must not reach a
    week months out — and a re-solve judged against bars of its own would be
    a different question, not the same one on better news.
    """
    if solved.draft_mode or not cfg.manager_enabled:
        return None

    # The imports come in two groups, and the split is not cosmetic. This one
    # is everything a decision needs in order to exist and be vetted, and it
    # is also the one that cannot be recovered from: without the manager's own
    # types there is nothing to build a labelled fallback out of, so the run
    # reads as a run with no manager and says why on the log. Not ImportError
    # alone — a half-installed dependency raises whatever it likes on the way
    # up, and none of it is worth the week's report.
    try:
        import anthropic

        from aigaffer.manager.agent import (
            FALLBACK,
            MANAGER,
            NO_VIEW,
            ManagerDecision,
            run_manager,
        )
        from aigaffer.manager.tools import NO_CHIP
    except Exception as error:  # a broken install, and still not a lost week
        print(f"{STOOD_DOWN}: {type(error).__name__}")
        return None

    def solver_view(reason: str, searches: int = 0) -> ManagerDecision:
        """The solver's own week, labelled with why it is being read instead."""
        return ManagerDecision(
            plan=solved.choice,
            lineup=solved.lineup,
            captain=solved.lineup.captain,
            vice=solved.lineup.vice,
            chip=NO_CHIP,
            chip_justification="",
            rationale=NO_VIEW,
            adjustments=[],
            searches=searches,
            source=f"{FALLBACK}: {reason}",
        )

    def resolver(
        overrides: dict[int, float],
    ) -> tuple[SolveResult, dict[int, PlayerProjection]]:
        """His minutes, projected and solved again — the whole point of the
        seam. The projections that come back are the ones the solve was run
        on, because the eleven he ends up with is picked from them. The
        ledger's selling prices ride along: a re-solve on fresh minutes is
        still spending the same money. So do the run's calendar and its fit
        of the team strengths: never recomputed here, so his minutes move
        the week and never the bars it is judged against."""
        _, adjusted = build_projections(
            inputs, cfg, overrides, strengths=strengths
        )
        return solve(inputs, adjusted, cfg, selling_prices, calendar), adjusted

    try:
        # The second group: what asking him needs, imported where it is used,
        # because a failure from here on is a fallback like any other and the
        # net below is what says so in the report.
        from aigaffer.manager.briefing import build_briefing

        decision = run_manager(
            # Bounded, because the SDK is not by default: ten minutes a request
            # and two retries is half an hour of one turn, inside a job that is
            # given thirty for the whole run.
            #
            # Five minutes, not the two this was first written with. Two was
            # chosen against a turn that hangs and never against a turn that
            # works: a turn of this model at high effort, running its web
            # searches on the server before a single token comes back, takes
            # minutes on purpose. Live it never once finished — two attempts of
            # two minutes each, no searches, no turns, and the fallback every
            # time, which is a manager who can never be reached wearing the
            # clothes of a manager who was unlucky.
            #
            # One retry stays, and the worst case adds up rather than overlaps:
            # the loop's own budget
            # (:data:`~aigaffer.manager.agent.TIME_BUDGET_SECONDS`, twelve
            # minutes) is checked before a request, a hung last request burns
            # ten more, and the tools that request asked for run after it —
            # a resolve sweeps the window again, two minutes at the outside.
            # Twenty-four minutes inside the manager, against a job that is
            # given thirty, which leaves the solver's own week time to be
            # rendered and sent: the thing that must not be missed.
            anthropic.Anthropic(
                api_key=cfg.anthropic_api_key, timeout=300.0, max_retries=1
            ),
            cfg,
            inputs,
            solved,
            projections,
            build_briefing(
                inputs, solved, projections, inputs.free_transfers, xmins=xmins
            ),
            resolver,
        )
    except Exception as error:  # the gaffer is a luxury; the report is not
        decision = solver_view(f"unexpected {type(error).__name__}")

    # The briefing tells him which chips he cannot play; this is the belt under
    # that brace, for a chip not held for this gameweek — spent, expired, or
    # the next set's, in hand but not yet playable — and it is checked here
    # rather than in the loop because the loop has no business knowing what
    # our chip history looks like. A week built on a chip we cannot play is
    # not a week anybody can enter, so the whole decision goes back to the
    # solver rather than just the chip.
    held = _held(cfg, inputs)
    if decision.chip != NO_CHIP and held_for(held, decision.chip, inputs.event.id) is None:
        decision = solver_view(CHIP_SPENT, decision.searches)

    # One line a run, on stdout, for the log nobody is watching live: either
    # the manager decided and what it cost, or he did not and why. The reason
    # is a class name or a phrase of ours, never an exception's own words.
    if decision.source == MANAGER:
        searches = decision.searches
        spent = "1 search" if searches == 1 else f"{searches} searches"
        print(f"{DECIDED}: {spent}")
    else:
        print(f"{STOOD_DOWN}: {decision.source.partition(': ')[2]}")
    return decision


def _undecided(
    cfg: Config, solved: SolveResult, gaffer: "ManagerDecision | None"
) -> str | None:
    """Why a manager who was asked did not decide — or None, because nobody
    asked him, or because he did.

    The reason is the one the report prints: a class name or a phrase of
    ours, never an exception's own words. It is also the key the phone is
    told the failure under, once, so two ticks that failed the same way are
    one alert and a tick that failed a new way is another.
    """
    if solved.draft_mode or not cfg.manager_enabled:
        return None
    if gaffer is None:
        return NOT_LOADED
    if gaffer.source == BY_MANAGER:
        return None
    return gaffer.source.partition(": ")[2] or gaffer.source


def _worth_another_tick(mode: str, event: Event, now: datetime | None) -> bool:
    """Whether ``mode``'s window has more than its floor left — enough for a
    later tick to ask the manager again and still land in time.

    The wall clock unless one is handed in, and read here, after the manager
    has had his turn: a run that spent twenty minutes on him is twenty
    minutes nearer the floor, which is the right side to err on. A naive
    clock is UTC, as a naive deadline is. A mode with no floor is never worth
    waiting on; the reminder never asks him.
    """
    floor = RETRY_FLOORS.get(mode)
    if floor is None:
        return False
    deadline = event.deadline_time
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return (deadline - now).total_seconds() / 3600 > floor


def _within(hours: float, window: tuple[float, float]) -> bool:
    """Is ``hours`` inside ``window``, open at the near end and closed at the
    far one? A deadline that has just gone is not a deadline to report on."""
    low, high = window
    return low < hours <= high


def _current_squad(
    client: FplClient, team_id: int, bootstrap: Bootstrap
) -> Squad | None:
    """The fifteen we hold, or None if there is no such thing yet.

    Pre-season there is no current gameweek to ask about, and a manager who
    joined mid-season 404s on a gameweek he did not play. Both mean there is
    nothing to transfer from.
    """
    current = bootstrap.current_event()
    if current is None:
        return None
    try:
        return client.picks(team_id, current.id)
    except httpx.HTTPStatusError:
        return None


def _standing(inputs: PipelineInputs) -> Standing | None:
    """Where the season stands, for the phone's opening line — or None for a
    manager with no squad, who stands nowhere yet."""
    return None if inputs.squad is None else inputs.squad.standing


def _free_transfers(
    client: FplClient,
    team_id: int,
    squad: Squad | None,
    event: Event,
    chips_used: list[dict],
) -> int | None:
    """Free transfers for ``event``, or None when there is no squad: nothing
    has been earned or spent by a manager who has not played yet.

    ``chips_used`` is handed in rather than fetched: the caller needs the same
    list for the manager's chip panel, and one season of chip history is worth
    one request.
    """
    if squad is None:
        return None
    return compute_free_transfers(client.transfers(team_id), chips_used, event.id)


def _summary(
    client: FplClient, pid: int
) -> tuple[list[GwHistory], list[PastSeason]]:
    """One player's season so far and the seasons behind it, or neither.

    Two hundred requests go out on a run and the API is somebody else's; one
    of them refusing after its retries is not a reason to lose the week's
    report. The minutes model already has a way to guess without a history —
    it is what a new signing gets — so the player is projected from his
    starts rather than dropped. Both halves go together because they come off
    one response: a player we could not fetch has no last season either, and
    pretending otherwise would be inventing one.
    """
    try:
        return client.element_summary(pid)
    except httpx.HTTPStatusError:
        return [], []


def _played(
    histories: dict[int, list[GwHistory]],
    bootstrap: Bootstrap,
    fixtures: list[Fixture],
) -> dict[int, list[GwHistory]]:
    """The histories with the matches nobody has played taken out.

    A history row appears the moment the deadline goes, not the moment the
    match does: for the hours or days between the two it says 0 minutes and 0
    points about a fixture that has not kicked off. Checked live on
    2026-08-21, five hours after GW1's deadline: Haaland's entire history was
    one round-1 row of 0 minutes for a match on the Sunday, while a player
    whose match had been played that evening had his real 67 minutes in the
    same round.

    Nothing downstream can tell those two rows apart — a 0 is a 0 — and the
    minutes model averages them as gameweeks the player sat out, so a fit
    starter is marked down or written off for a match nobody has played.

    The cut is made per **fixture**, which is the grain the question is
    actually asked at. A row carries the id of the match it belongs to, and
    :attr:`~aigaffer.data.models.Fixture.played` says whether that match has
    happened; a gameweek that runs from Saturday to Monday is then half
    evidence and half phantom rather than all one or the other. The rule this
    replaces was the event's own ``finished`` flag, which only goes up once
    every match in the round has been played and so threw Saturday's real
    minutes away with Monday's zeroes. A double gameweek is two rows in the
    same round, judged one at a time.

    A row with no fixture id at all falls back on that older rule — its round
    is in the finished-events set, or it goes. Payloads that predate the field
    and any surprise from upstream then land on the conservative side rather
    than being kept on a ``fixture`` of 0 that matches nothing, and the same
    goes for a row naming a fixture the payload does not carry. Conservative
    in the direction the model can recover from: the minutes it does not have
    it falls back on ``starts`` and last season for.
    """
    finished = {event.id for event in bootstrap.events if event.finished}
    played = {fixture.id for fixture in fixtures if fixture.played}

    def happened(entry: GwHistory) -> bool:
        if not entry.fixture:
            return entry.round in finished
        return entry.fixture in played

    return {
        pid: [entry for entry in history if happened(entry)]
        for pid, history in histories.items()
    }


def _warn_if_emptied(
    fetched: dict[int, list[GwHistory]], histories: dict[int, list[GwHistory]]
) -> None:
    """Say so, once, if the filter took every row the API served.

    The two ways that can happen are worlds apart and look identical from
    downstream. Either the season has genuinely not started — nothing to keep,
    and the fallbacks are exactly what should happen — or the cut has stopped
    working: a flag renamed, a fixture id that no longer matches, a payload
    shape that moved. In the second case the whole run's recommendation rests
    on priors, every player equally, and nothing in the report would look
    wrong.

    So the line is printed only when rows went in and none came out, which is
    the shape the second case has and the first never does. It is a warning
    and not an error: priors are a worse week than evidence, and no week at
    all is worse than both.
    """
    with_rows = sum(1 for history in fetched.values() if history)
    if with_rows and not any(histories.values()):
        print(HISTORY_FILTER_EMPTIED.format(players=with_rows))


def _warn_if_cost_changes_missing(bootstrap: Bootstrap) -> None:
    """Say so, once, if no price on the board claims to have moved all season.

    In August that is simply true — prices have not moved yet, and the ledger's
    seed of ``now_cost - 0`` is exact. Weeks in, it is the signature of a
    payload regression: the model defaults ``cost_change_start`` to 0, so a
    dropped or renamed column parses cleanly and degrades the seed to today's
    price without a word. "Weeks in" is the same
    :data:`~aigaffer.config.EARLY_SEASON_GWS` the briefing's early-season
    warning stands down by, because the two are one judgement about when a
    season's evidence should exist — read from config, not from the manager
    package, whose broken install must only ever cost the manager.
    """
    played = sum(1 for event in bootstrap.events if event.finished)
    if played < EARLY_SEASON_GWS:
        return
    if all(player.cost_change_start == 0 for player in bootstrap.elements):
        print(COST_CHANGES_MISSING)


def history_pool(players: dict[int, Player], held: list[int]) -> list[int]:
    """Whose history to fetch: the squad, and the best of each position.

    The solver cuts its own field by projected points, which is the thing
    this data is needed to compute, so the cut here is by points scored so
    far — a season's form is a decent proxy for the shortlist, and it needs
    nothing but the bootstrap. The two cuts are the same width, and anyone
    left out of this one projects at zero, so he is never the player the
    solver buys.

    Between seasons every total is zero and the cut has nothing to rank on;
    price breaks that tie, because a £14.5m striker is who the market expects
    to score and an id is nobody in particular.

    Public because :mod:`aigaffer.backtest` scores this same population: the
    players the pipeline would actually consider are the ones whose ranking
    is worth grading.
    """
    pool = {pid for pid in held if pid in players}

    by_position: dict[int, list[Player]] = defaultdict(list)
    for player in players.values():
        if player.status == AVAILABLE:
            by_position[player.element_type].append(player)
    for candidates in by_position.values():
        candidates.sort(
            key=lambda candidate: (
                -candidate.total_points,
                -candidate.now_cost,
                candidate.id,
            )
        )
        pool.update(candidate.id for candidate in candidates[:CANDIDATES_PER_POSITION])

    return sorted(pool)


def _held(cfg: Config, inputs: PipelineInputs) -> tuple[HeldChip, ...]:
    """The chips in hand this run, each with the window it must be played in.

    Empty when the switch is off — chips advisory only, Phase 2.5 to the byte —
    and empty for a draft, because a chip is played against a squad and there
    is not one yet. Otherwise every chip the bootstrap's rules hand out that the
    season's history has not spent and whose window has not closed: the first
    set's and, from the start of the season, the second's, which the window
    may not play before GW20 (:mod:`aigaffer.chips` has the rules). A chip not
    held is simply absent from the decision space, which is where the "not held
    for this gameweek" belt in :func:`_consult` gets its half of the guarantee.
    The derivation lives here, in one place, so the solver, the calendar and
    the belt read the same chip history the same way.
    """
    if not cfg.chips or inputs.squad is None:
        return ()
    return held_chips(
        chip_windows(inputs.bootstrap.chips), inputs.chips_used, inputs.event.id
    )


def _calendar(
    inputs: PipelineInputs,
    cfg: Config,
    projections: dict[int, PlayerProjection],
    selling_prices: dict[int, int] | None,
    strengths: "TeamStrengths | None | object" = _UNFITTED,
) -> ChipCalendar | None:
    """The chip calendar for this run, or None when no chip is held.

    Built once, here, from projections of its own: base minutes (the gaffer's
    are for the coming gameweek and must not reach GW17) and a horizon running
    to the furthest expiry the window can reach. The resolver closes over the
    result, so a re-solve on his minutes is judged against the same calendar.
    A failure costs the calendar, never the week: one line on stdout and the
    fallback bars, which still play a chip with no later week inside the
    window.

    ``strengths`` is the run's fit, handed on so the calendar never refits;
    the window is the one ``projections`` covers, which is the one the solve
    plans over.
    """
    held = _held(cfg, inputs)
    if not held:
        return None
    window = projected_events(projections)
    try:
        end = horizon_end(held, window)
        xmins, base = build_projections(
            inputs, cfg, horizon=end - inputs.event.id + 1, strengths=strengths
        )
        return build_calendar(
            held, window, inputs.players, base, xmins,
            inputs.squad.player_ids, inputs.squad.bank, selling_prices,
        )
    except Exception as error:  # the calendar is advice to the bars, not the week
        print(f"{CALENDAR_FAILED}: {type(error).__name__}")
        return fallback_calendar(held, window)


def _fielded_lineup(
    chip: str,
    choice: Plan,
    lineup: Lineup,
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    event_id: int,
) -> Lineup:
    """The eleven actually taken to the deadline.

    Almost always the one already decided — the standing squad's best eleven,
    whoever picked it. A free-hit week is the exception the whole of Task 3
    exists for: the plan fields a temporary fifteen the standing squad is not,
    priced by the solver and surfaced on the path, and it is that team's eleven
    (and armbands) that plays this week. The standing squad reverts and is left
    to ``choice`` and the record; here it is the team on the sheet that changes.

    A free hit with no temporary squad behind it — a chip a manager finalized
    that the solver never built a free-hit team for — leaves the standing eleven
    standing, because there is no other to field.
    """
    if chip != FREE_HIT or choice.path is None or not choice.path.week1_freehit_squad:
        return lineup
    positions = {pid: player.element_type for pid, player in players.items()}
    gw_xp = {
        pid: projection.per_gw.get(event_id, 0.0)
        for pid, projection in projections.items()
    }
    attacking = attacking_evs(projections, event_id)
    return pick_lineup(
        choice.path.week1_freehit_squad, positions, gw_xp, attacking
    )


def _plans(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    squad: Squad | None,
    free_transfers: int | None,
    cfg: Config,
    held_chips: tuple[HeldChip, ...] = (),
    bars: dict[str, dict[int, float]] | None = None,
    selling_prices: dict[int, int] | None = None,
) -> tuple[list[Plan], Plan]:
    """The shortlist, and the plan to recommend from it.

    ``free_transfers`` is None exactly when ``squad`` is, and then there is no
    shortlist to draw up: one draft is the whole answer, and it is its own
    recommendation. A draft is also the one week the window is never asked
    about — fifteen signings will not fit under a gameweek's transfer ceiling
    — which is why the drafting branch below does not pass one, and is also
    the one week ``held_chips`` is always empty for, and the one week
    ``selling_prices`` has nobody to price: a draft only buys. ``bars`` are
    the calendar's, keyed by held chip and gameweek, and ride with the chips
    they price.
    """
    if squad is None:
        draft = optimize(
            players,
            xp,
            [],
            DRAFT_BANK,
            DRAFT_TRANSFERS,
            forced_transfers=DRAFT_TRANSFERS,
        )
        if draft is None:
            raise PipelineError("no legal fifteen fits the opening budget")
        return [draft], draft

    plans = generate_plans(
        players,
        xp,
        squad.player_ids,
        squad.bank,
        free_transfers,
        projections_events=projected_events(xp),
        decay=cfg.decay,
        planner=cfg.planner,
        held_chips=held_chips,
        bars=bars,
        selling_prices=selling_prices,
    )
    if not plans:
        raise PipelineError("no legal squad is reachable from the current one")
    return plans, recommend(plans)


def _engine(choice: Plan) -> str:
    """Which solver the week's recommendation came off.

    The plan itself is the evidence: the window hangs a path on what it
    returns and the single-week solver has nothing to hang. So a run that
    asked for the window and fell back on the other engine is recorded as what
    happened rather than as what was configured, and a decision record can be
    read a season later without the environment it was produced in.

    A draft is the single-week solver too, and honestly so: fifteen signings
    are not a window's question.
    """
    return MULTI if choice.path is not None else SINGLE


def _planned(move: PlannedMove) -> dict:
    """One future gameweek of the path, as the record keeps it.

    Player ids and nothing else. A name is what the report is for, and a name
    in the record would be the one field that stops meaning what it said when
    the API renames somebody. ``in`` and ``out`` rather than the field's own
    names, because this is read beside ``transfers_in`` and ``transfers_out``
    — the moves that were actually made — and the two must not look alike.
    """
    return {
        "event": move.event,
        "in": move.transfers_in,
        "out": move.transfers_out,
        "hits": move.hits,
    }


def _chip_baseline(plans: list[Plan], choice: Plan) -> Plan:
    """The plan the chip numbers are measured against.

    The squad and the bank handed to :func:`chip_evs` have to describe the
    same moment, so the chips are priced off the plan that spends nothing —
    rolling the transfer — against the money we actually have. If the solver
    could not reach that plan, the recommendation stands in and the decision
    record says which it was.
    """
    return next((plan for plan in plans if not plan.transfers_in), choice)


def _baseline_label(solved: SolveResult) -> str | None:
    """Which plan the chips were priced against, for the record.

    A draft priced no chips, so it has no baseline to name.
    """
    if solved.draft_mode:
        return None
    baseline = _chip_baseline(solved.plans, solved.choice)
    return "roll" if not baseline.transfers_in else "recommended"


def _label(mode: str, drafting: bool) -> str:
    """The mode as the report header says it. A draft is not the week's
    transfer decision and must not be read as one; the early scout is the
    scout and says so."""
    name = MODE_LABELS.get(mode, mode)
    return f"{name} — {DRAFT_LABEL}" if drafting else name


def _write_report(cfg: Config, event_id: int, mode: str, report: str) -> None:
    """Keep the report as files: the repo is the managerial diary.

    Two copies. ``state/reports/gw{n}-{mode}.md`` is the history, one file per
    run, never overwritten by the other mode. ``GW{n}.md`` is the polished
    verdict, written at the root beside README.md so the GitHub homepage's
    file listing shows it: the early scout puts it there the evening after
    the round, the scout rewrites it midweek and the deadline run overwrites
    it with the operative plan, which the shared filename does on its own —
    latest wins. (A shelved early scout writes it too, labelled as the
    solver's; the diary is honest about a week nobody was told.) The root is
    the directory holding
    ``state_dir`` rather than the working directory by name: in CI those are
    the same place, since the run starts at the checkout root with
    ``state_dir="state"``, and in a test they are the tmp_path the test owns,
    which is what keeps a test run from leaving verdicts in the real repo.
    """
    path = cfg.state_dir / "reports" / f"gw{event_id}-{mode}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    (cfg.state_dir.parent / f"GW{event_id}.md").write_text(report, encoding="utf-8")
    _point_readme(cfg.state_dir.parent, event_id)


# GitHub's tab row (README / LICENSE / …) is a fixed set and GW{n}.md can
# never join it, so the README itself carries one marked line the run
# rewrites — the verdict, one click from the homepage. The marker is the
# whole contract: a README without it (a fork that deleted the line, a
# test that never wrote one) is left byte-for-byte alone.
LATEST_MARKER = "<!-- latest-verdict -->"


def _point_readme(root: Path, event_id: int) -> None:
    """Rewrite the README's marked line to point at ``GW{event_id}.md``."""
    readme = root / "README.md"
    if not readme.exists():
        return
    lines = readme.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if LATEST_MARKER in line:
            ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
            lines[i] = (
                f"📋 **Latest verdict: [GW{event_id}](GW{event_id}.md)**"
                f" {LATEST_MARKER}{ending}"
            )
            readme.write_text("".join(lines), encoding="utf-8")
            return


def _deliver(cfg: Config, report: str, unconfigured: str = NOT_CONFIGURED) -> bool:
    """Send the report on, if there is anywhere to send it.

    Half-configured — a token in the secrets and no chat id, or the other way
    about — looks exactly like a working bot until the phone stays quiet, so
    a run that meant to deliver and could not says so.

    The bot token is part of the URL, so it is inside any ``httpx`` error
    this can raise — which makes the exception text unprintable in a log
    anyone can read. The class name says what went wrong without saying it
    with the token attached, and the run survives either way.

    Returns whether the report is as delivered as it will ever be: True when
    the message went, and True too when there was nowhere to send it — an
    unconfigured phone stays unconfigured on the next tick, so waiting for
    one would be waiting forever — False only when a configured send failed
    and a retry might land. The full report ignores the answer, its save
    having deliberately come first; the reminder's save hangs off it.
    """
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        print(unconfigured)
        return True
    try:
        send_report(cfg.telegram_token, cfg.telegram_chat_id, report)
    except Exception as error:  # any failure, and none of them worth the run
        print(f"telegram send failed: {type(error).__name__}")
        return False
    return True
