"""The week's run, from the API to the phone.

Every module in the project answers one question; this is the one that asks
them, in order, and writes down what came back. There is no cleverness here
by design — the decisions are all made elsewhere — but there are three
judgements it has to make on its own:

* **When to run.** A report is only worth reading at two moments: two days
  out, when there is still time to plan, and on the day, when the team news
  is in. :func:`decide_mode` turns the hours to the deadline into one of
  those or into nothing at all, so the cron job can fire every three hours
  and stand down quietly most of the time.
* **Whose history to fetch.** A season of history is one request per player,
  and six hundred requests is not a polite thing to do to a public API every
  three hours. Only the squad and the players who could plausibly replace
  someone in it are asked for.
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
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime

import httpx

from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.free_transfers import compute_free_transfers
from aigaffer.data.models import Bootstrap, Event, Fixture, GwHistory, Player, Squad
from aigaffer.model.minutes import expected_minutes
from aigaffer.model.xp import PlayerProjection, project_all
from aigaffer.report.render import render_report
from aigaffer.report.telegram import send_report
from aigaffer.solver.lineup import ChipEvs, Lineup, chip_evs, pick_lineup
from aigaffer.solver.optimizer import (
    AVAILABLE,
    CANDIDATES_PER_POSITION,
    SQUAD_SIZE,
    Plan,
    optimize,
)
from aigaffer.solver.plans import generate_plans, recommend
from aigaffer.store import Store

DEADLINE_MODE, SCOUT_MODE = "deadline", "scout"
# Three hours, not six: the press conferences that decide the team news land
# the day before or the morning of, and a report written before them is a
# report written without the one thing the deadline run is for. One
# correctly-timed tick beats two early ones.
DEADLINE_WINDOW = (0, 3)
SCOUT_WINDOW = (36, 60)

NOT_CONFIGURED = "telegram not configured: the report was kept but not sent"

# A manager with no squad has the whole board and the opening budget: fifteen
# moves from nothing, £100.0m to make them with, and no hit for any of them.
DRAFT_BANK = 1000
DRAFT_TRANSFERS = SQUAD_SIZE
DRAFT_LABEL = "initial squad draft"

NO_CHIPS = ChipEvs(bench_boost=0.0, triple_captain=0.0, free_hit=0.0, wildcard=0.0)

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


@dataclass
class SolveResult:
    """The week's answer: what could be done, what to do, and who plays.

    ``draft_mode`` says which question was asked. A manager with no squad is
    not choosing between transfers — he is buying fifteen players — so the
    shortlist is one plan long, the chips are all worth nothing, and the
    report has to say so rather than read as this week's transfer advice.
    """

    plans: list[Plan]
    choice: Plan
    lineup: Lineup
    chips: ChipEvs
    draft_mode: bool


def decide_mode(now: datetime, deadline: datetime) -> str | None:
    """Which report ``now`` calls for, or None for none at all.

    The last three hours before a deadline are the deadline report — late
    enough that the press conferences have happened and the team news is in
    — and a window a day and a half to two and a half days out is the scout
    report. Between and either side of them there is nothing worth saying,
    which is most of the week. Both datetimes must be timezone-aware.
    """
    hours = (deadline - now).total_seconds() / 3600
    if _within(hours, DEADLINE_WINDOW):
        return DEADLINE_MODE
    if _within(hours, SCOUT_WINDOW):
        return SCOUT_MODE
    return None


def run_pipeline(
    cfg: Config,
    client: FplClient,
    store: Store,
    mode: str,
    send: bool = True,
    save: bool = True,
) -> str:
    """Run ``mode`` for the next gameweek and return the report.

    With ``save`` the report is written to ``state/reports/gw{n}-{mode}.md``
    and recorded in the store; with ``send`` it goes to Telegram, if a token
    and a chat are configured. A dry run turns both off and leaves nothing
    behind, which is what makes it safe to point at the live API.
    """
    inputs = fetch_inputs(cfg, client)
    _, projections = build_projections(inputs, cfg)
    solved = solve(inputs, projections, cfg)

    event, choice, lineup = inputs.event, solved.choice, solved.lineup
    report = render_report(
        _label(mode, drafting=solved.draft_mode),
        event,
        solved.plans,
        choice,
        lineup,
        solved.chips,
        inputs.bootstrap,
        projections,
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
        "xp_total": choice.xp_total,
        "objective": choice.objective,
        "chip_evs": asdict(solved.chips),
        "chip_baseline": _baseline_label(solved),
    }

    if save:
        _write_report(cfg, event.id, mode, report)
        store.save_run(event.id, mode, report, decision)
    if send:
        _deliver(cfg, report)
    return report


def fetch_inputs(cfg: Config, client: FplClient) -> PipelineInputs:
    """Ask the API everything the run needs, once.

    This is the only stage that talks to the network, so everything after it
    can be re-run for free. The histories are the expensive part — a request
    per player — and the bootstrap and the picks are asked for first because
    they are what decides whose history is worth asking for.
    """
    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        raise PipelineError("the API has no gameweek ahead")

    players = {player.id: player for player in bootstrap.elements}
    squad = _current_squad(client, cfg.team_id, bootstrap)
    # Asked for once and read twice: the free-transfer sum spends it here and
    # the manager's chip panel spends it later, and the entry endpoints do not
    # answer at all for a manager who has no squad.
    chips_used = [] if squad is None else client.chips_used(cfg.team_id)
    free_transfers = _free_transfers(client, cfg.team_id, squad, event, chips_used)

    held = [] if squad is None else squad.player_ids
    histories = {pid: _history(client, pid) for pid in history_pool(players, held)}
    fixtures = client.fixtures()

    return PipelineInputs(
        bootstrap=bootstrap,
        fixtures=fixtures,
        event=event,
        squad=squad,
        free_transfers=free_transfers,
        histories=histories,
        players=players,
        chips_used=chips_used,
    )


def build_projections(
    inputs: PipelineInputs,
    cfg: Config,
    minute_overrides: dict[int, float] | None = None,
) -> tuple[dict[int, float], dict[int, PlayerProjection]]:
    """Expected minutes, and the expected points that follow from them.

    ``minute_overrides`` replaces what the minutes model made of a player's
    history with what the caller knows about him — the manager who has read
    that he trained alone on Friday — as expected minutes for the gameweek,
    not as a multiplier, and clamped to a match. Everyone else is projected
    exactly as he would have been.

    A player nobody fetched a history for has no entry here at all, which
    projects him at zero: not a player the solver will buy, which is the
    point of leaving him out.
    """
    xmins = {
        pid: expected_minutes(history, inputs.players[pid])
        for pid, history in inputs.histories.items()
    }
    for pid, minutes in (minute_overrides or {}).items():
        xmins[pid] = min(FULL_MATCH, max(NO_MINUTES, float(minutes)))

    projections = project_all(
        inputs.bootstrap,
        inputs.fixtures,
        xmins,
        inputs.event.id,
        cfg.horizon,
        cfg.decay,
    )
    return xmins, projections


def solve(
    inputs: PipelineInputs,
    projections: dict[int, PlayerProjection],
    cfg: Config,
) -> SolveResult:
    """The shortlist, the plan to recommend, the eleven and the chip panel.

    Pure, and cheap enough to run more than once: given other projections it
    answers the same question about a different week, which is how an opinion
    about the team news becomes a different recommendation. ``cfg`` is not
    read here — the horizon and the decay are spent in the projection — but
    every stage takes it, so a caller re-running the last two has one shape
    to call them by.
    """
    plans, choice = _plans(
        inputs.players, projections, inputs.squad, inputs.free_transfers
    )
    positions = {pid: player.element_type for pid, player in inputs.players.items()}
    gw_xp = {
        pid: projection.per_gw[inputs.event.id]
        for pid, projection in projections.items()
    }
    lineup = pick_lineup(choice.squad, positions, gw_xp)

    # A draft has no chips to weigh: they are played against a squad, and
    # there is not one yet.
    chips = NO_CHIPS
    if inputs.squad is not None:
        baseline = _chip_baseline(plans, choice)
        chips = chip_evs(
            baseline,
            pick_lineup(baseline.squad, positions, gw_xp),
            gw_xp,
            inputs.players,
            projections,
            inputs.squad.bank,
            inputs.event.id,
        )

    return SolveResult(
        plans=plans,
        choice=choice,
        lineup=lineup,
        chips=chips,
        draft_mode=inputs.squad is None,
    )


def _within(hours: float, window: tuple[int, int]) -> bool:
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


def _history(client: FplClient, pid: int) -> list[GwHistory]:
    """One player's season so far, or none of it.

    Two hundred requests go out on a run and the API is somebody else's; one
    of them refusing after its retries is not a reason to lose the week's
    report. The minutes model already has a way to guess without a history —
    it is what a new signing gets — so the player is projected from his
    starts rather than dropped.
    """
    try:
        return client.element_history(pid)
    except httpx.HTTPStatusError:
        return []


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


def _plans(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    squad: Squad | None,
    free_transfers: int | None,
) -> tuple[list[Plan], Plan]:
    """The shortlist, and the plan to recommend from it.

    ``free_transfers`` is None exactly when ``squad`` is, and then there is no
    shortlist to draw up: one draft is the whole answer, and it is its own
    recommendation.
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

    plans = generate_plans(players, xp, squad.player_ids, squad.bank, free_transfers)
    if not plans:
        raise PipelineError("no legal squad is reachable from the current one")
    return plans, recommend(plans)


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
    transfer decision and must not be read as one."""
    return f"{mode} — {DRAFT_LABEL}" if drafting else mode


def _write_report(cfg: Config, event_id: int, mode: str, report: str) -> None:
    """Keep the report as a file: the repo is the managerial diary."""
    path = cfg.state_dir / "reports" / f"gw{event_id}-{mode}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _deliver(cfg: Config, report: str) -> None:
    """Send the report on, if there is anywhere to send it.

    Half-configured — a token in the secrets and no chat id, or the other way
    about — looks exactly like a working bot until the phone stays quiet, so
    a run that meant to deliver and could not says so.

    The bot token is part of the URL, so it is inside any ``httpx`` error
    this can raise — which makes the exception text unprintable in a log
    anyone can read. The class name says what went wrong without saying it
    with the token attached, and the run survives either way.
    """
    if not (cfg.telegram_token and cfg.telegram_chat_id):
        print(NOT_CONFIGURED)
        return
    try:
        send_report(cfg.telegram_token, cfg.telegram_chat_id, report)
    except Exception as error:  # any failure, and none of them worth the run
        print(f"telegram send failed: {type(error).__name__}")
