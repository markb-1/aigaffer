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
"""

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime

import httpx

from aigaffer.config import Config
from aigaffer.data.fpl_api import FplClient
from aigaffer.data.free_transfers import compute_free_transfers
from aigaffer.data.models import Bootstrap, Event, GwHistory, Player, Squad
from aigaffer.model.minutes import expected_minutes
from aigaffer.model.xp import PlayerProjection, project_all
from aigaffer.report.render import render_report
from aigaffer.report.telegram import send_report
from aigaffer.solver.lineup import ChipEvs, chip_evs, pick_lineup
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


class PipelineError(RuntimeError):
    """The run cannot produce a report: no gameweek ahead, or no legal squad."""


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
    bootstrap = client.bootstrap()
    event = bootstrap.next_event()
    if event is None:
        raise PipelineError("the API has no gameweek ahead")

    players = {player.id: player for player in bootstrap.elements}
    squad = _current_squad(client, cfg.team_id, bootstrap)
    free_transfers = _free_transfers(client, cfg.team_id, squad, event)

    held = [] if squad is None else squad.player_ids
    xmins = _expected_minutes(client, players, held)
    xp = project_all(
        bootstrap, client.fixtures(), xmins, event.id, cfg.horizon, cfg.decay
    )

    plans, choice = _plans(players, xp, squad, free_transfers)
    positions = {pid: player.element_type for pid, player in players.items()}
    gw_xp = {pid: projection.per_gw[event.id] for pid, projection in xp.items()}
    lineup = pick_lineup(choice.squad, positions, gw_xp)

    # A draft has no chips to weigh: they are played against a squad, and
    # there is not one yet.
    baseline, chips = None, NO_CHIPS
    if squad is not None:
        baseline = _chip_baseline(plans, choice)
        chips = chip_evs(
            baseline,
            pick_lineup(baseline.squad, positions, gw_xp),
            gw_xp,
            players,
            xp,
            squad.bank,
            event.id,
        )

    report = render_report(
        _label(mode, drafting=squad is None),
        event,
        plans,
        choice,
        lineup,
        chips,
        bootstrap,
        xp,
    )
    decision = {
        "mode": mode,
        "event": event.id,
        "initial_draft": squad is None,
        "free_transfers": free_transfers,
        "transfers_in": choice.transfers_in,
        "transfers_out": choice.transfers_out,
        "hits": choice.hits,
        "captain": lineup.captain,
        "vice": lineup.vice,
        "xp_total": choice.xp_total,
        "objective": choice.objective,
        "chip_evs": asdict(chips),
        "chip_baseline": _baseline_label(baseline),
    }

    if save:
        _write_report(cfg, event.id, mode, report)
        store.save_run(event.id, mode, report, decision)
    if send:
        _deliver(cfg, report)
    return report


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
    client: FplClient, team_id: int, squad: Squad | None, event: Event
) -> int | None:
    """Free transfers for ``event``, or None when there is no squad: nothing
    has been earned or spent by a manager who has not played yet."""
    if squad is None:
        return None
    return compute_free_transfers(
        client.transfers(team_id), client.chips_used(team_id), event.id
    )


def _expected_minutes(
    client: FplClient, players: dict[int, Player], held: list[int]
) -> dict[int, float]:
    """Expected minutes for everyone worth asking about, one request each.

    A player left out has no entry here at all, which projects him at zero —
    not a player the solver will buy, which is the point of leaving him out.
    """
    return {
        pid: expected_minutes(_history(client, pid), players[pid])
        for pid in history_pool(players, held)
    }


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
        candidates.sort(key=lambda candidate: (-candidate.total_points, candidate.id))
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


def _baseline_label(baseline: Plan | None) -> str | None:
    """Which plan the chips were priced against, for the record."""
    if baseline is None:
        return None
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
