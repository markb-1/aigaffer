"""Replay real squads through each policy, and judge the policies by it.

The question is whether a new way of valuing the bench and the free
transfers the plan keeps (WINDOW, or its alternative ALT) does at least as
well as the model it replaces (LEGACY) when it has to live with its own
decisions. So real managers' squads are taken as they stood before the first
gameweek of a finished range, and each policy plays that range as a season of
its own from there: the same start, its own transfers, its own bank and free
transfers carried week to week, and each gameweek scored on what the players
actually did. What comes back is a paired difference per squad, candidate
against LEGACY, and a gate that says whether a candidate ships.

The module is in two halves. The pure core is a gameweek's score with the
game's automatic substitutions, the step that carries a squad's money and free
transfers through a week, the gate, and the summary; it makes no requests and
runs no solver. The driver below it samples the squads from the overall
league, rebuilds each past deadline's board — the minutes, strengths and
projections as they could have been made then — and plays every policy through
the range from every start, one solve a gameweek, fanned out over processes.

**This is not a true backtest, and the gate is a sanity check, not a
measurement.** These are the stated leaks, as in :mod:`aigaffer.backtest`:

* **The rates and prices.** The per-90 rates and the prices come from the
  bootstrap as it stands today, which has seen every gameweek being replayed,
  except where a player's history gives that gameweek's own value. The
  projections the policies plan against know a little of how the range went,
  and they know it for every policy alike.
* **The shortlist is today's survivors.** The players a policy may buy are the
  ones available today, so nobody since injured or gone is ever on offer, and
  nobody is ever bought the week before he broke down.
* **Chips are off.** Every policy plays the range chip-free, so the replay
  measures the bench and free-transfer terms and nothing of chip timing. A
  manager who played a chip in the range is still replayed chip-free from his
  start, and flagged.
* **Selling prices are simplified.** A player is bought and sold at that
  gameweek's value; the game's rule that a seller keeps only half of a rise
  is not modelled, so a policy that churns risers sells a little dear.
* **The schedule is today's, and the window sees past the range.** Each
  deadline plans six gameweeks ahead, as the live run does, so the last
  replayed deadline plans over fixtures five weeks beyond it. Those are all
  known — ``client.fixtures()`` serves the whole season — but they are the
  fixture list as it stands today: a match since moved for a cup tie sits in
  its new week, and the policies plan as if it had always been there.

And two notes on what the comparison can and cannot say:

* **Policies diverge from the real managers after week 1.** From the second
  gameweek on, each policy holds a squad of its own making that no real
  manager held. The comparison is paired on start states, not on what the
  managers did, and says nothing about how any of them would have fared.
* **The gameweeks are shared shocks.** A haul or a blank lands on every squad
  that holds the player under every policy at once, so the squads are not
  independent draws. The standard error is clustered by gameweek: it is taken
  over the gameweek means, and six of those make the SE rough. The end-credit
  sensitivity — leftover free transfers credited at 0, 1.5 and 2 apiece —
  is the check on a policy that wins only by banking.
"""

import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from math import inf, sqrt
from statistics import mean, variance

import httpx

from aigaffer.backtest import _before, strengths_before
from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS, compute_free_transfers
from aigaffer.data.models import Bootstrap, Fixture, GwHistory, Player
from aigaffer.model.minutes import expected_minutes
from aigaffer.model.xp import PlayerProjection, project_all, projected_events
from aigaffer.orchestrator import history_pool
from aigaffer.solver.lineup import XI_RANGES, Lineup, attacking_evs, pick_lineup
from aigaffer.solver.multiweek import CANDIDATES_PER_POSITION, optimize_path
from aigaffer.solver.optimizer import (
    ALT,
    HIT_POINTS,
    LEGACY,
    SQUAD_SIZE,
    WINDOW,
    WindowSettings,
    candidate_pool,
)

# The policy every candidate is judged against: today's model.
BASELINE = "LEGACY"

# The group whose lead breaks a tie between two candidates that both ship,
# and the end credit it is read at — the middle of the three, the expert's
# own figure for a banked transfer.
TOP_GROUP = "top"
TIE_BREAK_CREDIT = 1.5


@dataclass(frozen=True)
class Start:
    """A real manager's squad as it stood before the replay's first deadline.

    ``group`` names the rank band he was sampled from; ``chipped`` says he
    played a chip somewhere in the range, which the replay does not (chips are
    off), so his real season and the replayed ones part company from the start.
    """

    entry: int
    group: str
    squad: tuple[int, ...]
    bank: int
    free_transfers: int
    chipped: bool


@dataclass(frozen=True)
class State:
    """What a policy carries from one gameweek to the next: the fifteen, the
    money in the bank (in tenths of a million, as the API counts it), and the
    free transfers held for the next deadline."""

    squad: tuple[int, ...]
    bank: int
    free_transfers: int


@dataclass(frozen=True)
class WeekResult:
    """One policy's gameweek for one squad: the points it scored after hits,
    the hits it took, the moves it made, and — where it is being compared —
    the policy it made a different decision from.

    ``ft_end`` is the free transfers it holds after the week, which the gate's
    end credit reads off the last one. ``unproven`` says the week's decision
    was not proven the best: CBC stopped on its time limit with a path in
    hand, or came back with none and the squad rolled (:func:`run_policy`)."""

    gw: int
    points: int
    hits: int
    moves: int
    differs_from: str | None = None
    ft_end: int = 0
    unproven: bool = False


# --- scoring a gameweek -----------------------------------------------------


def autosub_points(
    lineup: Lineup, positions: dict[int, int], actual: dict[int, tuple[int, int]]
) -> int:
    """The gameweek's points for ``lineup``, with the game's automatic
    substitutions and its armband rule, before any hits.

    ``positions`` maps a player to his element type; ``actual`` maps him to
    ``(minutes, points)`` for the gameweek, summed over a double. A player
    with no entry at all — his team blanked — played no minutes.

    The substitutions follow the game's sequential procedure, because the
    order matters: a bench player can come on only for a starter still on the
    pitch, and only if the eleven stays a legal shape.

    1. **The keeper.** If the starting keeper played no minutes and the
       substitute keeper (``bench[0]``) did, he comes on.
    2. **The outfield,** in bench order. Each bench outfielder who played
       takes the place of the first starter, in XI order (by id), who did not
       play and has not already been replaced — provided the swap leaves the
       eleven within the formation bounds. A starter who did not play still
       counts towards the shape until he is replaced, and a replaced one is
       off the pitch for every later check. If no starter can be swapped for
       him, he stays on the bench and the next one is tried.
    3. **The armband.** The captain's points count twice if he played;
       otherwise the vice-captain's do if he played; otherwise nobody's.

    The keeper is never swapped for an outfielder, or an outfielder for him:
    the formation bounds hold exactly one keeper, so the shape check refuses
    it without a special case.
    """

    def minutes(player: int) -> int:
        return actual.get(player, (0, 0))[0]

    def points(player: int) -> int:
        return actual.get(player, (0, 0))[1]

    def legal(eleven: list[int]) -> bool:
        counts = defaultdict(int)
        for player in eleven:
            counts[positions[player]] += 1
        return all(low <= counts[pos] <= high for pos, (low, high) in XI_RANGES.items())

    eleven = sorted(lineup.xi)
    # A starter once replaced is out of the reckoning for the rest of the
    # procedure; the ones still waiting are those who missed and are on.
    waiting = [p for p in eleven if minutes(p) == 0]

    keeper_sub = lineup.bench[0]
    for starter in waiting:
        swapped = [keeper_sub if p == starter else p for p in eleven]
        if minutes(keeper_sub) > 0 and legal(swapped):
            eleven = swapped
            waiting.remove(starter)
            break

    for sub in lineup.bench[1:]:
        if minutes(sub) == 0:
            continue
        for starter in waiting:
            swapped = [sub if p == starter else p for p in eleven]
            if legal(swapped):
                eleven = swapped
                waiting.remove(starter)
                break

    total = sum(points(p) for p in eleven)
    if minutes(lineup.captain) > 0:
        total += points(lineup.captain)
    elif minutes(lineup.vice) > 0:
        total += points(lineup.vice)
    return total


# --- a week's step ------------------------------------------------------------


def next_free_transfers(free_transfers: int, moves: int) -> int:
    """The free transfers held for next deadline after ``moves`` this week:
    the moves come off what was free, never below nothing (the rest were
    hits), the week's one more goes on, and five is the most kept — the same
    carry the optimizer plans with, chips aside, because chips are off."""
    return min(MAX_FREE_TRANSFERS, max(free_transfers - moves, 0) + 1)


def apply_moves(
    state: State,
    transfers_in: list[int],
    transfers_out: list[int],
    value: dict[int, int],
    hits: int,
) -> State:
    """``state`` after this week's transfers: the squad with the moves made,
    the bank paid and refunded at this gameweek's values (``value``, in the
    API's tenths), and the free transfers carried by :func:`next_free_transfers`.

    A sale is refunded at the same value a purchase pays — the game's rule
    that a seller keeps only half of a rise is not modelled, a stated leak.

    ``hits`` is the plan's own count, and it must be exactly the moves beyond
    the free ones (the optimizer reads at most five free, as the game does).
    The caller charges the hits against the week's points; they are checked
    here because a plan that disagreed with the bank it was solved against
    would be scored at the wrong price, and silently. So is a move that does
    not fit the squad — selling a man not held, buying one already held, or
    an unequal swap — which can only be a bug upstream. And so is a bank left
    below nothing: the plan was solved at prices other than the ones it is
    paying, and it has bought what it could not afford.
    """
    held = set(state.squad)
    if len(transfers_in) != len(transfers_out):
        raise ValueError(f"{len(transfers_in)} in for {len(transfers_out)} out")
    if not set(transfers_out) <= held or len(set(transfers_out)) != len(transfers_out):
        raise ValueError(f"selling players not held: {sorted(set(transfers_out) - held)}")
    if set(transfers_in) & held or len(set(transfers_in)) != len(transfers_in):
        raise ValueError(f"buying players already held: {sorted(set(transfers_in) & held)}")
    owed = max(len(transfers_in) - min(state.free_transfers, MAX_FREE_TRANSFERS), 0)
    if hits != owed:
        raise ValueError(
            f"{len(transfers_in)} moves on {state.free_transfers} free transfers"
            f" owe {owed} hits, not {hits}"
        )

    out = set(transfers_out)
    squad = tuple(p for p in state.squad if p not in out) + tuple(transfers_in)
    bank = (
        state.bank
        + sum(value[p] for p in transfers_out)
        - sum(value[p] for p in transfers_in)
    )
    if bank < 0:
        raise ValueError(f"the moves leave the bank at {bank}")
    return State(
        squad=squad,
        bank=bank,
        free_transfers=next_free_transfers(state.free_transfers, len(transfers_in)),
    )


# --- the gate ------------------------------------------------------------------


def _cell(
    baseline: dict[tuple[int, int], dict],
    candidate: dict[tuple[int, int], dict],
    entries: list[int],
    gws: list[int],
    credit: float,
) -> dict[str, float]:
    """One cell of the gate: a candidate against the baseline in one group at
    one end credit.

    Both mappings take ``(entry, gw)`` to that squad's row. ``entries`` are
    the squads with a whole season under both policies — a row for every one
    of ``gws``, the replay's range — and only they are paired, in every week
    alike (see :func:`_whole_seasons`).

    ``d̄_g`` is the mean over those squads of the candidate's points less the
    baseline's in gameweek g. In the last gameweek each squad's difference also
    carries ``credit`` times the difference in the free transfers left over,
    which is the end credit: points a policy banked as transfers rather than
    spent. Then ``d = Σ_g d̄_g``, the mean paired season difference, and
    ``SE = sqrt(G) · stdev(d̄_g)`` with the sample stdev over the G gameweek
    means — the SE of a sum of G gameweek shocks, each estimated from the
    spread between them. It is computed as ``sqrt(G · variance)``, which is
    the same number and exact where the variance is. With fewer than two
    gameweeks there is no spread to measure, so the SE is infinite: such a cell
    can never fail the gate, and never ship on it unless its d is ≥ 0 outright.
    """
    last = gws[-1]
    weekly = []
    for gw in gws:
        diffs = []
        for entry in entries:
            diff = candidate[entry, gw]["points"] - baseline[entry, gw]["points"]
            if gw == last:
                diff += credit * (candidate[entry, gw]["ft_end"] - baseline[entry, gw]["ft_end"])
            diffs.append(diff)
        weekly.append(mean(diffs))
    se = sqrt(len(weekly) * variance(weekly)) if len(weekly) > 1 else inf
    return {"d": sum(weekly), "se": se}


def _whole_seasons(
    baseline: dict[tuple[int, int], dict],
    candidate: dict[tuple[int, int], dict],
    gws: list[int],
) -> tuple[list[int], int]:
    """The squads in one group with a row for every gameweek in ``gws`` under
    both policies, and how many of the group's squads fall short.

    The pairing is of whole seasons, never part of one. A squad paired only in
    the weeks it has would make each ``d̄_g`` a mean over a different set of
    squads, so the sum would no longer be any squad's season; and a squad
    missing the last gameweek would lose its end credit without a word. Such a
    squad is dropped from every week instead, and counted, so the summary can
    say how much of the sample the cell stands on.
    """
    seen = {entry for entry, _ in baseline} | {entry for entry, _ in candidate}
    whole = sorted(
        entry
        for entry in seen
        if all((entry, gw) in baseline and (entry, gw) in candidate for gw in gws)
    )
    return whole, len(seen) - len(whole)


def _verdict(cells: list[dict[str, float]]) -> str:
    """A candidate's verdict over its cells: "no" if any cell falls more than
    an SE below zero (exactly one SE behind is not below), "ship" if every
    cell is ≥ 0, and otherwise the owner's call — some cell is behind, but
    within an SE of level. The gate asks "not worse", not "better": six
    gameweeks cannot separate a point or two a week, and the prior the
    candidates were built on carries the rest. A candidate short of a cell
    never reaches here (see :func:`gate`)."""
    if any(c["d"] + c["se"] < 0 for c in cells):
        return "no"
    if all(c["d"] >= 0 for c in cells):
        return "ship"
    return "owner"


def gate(rows: list[dict], credits=(0.0, 1.5, 2.0)) -> dict:
    """Judge every candidate policy in ``rows`` against LEGACY.

    ``rows`` are the replay's per-squad, per-gameweek rows, each with at least
    ``group``, ``entry``, ``gw``, ``policy``, ``points`` (after hits) and
    ``ft_end`` (the free transfers held after that gameweek). Every policy
    other than LEGACY is a candidate, and each is scored in one cell per group
    and end credit (see :func:`_cell`), over the squads with a whole season
    under both — the season being every gameweek any row has.

    Returns::

        {"cells": {(candidate, group, credit): {"d", "se"}},
         "verdict": {candidate: "ship" | "owner" | "no"},
         "reason": {candidate: "incomplete"},   # only for those it applies to
         "dropped": {(candidate, group): squads short of a whole season},
         "pick": candidate | None}

    A candidate needs a cell for every credit in every group the rows hold. One
    short of that — a group with no whole season under it, or no rows at all —
    is "no", for being incomplete: a group it was never measured in is not a
    group it passed, and it must never ship on ``all([])``.

    If more than one candidate ships, the pick is the one with the larger d in
    the top group at a credit of 1.5 (the first in name order on an exact
    tie). A shipped candidate with no such cell — no top group was replayed,
    or 1.5 is not among the credits — cannot be picked; if none ships, or none
    can be picked, nothing is, and the table is the owner's to read.
    """
    by_policy: dict[str, dict[str, dict[tuple[int, int], dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        by_policy[row["policy"]][row["group"]][row["entry"], row["gw"]] = row

    gws = sorted({row["gw"] for row in rows})
    groups = sorted({row["group"] for row in rows})
    baseline = by_policy.get(BASELINE, {})
    candidates = sorted(p for p in by_policy if p != BASELINE)

    cells = {}
    dropped = {}
    for candidate in candidates:
        for group in groups:
            ours = baseline.get(group, {})
            theirs = by_policy[candidate].get(group, {})
            whole, short = _whole_seasons(ours, theirs, gws)
            if short:
                dropped[candidate, group] = short
            if not whole:
                continue
            for credit in credits:
                cells[candidate, group, credit] = _cell(ours, theirs, whole, gws, credit)

    needed = len(credits) * len(groups)
    verdict, reason = {}, {}
    for candidate in candidates:
        mine = [cell for key, cell in cells.items() if key[0] == candidate]
        if len(mine) < needed or not mine:
            verdict[candidate], reason[candidate] = "no", "incomplete"
        else:
            verdict[candidate] = _verdict(mine)

    pickable = [
        c
        for c in candidates
        if verdict[c] == "ship" and (c, TOP_GROUP, TIE_BREAK_CREDIT) in cells
    ]
    pick = (
        max(pickable, key=lambda c: cells[c, TOP_GROUP, TIE_BREAK_CREDIT]["d"])
        if pickable
        else None
    )
    return {
        "cells": cells,
        "verdict": verdict,
        "reason": reason,
        "dropped": dropped,
        "pick": pick,
    }


# --- the summary -----------------------------------------------------------------


def render_summary(gate_result: dict, meta: dict) -> str:
    """The gate as markdown, for the terminal and the pull request: a table of
    the cells, each candidate's verdict, the pick, the run's counts from
    ``meta``, and the caveats a reader needs to weigh it.

    Nothing here names a manager: ``meta`` is counts, and the cells are means.
    """
    lines = [
        "## Replay gate",
        "",
        "Mean paired season difference against LEGACY (d) and its"
        " gameweek-clustered standard error (SE), in points.",
        "",
        "| candidate | group | credit | d | SE |",
        "|---|---|---|---|---|",
    ]
    for (candidate, group, credit), cell in gate_result["cells"].items():
        lines.append(
            f"| {candidate} | {group} | {credit:.1f} | {cell['d']:+.2f} | {cell['se']:.2f} |"
        )
    lines += ["", "**Verdicts**", ""]
    reason = gate_result.get("reason", {})
    lines += [
        f"- {candidate}: {v}" + (f" ({reason[candidate]})" if candidate in reason else "")
        for candidate, v in gate_result["verdict"].items()
    ]
    dropped = gate_result.get("dropped", {})
    if dropped:
        lines += ["", "**Squads short of a whole season** (left out of every week)", ""]
        lines += [
            f"- {candidate}, {group}: {n} squads dropped as incomplete"
            for (candidate, group), n in dropped.items()
        ]
    lines += ["", f"**Pick: {gate_result['pick'] or 'none'}**"]
    if meta:
        lines += ["", "**Run**", ""]
        lines += [f"- {key}: {value}" for key, value in meta.items()]
    lines += [
        "",
        "**Read with care**",
        "",
        "- The SE is clustered by gameweek, and six clusters make the SE rough.",
        "- Policies diverge from the real managers after week 1: the comparison"
        " is paired on start states, not on what the managers did.",
        "- The end credit at 0, 1.5 and 2 a leftover free transfer is the check"
        " on a policy that wins only by banking.",
        "- Rates and prices are today's except where history gives the"
        " gameweek's value; the shortlist is today's survivors; chips are off;"
        " sales are at the gameweek's value, without the half-rise rule.",
    ]
    return "\n".join(lines) + "\n"


# --- the driver ------------------------------------------------------------------

# The policies a replay plays, by the names the rows and the gate carry.
POLICIES: dict[str, WindowSettings] = {"LEGACY": LEGACY, "WINDOW": WINDOW, "ALT": ALT}

# The game's overall league, whose standings are every manager in rank order,
# fifty to a page. The top group is drawn from the first two hundred pages
# (ranks to 10k), the deep one from pages 2001 to 20000 (100k to 1M): far
# enough down to be a different kind of manager, not so far as the abandoned.
OVERALL_LEAGUE = 314
TOP_PAGES = (1, 200)
DEEP_PAGES = (2001, 20000)
DEEP_GROUP = "deep"

# What every deadline plans with, as the live run does: six gameweeks, decayed
# at the configuration's 0.85. The solve's time limit is a third of the live
# run's minute, because a replay is a few thousand solves; one that hits it
# is counted as unproven rather than trusted.
HORIZON = 6
DECAY = 0.85
TIME_LIMIT = 20


@dataclass(frozen=True)
class Board:
    """One past deadline as a policy faces it, and the week as it turned out.

    ``players`` is the bootstrap with each player's ``now_cost`` replaced by
    his price that gameweek where his history gives it (``value``, which also
    prices :func:`apply_moves`), so the solver plans at the prices it then
    pays. ``projections`` are six gameweeks from ``gw``, made on the history
    before it. ``actual`` is each fetched player's ``(minutes, points)`` for
    ``gw``, summed over a double; a player absent from it played no minutes.

    ``shortlist`` is the top thirty a position by this board's projections —
    the field :func:`~aigaffer.solver.multiweek.optimize_path` cuts its
    candidates to, less the squad it adds back — and so the only players any
    policy can buy at this deadline. Every one of them has ``actual`` read,
    and every later board projects them (see :func:`build_boards`).
    """

    gw: int
    players: dict[int, Player]
    projections: dict[int, PlayerProjection]
    positions: dict[int, int]
    actual: dict[int, tuple[int, int]]
    value: dict[int, int]
    shortlist: frozenset[int] = frozenset()


def sample_starts(
    client,
    first_gw: int,
    last_gw: int,
    top_n: int,
    deep_n: int,
    seed: int,
    own_entry: int | None,
    counts: dict | None = None,
) -> list[Start]:
    """The real squads the replay starts from: ``top_n`` drawn from the top of
    the overall league and ``deep_n`` from deep in it, plus ``own_entry`` (if
    given) in the top group.

    The draw is ``random.Random(seed)``'s, so a seed is a sample: that many
    distinct pages from each band, and one row at random from each page
    drawn. A page that fails or comes back empty draws nobody. A manager drawn
    twice is replayed once.

    Each start is the squad and bank from the picks for ``first_gw − 1`` —
    what he held at the first replayed deadline — with the free transfers the
    season's transfers and chips leave him for it, and ``chipped`` when he
    played a chip anywhere in ``[first_gw, last_gw]``. A manager whose picks
    are missing (the API's 404 for a week he did not play), empty, short of a
    fifteen, or whose requests fail over HTTP is skipped.

    ``counts``, when given, gathers how the draw went — ``sampled``,
    ``skipped``, ``empty_pages``, ``duplicates`` — counts only, for the meta.
    """
    rng = random.Random(seed)
    counts = counts if counts is not None else {}

    def count(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    drawn: list[tuple[str, int]] = []
    for group, (low, high), n in (
        (TOP_GROUP, TOP_PAGES, top_n),
        (DEEP_GROUP, DEEP_PAGES, deep_n),
    ):
        for page in rng.sample(range(low, high + 1), n):
            try:
                rows = client.classic_league(OVERALL_LEAGUE, page)
            except httpx.HTTPError:
                rows = []
            if not rows:
                count("empty_pages")
                continue
            drawn.append((group, rng.choice(rows)["entry"]))
    if own_entry is not None:
        drawn.append((TOP_GROUP, own_entry))

    starts: list[Start] = []
    seen: set[int] = set()
    for group, entry in drawn:
        if entry in seen:
            count("duplicates")
            continue
        seen.add(entry)
        count("sampled")
        try:
            squad = client.picks(entry, first_gw - 1)
            transfers = client.transfers(entry)
            chips = client.chips_used(entry)
        except httpx.HTTPError:
            count("skipped")
            continue
        if len(squad.picks) != SQUAD_SIZE:
            count("skipped")
            continue
        starts.append(
            Start(
                entry=entry,
                group=group,
                squad=tuple(squad.player_ids),
                bank=squad.bank,
                free_transfers=compute_free_transfers(transfers, chips, first_gw),
                chipped=any(first_gw <= chip["event"] <= last_gw for chip in chips),
            )
        )
    return starts


def _fetch(client, histories: dict[int, list[GwHistory]], players: set[int]) -> None:
    """Each of ``players``' season histories, into the cache, once a player
    for the whole replay: every board reads the same season, cut at its own
    gameweek."""
    for pid in sorted(players - set(histories)):
        histories[pid] = client.element_summary(pid)[0]


def build_board(
    client,
    bootstrap: Bootstrap,
    fixtures: list[Fixture],
    histories: dict[int, list[GwHistory]],
    gw: int,
    extra: set[int],
) -> Board:
    """``gw``'s deadline rebuilt from the history before it, as
    :func:`~aigaffer.backtest.backtest_gw` rebuilds one, over six gameweeks.

    The pool projected is the pipeline's shortlist
    (:func:`~aigaffer.orchestrator.history_pool`) and ``extra``: everyone a
    start squad holds and everyone a policy could have bought before ``gw``
    (:func:`build_boards`). A player in neither projects at nothing, which is
    right for a man the pipeline never looks at and wrong for one a squad
    holds — a doubtful starter the shortlist skips would be benched and sold
    as worthless, and score as a man who never played.

    The shortlist is cut from those projections the way the solver will cut
    them, by :func:`~aigaffer.solver.optimizer.candidate_pool` with the
    window's width, and every one of them has his history fetched too: the
    week is scored on what the bought man did, whoever he is. The cut reads
    only availability and the projections, never a price, so cutting it on
    today's prices names the same players the solver's cut on the board's
    will. A shortlisted man from outside the pool still projects at nothing
    *here* — he is in the cut only as a tie at nothing, and that is what the
    solver sees — but every later board projects him on his history.

    ``histories`` is the replay's cache, filled in place.
    """
    players = {player.id: player for player in bootstrap.elements}
    pool = set(history_pool(players, [])) | {pid for pid in extra if pid in players}
    _fetch(client, histories, pool)
    xmins = {
        pid: expected_minutes(_before(histories[pid], gw), players[pid])
        for pid in sorted(pool)
    }
    projections = project_all(
        bootstrap,
        fixtures,
        xmins,
        gw,
        horizon=HORIZON,
        decay=DECAY,
        strengths=strengths_before(bootstrap, fixtures, gw),
    )
    shortlist = frozenset(
        candidate_pool(players, projections, [], limit=CANDIDATES_PER_POSITION)
    )
    _fetch(client, histories, set(shortlist))

    value: dict[int, int] = {}
    actual: dict[int, tuple[int, int]] = {}
    for pid, player in players.items():
        week = [row for row in histories.get(pid, []) if row.round == gw]
        priced = next((row.value for row in week if row.value is not None), None)
        value[pid] = player.now_cost if priced is None else priced
        if week:
            actual[pid] = (
                sum(row.minutes for row in week),
                sum(row.total_points for row in week),
            )
    return Board(
        gw=gw,
        players={
            pid: player.model_copy(update={"now_cost": value[pid]})
            for pid, player in players.items()
        },
        projections=projections,
        positions={pid: player.element_type for pid, player in players.items()},
        actual=actual,
        value=value,
        shortlist=shortlist,
    )


def build_boards(
    client,
    bootstrap: Bootstrap,
    fixtures: list[Fixture],
    gws: list[int],
    starts: list[Start],
) -> list[Board]:
    """Every replayed deadline's board, in gameweek order, before any policy
    runs.

    The extra each board projects is **cumulative**: the start squads, and
    every earlier board's shortlist. A player is only ever in a policy's
    squad because he started in it or was bought at some deadline, and a buy
    can only come from that deadline's shortlist — the solver's candidates
    are the shortlist and the squad — so this covers every squad every policy
    can hold, without knowing what any of them did. A player bought at GW3
    who has dropped out of GW5's thirty is still projected at GW5, rather
    than projected at nothing and sold for it.
    """
    extra = {pid for start in starts for pid in start.squad}
    histories: dict[int, list[GwHistory]] = {}
    boards = []
    for gw in gws:
        board = build_board(client, bootstrap, fixtures, histories, gw, extra)
        extra |= board.shortlist
        boards.append(board)
    return boards


def run_policy(
    start: Start,
    boards: list[Board],
    settings: WindowSettings,
    time_limit: int = TIME_LIMIT,
) -> list[WeekResult]:
    """One policy's season from ``start``: a solve at each board's deadline,
    its first week's moves made, and the week scored on what happened.

    The solve is the live window's — the board's six gameweeks, decayed, chip
    free (``held_chips=()``) and under ``settings`` passed outright, never the
    module default — from the squad, bank and free transfers the policy's own
    earlier weeks left it. No answer is a roll: the squad stands, and the week
    is marked unproven alongside the ones CBC stopped on its time limit.

    The moves go through :func:`apply_moves` at the board's prices with the
    plan's own hits, which raises on a plan that disagrees with the bank or the
    squad (:func:`replay` drops such a run whole). The week is scored as the
    game would score the eleven :func:`~aigaffer.solver.lineup.pick_lineup`
    fields on the projections, autosubs and armband included, less four a hit.
    """
    state = State(squad=start.squad, bank=start.bank, free_transfers=start.free_transfers)
    weeks: list[WeekResult] = []
    for board in boards:
        answer = optimize_path(
            board.players,
            board.projections,
            list(state.squad),
            state.bank,
            state.free_transfers,
            projected_events(board.projections),
            decay=DECAY,
            time_limit=time_limit,
            held_chips=(),
            settings=settings,
        )
        if answer is None:
            transfers_in, transfers_out, hits, unproven = [], [], 0, True
        else:
            plan, path = answer
            transfers_in, transfers_out, hits = plan.transfers_in, plan.transfers_out, plan.hits
            unproven = not path.proven
        state = apply_moves(state, transfers_in, transfers_out, board.value, hits)

        squad = list(state.squad)
        gw_xp = {
            pid: board.projections[pid].per_gw.get(board.gw, 0.0)
            for pid in squad
            if pid in board.projections
        }
        lineup = pick_lineup(
            squad, board.positions, gw_xp, attacking_evs(board.projections, board.gw)
        )
        weeks.append(
            WeekResult(
                gw=board.gw,
                points=autosub_points(lineup, board.positions, board.actual) - HIT_POINTS * hits,
                hits=hits,
                moves=len(transfers_in),
                ft_end=state.free_transfers,
                unproven=unproven,
            )
        )
    return weeks


# The boards a worker process plays against, handed over once at its start
# (``_take_boards``) rather than pickled into every task.
_BOARDS: list[Board] = []


def _take_boards(boards: list[Board]) -> None:
    global _BOARDS
    _BOARDS = boards


def _attempt(
    start: Start, boards: list[Board], settings: WindowSettings, time_limit: int
) -> tuple[list[WeekResult] | None, str | None]:
    """One ``(start, policy)`` run, or the class name of what stopped it.

    One run that raises — a plan :func:`apply_moves` refuses, a squad holding
    a player the bootstrap has lost — must not cost the other few hundred, so
    it is caught here and the run is dropped whole: half a season kept would
    be paired as though it were one. Only the class name comes back, the
    house rule for what an exception's text may carry.
    """
    try:
        return run_policy(start, boards, settings, time_limit), None
    except Exception as error:  # noqa: BLE001 — one run's failure is counted, not fatal
        return None, type(error).__name__


def _attempt_in_worker(
    start: Start, settings: WindowSettings, time_limit: int
) -> tuple[list[WeekResult] | None, str | None]:
    return _attempt(start, _BOARDS, settings, time_limit)


def _run_all(
    tasks: list[tuple[Start, WindowSettings]],
    boards: list[Board],
    jobs: int,
    time_limit: int,
) -> list[tuple[list[WeekResult] | None, str | None]]:
    """Every task's outcome, in task order: in this process when ``jobs`` is
    one, otherwise over a pool of that many, each worker handed the boards
    once. A worker lost outright is a failed run like any other."""
    if jobs <= 1:
        return [_attempt(start, boards, settings, time_limit) for start, settings in tasks]
    outcomes = []
    with ProcessPoolExecutor(
        max_workers=jobs, initializer=_take_boards, initargs=(boards,)
    ) as pool:
        futures = [
            pool.submit(_attempt_in_worker, start, settings, time_limit)
            for start, settings in tasks
        ]
        for future in futures:
            try:
                outcomes.append(future.result())
            except Exception as error:  # noqa: BLE001 — a lost worker is one failed run
                outcomes.append((None, type(error).__name__))
    return outcomes


def replay(
    client,
    first_gw: int,
    last_gw: int,
    top_n: int,
    deep_n: int,
    seed: int,
    jobs: int,
    own_entry: int | None,
    *,
    policies: dict[str, WindowSettings] = POLICIES,
    time_limit: int = TIME_LIMIT,
) -> tuple[list[dict], dict]:
    """Play every policy through ``first_gw..last_gw`` from every sampled
    start, and return the rows and the run's meta.

    The boards are built here, in gameweek order, before any policy runs
    (:func:`build_boards`), then every ``(start, policy)`` pair is played as a
    season of its own over ``jobs`` processes. A run that fails keeps no rows,
    so the gate drops its squad as short of a whole season; the failures are
    counted by exception class.

    A row is ``{"group", "entry", "gw", "policy", "points", "hits", "moves",
    "ft_end", "unproven"}``, one per squad, gameweek and policy — the only
    place an entry id appears. The meta is counts and nothing else, so it can
    be printed and pasted where the rows cannot.
    """
    bootstrap = client.bootstrap()
    fixtures = client.fixtures()
    counts: dict[str, int] = {}
    starts = sample_starts(
        client, first_gw, last_gw, top_n, deep_n, seed, own_entry, counts=counts
    )
    boards = build_boards(
        client, bootstrap, fixtures, list(range(first_gw, last_gw + 1)), starts
    )

    tasks = [(start, name) for start in starts for name in policies]
    outcomes = _run_all(
        [(start, policies[name]) for start, name in tasks], boards, jobs, time_limit
    )

    rows: list[dict] = []
    failures: Counter = Counter()
    for (start, name), (weeks, failure) in zip(tasks, outcomes, strict=True):
        if weeks is None:
            failures[failure] += 1
            continue
        rows += [
            {
                "group": start.group,
                "entry": start.entry,
                "gw": week.gw,
                "policy": name,
                "points": week.points,
                "hits": week.hits,
                "moves": week.moves,
                "ft_end": week.ft_end,
                "unproven": week.unproven,
            }
            for week in weeks
        ]

    meta = {
        "gameweeks": f"{first_gw}-{last_gw}",
        "sampled": counts.get("sampled", 0),
        "skipped": counts.get("skipped", 0),
        "empty_pages": counts.get("empty_pages", 0),
        "duplicates": counts.get("duplicates", 0),
        "starts": len(starts),
        TOP_GROUP: sum(start.group == TOP_GROUP for start in starts),
        DEEP_GROUP: sum(start.group == DEEP_GROUP for start in starts),
        "chipped": sum(start.chipped for start in starts),
        "runs": len(tasks),
        "failed": sum(failures.values()),
        **{f"failed_{name}": n for name, n in sorted(failures.items())},
        "unproven": sum(row["unproven"] for row in rows),
    }
    return rows, meta
