"""The transfer decision as one mixed-integer program.

Who to sell, who to buy, whether an extra transfer earns back its four-point
hit and who lines up are all the same question: a signing is only worth its
price if he gets into the XI, and he only gets into the XI at someone's
expense. So squad, XI and transfer count are chosen together, by CBC, rather
than in a sequence of individually sensible steps that add up to a bad one.

Two decisions are deliberately blunt:

* Bench players earn ``BENCH_WEIGHT`` of their projection. They only score
  when someone ahead of them fails to play, but a squad that values them at
  nothing drifts towards eleven stars and four unplayable cast-offs.
* Selling price is whatever the caller has observed. ``selling_prices``
  carries the purchase ledger's answer (:mod:`aigaffer.ledger`) for the
  players we hold — FPL pays purchase plus half a rise, rounded down, never
  the listed price — and a player it does not cover falls back to
  ``now_cost``, which is Phase 1's approximation and is still exact for a
  price that has not risen. Buys are always at ``now_cost``: that is what
  the market charges.

Left to itself the model will make at most ``MAX_TRANSFERS`` moves and pay for
at most ``MAX_HITS`` of them. Past three moves a manager is wildcarding rather
than transferring, and eight points in one gameweek is as much as a plan is
allowed to burn: a third hit is not priced and rejected, it is off the board.

A manager with more than three free transfers banked lifts the move cap to
what he holds — a free move costs nothing to consider, and the bank tops out
at five (:data:`~aigaffer.data.free_transfers.MAX_FREE_TRANSFERS`), so that is
five and not a wildcard. Fifteen free transfers is not a bank at all: it is
how :mod:`aigaffer.solver.lineup` prices a wildcard or free hit, by asking for
the best squad reachable with the whole fifteen on the table.
``forced_transfers`` overrides the move cap, which is also how a squad is
drafted from nothing — pass an empty ``current_squad`` and
``forced_transfers=15``.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pulp

from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection

if TYPE_CHECKING:  # the multi-week solver imports this module, so never the reverse
    from aigaffer.solver.multiweek import PlannedPath

GOALKEEPER, DEFENDER, MIDFIELDER, FORWARD = 1, 2, 3, 4

SQUAD_SIZE = 15
SQUAD_QUOTAS = {GOALKEEPER: 2, DEFENDER: 5, MIDFIELDER: 5, FORWARD: 3}
MAX_PER_CLUB = 3

XI_SIZE = 11
XI_GOALKEEPERS = 1
MIN_XI_DEFENDERS = 3
MIN_XI_FORWARDS = 1

BENCH_WEIGHT = 0.1
HIT_POINTS = 4
MAX_TRANSFERS = 3
# The ceiling: -8 in a gameweek, whatever the projection says it would
# earn back. Two hits is a considered gamble; three is a manager tilting.
MAX_HITS = 2

CANDIDATES_PER_POSITION = 40
AVAILABLE = "a"

# CBC ships with PuLP. The command wrapper keeps no per-problem state — every
# solve writes its own temporary files — so one instance serves the process.
SOLVER = pulp.PULP_CBC_CMD(msg=0)


@dataclass
class Plan:
    """One transfer decision and the squad it leaves behind.

    ``xp_total`` is the projection the plan earns — the XI in full, the bench
    at ``BENCH_WEIGHT`` — and ``objective`` is that net of the hits taken.

    ``path`` is the rest of the story when a plan came from
    :func:`~aigaffer.solver.multiweek.optimize_path`: the gameweeks after this
    one, advisory and re-planned every run. A plan from the single-week solver
    has none, which is how a caller tells the two engines apart.
    """

    squad: list[int]
    xi: list[int]
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    xp_total: float
    objective: float
    path: "PlannedPath | None" = None


@dataclass(frozen=True)
class Week1Lock:
    """Moves the owner has already entered this gameweek, held still in week 1.

    "Transfers made" records what he did; the run after it solves from the
    squad it left, and must neither sell a signing he has just made nor buy
    back a player he has just sold — that is re-recommending, or reversing,
    a decision already in the app. ``keep`` are the signings, ``shun`` the
    sales, and ``hold`` is a recorded free-hit week, whose standing squad
    makes no transfers at all. Week 1 only: from week 2 the window is as free
    as ever, and a signing who has since been ruled out is the report's to
    say in words, never the lock's to undo. Empty, it adds nothing to a
    model, so a run with nothing recorded is the model it always was.
    """

    keep: frozenset[int] = frozenset()
    shun: frozenset[int] = frozenset()
    hold: bool = False


def projected_points(xp: dict[int, PlayerProjection], player_id: int) -> float:
    """A player with no projection is worth nothing, not a guess."""
    projection = xp.get(player_id)
    return projection.total if projection else 0.0


def candidate_pool(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
    limit: int = CANDIDATES_PER_POSITION,
) -> list[int]:
    """The players the model is allowed to consider.

    Six hundred elements make for a slow program and most of them are never
    the answer, so the field is cut to the best ``limit`` available players in
    each position. Our own squad is always in the pool whatever its state: an
    injured player we own is still a player we own, and without him the model
    could not even leave the squad alone.

    ``limit`` is a parameter and not a constant because the multi-week solver
    carries a copy of every one of these columns per gameweek and so cannot
    afford as wide a field.
    """
    pool = {pid for pid in current_squad if pid in players}

    by_position: dict[int, list[int]] = defaultdict(list)
    for pid, player in players.items():
        if player.status == AVAILABLE:
            by_position[player.element_type].append(pid)
    for candidates in by_position.values():
        candidates.sort(key=lambda pid: projected_points(xp, pid), reverse=True)
        pool.update(candidates[:limit])

    return sorted(pool)


def optimize(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    forced_transfers: int | None = None,
    selling_prices: dict[int, int] | None = None,
    lock: Week1Lock | None = None,
) -> Plan | None:
    """Best squad and XI reachable from ``current_squad``, or None.

    ``bank`` and prices are in tenths of a million. None means no legal squad
    exists — usually a forced transfer count the budget, the pool or
    :data:`MAX_HITS` cannot support — which is an answer, not an error: the
    caller asks for several transfer counts and keeps the ones that came back.

    ``selling_prices`` is what each squad member's sale would actually raise,
    from the purchase ledger; a squad member absent from it — a caller that
    has no ledger, a row maintenance somehow never wrote — sells at his
    ``now_cost``, which is the pre-ledger behaviour and never worse than a
    guess. The game's money rule is ``spent on buys <= bank + raised by
    sales``, and it is written below as one row by valuing every squad slot
    at what its player is worth *to us*: a bought player at the ``now_cost``
    the market charges, a kept player at his selling price — which appears on
    both sides of the inequality and cancels, so holding a riser at a paper
    loss costs nothing, exactly as it does in the app.

    ``lock`` holds the recorded moves still — this solver has no per-player
    buy and sell binaries, so a kept signing is pinned into the squad and a
    recorded sale out of it.
    """
    pool = candidate_pool(players, xp, current_squad)
    current = {pid for pid in current_squad if pid in players}
    points = {pid: projected_points(xp, pid) for pid in pool}
    sale = selling_prices or {}
    value = {
        pid: sale.get(pid, players[pid].now_cost)
        if pid in current
        else players[pid].now_cost
        for pid in pool
    }
    budget = bank + sum(value[pid] for pid in current)

    problem = pulp.LpProblem("aigaffer_transfers", pulp.LpMaximize)
    squad = problem.add_variable_dicts("squad", pool, cat=pulp.LpBinary)
    starting = problem.add_variable_dicts("xi", pool, cat=pulp.LpBinary)
    hits = problem.add_variable("hits", lowBound=0)
    transfers = SQUAD_SIZE - pulp.lpSum(squad[p] for p in pool if p in current)

    problem += (
        pulp.lpSum(points[p] * starting[p] for p in pool)
        + BENCH_WEIGHT * pulp.lpSum(points[p] * (squad[p] - starting[p]) for p in pool)
        - HIT_POINTS * hits
    )

    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)

    problem += pulp.lpSum(squad.values()) == SQUAD_SIZE
    for position, quota in SQUAD_QUOTAS.items():
        problem += pulp.lpSum(squad[p] for p in by_position[position]) == quota
    for club_mates in by_club.values():
        problem += pulp.lpSum(squad[p] for p in club_mates) <= MAX_PER_CLUB
    if lock is not None:
        for p in sorted(lock.keep & set(pool)):
            if p in current:
                problem += squad[p] == 1
        for p in sorted(lock.shun & set(pool)):
            if p not in current:
                problem += squad[p] == 0
    problem += pulp.lpSum(value[p] * squad[p] for p in pool) <= budget

    problem += pulp.lpSum(starting.values()) == XI_SIZE
    for p in pool:
        problem += starting[p] <= squad[p]
    problem += (
        pulp.lpSum(starting[p] for p in by_position[GOALKEEPER]) == XI_GOALKEEPERS
    )
    problem += (
        pulp.lpSum(starting[p] for p in by_position[DEFENDER]) >= MIN_XI_DEFENDERS
    )
    problem += pulp.lpSum(starting[p] for p in by_position[FORWARD]) >= MIN_XI_FORWARDS

    if forced_transfers is None:
        problem += transfers <= max(MAX_TRANSFERS, free_transfers)
    else:
        problem += transfers == forced_transfers
    problem += hits >= transfers - free_transfers
    problem += hits <= MAX_HITS

    status = problem.solve(SOLVER)
    if pulp.LpStatus[status] != "Optimal":
        return None

    chosen = _chosen(squad)
    xi = _chosen(starting)
    bench = set(chosen) - set(xi)
    xp_total = sum(points[p] for p in xi) + BENCH_WEIGHT * sum(points[p] for p in bench)
    # h is continuous and only pinned from below, so it lands on the integer
    # count of paid transfers with a solver tolerance either side of it.
    taken = round(hits.value() or 0.0)

    return Plan(
        squad=chosen,
        xi=xi,
        transfers_in=sorted(set(chosen) - current),
        transfers_out=sorted(current - set(chosen)),
        hits=taken,
        xp_total=xp_total,
        objective=xp_total - HIT_POINTS * taken,
    )


def _grouped(pool: list[int], key: Callable[[int], int]) -> dict[int, list[int]]:
    """Pool ids bucketed by ``key``; an empty bucket sums to nothing, which
    leaves the quota that needs it unsatisfiable, which is the right answer."""
    groups: dict[int, list[int]] = defaultdict(list)
    for pid in pool:
        groups[key(pid)].append(pid)
    return groups


def _chosen(variables: dict[int, pulp.LpVariable]) -> list[int]:
    """The player ids whose binary came back set."""
    return sorted(pid for pid, var in variables.items() if (var.value() or 0) > 0.5)
