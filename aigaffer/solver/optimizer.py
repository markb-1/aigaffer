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
* Selling price is taken as the current price. Phase 1 does not track what we
  paid, so a player who has risen is valued a shade high — an approximation,
  not an oversight.

Left to itself the model will make at most ``MAX_TRANSFERS`` moves: past three
a manager is wildcarding, not transferring. A manager who *is* wildcarding
says so by having more than three free transfers, and the cap follows him up
— that is how a wildcard or free hit is valued, with fifteen free moves and
the whole squad on the table. ``forced_transfers`` overrides both, which is
also how a squad is drafted from nothing — pass an empty ``current_squad``
and ``forced_transfers=15``.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import pulp

from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection

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
    """

    squad: list[int]
    xi: list[int]
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    xp_total: float
    objective: float


def projected_points(xp: dict[int, PlayerProjection], player_id: int) -> float:
    """A player with no projection is worth nothing, not a guess."""
    projection = xp.get(player_id)
    return projection.total if projection else 0.0


def candidate_pool(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
) -> list[int]:
    """The players the model is allowed to consider.

    Six hundred elements make for a slow program and most of them are never
    the answer, so the field is cut to the best ``CANDIDATES_PER_POSITION``
    available players in each position. Our own squad is always in the pool
    whatever its state: an injured player we own is still a player we own, and
    without him the model could not even leave the squad alone.
    """
    pool = {pid for pid in current_squad if pid in players}

    by_position: dict[int, list[int]] = defaultdict(list)
    for pid, player in players.items():
        if player.status == AVAILABLE:
            by_position[player.element_type].append(pid)
    for candidates in by_position.values():
        candidates.sort(key=lambda pid: projected_points(xp, pid), reverse=True)
        pool.update(candidates[:CANDIDATES_PER_POSITION])

    return sorted(pool)


def optimize(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    forced_transfers: int | None = None,
) -> Plan | None:
    """Best squad and XI reachable from ``current_squad``, or None.

    ``bank`` and prices are in tenths of a million. None means no legal squad
    exists — usually a forced transfer count the budget or the pool cannot
    support — which is an answer, not an error: the caller asks for several
    transfer counts and keeps the ones that came back.
    """
    pool = candidate_pool(players, xp, current_squad)
    current = {pid for pid in current_squad if pid in players}
    points = {pid: projected_points(xp, pid) for pid in pool}
    budget = bank + sum(players[pid].now_cost for pid in current)

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
    problem += pulp.lpSum(players[p].now_cost * squad[p] for p in pool) <= budget

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
