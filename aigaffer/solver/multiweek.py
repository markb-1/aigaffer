"""The transfer *path* as one mixed-integer program.

:mod:`~aigaffer.solver.optimizer` answers the best question it can about one
gameweek: given the squad, the bank and the free transfers in hand, what is the
best fifteen reachable this week? It cannot answer the question a manager
actually asks in front of a double gameweek — *what do I do this week so that
in three weeks' time I own the players I want?* — because a single-week model
has nowhere to put the three weeks. Money raised has to be spent immediately,
a free transfer rolled is a free transfer wasted, and a plan that needs five
moves and can make three is simply infeasible rather than staged.

So this module solves the window instead of the week. One squad, one XI and one
captain per gameweek; buys and sells that carry the squad from each gameweek to
the next; a bank and a free-transfer count that carry with them. Only the first
gameweek is ever executed — the rest is advice, re-planned from scratch every
run — but the first gameweek's move is chosen knowing what it is for.

Four parts of the formulation are worth stating plainly.

**Free transfers are carried by upper bound alone.** The rule is
``ft[w+1] = min(5, ft[w] − transfers[w] + paid[w] + 1)``, and a linear program
cannot write ``min``. What it can write is the pair of inequalities the ``min``
is made of, with nothing pinning ``ft`` from below at all. That is exact rather
than a relaxation, because more free transfers never hurt: ``ft`` appears
elsewhere only in the pricing of ``paid``, where a larger ``ft`` can only lower
the one quantity the objective is charging four points a unit for, and in the
next gameweek's ceiling, which it raises. The solver therefore has every reason
to push ``ft`` as high as those inequalities allow and none whatever to hold it
down, so at the optimum it sits exactly on the ``min``.

The monotonicity has one more channel than that reads, and it is the channel
that has to be closed by hand. ``paid`` appears in the carry too, so lowering
``ft[w]`` by one raises ``paid[w]`` by one and leaves the carry's right-hand
side exactly where it was — a gameweek could buy back its own ceiling for four
points if ``paid`` were free to rise. It is not, because of the pin below; with
``paid`` held to ``max(0, transfers − ft)`` a lower ``ft`` costs four points and
buys nothing, and the argument above is then the whole of it.

**But the hits have to be pinned from both sides.** ``paid[w] ≥ transfers[w] −
ft[w]`` is the obvious half and, on its own, wrong: ``paid`` also appears in the
carry above, so a plan could take a hit it did not owe in order to leave its
free-transfer bank untouched — buying a free transfer for four points, which
the game does not sell. Left in, the model dodges the two-hit ceiling by
scattering bought transfers across gameweeks. So ``paid[w]`` is held to exactly
``max(0, transfers[w] − ft[w])`` by the usual big-M pair on an indicator: one
binary a gameweek, which on a five-gameweek window is nothing, and without it
the FT dynamics are not the game's.

**Nobody caps the transfers in a gameweek, and nobody has to.** ``paid[w]`` is
capped at :data:`~aigaffer.solver.optimizer.MAX_HITS`, which with
``paid[w] ≥ transfers[w] − ft[w]`` caps the moves at whatever is free plus two
— the single-week solver's doctrine (eight points is as much as a gameweek may
burn) restated a gameweek at a time. That ceiling is the whole reason a plan
ever moves early: five moves do not fit into one gameweek, so the one that only
raises money goes first.

**The captain is continuous and lands on an integer anyway.** With the XI
fixed, ``0 ≤ captain ≤ xi`` and ``Σ captain = 1`` describe a simplex whose
vertices are single players, so the solver cannot gain by splitting an armband
and never does. Leaving it continuous keeps a fifth of the binaries out of the
branch-and-bound tree, which on a five-gameweek window is worth having.

Two approximations from the single-week model are inherited unchanged and are
worse here than there, because they are compounded over the window: selling
price is the current price, and prices do not move. Both are documented
simplifications, not oversights.
"""

from dataclasses import dataclass

import pulp

from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.optimizer import (
    BENCH_WEIGHT,
    DEFENDER,
    FORWARD,
    GOALKEEPER,
    HIT_POINTS,
    MAX_HITS,
    MAX_PER_CLUB,
    MAX_TRANSFERS,
    MIN_XI_DEFENDERS,
    MIN_XI_FORWARDS,
    SQUAD_QUOTAS,
    SQUAD_SIZE,
    XI_GOALKEEPERS,
    XI_SIZE,
    Plan,
    _chosen,
    _grouped,
    candidate_pool,
)

# The single-week solver looks at forty a position. This one carries every
# column it builds once per gameweek in the window, so it looks at thirty.
CANDIDATES_PER_POSITION = 30

# A minute is longer than the window has ever needed and shorter than the run
# can spare. A solve that hits it comes back not-Optimal, which is None, which
# is the caller falling back on the single-week solver — never an exception.
SOLVE_SECONDS = 60
SOLVER = pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVE_SECONDS)


def _solver(time_limit: int | None) -> pulp.LpSolver:
    """The module's solver, or one like it on a shorter leash.

    A caller solving the same window half a dozen times over — which is what
    :func:`~aigaffer.solver.plans.generate_plans` does — cannot give each of
    them the minute a single decision is worth. The command wrapper keeps no
    per-problem state, so a second one costs nothing to build and the silence
    is the only thing it has to inherit.
    """
    if time_limit is None:
        return SOLVER
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)


@dataclass
class PlannedMove:
    """The transfers one future gameweek of the path makes.

    Gameweeks the plan intends to leave alone have no move at all, rather than
    an empty one: a path is the list of things it means to do.
    """

    event: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int


@dataclass
class PlannedPath:
    """The gameweeks after the one being decided.

    ``moves`` covers the window from its second gameweek on — the first is the
    :class:`~aigaffer.solver.optimizer.Plan` itself. ``objective`` is the whole
    window's, the same number the plan carries, and ``weekly_xp`` is each
    gameweek's XI and captain undecayed, so a reader can see where the points
    the plan is chasing actually are.
    """

    moves: list[PlannedMove]
    objective: float
    weekly_xp: dict[int, float]


def optimize_path(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    events: list[int],
    decay: float,
    forced_first_transfers: int | None = None,
    time_limit: int | None = None,
) -> tuple[Plan, PlannedPath] | None:
    """The best sequence of squads over ``events``, or None.

    ``events`` are the gameweek ids of the window in order, the first of them
    the one being decided. ``bank`` and prices are in tenths of a million and
    ``free_transfers`` is the bank of free moves standing at the first
    deadline, read as the game reads it and clamped to
    :data:`~aigaffer.data.free_transfers.MAX_FREE_TRANSFERS`: the fifteen with
    which :mod:`aigaffer.solver.lineup` prices a wildcard is not a bank, and a
    window planned as though it were would be a window of illegal gameweeks.
    Chips are not this solver's business and drafting from an empty squad is
    not either — fifteen signings will not fit under a gameweek's move ceiling.
    A player with no projection for a gameweek is worth nothing in it, which is
    what a blank is.

    ``forced_first_transfers`` pins the opening gameweek's moves; left alone,
    the opening gameweek moves at most ``max(MAX_TRANSFERS, ft)`` times, which
    is the single-week solver's ceiling and is here for the same reason it is
    there — past three moves a manager is wildcarding — and for one more: the
    two engines' answers are ranked against each other on one number, and a
    window that outbids the other with moves the other was never allowed to
    consider is not a comparison. A caller asking for a count above the cap is
    asking a question, and gets an answer.

    None means no answer, not an error: an infeasible board, a
    ``forced_first_transfers`` the pool or the budget cannot support, or a
    solve that ran out of ``time_limit`` seconds — a minute by default — with
    nothing to show for it. The caller falls back on the single-week solver,
    which is the doctrine everywhere else too.
    """
    if not events:
        return None

    opening_bank = min(max(free_transfers, 0), MAX_FREE_TRANSFERS)
    pool = candidate_pool(
        players, projections, current_squad, limit=CANDIDATES_PER_POSITION
    )
    current = {pid for pid in current_squad if pid in players}
    weeks = list(range(1, len(events) + 1))
    points = {
        (p, w): _projected(projections, p, events[w - 1]) for p in pool for w in weeks
    }
    owned = {p: 1 if p in current else 0 for p in pool}

    problem = pulp.LpProblem("aigaffer_transfer_path", pulp.LpMaximize)

    def per_week(name: str, **bounds) -> dict[int, dict[int, pulp.LpVariable]]:
        """One variable a player a gameweek, built in a fixed order so that the
        same board always writes the same model file."""
        return {
            w: problem.add_variable_dicts(f"{name}{w}", pool, **bounds) for w in weeks
        }

    squad = per_week("squad", cat=pulp.LpBinary)
    starting = per_week("xi", cat=pulp.LpBinary)
    # Continuous on purpose — see the module docstring.
    captain = per_week("captain", lowBound=0, upBound=1)
    buy = per_week("buy", cat=pulp.LpBinary)
    sell = per_week("sell", cat=pulp.LpBinary)
    # Integer because hits are whole points off a whole gameweek, and because
    # a handful of variables cost nothing to branch on.
    paid = {
        w: problem.add_variable(
            f"paid{w}", lowBound=0, upBound=MAX_HITS, cat=pulp.LpInteger
        )
        for w in weeks
    }
    # owing[w] = 1 when the gameweek's moves outran its free transfers. It is
    # what pins paid to the max() it stands for; see the module docstring.
    owing = {w: problem.add_variable(f"owing{w}", cat=pulp.LpBinary) for w in weeks}
    big_m = MAX_HITS + MAX_FREE_TRANSFERS
    # ft[1] is what the manager holds, a constant; the rest are carried.
    banked: dict[int, float | pulp.LpVariable] = {1: opening_bank}
    banked.update(
        {
            w: problem.add_variable(f"ft{w}", lowBound=0, upBound=MAX_FREE_TRANSFERS)
            for w in weeks[1:]
        }
    )
    # Prices are whole tenths and buy/sell are binary, so the bank lands on an
    # integer without being told to.
    cash = {w: problem.add_variable(f"bank{w}", lowBound=0) for w in weeks}
    moves = {w: pulp.lpSum(buy[w][p] for p in pool) for w in weeks}

    problem += (
        pulp.lpSum(
            decay ** (w - 1)
            * pulp.lpSum(
                points[p, w]
                * (
                    starting[w][p]
                    + captain[w][p]
                    + BENCH_WEIGHT * (squad[w][p] - starting[w][p])
                )
                for p in pool
            )
            for w in weeks
        )
        - HIT_POINTS * pulp.lpSum(paid[w] for w in weeks)
    )

    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)

    for w in weeks:
        held = {p: owned[p] if w == 1 else squad[w - 1][p] for p in pool}

        problem += pulp.lpSum(squad[w][p] for p in pool) == SQUAD_SIZE
        for position, quota in SQUAD_QUOTAS.items():
            problem += pulp.lpSum(squad[w][p] for p in by_position[position]) == quota
        for club_mates in by_club.values():
            problem += pulp.lpSum(squad[w][p] for p in club_mates) <= MAX_PER_CLUB

        problem += pulp.lpSum(starting[w][p] for p in pool) == XI_SIZE
        for p in pool:
            problem += starting[w][p] <= squad[w][p]
            problem += captain[w][p] <= starting[w][p]
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[GOALKEEPER])
            == XI_GOALKEEPERS
        )
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[DEFENDER])
            >= MIN_XI_DEFENDERS
        )
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[FORWARD]) >= MIN_XI_FORWARDS
        )
        problem += pulp.lpSum(captain[w][p] for p in pool) == 1

        for p in pool:
            problem += squad[w][p] == held[p] + buy[w][p] - sell[w][p]
            # Nobody is bought who is already here and nobody sold who is not,
            # which between them already imply that nobody is both, fractions
            # included — so the third row is redundant and stays anyway: it is
            # worth a third off the solve time, which a five-gameweek window
            # has no business turning down.
            problem += buy[w][p] + sell[w][p] <= 1
            problem += buy[w][p] <= 1 - held[p]
            problem += sell[w][p] <= held[p]

        problem += cash[w] == (bank if w == 1 else cash[w - 1]) + pulp.lpSum(
            players[p].now_cost * (sell[w][p] - buy[w][p]) for p in pool
        )

        problem += paid[w] >= moves[w] - banked[w]
        problem += paid[w] <= moves[w] - banked[w] + big_m * (1 - owing[w])
        problem += paid[w] <= big_m * owing[w]
        if w > 1:
            problem += banked[w] <= banked[w - 1] - moves[w - 1] + paid[w - 1] + 1

    if forced_first_transfers is None:
        problem += moves[1] <= max(MAX_TRANSFERS, opening_bank)
    else:
        problem += moves[1] == forced_first_transfers

    status = problem.solve(_solver(time_limit))
    # CBC stopped on its time limit with a squad in hand reports as Optimal
    # here, and that is the answer we want: a plan the solver could not prove
    # best is still a plan, and the alternative is the single-week solver.
    # A timeout with nothing found reports as not-solved, and that is None.
    if pulp.LpStatus[status] != "Optimal":
        return None

    # The objective is added back up from the solution rather than read off the
    # solver, so that the number a report prints is the one its own squads earn.
    objective = 0.0
    hits = 0
    weekly_xp: dict[int, float] = {}
    path_moves: list[PlannedMove] = []
    opening: list[int] = []
    opening_xi: list[int] = []
    opening_hits = 0

    for w in weeks:
        event = events[w - 1]
        chosen = _chosen(squad[w])
        eleven = _chosen(starting[w])
        bench = set(chosen) - set(eleven)

        started = sum(points[p, w] for p in eleven)
        armband = sum(points[p, w] * (captain[w][p].value() or 0.0) for p in pool)
        benched = sum(points[p, w] for p in bench)
        # paid is integral by declaration; round() only clears solver dust.
        taken = round(paid[w].value() or 0.0)

        weekly_xp[event] = started + armband
        objective += decay ** (w - 1) * (started + armband + BENCH_WEIGHT * benched)
        hits += taken

        if w == 1:
            opening, opening_xi, opening_hits = chosen, eleven, taken
            continue
        incoming = _chosen(buy[w])
        outgoing = _chosen(sell[w])
        if incoming or outgoing:
            path_moves.append(
                PlannedMove(
                    event=event,
                    transfers_in=incoming,
                    transfers_out=outgoing,
                    hits=taken,
                )
            )

    objective -= HIT_POINTS * hits
    path = PlannedPath(moves=path_moves, objective=objective, weekly_xp=weekly_xp)
    plan = Plan(
        squad=opening,
        xi=opening_xi,
        transfers_in=sorted(set(opening) - current),
        transfers_out=sorted(current - set(opening)),
        hits=opening_hits,
        # The plan is ranked against single-week plans on ``objective``, so that
        # stays the window's own number, hits and all; ``xp_total`` is it with
        # the hits added back, which is what the field means everywhere else.
        xp_total=objective + HIT_POINTS * hits,
        objective=objective,
        path=path,
    )
    return plan, path


def _projected(
    projections: dict[int, PlayerProjection], player_id: int, event: int
) -> float:
    """What a player is projected for one gameweek, undecayed.

    A player with no projection at all, or none for this gameweek, is worth
    nothing in it: a blank gameweek and a missing model are the same fact as
    far as the squad is concerned. The decay belongs to the objective, which is
    why this reads ``per_gw`` and never ``total``.
    """
    projection = projections.get(player_id)
    return projection.per_gw.get(event, 0.0) if projection else 0.0
