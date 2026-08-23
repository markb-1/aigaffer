"""The shortlist a manager actually chooses from.

The optimizer answers one question at a time — the best squad reachable in
exactly n moves — and the decision lives in the comparison between those
answers, not in any one of them. A free transfer worth 1.2 points next to a
two-hit worth 9.6 is a judgement a reader can make; a single number handed
down is one they have to take on trust. So the same board is solved at every
transfer count a manager would consider, and the results are laid out
best first for the report to print.

Past ``MAX_TRANSFERS`` moves the answer is a wildcard, not a transfer plan,
which is why the shortlist stops at three — unless the manager has more free
transfers than that banked, in which case it stops where his bank does. Five
free moves are five moves that cost nothing, and a shortlist that never asked
for the fourth and fifth cannot recommend them.

Two engines can answer. The window goes first: it plans several gameweeks at
once and only the first of them is ever played, so what lands on the shortlist
is that opening gameweek with the rest of the window hanging off it as a path.
When the window has nothing to say — an infeasible board, or a sweep of solves
that all ran out of time, or a manager who asked for the other engine — the
same questions go to the single-week solver instead, because a gameweek with a
deadline needs a recommendation more than it needs the better model.

Which engine answered is written down nowhere. A plan off the window carries
its path and a plan off the single-week solver does not, and a caller wanting
to label a report reads that.
"""

from collections.abc import Iterable

from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.multiweek import FREE_HIT, _free_hit_prices, optimize_path
from aigaffer.solver.optimizer import MAX_TRANSFERS, Plan, optimize

# The window gets a minute for the one solve a decision is worth; a sweep is
# half a dozen of them and the deadline does not move. Twenty seconds is enough
# for every window the season has thrown at it, and a solve that overruns with
# a squad in hand still comes back with it — only one that overruns with
# nothing at all is dropped.
SWEEP_TIME_LIMIT = 20


def transfer_counts(free_transfers: int) -> tuple[int, ...]:
    """The forced transfer counts worth solving for, none to the most.

    The most is three, or the bank if it holds more: a plan that spends only
    free transfers is never off the table for costing too much.
    """
    return tuple(range(max(MAX_TRANSFERS, free_transfers) + 1))


def generate_plans(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    *,
    projections_events: list[int] | None = None,
    decay: float = 0.85,
    planner: str = "multi",
    available_chips: frozenset[str] = frozenset(),
) -> list[Plan]:
    """Candidate plans, best objective first.

    ``projections_events`` is the window to plan over — gameweek ids in order,
    the first of them the one being decided — and ``decay`` is what a gameweek
    further out is worth against it. Without a window there is nothing for the
    multi-week solver to plan across, so the single-week solver answers alone;
    a ``planner`` of ``"single"`` says the same thing on purpose, and anything
    else at all means the window, since a misspelling should not be able to
    quietly turn the better engine off.

    ``available_chips`` are the chips the window may schedule — the four
    :mod:`~aigaffer.solver.multiweek` plans, minus the ones already spent, or
    empty when the chip planner is switched off. It rides straight through to
    every windowed solve; empty, which is the default, builds the pre-chip
    model to the byte and leaves the single-week fallback untouched (it never
    saw a chip in the first place).

    A transfer count the budget or the pool cannot support comes back from
    the optimizer as None and is simply left out — infeasible is an answer,
    and a squad with nothing worth buying should say so by offering fewer
    plans, not by failing. A window where *every* count came back None is not
    an answer at all, and the single-week solver is asked instead; one
    surviving plan is enough to keep it out of it, since a shortlist of one
    real window beats a shortlist of four guesses at the wrong question.
    """
    if planner != "single" and projections_events:
        # The free hit is the one chip priced by a solve of its own — a CBC
        # sub-solve a gameweek — and its price depends on the board, not on the
        # opening move a sweep pins: the pool, the budget and each week's points
        # are the same across every count. So it is priced once here and handed
        # to all of them, sparing the sub-solves the five extra passes the sweep
        # would otherwise pay for. None is a window that cannot field a free-hit
        # squad, which every count would return None on — the empty sweep that
        # falls through to the single-week solver, reached here without the work.
        freehit_prices = (
            _free_hit_prices(
                players, xp, current_squad, bank, projections_events, SWEEP_TIME_LIMIT
            )
            if FREE_HIT in available_chips
            else None
        )
        if FREE_HIT not in available_chips or freehit_prices is not None:
            # The window reads a free-transfer bank as five at the most, so a
            # sixth forced opening move is a question about a board it does not
            # believe in. Below that the counts are the single-week solver's own.
            opened: list[Plan | None] = []
            for count in transfer_counts(min(free_transfers, MAX_FREE_TRANSFERS)):
                answer = optimize_path(
                    players,
                    xp,
                    current_squad,
                    bank,
                    free_transfers,
                    projections_events,
                    decay,
                    forced_first_transfers=count,
                    time_limit=SWEEP_TIME_LIMIT,
                    available_chips=available_chips,
                    freehit_prices=freehit_prices,
                )
                # The path comes back beside the plan and is already on it, so
                # the second half of the pair is nothing the shortlist carries.
                opened.append(answer[0] if answer is not None else None)

            planned = _shortlist(opened)
            if planned:
                return planned

    return _shortlist(
        optimize(
            players, xp, current_squad, bank, free_transfers, forced_transfers=count
        )
        for count in transfer_counts(free_transfers)
    )


def _shortlist(candidates: Iterable[Plan | None]) -> list[Plan]:
    """The plans worth printing out of what a sweep came back with.

    Two counts should not land on the same fifteen, since the moves a squad
    took are implied by the squad itself, but nothing in either solver's
    contract promises that and the same plan printed twice reads as a bug.
    The first sighting wins, and because both sweeps hand their answers over
    in ascending order of moves and the sort below is stable, that is the one
    that got there in fewer moves.
    """
    plans: list[Plan] = []
    seen: set[tuple[int, ...]] = set()

    for plan in candidates:
        if plan is None:
            continue
        squad = tuple(sorted(plan.squad))
        if squad in seen:
            continue
        seen.add(squad)
        plans.append(plan)

    return sorted(plans, key=lambda plan: plan.objective, reverse=True)


def recommend(plans: list[Plan]) -> Plan:
    """The plan to put at the top of the report.

    Best objective, and among plans worth the same the one that moves least:
    a transfer that buys no points still spends a free transfer that next
    week might need, and the squad it leaves behind is no better.
    """
    return max(plans, key=lambda plan: (plan.objective, -len(plan.transfers_in)))
