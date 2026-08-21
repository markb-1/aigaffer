"""The shortlist a manager actually chooses from.

The optimizer answers one question at a time — the best squad reachable in
exactly n moves — and the decision lives in the comparison between those
answers, not in any one of them. A free transfer worth 1.2 points next to a
two-hit worth 9.6 is a judgement a reader can make; a single number handed
down is one they have to take on trust. So the same board is solved at every
transfer count a manager would consider, and the results are laid out
best first for the report to print.

Past ``MAX_TRANSFERS`` moves the answer is a wildcard, not a transfer plan,
which is why the shortlist stops at three.
"""

from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.optimizer import MAX_TRANSFERS, Plan, optimize

TRANSFER_COUNTS = tuple(range(MAX_TRANSFERS + 1))


def generate_plans(
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
) -> list[Plan]:
    """Candidate plans, best objective first.

    A transfer count the budget or the pool cannot support comes back from
    the optimizer as None and is simply left out — infeasible is an answer,
    and a squad with nothing worth buying should say so by offering fewer
    plans, not by failing.

    Two counts should not land on the same fifteen, since the moves a squad
    took are implied by the squad itself, but nothing in the optimizer's
    contract promises that and the same plan printed twice reads as a bug.
    The first sighting wins, and because counts are tried in ascending order
    and the sort is stable, that is the one that got there in fewer moves.
    """
    plans: list[Plan] = []
    seen: set[tuple[int, ...]] = set()

    for count in TRANSFER_COUNTS:
        plan = optimize(
            players, xp, current_squad, bank, free_transfers, forced_transfers=count
        )
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
