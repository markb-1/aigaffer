"""The team sheet, and what a chip would be worth.

The optimizer already names an eleven, but it names it against the horizon:
six points a week for six weeks outranks ten points on Saturday and nothing
after. That is the right way to buy a player and the wrong way to pick a
side. So the eleven is chosen again here on next gameweek's projection alone
— every legal formation tried, each filled with the best players it has room
for — and the four who miss out are ordered as substitutes, keeper first,
because a keeper can only come on for a keeper.

The chip numbers are marginal points: what playing the chip this week adds
over the plan we already have. Two of them are exact. Bench boost pays the
bench, who otherwise score nothing; the triple captain pays one more helping
of a captain who is already doubled. The other two are rough by design —
each asks the optimizer for the best squad fifteen free moves can reach and
takes the difference. The free hit flatters itself a little — the squad it
picks is credited with a tenth of its bench, where the side we would field
is counted eleven men only — and neither number knows anything about the
fixtures past the horizon, which is most of what a wildcard is really about.
They feed a report panel, not a decision.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.optimizer import (
    DEFENDER,
    FORWARD,
    GOALKEEPER,
    MIDFIELDER,
    SQUAD_SIZE,
    XI_SIZE,
    Plan,
    optimize,
)

# How many of each position a legal XI may field, fewest to most.
XI_RANGES = {
    GOALKEEPER: (1, 1),
    DEFENDER: (3, 5),
    MIDFIELDER: (2, 5),
    FORWARD: (1, 3),
}

# Every shape those ranges allow that adds up to eleven — the eight familiar
# formations, as (keepers, defenders, midfielders, forwards).
FORMATIONS = tuple(
    counts
    for counts in product(*(range(low, high + 1) for low, high in XI_RANGES.values()))
    if sum(counts) == XI_SIZE
)

# A wildcard or free hit is as many free transfers as there are players: the
# whole squad can change and none of it costs a hit.
CHIP_TRANSFERS = SQUAD_SIZE


@dataclass
class Lineup:
    """The eleven to field, the armbands, and the bench in the order it is
    read out: ``bench[0]`` is the substitute keeper."""

    xi: list[int]
    captain: int
    vice: int
    bench: list[int]


@dataclass
class ChipEvs:
    """Points each chip would add if it were played next gameweek."""

    bench_boost: float
    triple_captain: float
    free_hit: float
    wildcard: float


def attacking_evs(
    projections: dict[int, PlayerProjection], event: int
) -> dict[int, float] | None:
    """Each player's goals-and-assists expected points for ``event``, or None.

    Read off the projections themselves — :attr:`PlayerProjection.attacking_per_gw`
    is the ceiling slice the model already computed — so no second pass over the
    fixtures is needed. None when no projection carries an attacking figure for
    the gameweek, which is how a set of projections built by hand for a test
    says "no ceiling claim here" and lets :func:`pick_lineup` fall back on the
    total. It is the basis the armband is chosen on.
    """
    ev = {
        pid: projection.attacking_per_gw[event]
        for pid, projection in projections.items()
        if event in projection.attacking_per_gw
    }
    return ev or None


def pick_lineup(
    squad: list[int],
    positions: dict[int, int],
    gw_xp: dict[int, float],
    attacking_ev: dict[int, float] | None = None,
) -> Lineup:
    """The best eleven from ``squad`` for one gameweek, and the bench behind it.

    ``positions`` and ``gw_xp`` are keyed by player id; a player with no
    projection is worth nothing, not a guess. ``squad`` is a legal fifteen —
    two keepers, five defenders, five midfielders, three forwards — which is
    what makes every formation fillable.

    The eleven and the bench are chosen on ``gw_xp``, the whole projected points
    — you field the side that scores most. The armband is not: a captain is
    doubled, and what doubles into a haul is a goal or an assist, so it goes to
    the biggest *attacking* threat, not the biggest total. Two reasons the two
    differ. A defender's total is mostly floor — appearance, a clean sheet, a
    defensive contribution — none of which a doubled score touches, so captain
    and vice are drawn from the midfielders and forwards only. And among those,
    a cheap player's total can be padded past a genuine goal threat's by that
    same floor and a kind fixture, so within them the armband is ranked by
    ``attacking_ev`` — the goals-and-assists points alone — when the caller has
    it, and only by ``gw_xp`` as a fallback. The any-position ordering behind
    the attackers is itself a last resort, for a malformed eleven holding fewer
    than two of them, which a legal formation — two midfielders and a forward at
    the very least — never is.
    """
    ranked = _ranked(squad, positions, gw_xp)
    xi = max(
        (_fill(ranked, formation) for formation in FORMATIONS),
        key=lambda eleven: sum(gw_xp.get(pid, 0.0) for pid in eleven),
    )

    # Ceiling when we have it, total as a fallback; a defender is never a
    # candidate, so his high floor cannot win the armband whichever key is used.
    key = attacking_ev if attacking_ev is not None else gw_xp
    attackers = sorted(
        (pid for pid in xi if positions[pid] in (MIDFIELDER, FORWARD)),
        key=lambda pid: (-key.get(pid, 0.0), pid),
    )
    others = sorted(
        (pid for pid in xi if positions[pid] not in (MIDFIELDER, FORWARD)),
        key=lambda pid: (-gw_xp.get(pid, 0.0), pid),
    )
    preferred = attackers + others
    bench = sorted(
        set(squad) - set(xi),
        key=lambda pid: (positions[pid] != GOALKEEPER, -gw_xp.get(pid, 0.0), pid),
    )
    return Lineup(
        xi=sorted(xi), captain=preferred[0], vice=preferred[1], bench=bench
    )


def chip_evs(
    current_plan: Plan,
    lineup: Lineup,
    gw_xp: dict[int, float],
    players: dict[int, Player],
    xp: dict[int, PlayerProjection],
    bank: int,
    next_event: int,
) -> ChipEvs:
    """What each chip would add to ``current_plan``, next gameweek.

    A rebuild the solver cannot answer — no legal squad within the budget —
    is worth 0.0 rather than the difference against nothing, which would read
    as a chip that loses us the season.
    """
    fielded = sum(gw_xp.get(pid, 0.0) for pid in lineup.xi)
    free_hit = optimize(
        players, _single_gw(xp, next_event), current_plan.squad, bank, CHIP_TRANSFERS
    )
    wildcard = optimize(players, xp, current_plan.squad, bank, CHIP_TRANSFERS)

    return ChipEvs(
        bench_boost=sum(gw_xp.get(pid, 0.0) for pid in lineup.bench),
        triple_captain=gw_xp.get(lineup.captain, 0.0),
        free_hit=free_hit.xp_total - fielded if free_hit else 0.0,
        wildcard=wildcard.xp_total - current_plan.xp_total if wildcard else 0.0,
    )


def _single_gw(
    xp: dict[int, PlayerProjection], event: int
) -> dict[int, PlayerProjection]:
    """``xp`` rewritten so every player is worth his ``event`` gameweek alone.

    A free hit lasts a week and is handed back, so the squad it picks is the
    best squad for that week — the fixtures after it are somebody else's
    problem. ``event`` must be a gameweek the projections cover.
    """
    single = {}
    for pid, projection in xp.items():
        points = projection.per_gw[event]
        single[pid] = PlayerProjection(
            player_id=pid, per_gw={event: points}, total=points
        )
    return single


def _ranked(
    squad: list[int], positions: dict[int, int], gw_xp: dict[int, float]
) -> dict[int, list[int]]:
    """Squad ids by position, best projected first."""
    ranked: dict[int, list[int]] = defaultdict(list)
    for pid in sorted(squad, key=lambda p: (-gw_xp.get(p, 0.0), p)):
        ranked[positions[pid]].append(pid)
    return ranked


def _fill(ranked: dict[int, list[int]], formation: tuple[int, ...]) -> list[int]:
    """The formation's places taken by the best players available for them."""
    return [
        pid
        for position, count in zip(XI_RANGES, formation, strict=True)
        for pid in ranked[position][:count]
    ]
