"""The rule that a call already made is kept unless something is new.

Run inside ``finalize_decision``, after every legality check has passed. What it
asks is small: a decision that changes the earlier call says why, and one that
changes it *inside the noise* also names a player something new is known about.
Nothing new to know is nothing to change a call on; the arithmetic that close is
the same arithmetic.

**News always gets through.** A record of adjust_players is news when it says
something the earlier run was not costed on, whatever day it was given, so a
quote that run missed counts as much as one given since. A flag, a price or a
fixture the API has changed is seen here without being told. A gap wider than
the noise needs no evidence at all, and what cannot be worked out is never a
reason to refuse. The caller bounds how often it refuses at all, and keeps the
decision it refused (:class:`aigaffer.manager.agent._Conversation`).
"""

from collections.abc import Iterable, Mapping

from aigaffer.continuity import (
    CAPTAIN_NOISE,
    CHEAPER,
    DEARER,
    FIXTURES,
    MINUTES_MOVED,
    NEWS,
    NOISE_POINTS,
    Call,
    Continuity,
    clean_reason,
    differences,
    fielded,
)
from aigaffer.executed import NO_CHIP
from aigaffer.manager.tools import MARKUP, ToolError
from aigaffer.news_ledger import Entry
from aigaffer.solver.optimizer import Plan

NAILED = "nailed"


def check_call(
    context: Continuity,
    call: Call,
    *,
    plan_id: int,
    registry: Mapping[int, Plan],
    solve_of: Mapping[int, int],
    xi: Iterable[int],
    gw_xp: Mapping[int, float],
    record: list[dict],
    reason: object,
    player_id: object,
) -> str | None:
    """The reason the report prints, or a ToolError that says what is missing.

    None when the call is the earlier one, or differs only in the vice: the
    report says that in words of its own and prints nobody's reason."""
    earlier = context.earlier
    differ = differences(earlier, call)
    if not differ - {"vice"}:
        return None

    kept = _plan_that_keeps(context, plan_id, registry, solve_of)
    said = reason if isinstance(reason, str) else ""
    cleaned = None if MARKUP.search(said) else clean_reason(said)
    if cleaned is None:
        raise ToolError(
            "That changes the call the owner already has"
            f" ({_spoken(differ)} differ). Say why in change_reason: plain words,"
            " under 80 characters, finishing the sentence \"because …\"."
            f" {_keeping(kept)}"
        )
    if not earlier.by_gaffer:
        return cleaned

    needs = [
        because
        for because in (
            _squad_inside_noise(differ, plan_id, kept, registry),
            _chip_dropped(context, call, differ),
            _captain_inside_noise(context, call, differ, xi, gw_xp),
        )
        if because is not None
    ]
    if needs and not _new_evidence(context, call, differ, record, player_id):
        raise ToolError(
            f"That changes the earlier call with nothing new behind it: {'; '.join(needs)}."
            " A change that small needs new evidence. Name in change_player_id a"
            " player you recorded new news about this run with adjust_players (news"
            " the earlier run was not costed on, not a fresh entry carried over),"
            " or one the API has changed since. If you have the news and have not"
            " recorded it, call adjust_players and finalize_decision in the same"
            f" turn. {_keeping(kept)}"
        )
    return cleaned


def _spoken(differ: set[str]) -> str:
    return ", ".join(part for part in ("squad", "chip", "captain", "vice") if part in differ)


def _keeping(kept: int | None) -> str:
    if kept is None:
        return "The earlier call's squad is not in this solve, so it cannot be kept as it was."
    return f"To keep the earlier call, finalize plan {kept} with its chip and captain."


def _plan_that_keeps(
    context: Continuity, plan_id: int, registry: Mapping[int, Plan], solve_of: Mapping[int, int]
) -> int | None:
    """The plan of the chosen one's own solve that leaves the earlier squad.

    Its own solve only: two solves are costed on two projections, and the
    difference of their nets is not a gap."""
    earlier = context.earlier
    solve = solve_of.get(plan_id)
    return next(
        (
            pid
            for pid, plan in sorted(registry.items())
            if solve_of.get(pid) == solve and fielded(plan, earlier.chip) == earlier.squad
        ),
        None,
    )


def _week1_chip(plan: Plan) -> str:
    return plan.path.week1_chip if plan.path is not None else NO_CHIP


def _squad_inside_noise(
    differ: set[str], plan_id: int, kept: int | None, registry: Mapping[int, Plan]
) -> str | None:
    if "squad" not in differ or kept is None:
        return None
    chosen, before = registry[plan_id], registry[kept]
    # A plan that plays a chip this week is worth the chip more than one that
    # does not: their nets differ by it and say nothing about the moves.
    if _week1_chip(chosen) != _week1_chip(before):
        return None
    gap = chosen.objective - before.objective
    if gap >= NOISE_POINTS:
        return None
    return f"plan {plan_id} is {gap:+.1f} net against plan {kept}, the earlier call, inside {NOISE_POINTS}"


def _chip_dropped(context: Continuity, call: Call, differ: set[str]) -> str | None:
    """A chip newly played is argued for in its own field. One taken off, or
    swapped for another, is argued for nowhere else."""
    earlier = context.earlier
    if "chip" not in differ or earlier.chip == NO_CHIP:
        return None
    if not context.chip_kept or context.chip_moved:
        return None
    return f"the earlier call played {earlier.chip} and the panel prices it as it did"


def _captain_inside_noise(
    context: Continuity, call: Call, differ: set[str], xi: Iterable[int], gw_xp: Mapping[int, float]
) -> str | None:
    if "captain" not in differ or "squad" in differ:
        return None
    before = context.earlier.captain
    if before not in set(xi):
        return None
    gap = gw_xp.get(call.captain, 0.0) - gw_xp.get(before, 0.0)
    if gap >= CAPTAIN_NOISE:
        return None
    return f"captain {call.captain} is {gap:+.1f} xP against {before}, inside {CAPTAIN_NOISE}"


def _new_evidence(
    context: Continuity, call: Call, differ: set[str], record: list[dict], player_id: object
) -> bool:
    if isinstance(player_id, bool) or not isinstance(player_id, int):
        return False
    involved = set(context.earlier.squad ^ call.squad)
    if "captain" in differ:
        involved |= {context.earlier.captain, call.captain}
    if any(
        _is_new(context, entry, player_id in involved)
        for entry in record
        if entry["player_id"] == player_id
    ):
        return True
    if context.since is None:
        return True
    # A price is evidence only for the change it argues for: a man the change
    # has that the earlier call did not, getting dearer, or one it gives up,
    # getting cheaper. The man the earlier call buys rising is a reason to keep
    # it. One risen out of reach is off the board, and nothing is asked then.
    counts = {NEWS, FIXTURES}
    if player_id in call.squad - context.earlier.squad:
        counts.add(DEARER)
    if player_id in context.earlier.squad - call.squad:
        counts.add(CHEAPER)
    return any(kind in counts for kind, _ in context.since.get(player_id, ()))


def _is_new(context: Continuity, record: dict, involved: bool) -> bool:
    """Whether one adjust_players record says something the earlier run did not
    know: other minutes than it was costed on, or other news than the ledger
    held when this run began. For a man the change moves, or whose armband it
    is, also the same news said again since that run."""
    pid = record["player_id"]
    used = context.used.get(pid)
    if used is None or abs(record["expected_minutes"] - used) >= MINUTES_MOVED:
        return True
    entry: Entry | None = context.known.get(pid)
    if entry is None:
        if record["category"] != NAILED:
            return True
    elif (
        record["category"] != entry.category
        or record["return_gw"] != entry.return_gw
        or record["tier"] < entry.tier
    ):
        return True
    first_or_later = entry is None or record["quote_date"] > entry.quote_date
    return (
        involved
        and first_or_later
        and record["quote_date"] >= context.earlier.ts.date().isoformat()
    )
