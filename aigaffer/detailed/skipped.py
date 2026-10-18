"""The transfers the plan did not make, and the one reason each lost.

The detailed report shows the plan taken; the next question is always "why not
him?". This answers it from the run snapshot alone, with no solver and no
model: the strongest players the chosen plan left alone, each paired with the
squad man he would most naturally have replaced, the points gap between them
over the window, and the first rule that explains why the swap did not happen.

Who is offered: the top three per position by decayed window total among
projected players who are neither owned nor bought by the chosen plan — those
come first — and then every player another plan in the registry bought. The
second group matters because a plan the gaffer weighed and passed over is
exactly the "what about..." the owner will ask. A player in both groups keeps
his per-position row (first wins).

The tags are checked in a fixed order and the first that applies is the answer,
because a man who is both unaffordable and a minutes risk is better explained
by the thing that made the move impossible than the thing that made it
unattractive: unaffordable, club limit, minutes risk, would sit on the bench, not worth a
transfer, and failing all of those, the plan does better without him.

"Would sit on the bench" is for a man whose weighted total is below that of the
weakest XI player at his position: he would not start for the owner, so the gap
to the lowest squad man (usually the benched backup) overstates what he is
worth. A backup keeper behind a starter is the classic case. It sits after
minutes risk (a man who may not play is better explained by that) and before
not worth a transfer (the bench is the truer reason a small gap is small).

"Not worth a transfer" mirrors what the solver itself charges for a move (R6):
a transfer that costs a banked free transfer loses that step's value, decayed
to the end of the window, and one that costs a hit loses the hit; plus the
small per-transfer penalty. A gap under that is a swap the objective would not
have made even with nothing else wrong.
"""

from aigaffer.report.render import POSITIONS
from aigaffer.solver.optimizer import HIT_POINTS, WindowSettings

UNAFFORDABLE = "unaffordable"
CLUB_LIMIT = "club limit"
MINUTES_RISK = "minutes risk"
BENCH = "would sit on the bench"
NOT_WORTH = "not worth a transfer"
BEATEN = "the plan does better without him"

# Below this expected minutes a man is as likely to sit as to play.
MINUTES_FLOOR = 60
PER_POSITION = 3
MAX_PER_CLUB = 3


def _threshold(
    chosen_plan: dict,
    free_transfers: int | None,
    decay: float,
    window: int,
    settings: WindowSettings,
) -> float:
    """The gap a swap has to clear to be worth making (see module docstring).

    ``free_transfers`` None (a week the record does not state one, such as a
    free hit) means no free move is known to be used up, so no hit is assumed.
    """
    kept = chosen_plan.get("ft_kept") or 0
    if kept >= 2 and settings.ft_values:
        # LEGACY has no free-transfer values at all, so no step is lost; a
        # count past the table's end loses the last step it has.
        step = settings.ft_values[min(kept, len(settings.ft_values)) - 1]
        base = step * decay**window
    elif free_transfers is not None and len(chosen_plan["in"]) >= free_transfers:
        base = float(HIT_POINTS)
    else:
        base = 0.0
    return base + settings.ft_use_penalty


def skipped_transfers(
    players: dict,
    squad: list[int],
    plans: dict,
    chosen: int,
    bank: int,
    free_transfers: int | None,
    decay: float,
    window: int,
    settings: WindowSettings,
    *,
    xi: list[int],
) -> list[dict]:
    """The alternatives not taken, as ``{"id","pos","replaces","gap","price_diff","tag"}``.

    ``replaces`` is the lowest-total squad man at the candidate's position and
    ``gap`` the candidate's total less his, to two decimals. ``price_diff`` is
    the candidate's price less the replaced man's *sale* price (his selling
    price when the snapshot has one, else his price), in tenths — what the swap
    would cost the bank. ``bank`` is in tenths. ``xi`` is the fielded eleven,
    against which the bench tag is judged.
    """
    chosen_plan = plans[chosen]
    owned = set(squad)
    bought = set(chosen_plan["in"])
    threshold = _threshold(chosen_plan, free_transfers, decay, window, settings)

    candidates: list[int] = []
    for pos in POSITIONS.values():
        pool = [
            pid
            for pid, row in players.items()
            if row["p"] == pos and pid not in owned and pid not in bought
        ]
        pool.sort(key=lambda pid: (-players[pid]["t"], pid))
        candidates.extend(pool[:PER_POSITION])
    for plan_id in sorted(plans):
        if plan_id == chosen:
            continue
        for pid in plans[plan_id]["in"]:
            if pid in players and pid not in owned and pid not in bought:
                candidates.append(pid)

    weakest_starter: dict[str, float] = {}
    for starter in xi:
        if starter in players:
            pos, total = players[starter]["p"], players[starter]["t"]
            weakest_starter[pos] = min(weakest_starter.get(pos, total), total)

    rows = []
    for pid in dict.fromkeys(candidates):
        row = players[pid]
        mates = [s for s in squad if s in players and players[s]["p"] == row["p"]]
        if not mates:
            continue
        out_id = min(mates, key=lambda s: (players[s]["t"], s))
        out = players[out_id]
        gap = round(row["t"] - out["t"], 2)
        sale = out["sp"] if out["sp"] is not None else out["pr"]
        same_club = sum(1 for s in squad if s in players and players[s]["c"] == row["c"])
        if row["pr"] > bank + sale:
            tag = UNAFFORDABLE
        elif same_club >= MAX_PER_CLUB and out["c"] != row["c"]:
            tag = CLUB_LIMIT
        elif row["xm"] < MINUTES_FLOOR:
            tag = MINUTES_RISK
        elif row["p"] in weakest_starter and row["t"] < weakest_starter[row["p"]]:
            tag = BENCH
        elif gap < threshold:
            tag = NOT_WORTH
        else:
            tag = BEATEN
        rows.append(
            {
                "id": pid,
                "pos": row["p"],
                "replaces": out_id,
                "gap": gap,
                "price_diff": row["pr"] - sale,
                "tag": tag,
            }
        )
    return rows
