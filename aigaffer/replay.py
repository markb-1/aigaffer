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

This module holds the pure core: a gameweek's score with the game's
automatic substitutions, the step that carries a squad's money and free
transfers through a week, the gate, and the summary. It makes no requests and
runs no solver.

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

from collections import defaultdict
from dataclasses import dataclass
from math import inf, sqrt
from statistics import mean, stdev

from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.solver.lineup import XI_RANGES, Lineup

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
    the policy it made a different decision from."""

    gw: int
    points: int
    hits: int
    moves: int
    differs_from: str | None = None


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
    an unequal swap — which can only be a bug upstream.
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
    return State(
        squad=squad,
        bank=bank,
        free_transfers=next_free_transfers(state.free_transfers, len(transfers_in)),
    )


# --- the gate ------------------------------------------------------------------


def _cell(
    baseline: dict[tuple[int, int], dict],
    candidate: dict[tuple[int, int], dict],
    credit: float,
) -> dict[str, float]:
    """One cell of the gate: a candidate against the baseline in one group at
    one end credit.

    Both arguments map ``(entry, gw)`` to that squad's row. A squad is paired
    only where both policies have a row for it, so a squad one policy lost is
    left out of every week rather than paired against nothing.

    ``d̄_g`` is the mean over squads of the candidate's points less the
    baseline's in gameweek g. In the last gameweek each squad's difference also
    carries ``credit`` times the difference in the free transfers left over,
    which is the end credit: points a policy banked as transfers rather than
    spent. Then ``d = Σ_g d̄_g``, the mean paired season difference, and
    ``SE = sqrt(G) · stdev(d̄_g)`` with the sample stdev over the G gameweek
    means — the SE of a sum of G gameweek shocks, each estimated from the
    spread between them. With fewer than two gameweeks there is no spread to
    measure, so the SE is infinite: such a cell can never fail the gate, and
    never ship on it unless its d is ≥ 0 outright.
    """
    paired = sorted(set(baseline) & set(candidate))
    gws = sorted({gw for _, gw in paired})
    last = gws[-1]
    weekly = []
    for gw in gws:
        diffs = []
        for key in (k for k in paired if k[1] == gw):
            diff = candidate[key]["points"] - baseline[key]["points"]
            if gw == last:
                diff += credit * (candidate[key]["ft_end"] - baseline[key]["ft_end"])
            diffs.append(diff)
        weekly.append(mean(diffs))
    se = sqrt(len(weekly)) * stdev(weekly) if len(weekly) > 1 else inf
    return {"d": sum(weekly), "se": se}


def _verdict(cells: list[dict[str, float]]) -> str:
    """A candidate's verdict over its cells: "no" if any cell falls more than
    an SE below zero, "ship" if every cell is ≥ 0, and otherwise the owner's
    call — some cell is behind, but within an SE of level. The gate asks "not
    worse", not "better": six gameweeks cannot separate a point or two a week,
    and the prior the candidates were built on carries the rest."""
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
    and end credit (see :func:`_cell`).

    Returns ``{"cells": {(candidate, group, credit): {"d", "se"}}, "verdict":
    {candidate: "ship" | "owner" | "no"}, "pick": candidate | None}``. If more
    than one candidate ships, the pick is the one with the larger d in the top
    group at a credit of 1.5 (the first in name order on an exact tie); if
    none ships, nothing is picked and the table is the owner's to read.
    """
    by_policy: dict[str, dict[str, dict[tuple[int, int], dict]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        by_policy[row["policy"]][row["group"]][row["entry"], row["gw"]] = row

    baseline = by_policy.get(BASELINE, {})
    candidates = sorted(p for p in by_policy if p != BASELINE)
    groups = sorted(baseline)

    cells = {}
    for candidate in candidates:
        for group in groups:
            if not set(baseline[group]) & set(by_policy[candidate].get(group, {})):
                continue
            for credit in credits:
                cells[candidate, group, credit] = _cell(
                    baseline[group], by_policy[candidate][group], credit
                )

    verdict = {
        candidate: _verdict([cell for key, cell in cells.items() if key[0] == candidate])
        for candidate in candidates
    }
    shipped = [c for c in candidates if verdict[c] == "ship"]
    if len(shipped) > 1:
        pick = max(shipped, key=lambda c: cells[c, TOP_GROUP, TIE_BREAK_CREDIT]["d"])
    else:
        pick = shipped[0] if shipped else None
    return {"cells": cells, "verdict": verdict, "pick": pick}


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
    lines += [f"- {candidate}: {v}" for candidate, v in gate_result["verdict"].items()]
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
