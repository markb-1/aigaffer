"""The transfer card: one block per transfer, so the move can be judged on the phone.

What the owner does with a suggested transfer is open the FPL app and look at
both players' fixtures. The card puts that in the text: each fixture the window
holds for the man going and the man coming, next to its projected points, then
the totals, the gain, and the next best alternatives. It only arranges numbers
the run already made; nothing here projects anything, and no model writes a
word of it, so it cannot get a fixture wrong.

This module is pure and sits below :mod:`aigaffer.report.render`, which imports
it. It therefore imports nothing from that module.

Pairing comes first because everything else hangs off it. A plan keeps its
sales and its purchases as two sorted id lists, not as pairs, and zipping two
sorted lists pairs strangers: on a week with a defender and a midfielder going
out, a defender can come out against a midfielder going in. :func:`pair_moves`
is the one rule the card, the "Do this" checklist, the "working from" line and
the detailed report all share, so the lines never disagree about who replaces
whom.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.data.schedule import phone_games


def pair_moves(
    out_ids: Sequence[int],
    in_ids: Sequence[int],
    position: Callable[[int], object],
) -> list[tuple[int, int]]:
    """This week's transfers as ``(out, in)`` pairs, like for like.

    FPL transfers are like for like by position, so each sale is paired with a
    purchase at the same ``position(pid)``; within a position they pair in
    ascending id order. Id order, not strongest first: inside one position the
    pairing is presentational, and ids need no projections, so the checklist,
    which holds none, can use the same rule as the card. The matched groups
    come first, ordered by their lowest outgoing id. Whatever is left over on
    either side (a wildcard-style odd lot, a position change the plan made) is
    zipped together afterwards in id order, so nothing is dropped when the two
    lists are the same length. With unequal lengths the surplus is dropped,
    which is what the detailed report's note about an odd lot expects.
    """
    gone: dict[object, list[int]] = defaultdict(list)
    came: dict[object, list[int]] = defaultdict(list)
    for pid in sorted(out_ids):
        gone[position(pid)].append(pid)
    for pid in sorted(in_ids):
        came[position(pid)].append(pid)
    pairs: list[tuple[int, int]] = []
    spare_out: list[int] = []
    spare_in: list[int] = []
    for pos in sorted(gone, key=lambda p: gone[p][0]):
        matched = min(len(gone[pos]), len(came.get(pos, [])))
        pairs.extend(zip(gone[pos][:matched], came[pos][:matched], strict=True))
        spare_out.extend(gone[pos][matched:])
    for pos, ids in came.items():
        spare_in.extend(ids[min(len(gone.get(pos, [])), len(ids)) :])
    pairs.extend(zip(sorted(spare_out), sorted(spare_in), strict=False))
    return pairs


WILDCARD_POINTER = (
    "Wildcard: {count} transfers, too many for cards here."
    " The detailed report lists each with its fixtures and points."
)

# A club may hold at most this many of a squad's fifteen.
CLUB_LIMIT = 3
SHOWN_ALTERNATIVES = 2


class CardInputs(NamedTuple):
    """What the card needs beyond the move itself.

    ``rows`` are the snapshot-shaped player rows (``n p c cf pr sp f x``, the
    keys every snapshot has held since the format began, so an old snapshot
    still renders). ``events`` is the window's gameweeks. ``squad`` and
    ``bank`` are the squad and the bank (in tenths) *before* this week's
    moves, as the snapshot keeps them.
    """

    rows: Mapping[int, dict]
    events: list[int]
    squad: list[int]
    bank: int


def _money(tenths: int) -> str:
    return f"£{tenths / 10:.1f}m"


def _window(row: dict, events: Sequence[int]) -> float:
    """The plain sum of the row's stored per-gameweek points. Never the
    decayed ``t``: the card's totals must be the numbers it prints."""
    return sum(row["x"].get(gw, 0.0) for gw in events)


def _sale(row: dict) -> int:
    """What selling him raises: the ledger's price, or the listed one when
    the run had none."""
    return row["sp"] if row.get("sp") is not None else row["pr"]


def bank_left(cards: CardInputs, out_ids: Sequence[int], in_ids: Sequence[int]) -> int:
    """The bank after all of this week's moves, in tenths.

    ``bank + Σ sale(out) − Σ price(in)``: the figure the owner sees once he
    has made them, the same on every card of the week, and the one the
    alternatives' affordability uses. The Week plan's first text prints it
    too, so it and the card cannot disagree.
    """
    rows = cards.rows
    return (
        cards.bank
        + sum(_sale(rows[pid]) for pid in out_ids)
        - sum(rows[pid]["pr"] for pid in in_ids)
    )


def _cell(row: dict, gw: int) -> str:
    games = row["f"].get(gw) or []
    if not games:
        return phone_games(games)
    return f"{phone_games(games)} {row['x'].get(gw, 0.0):.1f}"


def _line(label: str, row: dict, events: Sequence[int]) -> str:
    cells = " · ".join(_cell(row, gw) for gw in events)
    return f"{label}{cells} = {_window(row, events):.1f}"


def _alternatives(
    cards: CardInputs,
    gone: int,
    came: int,
    new_squad: Sequence[int],
    out_ids: Sequence[int],
    left: int,
) -> list[str]:
    """The next best players at the incoming player's position, as ``Name sum
    (£price)``, at most two.

    Eligible: in the run's rows, same position, not in the squad after this
    week's moves and not one of the men being sold (on a downgrade-to-fund week
    the man leaving would otherwise top the list), affordable in the incoming
    player's place, and not a fourth player from a club the rest of the new
    squad already holds three of. Ranked on plain window sums, ties by name; a
    man who projects higher than the pick is still listed, in rank order.
    """
    rows = cards.rows
    target = rows[came]
    budget = target["pr"] + left
    clubs = Counter(rows[pid]["c"] for pid in new_squad if pid != came and pid in rows)
    found = []
    for pid, row in rows.items():
        if pid in new_squad or pid in out_ids or row["p"] != target["p"]:
            continue
        if row["pr"] > budget or clubs[row["c"]] >= CLUB_LIMIT:
            continue
        found.append((-_window(row, cards.events), row["n"], row))
    found.sort(key=lambda item: item[:2])
    return [
        f"{name} {-neg:.1f} ({_money(row['pr'])})"
        for neg, name, row in found[:SHOWN_ALTERNATIVES]
    ]


def transfer_cards(
    cards: CardInputs, out_ids: Sequence[int], in_ids: Sequence[int], chip: str
) -> list[str]:
    """One card per transfer, each a block of lines joined by newlines.

    ``[]`` when there are no moves or on a Free Hit week (the temporary team
    is not a transfer to judge); a single pointer line on a Wildcard week,
    where up to fifteen cards would be a wall on a phone. Otherwise the cards
    come in :func:`pair_moves` order. A pair whose rows the run does not hold
    is left out rather than raised on, so a hole in the data costs a card, not
    the digest that carries it.
    """
    if not out_ids and not in_ids or chip == FREE_HIT:
        return []
    if chip == WILDCARD:
        return [WILDCARD_POINTER.format(count=len(in_ids))]
    rows = cards.rows
    pairs = [
        pair
        for pair in pair_moves(
            out_ids, in_ids, lambda pid: rows[pid]["p"] if pid in rows else "?"
        )
        if pair[0] in rows and pair[1] in rows
    ]
    out_ids = [gone for gone, _ in pairs]
    in_ids = [came for _, came in pairs]
    if not pairs:
        return []
    left = bank_left(cards, out_ids, in_ids)
    new_squad = [pid for pid in cards.squad if pid not in out_ids] + list(in_ids)
    events = cards.events
    out_cards = []
    for gone, came in pairs:
        out_row, in_row = rows[gone], rows[came]
        out_sum, in_sum = _window(out_row, events), _window(in_row, events)
        this_week = in_row["x"].get(events[0], 0.0) - out_row["x"].get(events[0], 0.0)
        lines = [
            f"{out_row['n']} → {in_row['n']} · {in_row['p']} · {_money(in_row['pr'])}"
            f" (bank {_money(left)} left)",
            _line("Out  ", out_row, events),
            _line("In   ", in_row, events),
            f"{this_week:+.1f} this week, {in_sum - out_sum:+.1f} over {len(events)} weeks",
        ]
        best = _alternatives(cards, gone, came, new_squad, out_ids, left)
        if best:
            lines.append("Next best: " + " · ".join(best))
        mates = sorted(
            rows[pid]["n"] for pid in new_squad if pid in rows and rows[pid]["c"] == in_row["c"]
        )
        if len(mates) == CLUB_LIMIT:
            lines.append(f"{CLUB_LIMIT} from {in_row['cf']} now ({', '.join(mates)})")
        out_cards.append("\n".join(lines))
    return out_cards
