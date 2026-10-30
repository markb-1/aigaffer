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

from collections import defaultdict
from collections.abc import Callable, Sequence


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
