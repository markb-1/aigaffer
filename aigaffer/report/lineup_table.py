"""The lineup table: who to field, with this week's fixture and xP beside each.

What the owner does with a lineup is set it in the FPL app, glancing at who
plays whom. The table puts that in the text: a row per player, the eleven by
position and then the bench in substitution order, each row the name (with the
armband as ``C`` or ``V``), the fixture as the phone writes it and the
projected points. It sits in a fenced block, as the transfer card does, so a
proportional font cannot break its columns: GitHub draws the fence as code,
and the Telegram sender turns it into ``<pre>``.

The first row says what the numbers are (``GW10 fixture, xP``) and is kept
short on purpose. Telegram draws a ``</>`` badge over the top right of a
``<pre>``, and a full-width player row in first place would have its xP
covered; the label row never reaches that corner, because it is never padded
and every row carrying a number is longer.

It takes the eleven as labelled groups of ids and a name lookup, not a
``Lineup`` and its players, because the Week plan has only a snapshot's dicts
to draw from. It formats numbers it is given and never computes one: the xP is
the row's stored ``x[gw]`` at one decimal, before the armband, so the table
reads the same figure the run's other cards do. A player with no row prints
his name with empty slots, and nothing truncates: a missing number or a long
name costs a tidy column, never the text.

This module is pure and sits below :mod:`aigaffer.report.render`, which imports
it. It therefore imports nothing from that module.
"""

from collections.abc import Callable, Mapping, Sequence

from aigaffer.data.schedule import phone_games
from aigaffer.report.transfer_card import FENCE

LABEL_WIDTH = 3


def lineup_table(
    groups: Sequence[tuple[str, Sequence[int]]],
    bench: Sequence[int],
    *,
    captain: int | None,
    vice: int | None,
    name: Callable[[int], str],
    rows: Mapping[int, dict] | None,
    gw: int | None,
    notes: Mapping[int, str] | None = None,
) -> list[str]:
    """The fenced table's lines, the fences included.

    ``groups`` are the eleven in print order as ``(label, ids)``, the label
    printed on each group's first row only; the bench follows one empty row
    down as ``B1``, ``B2``, ... in the order given, and an empty bench adds no
    empty row. ``rows`` are snapshot-shaped (``f`` and ``x`` by gameweek).

    There are numbers when ``rows`` and ``gw`` are given and at least one man
    in the eleven has a row; then the table opens with ``GW{gw} fixture, xP``
    and each row reads label, name, fixture, xP. Otherwise it is names only,
    with no first row and no fixture or xP slots, so a week with no numbers
    still gets its team sheet. ``notes`` (a doubt such as ``75%``) go in a
    last column, one space after the rest; an id not on show is ignored.

    The name, fixture and xP widths are measured across the eleven and the
    bench together. A blank gameweek prints the dash and leaves the xP empty,
    and every row is rstripped, so it ends at the dash.
    """
    notes = notes or {}
    xi = [pid for _, ids in groups for pid in ids]
    numbered = bool(rows) and gw is not None and any(pid in rows for pid in xi)

    labelled: list[tuple[str, int] | None] = []
    for label, ids in groups:
        for i, pid in enumerate(ids):
            labelled.append((label if i == 0 else "", pid))
    if bench:
        labelled.append(None)
        labelled += [(f"B{i}", pid) for i, pid in enumerate(bench, start=1)]

    def shown(pid: int) -> str:
        marker = " C" if pid == captain else " V" if pid == vice else ""
        return f"{name(pid)}{marker}"

    def cells(pid: int) -> tuple[str, str]:
        row = rows.get(pid) if numbered else None
        if row is None:
            return "", ""
        games = row["f"].get(gw) or []
        if not games:
            return phone_games(games), ""
        return phone_games(games), f"{row['x'].get(gw, 0.0):.1f}"

    players = [entry for entry in labelled if entry is not None]
    names = {pid: shown(pid) for _, pid in players}
    slots = {pid: cells(pid) for _, pid in players}
    name_w = max((len(text) for text in names.values()), default=0)
    fixture_w = max((len(fix) for fix, _ in slots.values()), default=0)
    points_w = max((len(pts) for _, pts in slots.values()), default=0)

    def line(label: str, pid: int) -> str:
        text = f"{label:<{LABEL_WIDTH}} {names[pid]:<{name_w}}"
        if numbered:
            fix, pts = slots[pid]
            text += f"  {fix:<{fixture_w}} {pts:>{points_w}}"
        if pid in notes:
            text += f" {notes[pid]}"
        return text.rstrip()

    body = ["" if entry is None else line(*entry) for entry in labelled]
    head = [f"GW{gw} fixture, xP"] if numbered else []
    return [FENCE, *head, *body, FENCE]
