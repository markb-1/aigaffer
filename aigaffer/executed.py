"""What the owner says he entered, and the squad later runs work from.

FPL's public API cannot see a transfer before the deadline it was made for,
so once the owner has entered the week's recommendation every later run
that gameweek would otherwise solve from the squad he no longer has. He
closes the gap himself: he texts "Transfers made" after entering the latest
verdict exactly, and the inbox (:mod:`aigaffer.inbox`,
:mod:`aigaffer.recording`) writes one row per gameweek — an
:class:`Executed` — holding the cumulative position: every move recorded so
far, the squad, bank and free transfers they leave, the chip, the armbands,
and the prices paid and raised at the moment of recording.

This module is a leaf on purpose. The store keeps the row, the ledger reads
its prices, the orchestrator and the renderers read its squad, and none of
them may form an import cycle through it: it imports the chip vocabulary and
the payload models and nothing else of ours.

The row is kept as a small JSON file per gameweek (``state/executed/gw{n}.json``,
:meth:`~aigaffer.store.Store.save_executed`) rather than a database table:
the inbox commits it at any minute while the other scheduler commits the
binary database, and two commits touching different text files always
rebase cleanly.

A :class:`Verdict` is a stored decision record and the two facts the runs
table keeps beside it — which report wrote it and when — because "the
verdict the owner saw" is a question about time as much as content.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from aigaffer.chips import CHIP_API_NAMES
from aigaffer.data.models import Pick, Squad

if TYPE_CHECKING:  # the orchestrator imports this module, never the reverse
    from aigaffer.orchestrator import PipelineInputs

# The chip a week plays when it plays none, in the record's own spelling.
# Repeated here rather than imported: render and the solver each keep the
# same word, and this module must import neither.
NO_CHIP = "none"


@dataclass(frozen=True)
class Verdict:
    """One stored decision record: the report that wrote it (``early``,
    ``scout`` or ``deadline``), when the runs table stamped it (ISO, UTC),
    and the record itself."""

    mode: str
    ts: str
    decision: dict


class StaleVerdict(ValueError):
    """A verdict solved from a squad that is neither the API's nor the
    recorded one: composing it would apply its moves to a position it never
    saw, so it is refused rather than guessed at."""


@dataclass(frozen=True)
class Executed:
    """The gameweek's cumulative recorded position.

    ``transfers_in``/``transfers_out`` are every move recorded so far against
    the API's squad, net of any reversal — and so also the week-1 lock set
    (:class:`~aigaffer.solver.optimizer.Week1Lock`). ``squad_after`` is the
    fifteen after them (the *standing* squad on a free-hit week, which
    reverts). ``chip`` is the chip recorded this gameweek, ``"none"`` most
    weeks. ``captain``/``vice`` are the latest recorded verdict's.
    ``buy_prices`` are what each incoming player cost on the live bootstrap
    when the owner texted, and ``sell_prices`` what each outgoing player
    raised by the ledger then — the closest the bot gets to the moment he
    clicked. ``bank_after`` and ``ft_after`` are after every recorded move
    (hits are not refunded on an ordinary week). ``ft_before`` is the API's
    own free-transfer count for the gameweek, before any recorded move: a
    wildcard or free hit played after earlier moves absorbs them, and the
    week's free transfers go back to it. ``verdicts`` are the ``[mode, ts]`` pairs
    composed in, oldest first. ``freehit_squad``/``freehit_xi`` are the
    temporary team on a free-hit week and None otherwise. ``arrival_status``
    is each incoming player's status flag at recording, which the T-3h
    reminder compares with today's.
    """

    gw: int
    mode: str
    recorded_at: str
    transfers_in: list[int]
    transfers_out: list[int]
    squad_after: list[int]
    chip: str
    captain: int
    vice: int
    buy_prices: dict[int, int]
    sell_prices: dict[int, int]
    bank_after: int
    ft_after: int
    ft_before: int
    verdicts: list[list[str]]
    freehit_squad: list[int] | None = None
    freehit_xi: list[int] | None = None
    arrival_status: dict[int, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        """The row as its file keeps it: JSON's own types, and dict keys as
        strings, because JSON has no other kind."""
        return {
            "gw": self.gw,
            "mode": self.mode,
            "recorded_at": self.recorded_at,
            "transfers_in": list(self.transfers_in),
            "transfers_out": list(self.transfers_out),
            "squad_after": list(self.squad_after),
            "chip": self.chip,
            "captain": self.captain,
            "vice": self.vice,
            "buy_prices": {str(k): v for k, v in self.buy_prices.items()},
            "sell_prices": {str(k): v for k, v in self.sell_prices.items()},
            "bank_after": self.bank_after,
            "ft_after": self.ft_after,
            "ft_before": self.ft_before,
            "verdicts": [list(pair) for pair in self.verdicts],
            "freehit_squad": self.freehit_squad,
            "freehit_xi": self.freehit_xi,
            "arrival_status": {str(k): v for k, v in self.arrival_status.items()},
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "Executed":
        """The row back from its file, string keys turned back into ids."""

        def keyed(value: object) -> dict:
            return {int(k): v for k, v in dict(value or {}).items()}

        return cls(
            gw=int(data["gw"]),
            mode=str(data["mode"]),
            recorded_at=str(data["recorded_at"]),
            transfers_in=list(data["transfers_in"]),
            transfers_out=list(data["transfers_out"]),
            squad_after=list(data["squad_after"]),
            chip=str(data["chip"]),
            captain=int(data["captain"]),
            vice=int(data["vice"]),
            buy_prices=keyed(data["buy_prices"]),
            sell_prices=keyed(data["sell_prices"]),
            bank_after=int(data["bank_after"]),
            ft_after=int(data["ft_after"]),
            ft_before=int(data["ft_before"]),
            verdicts=[list(pair) for pair in data["verdicts"]],
            freehit_squad=data.get("freehit_squad"),
            freehit_xi=data.get("freehit_xi"),
            arrival_status=keyed(data.get("arrival_status")),
        )


def apply_executed(
    inputs: "PipelineInputs", row: Executed | None
) -> tuple["PipelineInputs", Executed | None]:
    """The inputs as the owner's entered week leaves them, and the row applied.

    Applies only to the row's own gameweek: after the deadline the API shows
    the real picks and the free-transfer replay counts the real history, so a
    row from an earlier week would count everything twice. A draft has no
    squad to move. In every such case the inputs come back as they went in —
    the same object — with None.

    Otherwise: the squad is the row's fifteen, with its armbands; the bank
    is ``bank_after``, on the squad and on the standing line alike; the free
    transfers are ``ft_after`` (hits are not refunded; a wildcard or a free
    hit keeps them); a recorded chip joins the chip history, so it is no
    longer held. ``executed`` is set on the result, which is how the solve
    knows to lock the recorded moves and how every renderer knows to say
    what it is working from.
    """
    if row is None or inputs.squad is None or row.gw != inputs.event.id:
        return inputs, None
    squad = inputs.squad
    standing = (
        None
        if squad.standing is None
        else squad.standing.model_copy(update={"bank": row.bank_after})
    )
    effective_squad = Squad(
        picks=[
            Pick(
                element=pid,
                position=order,
                is_captain=pid == row.captain,
                is_vice_captain=pid == row.vice,
            )
            for order, pid in enumerate(row.squad_after, start=1)
        ],
        bank=row.bank_after,
        event=squad.event,
        standing=standing,
    )
    chips_used = list(inputs.chips_used)
    if row.chip != NO_CHIP:
        chips_used.append({"name": CHIP_API_NAMES[row.chip], "event": row.gw})
    effective = replace(
        inputs,
        squad=effective_squad,
        free_transfers=row.ft_after,
        chips_used=chips_used,
        executed=row,
    )
    return effective, row
