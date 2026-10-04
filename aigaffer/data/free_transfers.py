"""Free-transfer bank simulation.

The FPL API never tells us how many free transfers a manager has banked, so we
replay the season: managers earn their first free transfer for GW2 and one more
for every gameweek after that, banking at most ``MAX_FREE_TRANSFERS``. Each
transfer made spends one; extra transfers were points hits and cost nothing from
the bank.
"""

from collections import Counter

from aigaffer.chips import CHIP_API_NAMES, FREE_HIT, WILDCARD

MAX_FREE_TRANSFERS = 5
FREE_TRANSFER_CHIPS = (CHIP_API_NAMES[WILDCARD], CHIP_API_NAMES[FREE_HIT])


def compute_free_transfers(
    transfers: list[dict], chips: list[dict], next_event: int
) -> int:
    """Free transfers available for ``next_event``'s deadline.

    ``transfers`` and ``chips`` are the raw dicts from
    :meth:`~aigaffer.data.fpl_api.FplClient.transfers` and
    :meth:`~aigaffer.data.fpl_api.FplClient.chips_used`.
    """
    made_in = Counter(transfer["event"] for transfer in transfers)
    free_transfer_events = {
        chip["event"] for chip in chips if chip["name"] in FREE_TRANSFER_CHIPS
    }

    banked = 0
    for gw in range(2, next_event + 1):
        banked = min(MAX_FREE_TRANSFERS, banked + 1)
        # Transfers for next_event itself are still ahead of the deadline, so
        # nothing has been spent there yet.
        if gw < next_event and gw not in free_transfer_events:
            banked = max(0, banked - made_in[gw])
    return banked
