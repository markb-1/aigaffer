"""Free-transfer bank simulation.

The FPL API never tells us how many free transfers a manager has banked, so we
replay the season: managers earn their first free transfer for GW2 and one more
for every gameweek after that, banking at most ``MAX_FREE_TRANSFERS``. Each
transfer made spends one; extra transfers were points hits and cost nothing from
the bank.

FPL's rules treat a Wildcard or Free Hit week specially in both directions: the
transfers made in it are free and spend nothing from the bank, and the week after
it earns no extra free transfer, so the bank stands at its pre-chip value for
that week and only starts growing again the week after. Four saved ahead of a
Free Hit week is still four the week after it, not five.
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
        # The +1 is the transfer earned *for* ``gw``; the week after a chip week
        # earns none.
        earned = 0 if gw - 1 in free_transfer_events else 1
        banked = min(MAX_FREE_TRANSFERS, banked + earned)
        # Transfers for next_event itself are still ahead of the deadline, so
        # nothing has been spent there yet.
        if gw < next_event and gw not in free_transfer_events:
            banked = max(0, banked - made_in[gw])
    return banked
