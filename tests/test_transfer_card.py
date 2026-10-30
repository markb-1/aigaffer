"""The transfer card: the pairing rule and the card it feeds.

Boards are built by hand; the expected figures are worked in comments.
"""

from aigaffer.report.transfer_card import pair_moves


def test_pairs_like_for_like_across_interleaved_ids():
    # Out 10 (DEF) and 30 (MID); in 20 (MID) and 25 (DEF). A plain zip of the
    # sorted lists would give (10, 20) and (30, 25), a defender for a midfielder.
    # Like for like: the DEF 10 takes the DEF 25, the MID 30 takes the MID 20.
    positions = {10: "DEF", 30: "MID", 20: "MID", 25: "DEF"}

    assert pair_moves([10, 30], [20, 25], positions.__getitem__) == [(10, 25), (30, 20)]


def test_pairs_within_a_position_by_id():
    # All DEF: ascending id order on both sides, however the lists arrive.
    assert pair_moves([5, 3], [9, 7], lambda pid: "DEF") == [(3, 7), (5, 9)]


def test_unmatched_positions_are_still_paired_not_dropped():
    # A DEF out and a MID in share no position, but the transfer is real, so
    # the leftovers are zipped rather than lost.
    positions = {1: "DEF", 2: "MID"}

    assert pair_moves([1], [2], positions.__getitem__) == [(1, 2)]
