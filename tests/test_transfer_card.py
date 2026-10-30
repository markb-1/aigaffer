"""The transfer card: the pairing rule and the card it feeds.

Boards are built by hand; the expected figures are worked in comments.
"""

from aigaffer.chips import FREE_HIT, WILDCARD
from aigaffer.report.transfer_card import (
    WILDCARD_POINTER,
    CardInputs,
    pair_moves,
    transfer_cards,
)


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


NO_CHIP = "none"


# --- the card ---------------------------------------------------------------
#
# The board. Calafiori (1) goes out for Gabriel (2); the squad also holds Rice
# (3, MID ARS) and a fixed filler. Events are GW9 and GW10. Prices are in
# tenths and the bank before the move is 59 (£5.9m).
#
#   bank_left = bank + sp(out) - pr(in) = 59 + 57 - 79 = 37  (£3.7m)
#
# Calafiori's stored x are 0.81 and 3.62, which sum to 4.43 and print 4.4.
# Gabriel's are 4.82 and 5.86: 10.68, printing 10.7. Gain 10.68 - 4.43 = 6.25,
# printing 6.2 under the f-string's round-half-even on the binary float, and
# this week 4.82 - 0.81 = 4.01, printing +4.0.


def row(name, pos, club, price, f, x, sp=None, cf=None):
    return {
        "n": name, "p": pos, "c": club, "cf": cf or club, "pr": price, "sp": sp,
        "f": f, "x": x,
    }


def board():
    return {
        1: row("Calafiori", "DEF", "ARS", 59, {9: [("LIV", "a")], 10: [("HUL", "h")]},
               {9: 0.81, 10: 3.62}, sp=57, cf="Arsenal"),
        2: row("Gabriel", "DEF", "ARS", 79, {9: [("LIV", "a")], 10: [("HUL", "h")]},
               {9: 4.82, 10: 5.86}, cf="Arsenal"),
        3: row("Rice", "MID", "ARS", 70, {9: [("LIV", "a")], 10: [("HUL", "h")]},
               {9: 4.0, 10: 4.0}, cf="Arsenal"),
    }


def cards_for(rows, out_ids, in_ids, squad, bank, chip=NO_CHIP, events=(9, 10)):
    return transfer_cards(CardInputs(rows, list(events), squad, bank), out_ids, in_ids, chip)


def test_card_matches_the_approved_shape():
    [card] = cards_for(board(), [1], [2], [1, 3], 59)

    assert card.splitlines() == [
        "Calafiori → Gabriel · DEF · £7.9m (bank £3.7m left)",
        "Out  liv 0.8 · HUL 3.6 = 4.4",
        "In   liv 4.8 · HUL 5.9 = 10.7",
        "+4.0 this week, +6.2 over 2 weeks",
    ]


def test_double_shows_the_weeks_number_once():
    rows = board()
    # A double in GW9: x[9] is already the week's one number, 7.1.
    rows[2]["f"][9] = [("BHA", "h"), ("ARS", "a")]
    rows[2]["x"][9] = 7.1

    [card] = cards_for(rows, [1], [2], [1, 3], 59)

    # 7.1 + 5.86 = 12.96, printed 13.0: the double is one number, not two.
    assert card.splitlines()[2] == "In   BHA+ars 7.1 · HUL 5.9 = 13.0"


def test_blank_cell_is_a_dash():
    rows = board()
    rows[2]["f"][10] = []
    rows[2]["x"][10] = 0.0

    [card] = cards_for(rows, [1], [2], [1, 3], 59)

    # 4.82 + 0.0 = 4.82, printed 4.8.
    assert card.splitlines()[2] == "In   liv 4.8 · — = 4.8"


def alt(name, price, total, club="XXX", pos="DEF", x=None):
    """A same-position alternative whose two gameweeks sum to ``total``."""
    return row(name, pos, club, price, {9: [], 10: []}, x or {9: total / 2, 10: total / 2})


def next_best(card):
    return next(line for line in card.splitlines() if line.startswith("Next best:"))


def test_next_best_excludes_the_outgoing_player_on_a_downgrade():
    # Out Saliba (pr 60, sp 60, sum 10.0) for Cheap (pr 45, sum 6.0), bank 0.
    # bank_left = 0 + 60 - 45 = 15, so the budget is 45 + 15 = 60, and Saliba
    # at 60 would qualify on price. He is being sold, so he must not show.
    rows = {
        1: alt("Saliba", 60, 10.0, "AAA"),
        2: alt("Cheap", 45, 6.0, "BBB"),
        4: alt("Mid", 55, 8.0, "CCC"),
    }
    rows[1]["sp"] = 60

    [card] = cards_for(rows, [1], [2], [1], 0)

    assert next_best(card) == "Next best: Mid 8.0 (£5.5m)"


def test_next_best_affordability_boundary():
    # Out 1 (pr 50, sp 50), in 2 (pr 50), bank 20: bank_left = 20 + 50 - 50 = 20,
    # budget 50 + 20 = 70. Exactly 70 is in; 71 is out.
    rows = {
        1: {**alt("Out", 50, 4.0, "AAA"), "sp": 50},
        2: alt("In", 50, 6.0, "BBB"),
        5: alt("Edge", 70, 9.0, "CCC"),
        6: alt("Over", 71, 12.0, "DDD"),
    }

    [card] = cards_for(rows, [1], [2], [1], 20)

    assert next_best(card) == "Next best: Edge 9.0 (£7.0m)"


def test_next_best_uses_listed_price_when_no_selling_price():
    # sp is None, so the sale raises the listed 60: bank_left = 10 + 60 - 80 = -10
    # and the budget is 80 - 10 = 70. (Had it used some other figure the
    # boundary would move.) 70 is in, 71 is out.
    rows = {
        1: alt("Out", 60, 4.0, "AAA"),
        2: alt("In", 80, 6.0, "BBB"),
        5: alt("Edge", 70, 9.0, "CCC"),
        6: alt("Over", 71, 12.0, "DDD"),
    }
    assert rows[1]["sp"] is None

    [card] = cards_for(rows, [1], [2], [1], 10)

    assert "bank £-1.0m left" in card
    assert next_best(card) == "Next best: Edge 9.0 (£7.0m)"


def test_next_best_respects_three_per_club():
    # The squad after the move holds ZZZ players 10, 11, 12: three, so a ZZZ
    # defender would be a fourth. WWW has two (13, 14), so one more is a third.
    rows = {
        1: {**alt("Out", 50, 4.0, "AAA"), "sp": 50},
        2: alt("In", 50, 6.0, "BBB"),
        7: alt("FourthZ", 50, 9.0, "ZZZ"),
        8: alt("ThirdW", 50, 8.0, "WWW"),
        **{pid: alt(f"Z{pid}", 40, 1.0, "ZZZ", pos="MID") for pid in (10, 11, 12)},
        **{pid: alt(f"W{pid}", 40, 1.0, "WWW", pos="MID") for pid in (13, 14)},
    }
    squad = [1, 10, 11, 12, 13, 14]

    [card] = cards_for(rows, [1], [2], squad, 0)

    assert next_best(card) == "Next best: ThirdW 8.0 (£5.0m)"


def test_higher_projected_alternative_is_still_listed_in_rank_order():
    # Alt (12.0) beats the pick (6.0) on plain sums. It is shown, first.
    rows = {
        1: {**alt("Out", 50, 4.0, "AAA"), "sp": 50},
        2: alt("In", 50, 6.0, "BBB"),
        5: alt("Alt", 50, 12.0, "CCC"),
        6: alt("Next", 50, 8.0, "DDD"),
        7: alt("Third", 50, 7.0, "EEE"),
    }

    [card] = cards_for(rows, [1], [2], [1], 0)

    assert next_best(card) == "Next best: Alt 12.0 (£5.0m) · Next 8.0 (£5.0m)"


def test_injured_player_with_stale_total_does_not_lead():
    # The injured man still carries a high decayed total (t) but his projected
    # points are all zero. Ranking is on the x sums, so he sits behind Real.
    rows = {
        1: {**alt("Out", 50, 4.0, "AAA"), "sp": 50},
        2: alt("In", 50, 6.0, "BBB"),
        5: {**alt("Injured", 50, 0.0, "CCC", x={9: 0.0, 10: 0.0}), "t": 50.0},
        6: alt("Real", 50, 2.0, "DDD"),
    }

    [card] = cards_for(rows, [1], [2], [1], 0)

    assert next_best(card) == "Next best: Real 2.0 (£5.0m) · Injured 0.0 (£5.0m)"


def test_no_next_best_line_when_nobody_is_eligible():
    [card] = cards_for(board(), [1], [2], [1, 3], 59)

    assert not any(line.startswith("Next best:") for line in card.splitlines())


def test_club_line_only_at_three():
    rows = board()
    rows[9] = row("Zed", "FWD", "ARS", 60, {9: [], 10: []}, {9: 1.0, 10: 1.0}, cf="Arsenal")

    # Squad 1 (out), 3, 9 -> after the move 2, 3, 9: three Arsenal.
    [three] = cards_for(rows, [1], [2], [1, 3, 9], 59)
    # Squad 1 (out), 3 -> after the move 2, 3: two Arsenal.
    [two] = cards_for(rows, [1], [2], [1, 3], 59)

    assert three.splitlines()[-1] == "3 from Arsenal now (Gabriel, Rice, Zed)"
    assert not any("from Arsenal" in line for line in two.splitlines())


def test_two_moves_share_one_bank_figure():
    # Out 1 (DEF, sp 57) and 4 (MID, sp 70); in 2 (DEF, pr 79) and 5 (MID, pr 80).
    # Bank 50: 50 + 57 + 70 - 79 - 80 = 18, £1.8m on both cards.
    rows = {
        1: {**alt("DefOut", 59, 4.0, "AAA"), "sp": 57},
        2: alt("DefIn", 79, 6.0, "BBB"),
        4: {**alt("MidOut", 70, 4.0, "CCC", pos="MID"), "sp": 70},
        5: alt("MidIn", 80, 6.0, "DDD", pos="MID"),
    }

    cards = cards_for(rows, [1, 4], [5, 2], [1, 4], 50)

    assert [card.splitlines()[0] for card in cards] == [
        "DefOut → DefIn · DEF · £7.9m (bank £1.8m left)",
        "MidOut → MidIn · MID · £8.0m (bank £1.8m left)",
    ]


def test_wildcard_gives_the_pointer_and_free_hit_gives_nothing():
    rows = board()

    assert cards_for(rows, [1], [2], [1, 3], 59, chip=WILDCARD) == [
        WILDCARD_POINTER.format(count=1)
    ]
    assert cards_for(rows, [1], [2], [1, 3], 59, chip=FREE_HIT) == []
    assert cards_for(rows, [], [], [1, 3], 59) == []
