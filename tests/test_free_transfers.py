from aigaffer.data.free_transfers import compute_free_transfers


def t(event):
    return {"event": event}


def test_no_transfers_accumulates_capped():
    # GW2..GW8 accrual before GW9 deadline: capped at 5
    assert compute_free_transfers([], [], next_event=9) == 5


def test_gw2_has_one():
    assert compute_free_transfers([], [], next_event=2) == 1


def test_transfers_spend():
    assert compute_free_transfers([t(2)], [], next_event=3) == 1  # +1(gw2) -1 +1(gw3)


def test_hits_dont_go_negative():
    assert compute_free_transfers([t(2), t(2), t(2)], [], next_event=3) == 1


def test_wildcard_transfers_free():
    # GW2 earns 1 and the chip week spends nothing; GW3 is the week after the
    # chip, so it earns no +1: 1 stands.
    chips = [{"name": "wildcard", "event": 2}]
    assert compute_free_transfers([t(2), t(2), t(2)], chips, next_event=3) == 1


def test_freehit_transfers_free():
    # As the wildcard: 1 banked in GW2, kept through the chip week, and no +1
    # for GW3, the week after.
    chips = [{"name": "freehit", "event": 2}]
    assert compute_free_transfers([t(2), t(2), t(2)], chips, next_event=3) == 1


def test_other_chips_do_not_make_transfers_free():
    chips = [{"name": "bboost", "event": 2}]
    assert compute_free_transfers([t(2)], chips, next_event=3) == 1


def test_transfers_in_next_event_are_not_spent_yet():
    # The GW3 deadline has not passed, so a GW3 transfer has not been paid for.
    assert compute_free_transfers([t(3)], [], next_event=3) == 2


def test_accrual_resumes_after_spending():
    # Banked to the cap by GW9, two GW9 transfers leave 3, then +1 each for
    # GW10 and GW11.
    assert compute_free_transfers([t(9), t(9)], [], next_event=11) == 5


def test_before_gw2_there_are_no_free_transfers():
    assert compute_free_transfers([], [], next_event=1) == 0


def test_free_hit_keeps_four_but_the_next_week_gains_nothing():
    # The league's own example. GW2..GW5 earn 1 each, so 4 are saved ahead of
    # GW5 with no transfers made. A free hit in GW5 spends nothing and the week
    # after it earns no +1, so GW6 still has 4 (an ordinary week gives 5).
    chips = [{"name": "freehit", "event": 5}]
    assert compute_free_transfers([], chips, next_event=6) == 4
    assert compute_free_transfers([], [], next_event=6) == 5


def test_wildcard_keeps_four_but_the_next_week_gains_nothing():
    chips = [{"name": "wildcard", "event": 5}]
    assert compute_free_transfers([], chips, next_event=6) == 4


def test_the_plus_one_resumes_the_week_after_the_skipped_one():
    # Wildcard in GW5 (4 banked going in): GW6 stays 4, GW7 earns +1 again -> 5,
    # and GW8 is held at the cap of 5.
    chips = [{"name": "wildcard", "event": 5}]
    assert compute_free_transfers([], chips, next_event=7) == 5
    assert compute_free_transfers([], chips, next_event=8) == 5


def test_a_chip_week_does_not_change_the_chip_week_itself():
    # The chip is played in GW5, so the GW5 deadline is untouched: 4 earned.
    chips = [{"name": "freehit", "event": 5}]
    assert compute_free_transfers([], chips, next_event=5) == 4


def test_a_chip_recorded_at_gw1_does_not_take_gw2s_first_free_transfer():
    # Every manager is granted a free transfer for GW2 whatever the history
    # says; a chip at GW1 (impossible in the game, present in fixtures) must not
    # take it away: 1 for GW2.
    chips = [{"name": "wildcard", "event": 1}]
    assert compute_free_transfers([], chips, next_event=2) == 1
