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
    chips = [{"name": "wildcard", "event": 2}]
    assert compute_free_transfers([t(2), t(2), t(2)], chips, next_event=3) == 2


def test_freehit_transfers_free():
    chips = [{"name": "freehit", "event": 2}]
    assert compute_free_transfers([t(2), t(2), t(2)], chips, next_event=3) == 2


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
