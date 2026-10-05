"""Which chips are in hand, and when each may be played."""

import pytest

from aigaffer.chips import (
    BENCH_BOOST,
    FREE_HIT,
    TRIPLE_CAPTAIN,
    WILDCARD,
    ChipWindow,
    HeldChip,
    chip_windows,
    held_chips,
    held_for,
    playable_in,
    rule_gaps,
    whole_season,
)
from aigaffer.data.models import Bootstrap, ChipRule

# The live 2026-27 rules, as the bootstrap serves them.
LIVE_RULES = [
    ChipRule(name="wildcard", start_event=2, stop_event=19),
    ChipRule(name="wildcard", start_event=20, stop_event=38),
    ChipRule(name="freehit", start_event=2, stop_event=19),
    ChipRule(name="bboost", start_event=1, stop_event=19),
    ChipRule(name="3xc", start_event=1, stop_event=19),
    ChipRule(name="freehit", start_event=20, stop_event=38),
    ChipRule(name="bboost", start_event=20, stop_event=38),
    ChipRule(name="3xc", start_event=20, stop_event=38),
]
WINDOWS = chip_windows(LIVE_RULES)


def ids(held):
    return [chip.id for chip in held]


def test_the_live_rules_read_as_two_sets_of_four():
    assert len(WINDOWS) == 8
    assert ChipWindow(BENCH_BOOST, 1, 19) in WINDOWS
    assert ChipWindow(FREE_HIT, 20, 38) in WINDOWS


def test_an_untouched_season_holds_all_eight_ordered_by_expiry_then_kind():
    assert ids(held_chips(WINDOWS, [], 6)) == [
        "bench_boost@19", "triple_captain@19", "wildcard@19", "free_hit@19",
        "bench_boost@38", "triple_captain@38", "wildcard@38", "free_hit@38",
    ]


def test_a_chip_played_in_the_first_half_spends_only_the_first_half_one():
    held = held_chips(WINDOWS, [{"name": "bboost", "event": 7}], 8)
    assert "bench_boost@19" not in ids(held)
    assert "bench_boost@38" in ids(held)


def test_a_chip_played_in_the_second_half_spends_only_the_second_half_one():
    # Asked from GW6, where both windows are open, so the only thing that can
    # tell them apart is which one the GW25 play falls inside.
    held = held_chips(WINDOWS, [{"name": "bboost", "event": 25}], 6)
    assert "bench_boost@19" in ids(held)
    assert "bench_boost@38" not in ids(held)


@pytest.mark.parametrize(
    ("played_at", "spent", "kept"),
    [
        (19, HeldChip(BENCH_BOOST, 1, 19), HeldChip(BENCH_BOOST, 20, 38)),
        (20, HeldChip(BENCH_BOOST, 20, 38), HeldChip(BENCH_BOOST, 1, 19)),
    ],
)
def test_a_play_on_the_boundary_spends_the_set_whose_window_holds_it(
    played_at, spent, kept
):
    # GW19 is the first set's last week and GW20 the second's first: a play
    # on either edge spends exactly the window it is inside, never both.
    held = held_chips(WINDOWS, [{"name": "bboost", "event": played_at}], 6)
    assert spent not in held
    assert kept in held


@pytest.mark.parametrize("event", ["7", 7.0, None])
def test_a_play_with_no_whole_gameweek_marks_nothing(event):
    # A gameweek that is not an int — a string, a float, missing — cannot be
    # placed in a window, and guessing would cost a chip.
    assert len(held_chips(WINDOWS, [{"name": "bboost", "event": event}], 6)) == 8


def test_a_closed_window_is_dropped():
    assert all(chip.stop_event == 38 for chip in held_chips(WINDOWS, [], 20))


def test_a_second_half_chip_is_held_but_not_playable_before_gw20():
    held = held_chips(WINDOWS, [], 6)
    assert held_for(held, BENCH_BOOST, 6).id == "bench_boost@19"
    assert held_for(held, BENCH_BOOST, 20).id == "bench_boost@38"
    spent = held_chips(WINDOWS, [{"name": "bboost", "event": 3}], 6)
    assert held_for(spent, BENCH_BOOST, 6) is None


def test_history_we_do_not_plan_or_cannot_place_marks_nothing():
    history = [{"name": "manager", "event": 4}, {"name": "3xc"}, {}]
    assert len(held_chips(WINDOWS, history, 6)) == 8


def test_no_rules_falls_back_to_one_whole_season_window_per_chip():
    held = held_chips(chip_windows([]), [], 6)
    assert held == whole_season(BENCH_BOOST, TRIPLE_CAPTAIN, WILDCARD, FREE_HIT)
    assert all((chip.start_event, chip.stop_event) == (1, 38) for chip in held)


def test_unknown_api_names_are_ignored():
    rules = LIVE_RULES + [ChipRule(name="assistant_manager", start_event=1, stop_event=38)]
    assert len(chip_windows(rules)) == 8


def test_playable_in_keeps_chips_with_a_window_week():
    held = held_chips(WINDOWS, [], 15)
    assert ids(playable_in(held, [15, 16, 17, 18, 19, 20])) == ids(held)
    assert ids(playable_in(held, [6, 7, 8, 9, 10, 11])) == [
        "bench_boost@19", "triple_captain@19", "wildcard@19", "free_hit@19",
    ]


def test_the_bootstrap_reads_its_chips_and_defaults_to_none():
    payload = {"events": [], "teams": [], "elements": []}
    assert Bootstrap.model_validate(payload).chips == []
    payload["chips"] = [{"name": "bboost", "start_event": 1, "stop_event": 19, "id": 3, "number": 1, "chip_type": "team"}]
    assert Bootstrap.model_validate(payload).chips == [ChipRule(name="bboost", start_event=1, stop_event=19)]


def test_the_live_rules_leave_no_gap():
    assert rule_gaps(LIVE_RULES) == ((), ())


def test_no_rules_at_all_is_the_fallback_not_a_gap():
    # A payload recorded before the field was read: chip_windows falls back to
    # whole-season windows, which is a known model, not a chip gone missing.
    assert rule_gaps([]) == ((), ())


def test_a_renamed_chip_reads_as_ours_missing_and_theirs_unknown():
    # The case worth shouting about: FPL renames bboost and our map no longer
    # matches, so the bench boost silently leaves the plan. Both halves of the
    # evidence are named, since neither alone says "renamed".
    renamed = [
        ChipRule(name="benchboost" if rule.name == "bboost" else rule.name,
                 start_event=rule.start_event, stop_event=rule.stop_event)
        for rule in LIVE_RULES
    ]
    assert rule_gaps(renamed) == ((BENCH_BOOST,), ("benchboost",))


def test_a_chip_the_game_stopped_offering_is_missing_with_nothing_unknown():
    dropped = [rule for rule in LIVE_RULES if rule.name != "freehit"]
    assert rule_gaps(dropped) == ((FREE_HIT,), ())


def test_a_new_chip_beside_all_of_ours_is_no_gap():
    # A chip we do not plan, with our four all present, costs the plan nothing.
    extra = LIVE_RULES + [ChipRule(name="assistant_manager", start_event=1, stop_event=38)]
    assert rule_gaps(extra) == ((), ())


# --------------------------------------------------------------------------
# No free hit the week after a free hit
# --------------------------------------------------------------------------


def test_a_free_hit_at_19_holds_the_second_sets_free_hit_back_to_21():
    # The official rule: a first-set free hit in GW19 bars the second set's in
    # GW20. The first set's is spent, so only the second remains, its window
    # opening at 20 + 1 = 21 instead of 20. Its id is unchanged.
    held = held_chips(WINDOWS, [{"name": "freehit", "event": 19}], 20)

    free_hits = [chip for chip in held if chip.chip == FREE_HIT]
    assert free_hits == [HeldChip(FREE_HIT, 21, 38)]
    assert free_hits[0].id == "free_hit@38"
    assert held_for(held, FREE_HIT, 20) is None
    assert held_for(held, FREE_HIT, 21) == free_hits[0]


def test_a_free_hit_at_18_leaves_the_second_sets_free_hit_alone():
    # 18 + 1 = 19 is not where the second set opens, so nothing shifts.
    held = held_chips(WINDOWS, [{"name": "freehit", "event": 18}], 20)

    assert [c for c in held if c.chip == FREE_HIT] == [HeldChip(FREE_HIT, 20, 38)]


def test_other_chips_played_at_19_shift_nothing():
    history = [
        {"name": "bboost", "event": 19},
        {"name": "wildcard", "event": 19},
        {"name": "3xc", "event": 19},
    ]
    held = held_chips(WINDOWS, history, 20)

    assert HeldChip(FREE_HIT, 20, 38) in held
    assert held_for(held, FREE_HIT, 20) is not None
