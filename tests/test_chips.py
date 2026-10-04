"""Which chips are in hand, and when each may be played."""

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
