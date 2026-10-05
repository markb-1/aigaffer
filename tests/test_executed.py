"""Tests for the effective squad: what the API shows, plus what was entered.

``inputs`` stay the API's truth; :func:`apply_executed` lays the recorded
position over them for one gameweek. The universe is the pipeline's: the
fifteen of ``PICKS_15_JSON`` with £2.8m in the bank and one free transfer,
on the live chip rules (wildcard and free hit from GW2, a second set from
GW20).
"""

from aigaffer.chips import CHIP_ORDER, held_by_rules, held_for
from aigaffer.data.models import Bootstrap, Player, Squad, Standing
from aigaffer.executed import apply_executed
from aigaffer.orchestrator import PipelineInputs
from tests.fixtures import (
    HISTORY_JSON,
    PICKS_15_IDS,
    PICKS_15_JSON,
    PIPELINE_BOOTSTRAP_JSON,
    make_executed,
)

BOOTSTRAP = Bootstrap.model_validate(PIPELINE_BOOTSTRAP_JSON)
FH_SQUAD = [1, 9, 3, 4, 10, 12, 13, 5, 6, 11, 14, 17, 7, 15, 18]
FH_XI = [1, 3, 4, 10, 12, 5, 6, 11, 14, 17, 7]


def real_inputs() -> PipelineInputs:
    return PipelineInputs(
        bootstrap=BOOTSTRAP,
        fixtures=[],
        event=BOOTSTRAP.next_event(),
        squad=Squad(
            picks=PICKS_15_JSON["picks"],
            bank=28,
            event=1,
            standing=Standing.model_validate(PICKS_15_JSON["entry_history"]),
        ),
        free_transfers=1,
        histories={},
        players={p.id: p for p in BOOTSTRAP.elements},
        chips_used=list(HISTORY_JSON["chips"]),
    )


def test_an_ordinary_week_moves_the_squad_the_bank_and_the_free_transfers():
    inputs = real_inputs()
    row = make_executed()

    effective, executed = apply_executed(inputs, row)

    assert executed is row and effective.executed is row
    assert sorted(effective.squad.player_ids) == sorted(set(PICKS_15_IDS) - {6} | {17})
    assert effective.squad.bank == 8
    assert effective.squad.standing.bank == 8, "the phone's opening line agrees"
    assert effective.squad.standing.value == inputs.squad.standing.value
    assert effective.free_transfers == 0
    assert effective.chips_used == inputs.chips_used
    # The API's truth is untouched: the ledger keeps reading it.
    assert inputs.squad.bank == 28 and 6 in inputs.squad.player_ids
    assert inputs.executed is None


def test_a_wildcard_spends_the_chip_and_keeps_the_free_transfers():
    row = make_executed(chip="wildcard", ft_after=1)

    effective, _ = apply_executed(real_inputs(), row)

    assert effective.free_transfers == 1
    assert {"name": "wildcard", "event": 2} in effective.chips_used


def test_a_recorded_chip_is_the_only_chip_this_gameweek():
    # One chip a gameweek: with the wildcard recorded for GW2, no chip is
    # playable in GW2, while the bench boost still is from GW3.
    effective, _ = apply_executed(real_inputs(), make_executed(chip="wildcard", ft_after=1))

    held = held_by_rules(effective)

    assert all(held_for(held, chip, 2) is None for chip in CHIP_ORDER)
    assert held_for(held, "bench_boost", 3) is not None
    assert held_for(held, "wildcard", 3) is None, "the first-set wildcard is spent"


def test_a_week_with_no_chip_recorded_leaves_the_chips_alone():
    inputs = real_inputs()
    effective, _ = apply_executed(inputs, make_executed())

    assert held_by_rules(effective) == held_by_rules(inputs)


def test_a_free_hit_leaves_the_standing_squad_and_spends_the_chip():
    row = make_executed(
        chip="free_hit", transfers_in=[], transfers_out=[],
        squad_after=sorted(PICKS_15_IDS), bank_after=28, ft_after=1,
        freehit_squad=FH_SQUAD, freehit_xi=FH_XI, buy_prices={}, sell_prices={},
    )

    effective, _ = apply_executed(real_inputs(), row)

    assert sorted(effective.squad.player_ids) == sorted(PICKS_15_IDS)
    assert effective.squad.bank == 28 and effective.free_transfers == 1
    assert {"name": "freehit", "event": 2} in effective.chips_used


def test_a_row_for_another_gameweek_is_not_applied():
    inputs = real_inputs()

    assert apply_executed(inputs, make_executed(gw=1)) == (inputs, None)
    assert apply_executed(inputs, None) == (inputs, None)


def test_a_draft_has_nothing_to_apply_to():
    from dataclasses import replace

    inputs = replace(real_inputs(), squad=None, free_transfers=None)

    effective, executed = apply_executed(inputs, make_executed())

    assert effective is inputs and executed is None
