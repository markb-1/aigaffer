"""Tests for candidate plan generation, with the MILP stubbed out.

The optimizer has its own tests; what matters here is what the caller does
with its answers, so ``optimize`` is replaced by a lookup from forced
transfer count to a canned plan. That also lets a test hand back answers a
real board would rarely produce — two counts landing on the same fifteen, or
no feasible plan at all — and pin down what happens then.
"""

from aigaffer.solver import plans as plans_module
from aigaffer.solver.optimizer import Plan
from aigaffer.solver.plans import generate_plans, recommend

SQUAD = list(range(1, 16))
PLAYERS = {"players": "stand-in"}
XP = {"xp": "stand-in"}


def canned(transfers: int, objective: float, squad: list[int] | None = None) -> Plan:
    """A plan making ``transfers`` moves, worth ``objective``.

    Sold ids come off the front of the squad and signings are numbered from
    100, so the fifteen a plan leaves behind is implied by its transfer count
    unless a test says otherwise.
    """
    out = SQUAD[:transfers]
    into = list(range(101, 101 + transfers))
    return Plan(
        squad=squad if squad is not None else SQUAD[transfers:] + into,
        xi=[],
        transfers_in=into,
        transfers_out=out,
        hits=max(0, transfers - 1),
        xp_total=objective,
        objective=objective,
    )


def stub_optimize(monkeypatch, answers: dict[int, Plan | None]) -> list[dict]:
    """Answer each forced transfer count from ``answers``; record the calls."""
    calls: list[dict] = []

    def fake_optimize(
        players, xp, current_squad, bank, free_transfers, forced_transfers=None
    ):
        calls.append(
            {
                "players": players,
                "xp": xp,
                "current_squad": current_squad,
                "bank": bank,
                "free_transfers": free_transfers,
                "forced_transfers": forced_transfers,
            }
        )
        return answers[forced_transfers]

    monkeypatch.setattr(plans_module, "optimize", fake_optimize)
    return calls


def test_every_transfer_count_is_asked_for(monkeypatch):
    calls = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})

    generate_plans(PLAYERS, XP, SQUAD, bank=25, free_transfers=2)

    assert [call["forced_transfers"] for call in calls] == [0, 1, 2, 3]
    assert all(call["players"] is PLAYERS for call in calls)
    assert all(call["xp"] is XP for call in calls)
    assert all(call["current_squad"] is SQUAD for call in calls)
    assert all(call["bank"] == 25 for call in calls)
    assert all(call["free_transfers"] == 2 for call in calls)


def test_plans_come_back_best_objective_first(monkeypatch):
    stub_optimize(
        monkeypatch,
        {
            0: canned(0, 100.0),
            1: canned(1, 103.0),
            2: canned(2, 101.5),
            3: canned(3, 99.0),
        },
    )

    result = generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1)

    assert [plan.objective for plan in result] == [103.0, 101.5, 100.0, 99.0]
    assert [len(plan.transfers_in) for plan in result] == [1, 2, 0, 3]


def test_transfer_counts_the_squad_cannot_afford_are_dropped(monkeypatch):
    stub_optimize(
        monkeypatch,
        {0: canned(0, 100.0), 1: canned(1, 102.0), 2: None, 3: None},
    )

    result = generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1)

    assert [plan.objective for plan in result] == [102.0, 100.0]


def test_no_feasible_plan_is_an_empty_list(monkeypatch):
    stub_optimize(monkeypatch, dict.fromkeys(range(4)))

    assert generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1) == []


def test_the_same_fifteen_is_only_reported_once(monkeypatch):
    # Counts 1 and 2 come back with one squad between them, listed in
    # different orders, and only the cheaper way of reaching it survives.
    squad = SQUAD[1:] + [101]
    stub_optimize(
        monkeypatch,
        {
            0: canned(0, 100.0),
            1: canned(1, 104.0, squad=squad),
            2: canned(2, 104.0, squad=list(reversed(squad))),
            3: canned(3, 99.0),
        },
    )

    result = generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1)

    assert [plan.objective for plan in result] == [104.0, 100.0, 99.0]
    assert [len(plan.transfers_in) for plan in result] == [1, 0, 3]


def test_recommend_takes_the_highest_objective():
    best = canned(2, 104.0)

    assert recommend([canned(0, 100.0), best, canned(1, 101.0)]) is best


def test_recommend_breaks_a_tie_on_fewer_transfers():
    # An extra move that buys nothing is not worth making.
    still = canned(0, 100.0)

    assert recommend([canned(2, 100.0), still, canned(1, 100.0)]) is still
