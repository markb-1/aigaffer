"""Tests for candidate plan generation, with both MILPs stubbed out.

The two solvers have their own tests; what matters here is what the caller
does with their answers, so each is replaced by a lookup from forced transfer
count to a canned plan. That also lets a test hand back answers a real board
would rarely produce — two counts landing on the same fifteen, or no feasible
plan at all — and pin down what happens then.

The fallback is the point of half of these. A window that answers nothing is
a run that still has to recommend a transfer, so the single-week solver is
asked the same questions instead, and the plans that come back say which
engine ran by whether they carry a path.
"""

from aigaffer.solver import plans as plans_module
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan
from aigaffer.solver.plans import SWEEP_TIME_LIMIT, generate_plans, recommend

SQUAD = list(range(1, 16))
PLAYERS = {"players": "stand-in"}
XP = {"xp": "stand-in"}
EVENTS = [10, 11, 12]


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


def canned_path(
    transfers: int, objective: float, squad: list[int] | None = None
) -> tuple[Plan, PlannedPath]:
    """A window's answer: the opening plan and the path hanging off it.

    The path holds one move a gameweek from now on, which is not arithmetic
    any of these tests check — it is here so that a plan which came from the
    window is telling a caller so.
    """
    plan = canned(transfers, objective, squad)
    path = PlannedPath(
        moves=[
            PlannedMove(event=EVENTS[1], transfers_in=[201], transfers_out=[2], hits=0)
        ],
        objective=objective,
        weekly_xp={event: objective / len(EVENTS) for event in EVENTS},
    )
    plan.path = path
    return plan, path


def stub_optimize_path(monkeypatch, answers: dict[int, Plan | None]) -> list[dict]:
    """Answer each forced opening count from ``answers``; record the calls.

    ``answers`` holds plans rather than the pairs the real solver returns,
    since a None is the interesting half and a pair with a None in it is not a
    thing :func:`~aigaffer.solver.multiweek.optimize_path` can hand back.
    """
    calls: list[dict] = []

    def fake_optimize_path(
        players,
        projections,
        current_squad,
        bank,
        free_transfers,
        events,
        decay,
        forced_first_transfers=None,
        time_limit=None,
        available_chips=frozenset(),
        freehit_prices=None,
    ):
        calls.append(
            {
                "players": players,
                "projections": projections,
                "current_squad": current_squad,
                "bank": bank,
                "free_transfers": free_transfers,
                "events": events,
                "decay": decay,
                "forced_first_transfers": forced_first_transfers,
                "time_limit": time_limit,
                "available_chips": available_chips,
                "freehit_prices": freehit_prices,
            }
        )
        plan = answers[forced_first_transfers]
        return None if plan is None else (plan, plan.path)

    monkeypatch.setattr(plans_module, "optimize_path", fake_optimize_path)
    return calls


def stub_free_hit_prices(monkeypatch, value) -> list[dict]:
    """Answer the hoisted free-hit pricing with ``value``; record the calls.

    The real pricing runs CBC sub-solves on a real board; these tests hand the
    window canned plans, so the price it is handed matters only in that it is
    computed once and reaches every solve. A ``value`` of None is the window
    that cannot field a free-hit squad at all.
    """
    calls: list[dict] = []

    def fake_free_hit_prices(players, xp, current_squad, bank, events, time_limit):
        calls.append(
            {
                "players": players,
                "xp": xp,
                "current_squad": current_squad,
                "bank": bank,
                "events": events,
                "time_limit": time_limit,
            }
        )
        return value

    monkeypatch.setattr(plans_module, "_free_hit_prices", fake_free_hit_prices)
    return calls


def windows(counts: range, objective=lambda n: 100.0 + n) -> dict[int, Plan]:
    """A window answer for every count in ``counts``."""
    return {n: canned_path(n, objective(n))[0] for n in counts}


def test_every_transfer_count_is_asked_for(monkeypatch):
    calls = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})

    generate_plans(PLAYERS, XP, SQUAD, bank=25, free_transfers=2)

    assert [call["forced_transfers"] for call in calls] == [0, 1, 2, 3]
    assert all(call["players"] is PLAYERS for call in calls)
    assert all(call["xp"] is XP for call in calls)
    assert all(call["current_squad"] is SQUAD for call in calls)
    assert all(call["bank"] == 25 for call in calls)
    assert all(call["free_transfers"] == 2 for call in calls)


def test_a_bank_of_free_transfers_is_asked_for_in_full(monkeypatch):
    # Five banked free transfers are five moves that cost nothing, and a
    # shortlist that stopped at three never asked for the best of them. The
    # bank tops out at five, so five is as far as this goes.
    calls = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(6)})

    result = generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=5)

    assert [call["forced_transfers"] for call in calls] == [0, 1, 2, 3, 4, 5]
    assert [len(plan.transfers_in) for plan in result] == [5, 4, 3, 2, 1, 0]


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


def test_the_window_is_solved_at_every_opening_count(monkeypatch):
    single = stub_optimize(monkeypatch, {})
    calls = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=25, free_transfers=2,
        projections_events=EVENTS, decay=0.9,
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3]
    assert all(call["players"] is PLAYERS for call in calls)
    assert all(call["projections"] is XP for call in calls)
    assert all(call["current_squad"] is SQUAD for call in calls)
    assert all(call["bank"] == 25 for call in calls)
    assert all(call["free_transfers"] == 2 for call in calls)
    assert all(call["events"] is EVENTS for call in calls)
    assert all(call["decay"] == 0.9 for call in calls)
    # Half a dozen solves in a run that has one deadline to make: a minute
    # apiece is the single solve's budget, not the sweep's.
    assert all(call["time_limit"] == SWEEP_TIME_LIMIT for call in calls)
    assert single == []


def test_the_available_chips_ride_through_to_every_windowed_solve(monkeypatch):
    # The chips the window may schedule are handed to it on every opening count,
    # unchanged: one place derives them and the sweep only carries them.
    chips = frozenset({"bench_boost", "free_hit"})
    calls = stub_optimize_path(monkeypatch, windows(range(4)))
    stub_free_hit_prices(monkeypatch, {1: (0.0, [], [])})

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, available_chips=chips,
    )

    assert [call["available_chips"] for call in calls] == [chips] * 4


def test_the_free_hit_price_is_computed_once_and_rides_the_whole_sweep(monkeypatch):
    # The perf hoist. The free hit is the one chip priced by a solve of its own,
    # and that price does not move with the opening count — so it is computed
    # once and the same object reaches every windowed solve, sparing the sweep
    # five extra passes of the CBC sub-solves. One pricing call; one dict, shared.
    prices = {1: (99.0, list(range(16, 31)), list(range(16, 27)))}
    calls = stub_optimize_path(monkeypatch, windows(range(4)))
    priced = stub_free_hit_prices(monkeypatch, prices)

    generate_plans(
        PLAYERS, XP, SQUAD, bank=25, free_transfers=1,
        projections_events=EVENTS, available_chips=frozenset({"free_hit"}),
    )

    # Priced exactly once, off the real inputs and the sweep's own time budget.
    assert len(priced) == 1
    assert priced[0]["players"] is PLAYERS
    assert priced[0]["bank"] == 25
    assert priced[0]["events"] is EVENTS
    assert priced[0]["time_limit"] == SWEEP_TIME_LIMIT
    # And that one dict is the very object every opening count is handed.
    assert [call["freehit_prices"] for call in calls] == [prices] * 4
    assert all(call["freehit_prices"] is prices for call in calls)


def test_a_free_hit_that_cannot_be_priced_falls_back_to_the_single_week_solver(monkeypatch):
    # A None price is a window that cannot field a free-hit squad — every count
    # would return None, an empty sweep. Rather than run four doomed solves, the
    # run skips straight to the single-week engine, which is where an empty sweep
    # lands anyway. The window is never asked; the single-week solver answers.
    single = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})
    window = stub_optimize_path(monkeypatch, windows(range(4)))
    stub_free_hit_prices(monkeypatch, None)

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, available_chips=frozenset({"free_hit"}),
    )

    assert window == []
    assert [call["forced_transfers"] for call in single] == [0, 1, 2, 3]
    assert [plan.objective for plan in result] == [103.0, 102.0, 101.0, 100.0]
    assert all(plan.path is None for plan in result)


def test_no_available_chips_is_the_default_and_reaches_the_solve(monkeypatch):
    # The gate closed: the sweep asks for the pre-chip model, empty set and all.
    calls = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert all(call["available_chips"] == frozenset() for call in calls)


def test_the_openings_asked_for_stop_where_the_free_transfer_bank_does(monkeypatch):
    calls = stub_optimize_path(monkeypatch, windows(range(6)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=5, projections_events=EVENTS
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3, 4, 5]

    # Fifteen is how aigaffer.solver.lineup prices a wildcard and not a bank
    # anybody holds. The window reads it as five, so a sixth opening move is a
    # question about a board the solver does not believe in.
    calls.clear()
    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=15, projections_events=EVENTS
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3, 4, 5]


def test_window_plans_come_back_best_first_and_carry_their_paths(monkeypatch):
    single = stub_optimize(monkeypatch, {})
    stub_optimize_path(
        monkeypatch,
        windows(range(4), objective={0: 100.0, 1: 103.0, 2: 101.5, 3: 99.0}.get),
    )

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert [plan.objective for plan in result] == [103.0, 101.5, 100.0, 99.0]
    assert [len(plan.transfers_in) for plan in result] == [1, 2, 0, 3]
    assert all(plan.path is not None for plan in result)
    assert result[0].path.moves[0].event == EVENTS[1]
    assert single == []


def test_an_opening_the_window_cannot_make_is_dropped(monkeypatch):
    stub_optimize(monkeypatch, {})
    stub_optimize_path(
        monkeypatch,
        {
            0: canned_path(0, 100.0)[0],
            1: canned_path(1, 102.0)[0],
            2: None,
            3: None,
        },
    )

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert [plan.objective for plan in result] == [102.0, 100.0]
    assert all(plan.path is not None for plan in result)


def test_the_same_opening_fifteen_is_only_reported_once(monkeypatch):
    # The window's answers are deduped the way the single-week solver's are:
    # by the fifteen the first gameweek leaves behind, first sighting winning,
    # which in ascending order is the one that got there in fewer moves.
    squad = SQUAD[1:] + [101]
    stub_optimize(monkeypatch, {})
    stub_optimize_path(
        monkeypatch,
        {
            0: canned_path(0, 100.0)[0],
            1: canned_path(1, 104.0, squad=squad)[0],
            2: canned_path(2, 104.0, squad=list(reversed(squad)))[0],
            3: canned_path(3, 99.0)[0],
        },
    )

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert [plan.objective for plan in result] == [104.0, 100.0, 99.0]
    assert [len(plan.transfers_in) for plan in result] == [1, 0, 3]


def test_a_window_that_answers_nothing_falls_back_on_the_single_week_solver(monkeypatch):
    # An infeasible window, or six solves that all ran out of time, is still a
    # gameweek with a deadline: the run asks the other engine the same
    # questions rather than recommending nothing.
    single = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})
    window = stub_optimize_path(monkeypatch, dict.fromkeys(range(4)))

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert [call["forced_first_transfers"] for call in window] == [0, 1, 2, 3]
    assert [call["forced_transfers"] for call in single] == [0, 1, 2, 3]
    assert [plan.objective for plan in result] == [103.0, 102.0, 101.0, 100.0]
    # No path is how a caller knows which engine ended up answering.
    assert all(plan.path is None for plan in result)


def test_one_surviving_window_plan_is_enough_to_keep_the_other_engine_out(monkeypatch):
    single = stub_optimize(monkeypatch, {})
    stub_optimize_path(
        monkeypatch, {0: None, 1: None, 2: canned_path(2, 99.0)[0], 3: None}
    )

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert [plan.objective for plan in result] == [99.0]
    assert single == []


def test_the_single_week_planner_never_looks_at_the_window(monkeypatch):
    single = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})
    window = stub_optimize_path(monkeypatch, windows(range(4)))

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, planner="single",
    )

    assert window == []
    assert [call["forced_transfers"] for call in single] == [0, 1, 2, 3]
    assert all(plan.path is None for plan in result)


def test_without_a_window_there_is_only_the_single_week_solver(monkeypatch):
    single = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})
    window = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1)
    generate_plans(PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=[])

    assert window == []
    assert len(single) == 8


def test_recommend_takes_the_highest_objective():
    best = canned(2, 104.0)

    assert recommend([canned(0, 100.0), best, canned(1, 101.0)]) is best


def test_recommend_breaks_a_tie_on_fewer_transfers():
    # An extra move that buys nothing is not worth making.
    still = canned(0, 100.0)

    assert recommend([canned(2, 100.0), still, canned(1, 100.0)]) is still
