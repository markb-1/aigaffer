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

from aigaffer.chips import BENCH_BOOST, FREE_HIT, WILDCARD, HeldChip, whole_season
from aigaffer.solver import plans as plans_module
from aigaffer.solver.multiweek import PlannedMove, PlannedPath
from aigaffer.solver.optimizer import Plan, Week1Lock
from aigaffer.solver.plans import SWEEP_TIME_LIMIT, generate_plans, recommend
from tests.test_multiweek import DECAY, SQUAD as SPINE_SQUAD, six_arrivals, spine

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
        players, xp, current_squad, bank, free_transfers, forced_transfers=None,
        selling_prices=None, lock=None,
    ):
        calls.append(
            {
                "players": players,
                "xp": xp,
                "current_squad": current_squad,
                "bank": bank,
                "free_transfers": free_transfers,
                "forced_transfers": forced_transfers,
                "selling_prices": selling_prices,
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
        held_chips=(),
        freehit_prices=None,
        selling_prices=None,
        bars=None,
        lock=None,
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
                "held_chips": held_chips,
                "freehit_prices": freehit_prices,
                "selling_prices": selling_prices,
                "bars": bars,
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

    def fake_free_hit_prices(
        players, xp, current_squad, bank, events, time_limit, selling_prices=None,
        held_chips=None,
    ):
        calls.append(
            {
                "players": players,
                "xp": xp,
                "current_squad": current_squad,
                "bank": bank,
                "events": events,
                "time_limit": time_limit,
                "selling_prices": selling_prices,
                "held_chips": held_chips,
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


def test_the_held_chips_ride_through_to_every_windowed_solve(monkeypatch):
    # The chips the window may schedule are handed to it on every opening count,
    # unchanged: one place derives them and the sweep only carries them.
    chips = whole_season(BENCH_BOOST, FREE_HIT)
    calls = stub_optimize_path(monkeypatch, windows(range(4)))
    stub_free_hit_prices(monkeypatch, {1: (0.0, [], [])})

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=chips,
    )

    assert [call["held_chips"] for call in calls] == [chips] * 4


def test_the_selling_prices_ride_through_to_every_solve(monkeypatch):
    # The ledger's prices are derived in one place and only carried here: the
    # same dict reaches every windowed solve, the hoisted free-hit pricing,
    # and — asked separately below — the single-week fallback, so no engine is
    # ever spending money another engine was refused.
    sales = {1: 45, 2: 51}
    windowed = stub_optimize_path(monkeypatch, windows(range(4)))
    priced = stub_free_hit_prices(monkeypatch, {1: (0.0, [], [])})

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=whole_season(FREE_HIT),
        selling_prices=sales,
    )

    assert [call["selling_prices"] for call in priced] == [sales]
    assert [call["selling_prices"] for call in windowed] == [sales] * 4

    single = stub_optimize(monkeypatch, {n: canned(n, 100.0 + n) for n in range(4)})
    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, planner="single",
        selling_prices=sales,
    )

    assert [call["selling_prices"] for call in single] == [sales] * 4


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
        projections_events=EVENTS, held_chips=whole_season(FREE_HIT),
    )

    # Priced exactly once, off the real inputs and the sweep's own time budget.
    assert len(priced) == 1
    assert priced[0]["players"] is PLAYERS
    assert priced[0]["bank"] == 25
    assert priced[0]["events"] is EVENTS
    assert priced[0]["time_limit"] == SWEEP_TIME_LIMIT
    # Handed the held chips, so it prices only the weeks a held free hit may
    # be played in — the weeks each solve would have priced for itself.
    assert priced[0]["held_chips"] == whole_season(FREE_HIT)
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
        projections_events=EVENTS, held_chips=whole_season(FREE_HIT),
    )

    assert window == []
    assert [call["forced_transfers"] for call in single] == [0, 1, 2, 3]
    assert [plan.objective for plan in result] == [103.0, 102.0, 101.0, 100.0]
    assert all(plan.path is None for plan in result)


def test_no_held_chips_is_the_default_and_reaches_the_solve(monkeypatch):
    # The gate closed: the sweep asks for the pre-chip model, empty and all,
    # and with no calendar bars behind it.
    calls = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS
    )

    assert all(call["held_chips"] == () for call in calls)
    assert all(call["bars"] is None for call in calls)


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


def test_the_bars_ride_through_to_every_windowed_solve(monkeypatch):
    # The calendar's per-week bars are priced in one place and only carried
    # here: the very same dict reaches every opening count, beside the held
    # chips it is keyed by.
    chips = whole_season(BENCH_BOOST)
    bars = {"bench_boost@38": {10: 4.0, 11: 9.0, 12: 0.0}}
    calls = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=chips, bars=bars,
    )

    assert len(calls) == 4
    assert all(call["bars"] is bars for call in calls)
    assert all(call["held_chips"] == chips for call in calls)


def test_a_free_hit_held_only_outside_the_window_is_never_priced(monkeypatch):
    # A second-half free hit seen from GW10-12: it is in hand but no week of
    # this window may play it, so the window builds no free-hit binary and the
    # sweep has no price to hoist. The sub-solves are not run, and the solves
    # are handed no prices.
    chips = (HeldChip(FREE_HIT, 20, 38),)
    calls = stub_optimize_path(monkeypatch, windows(range(4)))
    priced = stub_free_hit_prices(monkeypatch, {1: (0.0, [], [])})

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=chips,
    )

    assert priced == []
    assert len(calls) == 4
    assert all(call["freehit_prices"] is None for call in calls)


def test_a_wildcard_eligible_this_week_adds_one_uncapped_solve(monkeypatch):
    # The sweep pins the opening count, and a pinned count is capped at three
    # moves (the wildcard's lift of the cap lives only in the unpinned solve).
    # So with a wildcard playable in the first event the sweep makes the four
    # pinned solves 0..3 and then exactly one more, unpinned, with every other
    # argument the sweep's own: 4 + 1 = 5 calls, the last forced=None.
    chips = whole_season(WILDCARD)
    answers = windows(range(4))
    answers[None] = canned_path(7, 120.0)[0]
    calls = stub_optimize_path(monkeypatch, answers)

    result = generate_plans(
        PLAYERS, XP, SQUAD, bank=25, free_transfers=1,
        projections_events=EVENTS, decay=0.9, held_chips=chips,
        selling_prices={1: 45}, bars={"wildcard@38": {10: 3.0}},
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3, None]
    extra = calls[-1]
    assert extra["time_limit"] == SWEEP_TIME_LIMIT
    assert extra["held_chips"] == chips
    assert extra["selling_prices"] == {1: 45}
    assert extra["bars"] == {"wildcard@38": {10: 3.0}}
    assert extra["bank"] == 25 and extra["free_transfers"] == 1
    assert extra["decay"] == 0.9 and extra["events"] is EVENTS
    # The 120.0 plan beats every capped one (100..103), so it tops the list.
    assert result[0].objective == 120.0


def test_a_wildcard_that_starts_after_this_week_leaves_the_sweep_alone(monkeypatch):
    # GW12-19 wildcard seen from a GW10-12 window: playable in the window but
    # not in its first event, so no solve this week could play it and the
    # sweep is exactly the four pinned solves, none unpinned.
    calls = stub_optimize_path(monkeypatch, windows(range(4)))

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=(HeldChip(WILDCARD, 12, 19),),
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3]


def test_no_wildcard_held_leaves_the_sweep_alone(monkeypatch):
    # Other chips, even ones eligible this week, do not buy the extra solve.
    calls = stub_optimize_path(monkeypatch, windows(range(4)))
    stub_free_hit_prices(monkeypatch, {1: (0.0, [], [])})

    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1,
        projections_events=EVENTS, held_chips=whole_season(BENCH_BOOST, FREE_HIT),
    )
    generate_plans(
        PLAYERS, XP, SQUAD, bank=0, free_transfers=1, projections_events=EVENTS,
    )

    assert [call["forced_first_transfers"] for call in calls] == [0, 1, 2, 3] * 2


def test_a_recorded_free_hit_week_is_a_hold_week():
    # A free hit was entered: the standing squad makes no transfers this
    # week, whatever the board would otherwise buy. Unlocked, the spine with
    # 8 sold for 16 buys 8 straight back.
    players, projections = spine([5, 6, 7])
    current = [pid for pid in SPINE_SQUAD if pid != 8] + [16]

    free = generate_plans(
        players, projections, current, 0, 1,
        projections_events=[5, 6, 7], decay=DECAY,
    )
    held = generate_plans(
        players, projections, current, 0, 1,
        projections_events=[5, 6, 7], decay=DECAY, lock=Week1Lock(hold=True),
    )

    assert recommend(free).transfers_in == [8]
    assert len(held) == 1 and held[0].transfers_in == [] and held[0].transfers_out == []


def test_the_single_week_fallback_holds_too():
    players, projections = spine([5, 6, 7])
    current = [pid for pid in SPINE_SQUAD if pid != 8] + [16]

    held = generate_plans(
        players, projections, current, 0, 1,
        projections_events=[5, 6, 7], decay=DECAY, planner="single",
        lock=Week1Lock(hold=True),
    )

    assert len(held) == 1 and held[0].transfers_in == []


def test_a_recorded_wildcard_week_is_free_in_both_engines():
    players, projections = six_arrivals([5, 6], opening=20.0)

    for planner in ("multi", "single"):
        plans = generate_plans(
            players, projections, SPINE_SQUAD, 0, 1,
            projections_events=[5, 6], decay=DECAY, planner=planner,
            lock=Week1Lock(free=True),
        )
        best = recommend(plans)
        assert best.hits == 0, planner
        assert len(best.transfers_in) == 3, planner  # the sweep stops at three
