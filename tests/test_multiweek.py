"""Tests for the multi-week transfer MILP, on hand-built universes.

Every optimum below is worked out with a pencil in the comment above the test,
so a failure says which arithmetic the model disagrees with rather than only
that it disagrees. Four boards are used.

**The spine** (:func:`spine`) is a fifteen where nothing is worth buying: two
keepers, five defenders, five midfielders, three forwards, all at 50, each on
a club of his own so the three-per-club rule never binds, and projections
identical in every gameweek.

==  ===  ====  =====  =============================================
id  pos  cost     xP  role
==  ===  ====  =====  =============================================
1   GK     50    5.0  starts
2   GK     50    0.5  bench
3   DEF    50    4.4  starts
4   DEF    50    4.3  starts
5   DEF    50    4.2  starts
6   DEF    50    0.5  bench
7   DEF    50    0.5  bench
8   MID    50    5.6  starts — the first midfielder a plan gives up
9   MID    50    5.7  starts
10  MID    50    5.8  starts
11  MID    50    5.9  starts
12  MID    50    6.0  starts — captain
13  FWD    50    4.1  starts
14  FWD    50    4.0  starts
15  FWD    50    3.9  bench — the man who steps up when a midfielder goes
==  ===  ====  =====  =============================================

Its XI is ``1 | 3 4 5 | 8 9 10 11 12 | 13 14`` — every midfielder starts,
because the worst of them (5.6) beats the best man on the bench (3.9). One
gameweek is worth 55.0 started, 6.0 for the captain and 0.1 x 5.4 benched:
**61.54**, and 61.0 of XI-and-captain xP.

**The bloomers** (:func:`blooming`) add midfielders who are worth nothing in
the first gameweek of the window and 20.0 in every one after it — the reason
a plan waits, and, when there are five of them, the reason it cannot wait.

**The six** (:func:`six_arrivals`) is the same idea with a forward on the end
of it, because six is one more than a free-transfer bank can ever hold.

**The double** (:func:`double_gameweek`) is its own board and draws its own
table, below :func:`double_gameweek`.

:func:`assert_legal_path` replays a plan and its path gameweek by gameweek and
checks the squad, the bank, the free transfers and the hits against the game's
rules rather than against the solver's own arithmetic, and hands back the
free-transfer bank standing at each deadline.
"""

from collections import Counter

import pytest

from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection, decayed_total
from aigaffer.solver.multiweek import (
    BENCH_BOOST,
    CANDIDATES_PER_POSITION,
    CHURN_EPSILON,
    FREE_HIT,
    MAX_HITS,
    SOLVER,
    TRIPLE_CAPTAIN,
    WILDCARD,
    PlannedMove,
    PlannedPath,
    _best_one_week_squad,
    _free_hit_prices,
    _solver,
    optimize_path,
)
from aigaffer.solver.optimizer import (
    MAX_PER_CLUB,
    MAX_TRANSFERS,
    SQUAD_QUOTAS,
    SQUAD_SIZE,
    _grouped,
    candidate_pool,
    optimize,
)
from aigaffer.solver.plans import (
    SWEEP_TIME_LIMIT,
    _shortlist,
    generate_plans,
    transfer_counts,
)

GK, DEF, MID, FWD = 1, 2, 3, 4
DECAY = 0.85
SQUAD = list(range(1, 16))


def _player(pid: int, position: int, cost: int) -> Player:
    """Every player on his own club: ``team=pid`` puts the three-per-club rule
    permanently out of the way, so only price and projection decide."""
    return Player(
        id=pid,
        web_name=f"P{pid}",
        team=pid,
        element_type=position,
        now_cost=cost,
        status="a",
        minutes=900,
        starts=10,
        total_points=0,
        bonus=0,
        saves=0,
    )


def _build(
    rows: list[tuple[int, int, int, dict[int, float]]],
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """``(id, position, cost, per_gw)`` rows into the two dicts the solvers take.

    ``total`` is the decayed sum the single-week solver reads, so a greedy
    contrast on the same board sees the same horizon this one does.
    """
    players = {pid: _player(pid, position, cost) for pid, position, cost, _ in rows}
    projections = {
        pid: PlayerProjection(
            player_id=pid, per_gw=dict(per_gw), total=decayed_total(per_gw, DECAY)
        )
        for pid, _, _, per_gw in rows
    }
    return players, projections


SPINE = [
    (1, GK, 5.0),
    (2, GK, 0.5),
    (3, DEF, 4.4),
    (4, DEF, 4.3),
    (5, DEF, 4.2),
    (6, DEF, 0.5),
    (7, DEF, 0.5),
    (8, MID, 5.6),
    (9, MID, 5.7),
    (10, MID, 5.8),
    (11, MID, 5.9),
    (12, MID, 6.0),
    (13, FWD, 4.1),
    (14, FWD, 4.0),
    (15, FWD, 3.9),
]

SPINE_WEEK_XI = 61.0


def spine(
    events: list[int], spare: float = 0.4
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The fifteen, plus one midfielder at ``spare``.

    Nobody worth having, at the default 0.4. At 5.6 he is instead the exact
    twin of 8, the cheapest starter here — which makes a transfer between them
    worth nothing at all rather than worth less than nothing, and that is the
    board a tiebreak has to be tested on.
    """
    rows = [
        (pid, position, 50, {event: points for event in events})
        for pid, position, points in SPINE
    ]
    rows.append((16, MID, 50, {event: spare for event in events}))
    return _build(rows)


def blooming(
    events: list[int], count: int, bloom: float = 20.0, early: float = 0.0
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The fifteen plus ``count`` midfielders who arrive a gameweek late.

    Each is worth ``bloom`` in every gameweek after ``events[0]`` and nothing
    in ``events[0]`` itself — except the first of them, 16, who is worth
    ``early`` there: the one it costs least to sign a gameweek ahead of time.
    Everyone costs 50, so no transfer on this board is a question of money.
    """
    rows = [
        (pid, position, 50, {event: points for event in events})
        for pid, position, points in SPINE
    ]
    for index in range(count):
        arrival = {events[0]: early if index == 0 else 0.0}
        arrival.update({event: bloom for event in events[1:]})
        rows.append((16 + index, MID, 50, arrival))
    return _build(rows)


def bloomers(count: int) -> list[int]:
    return list(range(16, 16 + count))


SIX = ((16, MID), (17, MID), (18, MID), (19, MID), (20, MID), (21, FWD))


def six_arrivals(
    events: list[int], opening: float = 0.0
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The fifteen plus six arrivals: five midfielders and a forward, each on
    20.0 from the second gameweek of the window on.

    Six is one more than the free-transfer bank can ever hold, which is the
    point of the board. ``opening`` is what they are worth in the first
    gameweek — nothing by default, so there is no reason to sign one early.
    """
    rows = [
        (pid, position, 50, {event: points for event in events})
        for pid, position, points in SPINE
    ]
    for pid, position in SIX:
        arrival = {events[0]: opening}
        arrival.update({event: 20.0 for event in events[1:]})
        rows.append((pid, position, 50, arrival))
    return _build(rows)


SEVEN = SIX + ((22, FWD),)


def seven_arrivals(
    events: list[int],
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The fifteen plus seven men worth 20.0 in every gameweek of the window.

    Seven is what a full bank of five free transfers plus the two-hit ceiling
    would buy in a single gameweek, and at 20.0 a man against four points a hit
    every one of them is worth having at once. It is the board on which the
    opening gameweek's transfer cap is the only thing standing in the way.
    """
    rows = [
        (pid, position, 50, {event: points for event in events})
        for pid, position, points in SPINE
    ]
    for pid, position in SEVEN:
        rows.append((pid, position, 50, {event: 20.0 for event in events}))
    return _build(rows)


# ==  ===  ====  ====  =====  =====  ===================================
# id  pos  club  cost   GW10   GW11  role
# ==  ===  ====  ====  =====  =====  ===================================
# 1   GK     1     50    4.0    4.0  starts
# 2   GK     2     50    1.0    1.0  bench
# 3-5 DEF  3-5     50    4.0    4.0  start
# 6,7 DEF  6,7     50    1.0    1.0  bench
# 8   MID    8    130    6.0    6.0  the money: 130 tied up in a 6.0
# 9   MID    9     50    4.7    4.7  starts
# 10  MID   10     50    4.8    4.8  starts
# 11  MID   11     50    4.9    4.9  starts
# 12  MID   12     50    5.0    5.0  starts
# 13,14 FWD      50    4.0    4.0  start
# 15  FWD   15     50    1.0    1.0  bench
# 16  MID   16     30    5.5    5.5  the filler whose 30 frees 100
# 17  MID   17     75    3.0   22.0  the double, and the best of them now
# 18-20 MID       75    2.0   22.0  the double
# ==  ===  ====  ====  =====  =====  ===================================
DOUBLE_ROWS = [
    (1, GK, 50, 4.0, 4.0),
    (2, GK, 50, 1.0, 1.0),
    (3, DEF, 50, 4.0, 4.0),
    (4, DEF, 50, 4.0, 4.0),
    (5, DEF, 50, 4.0, 4.0),
    (6, DEF, 50, 1.0, 1.0),
    (7, DEF, 50, 1.0, 1.0),
    (8, MID, 130, 6.0, 6.0),
    (9, MID, 50, 4.7, 4.7),
    (10, MID, 50, 4.8, 4.8),
    (11, MID, 50, 4.9, 4.9),
    (12, MID, 50, 5.0, 5.0),
    (13, FWD, 50, 4.0, 4.0),
    (14, FWD, 50, 4.0, 4.0),
    (15, FWD, 50, 1.0, 1.0),
    (16, MID, 30, 5.5, 5.5),
    (17, MID, 75, 3.0, 22.0),
    (18, MID, 75, 2.0, 22.0),
    (19, MID, 75, 2.0, 22.0),
    (20, MID, 75, 2.0, 22.0),
]
DOUBLE_EVENTS = [10, 11]


def double_gameweek() -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """Twenty players; the four doubles cost more than the squad can raise.

    The fifteen costs 830 with nothing in the bank. Four doubles at 75 plus a
    fifth midfielder is 300 + x, and the ten men outside midfield cost 500, so
    the only fifth midfielder the budget allows is 16 at 30 — which means
    selling 8 (130) and buying 16 is not one transfer among many, it is the
    transfer that pays for the other four.
    """
    rows = [
        (pid, position, cost, {DOUBLE_EVENTS[0]: first, DOUBLE_EVENTS[1]: second})
        for pid, position, cost, first, second in DOUBLE_ROWS
    ]
    return _build(rows)


def assert_legal(players: dict[int, Player], squad: list[int], cash: int) -> None:
    """One gameweek's fifteen, as the FPL website would check it."""
    assert len(squad) == SQUAD_SIZE
    assert Counter(players[p].element_type for p in squad) == SQUAD_QUOTAS
    assert max(Counter(players[p].team for p in squad).values()) <= MAX_PER_CLUB
    assert cash >= 0


def assert_legal_path(
    players: dict[int, Player],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    events: list[int],
    plan,
    path: PlannedPath,
) -> list[int]:
    """Replay the path a gameweek at a time and check everything it implies.

    The week-one move lives on the plan and the rest on the path, so this walks
    both: fifteen legal men, the quotas, the club limit, a bank that never goes
    below zero, and — worked out here from the game's rules rather than read
    back off the solver — the free transfers and the hits. Every gameweek's
    stated hits must be exactly the moves it makes beyond the free ones it
    holds, and the bank it leaves behind must be
    ``min(5, ft - moves + hits + 1)``.

    A wildcard gameweek is the exception the game writes into both of those: its
    transfers are all free however many it makes, so it charges no hits at all,
    and it spends none of its free-transfer bank, so the bank carries as though
    the gameweek had moved no one — ``min(5, ft + 1)``. The chip is read from
    ``path.week1_chip`` for the opening gameweek and from each move's ``chip``
    thereafter.

    Returns the free-transfer bank standing at each gameweek's deadline, which
    is how a test can pin the way the bank fills and where it stops.
    """
    moves = {move.event: move for move in path.moves}
    squad = set(current_squad)
    cash = bank
    banked = free_transfers
    series: list[int] = []

    for index, event in enumerate(events):
        if index == 0:
            incoming, outgoing, hits = (
                plan.transfers_in, plan.transfers_out, plan.hits
            )
            chip = path.week1_chip
        else:
            move = moves.get(event)
            incoming = move.transfers_in if move else []
            outgoing = move.transfers_out if move else []
            hits = move.hits if move else 0
            chip = move.chip if move else "none"

        assert set(outgoing) <= squad
        assert not set(incoming) & squad
        assert len(incoming) == len(outgoing)
        squad = (squad - set(outgoing)) | set(incoming)
        cash += sum(players[p].now_cost for p in outgoing)
        cash -= sum(players[p].now_cost for p in incoming)
        assert_legal(players, sorted(squad), cash)

        series.append(banked)
        if chip == WILDCARD:
            assert hits == 0, f"GW{event}: a wildcard charges no hits"
            banked = min(MAX_FREE_TRANSFERS, banked + 1)
        else:
            owed = max(0, len(incoming) - banked)
            assert hits == owed, f"GW{event}: {len(incoming)} moves on {banked} free"
            assert owed <= MAX_HITS
            banked = min(MAX_FREE_TRANSFERS, banked - len(incoming) + owed + 1)

    assert set(plan.squad) == set(current_squad) - set(plan.transfers_out) | set(
        plan.transfers_in
    )
    return series


# --------------------------------------------------------------------------
# A world where nothing is worth doing
# --------------------------------------------------------------------------


def test_a_static_world_rolls_the_transfer_and_plans_nothing():
    # 16 is the only signing on offer and the worst midfielder alive, so the
    # optimum is the squad as it stands in all three gameweeks: 61.54 a week,
    # discounted 1 + 0.85 + 0.7225 = 2.5725, for 158.31165.
    players, projections = spine([5, 6, 7])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY,
    )

    assert plan.transfers_in == []
    assert plan.transfers_out == []
    assert plan.hits == 0
    assert plan.squad == SQUAD
    assert plan.xi == [1, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    assert path.moves == []
    assert plan.objective == pytest.approx(158.31165, abs=1e-4)
    assert plan.xp_total == pytest.approx(158.31165, abs=1e-4)
    assert path.objective == pytest.approx(158.31165, abs=1e-4)
    assert path.weekly_xp == {
        5: pytest.approx(SPINE_WEEK_XI),
        6: pytest.approx(SPINE_WEEK_XI),
        7: pytest.approx(SPINE_WEEK_XI),
    }
    assert_legal_path(players, SQUAD, 0, 1, [5, 6, 7], plan, path)


def test_the_objective_counts_the_captain_twice():
    # Two gameweeks, nothing to do in either. The XI is 5.0 + 4.4 + 4.3 + 4.2 +
    # 5.6 + 5.7 + 5.8 + 5.9 + 6.0 + 4.1 + 4.0 = 55.0; the captain (12, on 6.0)
    # scores his 6.0 again; the bench (0.5 x 3 + 3.9 = 5.4) pays a tenth, 0.54.
    # 61.54 a gameweek, discounted 1 + 0.85 = 1.85, is 113.849. Without the
    # captain it would be 55.54 x 1.85 = 102.749, which is the number this test
    # exists to rule out.
    players, projections = spine([5, 6])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY,
    )

    assert plan.objective == pytest.approx(113.849, abs=1e-4)
    assert path.weekly_xp[5] == pytest.approx(61.0, abs=1e-4)
    assert path.weekly_xp[6] == pytest.approx(61.0, abs=1e-4)


def test_the_plan_carries_the_path_it_was_solved_with():
    players, projections = spine([5, 6])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY,
    )

    assert plan.path is path


def test_the_same_board_solves_the_same_way_twice():
    players, projections = double_gameweek()

    first = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY,
    )
    second = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY,
    )

    assert first[0].squad == second[0].squad
    assert first[0].xi == second[0].xi
    assert first[0].transfers_in == second[0].transfers_in
    assert first[0].transfers_out == second[0].transfers_out
    assert first[1].moves == second[1].moves
    assert first[0].objective == second[0].objective


# --------------------------------------------------------------------------
# Banking free transfers, and paying for the ones the bank cannot cover
# --------------------------------------------------------------------------


def test_free_transfers_are_banked_for_a_double_move_next_week():
    # Two midfielders worth nothing in GW5 and 20.0 after it. Signing one now
    # costs 5.6 out of the XI for 3.9 off the bench and 3.5 of bench weight —
    # 2.09 — and buys nothing, so the plan rolls its free transfer and spends
    # two of them in GW6 for no hit at all. Once they are in, a gameweek is
    # worth 83.7 started, 20.0 for the captain and 0.54 benched: 61.54 + 0.85 x
    # 104.24 + 0.7225 x 104.24 = 225.4574, less a hundredth of a point for each
    # of the two men bought: 225.4374.
    players, projections = blooming([5, 6, 7], count=2)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY,
    )

    assert plan.transfers_in == []
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(event=6, transfers_in=[16, 17], transfers_out=[8, 9], hits=0)
    ]
    assert plan.objective == pytest.approx(225.4374, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [5, 6, 7], plan, path)


def test_a_hit_is_taken_in_the_week_that_earns_it():
    # Same board, but with no free transfer banked: GW5 rolls one into GW6, and
    # the second of the two moves there is a hit. Waiting a further week to make
    # both free would cost 0.85 x (104.24 - 61.54) = 36.295 to save four points,
    # so the plan pays. The hit belongs to GW6, not to the decision week.
    players, projections = blooming([5, 6, 7], count=2)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[5, 6, 7],
        decay=DECAY,
    )

    assert plan.transfers_in == []
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(event=6, transfers_in=[16, 17], transfers_out=[8, 9], hits=1)
    ]
    assert isinstance(path.moves[0].hits, int)


def test_a_gameweek_that_cannot_hold_five_moves_starts_early():
    # Five late bloomers and two gameweeks. Rolling into GW6 leaves two free
    # transfers, and two free plus the two-hit ceiling is four moves — one short.
    # So the plan opens with two moves in GW5 (one hit) and finishes with three
    # in GW6 (two hits). 16 is worth 3.0 in GW5 and the others nothing, so he is
    # the one signed early, and the opening gameweek is 50.6 started + 6.0 for
    # the captain + 0.15 benched = 56.75: 56.75 + 0.85 x 146.54 - 12 = 169.309.
    #
    # The rival is to touch nothing in GW5 and take four bloomers in GW6 on two
    # free transfers and two hits — 61.54 + 0.85 x 132.54 - 8 = 166.199, which
    # is the number to beat. Opening with one move is 164.409 and with three is
    # 163.959.
    #
    # Every figure above is before the churn tiebreak, which charges a
    # hundredth of a point a buy. This plan buys five men (169.309 - 0.05 =
    # 169.259); the rival buys four (166.199 - 0.04 = 166.159); the one-move
    # opening buys four (164.369) and the three-move opening five (163.909).
    # Three points of margin against five hundredths of a point of tiebreak,
    # which is the whole argument for the size of it.
    players, projections = blooming([5, 6], count=5, early=3.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY,
    )

    assert plan.transfers_out == [8, 9]
    assert 16 in plan.transfers_in
    assert len(plan.transfers_in) == 2
    assert plan.hits == 1
    assert path.moves == [
        PlannedMove(
            event=6, transfers_in=sorted(set(bloomers(5)) - set(plan.transfers_in)),
            transfers_out=[10, 11, 12], hits=2,
        )
    ]
    assert set(bloomers(5)) <= set(plan.squad) | set(path.moves[0].transfers_in)
    assert plan.hits <= MAX_HITS
    assert all(move.hits <= MAX_HITS for move in path.moves)
    assert plan.objective == pytest.approx(169.259, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [5, 6], plan, path)


def test_the_free_transfer_bank_stops_at_five():
    # Five gameweeks of rolling from a bank that is already full. A transfer a
    # gameweek accrues, but never a sixth in hand: the bank stands at five at
    # every one of the five deadlines, and a rolled gameweek at the ceiling is
    # a free transfer thrown away rather than saved.
    players, projections = spine([5, 6, 7, 8, 9])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=MAX_FREE_TRANSFERS,
        events=[5, 6, 7, 8, 9], decay=DECAY,
    )

    assert plan.transfers_in == []
    assert path.moves == []
    banked = assert_legal_path(
        players, SQUAD, 0, MAX_FREE_TRANSFERS, [5, 6, 7, 8, 9], plan, path
    )
    assert banked == [MAX_FREE_TRANSFERS] * 5


def test_a_sixth_move_costs_a_hit_because_the_bank_stops_at_five():
    # Six players worth having in GW6 — five midfielders and a forward — and a
    # full bank of five going into GW5, which GW5 is pinned to roll. An
    # uncapped bank would carry 5 - 0 + 1 = 6 into GW6 and buy all six for
    # nothing; the real one carries five, so the sixth move is a hit. It is
    # still worth making: 0.85 x (162.55 - 146.54) = 13.61 against four points.
    # 61.54 + 0.85 x 162.55 - 4 = 195.7075, less six hundredths for the six men
    # bought: 195.6475.
    players, projections = six_arrivals([5, 6])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=MAX_FREE_TRANSFERS,
        events=[5, 6], decay=DECAY, forced_first_transfers=0,
    )

    assert path.moves == [
        PlannedMove(
            event=6,
            transfers_in=[16, 17, 18, 19, 20, 21],
            transfers_out=[8, 9, 10, 11, 12, 15],
            hits=1,
        )
    ]
    assert plan.objective == pytest.approx(195.6475, abs=1e-4)
    banked = assert_legal_path(
        players, SQUAD, 0, MAX_FREE_TRANSFERS, [5, 6], plan, path
    )
    assert banked == [MAX_FREE_TRANSFERS, MAX_FREE_TRANSFERS]


def test_a_bank_bigger_than_the_game_allows_is_taken_as_five():
    # 15 is how aigaffer.solver.lineup prices a wildcard, and it is not a bank:
    # this solver reads the number as the free transfers a manager actually
    # holds, so anything above the ceiling is the ceiling. Six arrivals worth
    # 20.0 straight away are all worth signing, and a five-strong bank is what
    # says five of them go now — the opening gameweek moves at most
    # max(MAX_TRANSFERS, ft) times, and here that is the clamped five.
    #
    # GW5 takes four of the midfielders and the forward (21 for 15, worth 20.0
    # against 3.9, beats a fifth midfielder by 2.01), keeping 12 as the fifth:
    # 128.0 started + 20.0 for the captain + 0.55 benched = 148.55. GW6 rolls
    # the one free transfer into the last arrival for 12: 162.55.
    # 148.55 + 0.85 x 162.55 = 286.7175, less six hundredths for the six men
    # bought: 286.6575. Read literally, fifteen free moves would have bought all
    # six in GW5 for nothing: 300.7175, or 300.6575 net of the same six.
    players, projections = six_arrivals([5, 6], opening=20.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=15, events=[5, 6],
        decay=DECAY,
    )

    assert len(plan.transfers_in) == 5
    assert 21 in plan.transfers_in
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(
            event=6,
            transfers_in=sorted({16, 17, 18, 19, 20} - set(plan.transfers_in)),
            transfers_out=[12],
            hits=0,
        )
    ]
    assert plan.objective == pytest.approx(286.6575, abs=1e-4)
    assert_legal_path(
        players, SQUAD, 0, MAX_FREE_TRANSFERS, [5, 6], plan, path
    )


def test_hits_reconcile_with_the_objective():
    # xp_total is the objective with the hits added back, so that a caller
    # ranking plans by objective is ranking them net of what they cost. The
    # board is the one above: one hit in GW5 and two in GW6.
    players, projections = blooming([5, 6], count=5, early=3.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY,
    )

    taken = plan.hits + sum(move.hits for move in path.moves)
    assert taken == 3
    assert plan.xp_total == pytest.approx(plan.objective + 4 * taken, abs=1e-4)


# --------------------------------------------------------------------------
# Money that outlives the gameweek that raised it
# --------------------------------------------------------------------------


def test_week_one_proceeds_pay_for_a_week_two_signing():
    # 8 is 120 of dead weight at 0.0; 16 is 6.5 for 45 and the only man the
    # squad can afford, since no other midfielder sells for more than 40. That
    # move banks 75. In GW6, 17 arrives on 12.0 for 115 — and 12 raises only 40,
    # so 75 of the money that pays for him was raised a gameweek earlier.
    # 56.65 + 0.85 x 70.7 = 116.745, against 112.445 for leaving GW5 alone.
    # Both buy two men, so the churn tiebreak takes two hundredths off each and
    # the margin is untouched: 116.725 against 112.425.
    rows = [
        (1, GK, 50, {5: 5.0, 6: 5.0}),
        (2, GK, 50, {5: 0.5, 6: 0.5}),
        (3, DEF, 50, {5: 4.2, 6: 4.2}),
        (4, DEF, 50, {5: 4.1, 6: 4.1}),
        (5, DEF, 50, {5: 4.0, 6: 4.0}),
        (6, DEF, 50, {5: 0.5, 6: 0.5}),
        (7, DEF, 50, {5: 0.5, 6: 0.5}),
        (8, MID, 120, {5: 0.0, 6: 0.0}),
        (9, MID, 40, {5: 6.0, 6: 6.0}),
        (10, MID, 40, {5: 5.0, 6: 5.0}),
        (11, MID, 40, {5: 3.5, 6: 3.5}),
        (12, MID, 40, {5: 3.0, 6: 3.0}),
        (13, FWD, 50, {5: 3.9, 6: 3.9}),
        (14, FWD, 50, {5: 3.8, 6: 3.8}),
        (15, FWD, 50, {5: 3.7, 6: 3.7}),
        (16, MID, 45, {5: 6.5, 6: 6.5}),
        (17, MID, 115, {5: 0.0, 6: 12.0}),
    ]
    players, projections = _build(rows)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY,
    )

    assert plan.transfers_in == [16]
    assert plan.transfers_out == [8]
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(event=6, transfers_in=[17], transfers_out=[12], hits=0)
    ]
    # 17 costs 115 and 12 sells for 40: the other 75 came out of GW5's sale.
    assert players[17].now_cost - players[12].now_cost == 75
    assert plan.objective == pytest.approx(116.725, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [5, 6], plan, path)


# --------------------------------------------------------------------------
# The double gameweek: the whole point of the exercise
# --------------------------------------------------------------------------


def test_it_stages_a_transfer_early_for_a_double_gameweek():
    # Four doubles in GW11, and the fifteen cannot hold them without first
    # turning 8 (130, and 6.0) into 16 (30, and 5.5). That is five moves in all
    # — one to raise the money, four to spend it — and one free transfer plus
    # the two-hit ceiling is three moves in GW10 and, after them, three in GW11.
    # Five will not fit in one gameweek, so the funding move goes early.
    #
    # GW10: out 8 and 9 (the cheapest starter to lose), in 16 and 17 (the best
    # of the doubles right now, 3.0) — one hit, and the bank goes 0 -> 75.
    # GW11: out 10, 11, 12, in 18, 19, 20 — two hits, and the bank returns to 0.
    # 53.1 + 0.85 x 139.9 - 12 = 160.015. Three doubles bought in GW11 alone is
    # 155.755; opening with three instead of two is 157.215.
    #
    # Net of the churn tiebreak, a hundredth of a point a buy: this plan buys
    # five men for 159.965, the GW11-only rival three for 155.725, and the
    # three-move opening five for 157.165. Four points of margin against five
    # hundredths — the tiebreak cannot reach a decision like this one, which is
    # what this test is here to keep true.
    players, projections = double_gameweek()

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY,
    )

    assert plan.transfers_out == [8, 9]
    assert plan.transfers_in == [16, 17]
    assert plan.hits == 1
    assert path.moves == [
        PlannedMove(
            event=11, transfers_in=[18, 19, 20], transfers_out=[10, 11, 12], hits=2
        )
    ]
    assert {17, 18, 19, 20} <= set(plan.squad) | set(path.moves[0].transfers_in)
    assert plan.objective == pytest.approx(159.965, abs=1e-4)
    assert plan.xp_total == pytest.approx(171.965, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(52.7, abs=1e-4)
    assert path.weekly_xp[11] == pytest.approx(139.5, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, DOUBLE_EVENTS, plan, path)


def test_the_greedy_solver_never_makes_the_funding_move():
    # The contrast that justifies the whole model. One gameweek at a time, 16 is
    # a 5.5 replacing a 6.0 and nothing else: a downgrade nobody would make. So
    # the single-week solver spends its three moves on doubles it can reach
    # without him — 8, 9 and 10 out for three of 17-20 — and stops one double
    # short of the squad the multi-week plan reaches, because the money to go
    # further was never raised.
    players, projections = double_gameweek()

    greedy = optimize(players, projections, SQUAD, bank=0, free_transfers=1)
    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY,
    )

    assert 16 not in greedy.transfers_in
    assert 16 in plan.transfers_in
    assert len(set(greedy.squad) & {17, 18, 19, 20}) == 3
    assert {17, 18, 19, 20} <= set(path.moves[0].transfers_in) | set(plan.squad)


# --------------------------------------------------------------------------
# The opening gameweek plays by the single-week solver's rules
# --------------------------------------------------------------------------


def test_the_opening_gameweek_moves_no_more_than_the_single_week_solver_could():
    # Seven men worth 20.0 straight away, a full bank of five free transfers and
    # the two-hit ceiling: seven moves in GW5 are legal FPL, and at 20.0 a man
    # against four points a hit the arithmetic wants all of them now. The
    # single-week solver would never offer more than max(MAX_TRANSFERS, ft) —
    # five here — and a window that outbids it with moves the other engine was
    # never allowed to consider is not being ranked against it fairly. So the
    # opening gameweek stops at five too, and the rest wait for GW6.
    players, projections = seven_arrivals([5, 6])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=MAX_FREE_TRANSFERS,
        events=[5, 6], decay=DECAY,
    )

    assert len(plan.transfers_in) == 5
    assert plan.hits == 0
    assert len(path.moves) == 1
    assert len(path.moves[0].transfers_in) == 2
    assert_legal_path(
        players, SQUAD, 0, MAX_FREE_TRANSFERS, [5, 6], plan, path
    )


def test_a_forced_opening_is_not_bound_by_that_cap():
    # The cap is on what the model chooses for itself, not on what a caller may
    # ask it. A forced count is a question — what is the best window that opens
    # with exactly this many moves? — and the answer to a legal question is a
    # plan, hits and all.
    players, projections = seven_arrivals([5, 6])

    plan, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=MAX_FREE_TRANSFERS,
        events=[5, 6], decay=DECAY, forced_first_transfers=7,
    )

    assert len(plan.transfers_in) == 7
    assert plan.hits == MAX_HITS


# --------------------------------------------------------------------------
# A gameweek with nothing to do does nothing
# --------------------------------------------------------------------------


def test_a_pointless_round_trip_is_not_planned():
    # 16 here is projected at exactly 5.6 in every gameweek: the twin of 8, the
    # cheapest starter the spine has. One opening move is demanded, so 8 goes
    # and 16 arrives and the eleven is worth exactly what it was worth before.
    #
    # From GW6 on there is nothing left to decide. Buying 8 straight back is
    # worth precisely nothing, and before the churn tiebreak the model had no
    # reason not to and did — a round trip printed under "the road ahead" as
    # though it were advice, when it was only the solver reporting which of a
    # great many equal optima it happened to land on. Now the quiet gameweeks
    # are empty, and the objective is the do-nothing 158.31165 less the one
    # hundredth of a point the one forced buy is charged.
    players, projections = spine([5, 6, 7], spare=5.6)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY, forced_first_transfers=1,
    )

    assert plan.transfers_in == [16]
    assert plan.transfers_out == [8]
    assert path.moves == []
    assert plan.objective == pytest.approx(158.31165 - CHURN_EPSILON, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [5, 6, 7], plan, path)


def test_the_tiebreak_is_too_small_to_reach_a_real_decision():
    # The other half of the epsilon's contract, on the board the whole model
    # exists for: it still stages the funding move a gameweek early. The plan
    # that does not stage it — the one that rolls GW10 and buys what it can
    # afford in GW11 — is more than four points worse, and no plan on this
    # board buys more than five men, so the most the tiebreak can move any
    # number here is five hundredths of a point. A hundred epsilons is the
    # margin asked for below, and the real one is eighty-five times that.
    players, projections = double_gameweek()

    plan, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY,
    )
    rolled, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY, forced_first_transfers=0,
    )

    assert plan.transfers_in == [16, 17]
    assert plan.objective - rolled.objective > 100 * CHURN_EPSILON


# --------------------------------------------------------------------------
# The leash on a solve
# --------------------------------------------------------------------------


def test_a_solve_can_be_put_on_a_shorter_leash():
    # One solve is one gameweek's decision and can have the minute; a sweep is
    # half a dozen of them and cannot. Nothing else about the solver changes,
    # the silence included.
    assert _solver(None) is SOLVER
    assert _solver(20).timeLimit == 20
    assert not _solver(20).msg


def test_a_time_limit_reaches_the_solve_itself():
    # Twenty seconds is far longer than the spine has ever needed, so the answer
    # is the one the default leash gives. What this pins is that the parameter
    # reaches the solver rather than being accepted and dropped.
    players, projections = spine([5, 6])

    plan, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6],
        decay=DECAY, time_limit=20,
    )

    assert plan.transfers_in == []
    assert plan.objective == pytest.approx(113.849, abs=1e-4)


# --------------------------------------------------------------------------
# Pinning the opening move, and saying no
# --------------------------------------------------------------------------


def test_forced_first_transfers_pins_the_opening_move():
    # One move demanded in a world with nothing worth buying: the plan gives up
    # the cheapest starter it has (8, on 5.6) for the only man on offer, and
    # buys him straight back the following gameweek with the free transfer that
    # rolls in.
    players, projections = spine([5, 6, 7])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY, forced_first_transfers=1,
    )

    assert plan.transfers_in == [16]
    assert plan.transfers_out == [8]
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(event=6, transfers_in=[8], transfers_out=[16], hits=0)
    ]


def test_forced_first_transfers_of_zero_leaves_the_squad_alone():
    players, projections = double_gameweek()

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1,
        events=DOUBLE_EVENTS, decay=DECAY, forced_first_transfers=0,
    )

    assert plan.transfers_in == []
    assert plan.transfers_out == []
    assert plan.squad == SQUAD
    assert path.moves != []


def test_more_forced_transfers_than_the_pool_allows_returns_none():
    # One player outside the squad, two moves demanded.
    players, projections = spine([5, 6, 7])

    assert (
        optimize_path(
            players, projections, SQUAD, bank=0, free_transfers=2, events=[5, 6, 7],
            decay=DECAY, forced_first_transfers=2,
        )
        is None
    )


def test_a_window_with_no_gameweeks_in_it_is_no_plan_at_all():
    players, projections = spine([5, 6, 7])

    assert (
        optimize_path(
            players, projections, SQUAD, bank=0, free_transfers=1, events=[],
            decay=DECAY,
        )
        is None
    )


def test_a_gameweek_a_player_has_no_projection_for_is_a_blank():
    # 16's projection stops after GW5: 9.0 in the gameweek he has, nothing in
    # the two he does not. So he is worth signing for one week — 9.0 into the XI
    # for 8's 5.6, and the armband on top, 67.94 against 61.54 — and worth
    # selling straight back out of it, which is what the plan does. The two
    # buys cost two hundredths of the churn tiebreak against 5.2 a gameweek of
    # real gain, which is the ordering the size of it is chosen for.
    players, projections = spine([5, 6, 7])
    projections[16] = PlayerProjection(
        player_id=16, per_gw={5: 9.0}, total=decayed_total({5: 9.0}, DECAY)
    )

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY,
    )

    assert plan.transfers_in == [16]
    assert plan.transfers_out == [8]
    assert plan.hits == 0
    assert path.moves == [
        PlannedMove(event=6, transfers_in=[8], transfers_out=[16], hits=0)
    ]
    assert path.weekly_xp[5] == pytest.approx(67.4, abs=1e-4)
    assert path.weekly_xp[6] == pytest.approx(SPINE_WEEK_XI, abs=1e-4)
    assert path.weekly_xp[7] == pytest.approx(SPINE_WEEK_XI, abs=1e-4)
    assert plan.objective == pytest.approx(164.69165, abs=1e-4)


# --------------------------------------------------------------------------
# Chips: bench boost and triple captain, planned into the window
# --------------------------------------------------------------------------


def test_no_chips_available_is_byte_for_byte_the_old_solve():
    # The fallback guarantee. An empty ``available_chips`` — the default, and
    # what every pre-chip caller passes without knowing it — must leave the
    # objective exactly where it was: the spine's 158.31165 do-nothing, whether
    # the argument is omitted or handed in empty. No chip is planned, and the
    # path's weeks carry no chip.
    players, projections = spine([5, 6, 7])

    default = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY,
    )
    empty = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[5, 6, 7],
        decay=DECAY, available_chips=frozenset(),
    )

    assert default[0].objective == empty[0].objective
    assert default[0].objective == pytest.approx(158.31165, abs=1e-4)
    assert empty[1].week1_chip == "none"
    assert all(move.chip == "none" for move in empty[1].moves)


# --------------------------------------------------------------------------
# The early-season chip floor: no chip is played before CHIP_FLOOR_GW
# --------------------------------------------------------------------------


def test_a_chip_before_the_floor_is_held_however_good():
    # The floor, and the live GW2 free hit it exists to stop, reproduced offline.
    # GW2 is lopsided — a current fifteen flat at 4.0 and a fifteen of heroes
    # worth 8.0 there and nothing after — so a free hit's best one-week squad is
    # 12.4 x 8.0 = 99.2 against a 12.4 x 4.0 = 49.6 do-nothing week, a gain of
    # 49.6 that clears the 25.0 reservation with room to spare. Left to the bar
    # alone the free hit would be played in GW2 (the RED half of this test before
    # the floor existed). But every gameweek in the window is before
    # CHIP_FLOOR_GW, so no chip may be played at all: the free hit is held, and
    # with its play binary pinned to zero the whole solve collapses onto the
    # no-chip one — the same feasible region and the same objective to the point,
    # whatever permanent moves each makes.
    players, projections = flat_with_heroes(
        [2, 3], spike_event=2, base=4.0, hero_value=8.0
    )

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[2, 3],
        decay=DECAY, available_chips=frozenset({FREE_HIT}),
    )
    no_chip, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[2, 3],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert path.week1_freehit_squad is None
    # The floor removes the chip, so the chip-available objective is exactly the
    # no-chip solve's — the whole point of "held".
    assert plan.objective == pytest.approx(no_chip.objective, abs=1e-4)


def test_the_floor_lets_the_first_eligible_week_play():
    # The floor's edge. Two flat gameweeks worth 6.0 a man and a bench boost in
    # hand: the boost clears its bar in both weeks (0.9 x 24.0 = 21.6, over the
    # 20.0 reservation). GW9 is the week the model would want — earlier, so the
    # decay bites least — but GW9 is before CHIP_FLOOR_GW and blocked; GW10 is
    # the first eligible week, so the boost lands there and nowhere earlier.
    # The two weeks are identical, so the floor is the whole of why it is GW10.
    players, projections = flat([9, 10], 6.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[9, 10],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )

    assert path.week1_chip == "none"
    assert path.moves == [
        PlannedMove(
            event=10, transfers_in=[], transfers_out=[], hits=0, chip=BENCH_BOOST
        )
    ]
    # do-nothing 12.4 x 6.0 x (1 + 0.85) = 137.64, plus the boost in GW10 decayed
    # 0.85 x (0.9 x 24.0 - 20.0) = 0.85 x 1.6 = 1.36: 139.0.
    assert plan.objective == pytest.approx(139.0, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [9, 10], plan, path)


def test_a_bench_boost_between_the_old_and_new_bar_is_held():
    # The raised reservation, the second line of defence past the floor. A flat
    # board at 4.0 a man in two eligible weeks: the boosted bench is worth 0.9 x
    # 16.0 = 14.4, which cleared the old 12.0 bar but sits under the raised 20.0
    # one. So the boost is held and the objective is exactly the no-chip solve:
    # 12.4 x 4.0 x (1 + 0.85) = 91.76.
    players, projections = flat([10, 11], 4.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )
    held = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert plan.objective == pytest.approx(held[0].objective, abs=1e-4)
    assert plan.objective == pytest.approx(91.76, abs=1e-4)


# The bench boost/triple captain reservation is 12.0 undecayed points: a chip
# is planned only where its marginal beats that bar, and held otherwise. The
# boards below straddle it on purpose.
FLAT_POSITIONS = (
    [GK, GK] + [DEF] * 5 + [MID] * 5 + [FWD] * 3
)


def flat(events: list[int], value: float) -> tuple[dict, dict]:
    """Fifteen players, the legal quota, every one worth ``value`` every week.

    The pool is the squad exactly, so nothing is ever bought and the only
    decision left is the chip. A flat board pins the four-man bench at
    ``4 x value`` however the XI is split, which is what lets a bench-boost
    objective be worked out with a pencil.
    """
    rows = [
        (pid, FLAT_POSITIONS[pid - 1], 50, {event: value for event in events})
        for pid in range(1, 16)
    ]
    return _build(rows)


def test_bench_boost_below_its_bar_is_held():
    # Two eligible weeks (past the floor) at 3.0 a man: the four-man bench is
    # worth 12.0, boosted 0.9 x 12.0 = 10.8, which does not clear the 20.0
    # reservation. So the chip is held — not played in any week — and the
    # objective is exactly the no-bench-boost solve: 12.4 x 3.0 = 37.2 a week,
    # discounted 1 + 0.85 = 1.85, is 68.82.
    players, projections = flat([10, 11], 3.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )
    held = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert plan.objective == pytest.approx(held[0].objective, abs=1e-4)
    assert plan.objective == pytest.approx(68.82, abs=1e-4)


def test_bench_boost_above_its_bar_is_played():
    # The threshold-crossing case past the raised bar, hand-computed. A flat
    # board at 6.0 a man in two eligible weeks: the bench is worth 24.0, boosted
    # 0.9 x 24.0 = 21.6, which clears the 20.0 bar by 1.6. It is played in GW10,
    # the window's first eligible week, where the decay bites least; the
    # objective gains decay^0 x (0.9 x 24.0 - 20.0) = 1.6 over the do-nothing
    # 12.4 x 6.0 x 1.85 = 137.64: 139.24. The bench pays in full only for the
    # objective's ranking; weekly_xp shows the fifteen's 66 + 6 armband + 24
    # bench = 96.
    players, projections = flat([10, 11], 6.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )

    assert path.week1_chip == BENCH_BOOST
    assert path.moves == []
    assert plan.objective == pytest.approx(139.24, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(96.0, abs=1e-4)
    assert path.weekly_xp[11] == pytest.approx(72.0, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [10, 11], plan, path)


def test_triple_captain_below_its_bar_is_held():
    # The spine over two eligible weeks, whose captain (12) is worth 6.0: a third
    # of the 18.0 reservation, so the extra armband a triple captain buys is not
    # worth the chip. It is held, and the objective is the plain two-gameweek
    # do-nothing 113.849.
    players, projections = spine([10, 11])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({TRIPLE_CAPTAIN}),
    )
    held = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert plan.objective == pytest.approx(held[0].objective, abs=1e-4)
    assert plan.objective == pytest.approx(113.849, abs=1e-4)


def test_triple_captain_above_its_bar_lands_on_the_monster_week():
    # The spine over two eligible weeks, but the captain (12) is worth 6.0 in
    # GW10 and 40.0 in GW11. In GW10 the extra armband (6.0) does not clear the
    # 18.0 bar; in GW11 it clears it by 22.0, and decayed that is 0.85 x 22.0 =
    # 18.7. So the chip waits for the monster: the do-nothing 171.649 plus 18.7
    # is 190.349. It is not the first week's, so week1_chip stays "none" and the
    # chip rides a move that makes no transfers at all — the road-ahead entry
    # exists only to name the chip.
    players, projections = spine([10, 11])
    monster = {10: 6.0, 11: 40.0}
    projections[12] = PlayerProjection(
        player_id=12, per_gw=monster, total=decayed_total(monster, DECAY)
    )

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({TRIPLE_CAPTAIN}),
    )

    assert path.week1_chip == "none"
    assert path.moves == [
        PlannedMove(
            event=11, transfers_in=[], transfers_out=[], hits=0, chip=TRIPLE_CAPTAIN
        )
    ]
    assert plan.objective == pytest.approx(190.349, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(61.0, abs=1e-4)
    assert path.weekly_xp[11] == pytest.approx(169.0, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [10, 11], plan, path)


def test_a_single_gameweek_holds_at_most_one_chip():
    # One eligible gameweek, both chips in hand, and a board where each clears
    # its bar on its own: fourteen men on 10.0 and a captain (12) on 30.0. The
    # bench is 4 x 10.0 = 40.0, boosted 0.9 x 40.0 = 36.0 and worth 36.0 - 20.0 =
    # 16.0 net; the extra armband is worth 30.0 - 18.0 = 12.0 net. Both would be
    # played were there room, but the one-chip-a-week rule is the whole of what
    # stops it here — there is no other week to send the loser to. Bench boost is
    # the bigger, so it alone is played: 130 started + 30 armband + 4.0 bench =
    # 164.0 do-nothing, plus 16.0, is 180.0 — not the 192.0 that both would earn.
    rows = [
        (pid, FLAT_POSITIONS[pid - 1], 50, {10: 30.0 if pid == 12 else 10.0})
        for pid in range(1, 16)
    ]
    players, projections = _build(rows)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST, TRIPLE_CAPTAIN}),
    )

    assert path.week1_chip == BENCH_BOOST
    assert plan.objective == pytest.approx(180.0, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [10], plan, path)


def test_two_chips_find_their_two_best_weeks():
    # A fifteen with no one to buy — the pool is the squad exactly — and both
    # chips in hand, over two eligible weeks. GW10 is flat at 10.0 a man, so its
    # four-man bench is worth 0.9 x 40 = 36.0 boosted (net 16.0 over the bar) and
    # its captain only 10.0 tripled (below the bar): bench boost's week. GW11 is
    # 1.0 a man but for the captain (12) at 100.0, so the extra armband is worth
    # 100.0 - 18.0 = 82.0 net, decayed 0.85 x 82.0 = 69.7, and the bench a
    # rounding error: triple captain's week. Different weeks, no clash, both
    # played.
    #
    # GW10 do-nothing is 110 started + 10 armband + 4.0 bench = 124.0; boosted,
    # +16.0 net. GW11 is 110 + 100 + 0.4 = 210.4; tripled, +82.0 net. So
    # 124.0 + 16.0 + 0.85 x (210.4 + 82.0) = 140.0 + 248.54 = 388.54.
    rows = [
        (1, GK, 50, {10: 10.0, 11: 1.0}),
        (2, GK, 50, {10: 10.0, 11: 1.0}),
        (3, DEF, 50, {10: 10.0, 11: 1.0}),
        (4, DEF, 50, {10: 10.0, 11: 1.0}),
        (5, DEF, 50, {10: 10.0, 11: 1.0}),
        (6, DEF, 50, {10: 10.0, 11: 1.0}),
        (7, DEF, 50, {10: 10.0, 11: 1.0}),
        (8, MID, 50, {10: 10.0, 11: 1.0}),
        (9, MID, 50, {10: 10.0, 11: 1.0}),
        (10, MID, 50, {10: 10.0, 11: 1.0}),
        (11, MID, 50, {10: 10.0, 11: 1.0}),
        (12, MID, 50, {10: 10.0, 11: 100.0}),
        (13, FWD, 50, {10: 10.0, 11: 1.0}),
        (14, FWD, 50, {10: 10.0, 11: 1.0}),
        (15, FWD, 50, {10: 10.0, 11: 1.0}),
    ]
    players, projections = _build(rows)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST, TRIPLE_CAPTAIN}),
    )

    assert path.week1_chip == BENCH_BOOST
    assert path.moves == [
        PlannedMove(
            event=11, transfers_in=[], transfers_out=[], hits=0, chip=TRIPLE_CAPTAIN
        )
    ]
    assert plan.objective == pytest.approx(388.54, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(160.0, abs=1e-4)
    assert path.weekly_xp[11] == pytest.approx(310.0, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [10, 11], plan, path)


def test_a_chip_is_played_at_most_once_across_the_horizon():
    # Bench boost only, three eligible flat gameweeks at 6.0 a man. The bench is
    # worth 0.9 x 24.0 = 21.6 a week, clearing the 20.0 bar by 1.6 in all three,
    # and the chip would be welcome in every one — but the horizon limit lets it
    # be played once, in the first, where the decay bites least. The do-nothing
    # 12.4 x 6.0 x (1 + 0.85 + 0.7225) = 191.394 plus 1.6 is 192.994, not the
    # 191.394 + 1.6 x 2.5725 = 195.51 that three plays would earn.
    players, projections = flat([10, 11, 12], 6.0)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )

    assert path.week1_chip == BENCH_BOOST
    assert all(move.chip == "none" for move in path.moves)
    assert plan.objective == pytest.approx(192.994, abs=1e-4)
    assert_legal_path(players, SQUAD, 0, 1, [10, 11, 12], plan, path)


# --------------------------------------------------------------------------
# Wildcard: free, uncapped transfers for one gameweek of the window
# --------------------------------------------------------------------------


def _rebuild_and_arrivals(
    events: list[int], squad_value: float, arrival_value: float
) -> tuple[dict, dict]:
    """The legal fifteen, every man worth ``squad_value``, plus five midfielders
    and two forwards worth ``arrival_value``.

    The seven arrivals are exactly the men an XI wants — five midfield slots and
    two forward — so a full rebuild starts all seven, and everyone costs 50, so
    money never decides. It is the board a wildcard exists for: a rebuild that
    needs more moves than a gameweek's cap allows.
    """
    rows = [
        (pid, FLAT_POSITIONS[pid - 1], 50, {event: squad_value for event in events})
        for pid in range(1, 16)
    ]
    for pid, position in SEVEN:
        rows.append((pid, position, 50, {event: arrival_value for event in events}))
    return _build(rows)


def test_a_wildcard_rebuilds_the_whole_squad_in_one_week():
    # Seven men worth 20.0 straight away — five midfielders and two forwards —
    # and a squad on the spine, one free transfer, one eligible gameweek. Without
    # a chip the opening gameweek moves at most three; the wildcard makes every
    # transfer free and lifts the cap, so all seven arrive at once for no hit.
    #
    # The XI is 1 | 3 4 5 | 16 17 18 19 20 | 21 22: 5.0 + (4.4 + 4.3 + 4.2) +
    # 100.0 + 40.0 = 157.9 started, 20.0 for the captain, and a bench of
    # 2 6 7 13 worth 0.5 + 0.5 + 0.5 + 4.1 = 5.6 at a tenth, 0.56. The week
    # scores 157.9 + 20.0 + 0.56 = 178.46; the wildcard's 35.0 reservation comes
    # off, and seven bought cost seven hundredths of the churn tiebreak:
    # 178.46 - 35.0 - 0.07 = 143.39.
    players, projections = seven_arrivals([10])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )
    no_chip, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == WILDCARD
    assert len(plan.transfers_in) == 7
    assert plan.hits == 0
    assert plan.objective == pytest.approx(143.39, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(177.9, abs=1e-4)
    # Without the chip the same rebuild is throttled to the opening cap and can
    # never take all seven in a single gameweek.
    assert len(no_chip.transfers_in) <= MAX_TRANSFERS
    assert_legal_path(players, SQUAD, 0, 1, [10], plan, path)


def test_a_wildcard_uncaps_the_gameweek():
    # A seven-move gameweek is more than the three the opening cap allows and
    # more than a one-transfer bank plus the two-hit ceiling can buy. It is legal
    # only under the wildcard, which is the one thing that lifts the cap.
    players, projections = seven_arrivals([10])

    on, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )
    off, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset(),
    )

    assert len(on.transfers_in) == 7
    assert on.hits == 0
    assert len(off.transfers_in) < 7


def test_a_wildcard_week_spends_no_free_transfers_and_no_hits():
    # The carry across a wildcard gameweek, pinned. The rebuild lands in GW10 —
    # the earliest eligible week, where the horizon's decay bites least — takes
    # all seven for no hit, and leaves the two later gameweeks with nothing to
    # do. A normal seven-move gameweek would empty the bank; a wildcard spends
    # none of it, so the bank carries as though no one moved: 1 into GW10, then
    # min(5, 1 + 1) = 2 at GW11's deadline and min(5, 2 + 1) = 3 at GW12's.
    players, projections = seven_arrivals([10, 11, 12])

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )

    assert path.week1_chip == WILDCARD
    assert len(plan.transfers_in) == 7
    assert plan.hits == 0
    assert path.moves == []
    banked = assert_legal_path(players, SQUAD, 0, 1, [10, 11, 12], plan, path)
    assert banked == [1, 2, 3]


def test_a_wildcard_and_a_bench_boost_cannot_share_a_gameweek():
    # One eligible gameweek, both chips in hand, and a board where each earns its
    # keep on its own: fifteen men worth 10.0 (a bench of 4 x 10.0 = 40.0,
    # boosted 0.9 x 40.0 = 36.0, well over the 20.0 bar) and seven arrivals worth
    # 30.0 (a rebuild worth far more than the 35.0 wildcard bar). Both would be
    # played were there room; the one-chip-a-week rule is the whole of what stops
    # it, and the wildcard, worth the most, takes the week. The bench boost is
    # held.
    players, projections = _rebuild_and_arrivals(
        [10], squad_value=10.0, arrival_value=30.0
    )

    both, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({WILDCARD, BENCH_BOOST}),
    )
    boost_only, boost_path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST}),
    )

    # The bench boost is worth playing on this board — it is played when it has
    # the week to itself — so its absence when the wildcard is available is the
    # one-chip-a-week rule doing its work.
    assert boost_path.week1_chip == BENCH_BOOST
    assert path.week1_chip == WILDCARD
    assert_legal_path(players, SQUAD, 0, 1, [10], both, path)


def test_a_wildcard_is_played_at_most_once_across_the_horizon():
    # Two eligible gameweeks, each with its own set of seven arrivals worth 20.0
    # in that gameweek and nothing in the other. Each week would take its own
    # wildcard rebuild if it could; the horizon allows one wildcard in all, so
    # the model spends it on a single week and never twice.
    rows = [
        (pid, FLAT_POSITIONS[pid - 1], 50, {10: 4.0, 11: 4.0}) for pid in range(1, 16)
    ]
    for pid, position in SEVEN:
        rows.append((pid, position, 50, {10: 20.0, 11: 0.0}))
    for offset, (_, position) in enumerate(SEVEN):
        rows.append((23 + offset, position, 50, {10: 0.0, 11: 20.0}))
    players, projections = _build(rows)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )

    played = (1 if path.week1_chip == WILDCARD else 0) + sum(
        1 for move in path.moves if move.chip == WILDCARD
    )
    assert played <= 1


def test_a_rebuild_worth_less_than_the_bar_holds_the_wildcard():
    # Three midfielders worth 20.0 replacing the spine's cheapest three (5.6,
    # 5.7, 5.8) is a rebuild worth having — 42.9 in the XI — but it fits inside
    # the opening cap in an eligible week: three moves on one free transfer is
    # two hits, eight points. A wildcard would save those eight and no more, and
    # eight is a long way under its 35.0 bar, so the chip is held and the hits
    # are paid instead.
    rows = [
        (pid, position, 50, {10: points}) for pid, position, points in SPINE
    ]
    for pid in (16, 17, 18):
        rows.append((pid, MID, 50, {10: 20.0}))
    players, projections = _build(rows)

    with_wc, wc_path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )
    without, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, available_chips=frozenset(),
    )

    assert wc_path.week1_chip == "none"
    assert len(with_wc.transfers_in) == 3
    assert with_wc.hits == MAX_HITS
    assert with_wc.objective == pytest.approx(without.objective, abs=1e-4)


def test_a_wildcard_in_hand_but_unused_matches_the_plain_solve():
    # The byte-for-byte guarantee at the wildcard's own reservation, in eligible
    # weeks so the hold is the value's doing and not the floor's. The spine has
    # nothing worth buying, so a wildcard buys nothing and is held; the objective
    # is exactly the do-nothing solve, whether the chip is offered or withheld.
    players, projections = spine([10, 11, 12])

    offered, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({WILDCARD}),
    )
    withheld, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert offered.objective == pytest.approx(withheld.objective, abs=1e-4)
    assert offered.objective == pytest.approx(158.31165, abs=1e-4)


# --------------------------------------------------------------------------
# Free hit: a one-week squad from the whole pool that reverts after the week
# --------------------------------------------------------------------------

# A full legal fifteen of "heroes" — two keepers, five defenders, five
# midfielders, three forwards, each on his own club — that a free-hit week can
# field all at once, which no run of capped transfers ever could. Their ids sit
# above the spine's, so a board is the spine plus these.
FIFTEEN_HEROES = (
    (16, GK), (17, GK),
    (18, DEF), (19, DEF), (20, DEF), (21, DEF), (22, DEF),
    (23, MID), (24, MID), (25, MID), (26, MID), (27, MID),
    (28, FWD), (29, FWD), (30, FWD),
)
# A second such fifteen, ids 31-45, for the board with a lopsided week apiece.
FIFTEEN_HEROES_B = tuple((pid + 15, position) for pid, position in FIFTEEN_HEROES)


def spine_with_heroes(
    events: list[int], spike_event: int, hero_value: float
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """The spine in every gameweek, plus a full legal fifteen worth
    ``hero_value`` in ``spike_event`` alone and nothing in any other gameweek.

    The heroes dominate the spine only in the spike gameweek and are dead weight
    everywhere else, so no permanent transfer ever wants them — but a free hit
    can field all eleven for that one week, which is the board free hit exists
    for. Everyone costs 50, so money never decides.
    """
    rows = [
        (pid, position, 50, {event: points for event in events})
        for pid, position, points in SPINE
    ]
    for pid, position in FIFTEEN_HEROES:
        arrival = {
            event: (hero_value if event == spike_event else 0.0)
            for event in events
        }
        rows.append((pid, position, 50, arrival))
    return _build(rows)


def flat_with_heroes(
    events: list[int], spike_event: int, base: float, hero_value: float
) -> tuple[dict[int, Player], dict[int, PlayerProjection]]:
    """A current fifteen flat at ``base`` in every gameweek, plus a full legal
    fifteen of heroes worth ``hero_value`` in ``spike_event`` alone.

    Chosen so a free hit is worth playing but a capped punt is not: eleven heroes
    together clear the reservation, while the two a gameweek's hit ceiling allows
    gain less per man than the four points a hit costs, so the no-free-hit solve
    leaves the squad alone and the revert is clean to read.
    """
    rows = [
        (pid, FLAT_POSITIONS[pid - 1], 50, {event: base for event in events})
        for pid in range(1, 16)
    ]
    for pid, position in FIFTEEN_HEROES:
        arrival = {
            event: (hero_value if event == spike_event else 0.0)
            for event in events
        }
        rows.append((pid, position, 50, arrival))
    return _build(rows)


def test_the_free_hit_xi_is_a_legal_fifteen_drawn_from_the_pool():
    # The side-calc that prices the free-hit week, checked on its own. In GW5 the
    # heroes are worth 8.0 and the spine at most 6.0, so the best one-week squad
    # is fifteen heroes: an XI of eleven (88.0) plus the armband (8.0) plus four
    # on the bench at a tenth (3.2) is 99.2. It must be a legal fifteen — the
    # 2/5/5/3 quota, three-a-club, inside the budget — with a legal eleven, and
    # every man drawn from the pool.
    players, projections = spine_with_heroes([5], spike_event=5, hero_value=8.0)
    pool = candidate_pool(players, projections, SQUAD, limit=CANDIDATES_PER_POSITION)
    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)
    week_points = {p: projections[p].per_gw.get(5, 0.0) for p in pool}
    budget = sum(players[p].now_cost for p in SQUAD)

    result = _best_one_week_squad(
        pool, players, by_position, by_club, week_points, budget, SOLVER
    )

    assert result is not None
    value, squad, xi = result
    assert value == pytest.approx(99.2, abs=1e-4)
    assert set(squad) <= set(pool)
    assert set(xi) <= set(squad)
    # A legal fifteen, checked exactly as the FPL website would.
    spent = sum(players[p].now_cost for p in squad)
    assert_legal(players, squad, budget - spent)
    # A legal eleven: eleven men, one keeper, at least three defenders and a
    # forward.
    assert len(xi) == 11
    assert sum(1 for p in xi if players[p].element_type == GK) == 1
    assert sum(1 for p in xi if players[p].element_type == DEF) >= 3
    assert sum(1 for p in xi if players[p].element_type == FWD) >= 1
    # The heroes dominate the spine that week, so the free-hit squad is theirs.
    assert set(squad) == {pid for pid, _ in FIFTEEN_HEROES}


def test_free_hit_fields_a_temp_squad_that_reverts():
    # The whole point of the chip, hand-computed and its revert pinned, in
    # eligible weeks. The current fifteen is flat at 4.0 in every gameweek; GW10
    # is lopsided — fifteen heroes worth 8.0 there and nothing after — and GW11,
    # GW12 are the ordinary 4.0. A free hit fields all fifteen heroes in GW10 for
    # nothing and reverts, which no run of capped transfers can match: two paid
    # heroes gain 2 x (8.0 - 4.0) = 8.0 for their two hits (8 points) and would
    # revert anyway, a wash, so the whole spike is the free hit's alone.
    #
    # The best one-week squad in GW10 is fifteen heroes: an XI of 11 x 8.0 =
    # 88.0, the armband 8.0 and four benched at a tenth (3.2), 99.2. As a free
    # hit that is 99.2 less the 25.0 reservation: 74.2 — far past any capped
    # punt of the same heroes, so the chip is played. GW11 and GW12 are the
    # do-nothing 12.4 x 4.0 = 49.6. Undiscounted GW10, then 0.85 and 0.7225:
    # 74.2 + 0.85 x 49.6 + 0.7225 x 49.6 = 152.196.
    #
    # The revert is the invariant: the free-hit heroes never enter the standing
    # squad, so the fifteen carried into GW11 and GW12 is the one that started
    # the window, and the plan makes no permanent transfer in the free-hit week
    # or after it.
    players, projections = flat_with_heroes(
        [10, 11, 12], spike_event=10, base=4.0, hero_value=8.0
    )

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({FREE_HIT}),
    )

    assert path.week1_chip == FREE_HIT
    assert plan.transfers_in == []
    assert plan.transfers_out == []
    assert plan.hits == 0
    assert plan.squad == SQUAD
    assert path.moves == []
    assert plan.objective == pytest.approx(152.196, abs=1e-4)
    assert path.weekly_xp[10] == pytest.approx(99.2, abs=1e-4)
    assert path.weekly_xp[11] == pytest.approx(48.0, abs=1e-4)
    assert path.weekly_xp[12] == pytest.approx(48.0, abs=1e-4)

    # The revert invariant: none of the temp heroes persist into the standing
    # squad, which is the fifteen the window carries past the free-hit week.
    assert not ({pid for pid, _ in FIFTEEN_HEROES} & set(plan.squad))
    assert_legal_path(players, SQUAD, 0, 0, [10, 11, 12], plan, path)

    # The temporary team it fields is surfaced for the report — the heroes, not
    # the standing squad — and it is a legal fifteen with a legal eleven inside
    # it. This is the eleven Mark takes to the deadline, and the plan's own
    # squad/xi are the standing team that reverts.
    heroes = {pid for pid, _ in FIFTEEN_HEROES}
    assert set(path.week1_freehit_squad) == heroes
    assert len(path.week1_freehit_squad) == 15
    assert set(path.week1_freehit_xi) <= set(path.week1_freehit_squad)
    assert len(path.week1_freehit_xi) == 11
    assert not (set(path.week1_freehit_squad) & set(plan.squad))


def test_a_played_chip_that_is_not_a_free_hit_surfaces_no_temp_squad():
    # The free-hit fields are None on every other opening chip: a bench boost is
    # played on the team as it stands, so there is no temporary eleven to field.
    # A flat board at 6.0 in eligible weeks plays the boost in GW10 (0.9 x 24.0 =
    # 21.6, over the 20.0 bar) and holds the free hit, whose best one-week squad
    # is the board itself and gains nothing.
    players, projections = flat([10, 11], 6.0)

    _, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11],
        decay=DECAY, available_chips=frozenset({BENCH_BOOST, FREE_HIT}),
    )

    assert path.week1_chip == BENCH_BOOST
    assert path.week1_freehit_squad is None
    assert path.week1_freehit_xi is None


def test_a_free_hit_below_its_bar_is_held():
    # The reservation brake, in an eligible week so it is the bar and not the
    # floor doing the holding. Heroes worth 6.4 in GW10 against a spine topping
    # out at 6.0: the best one-week squad is 12.4 x 6.4 = 79.36, a gain of
    # 79.36 - 61.54 = 17.82 over the do-nothing week, which is under the raised
    # 25.0 bar. So the chip is held and the week is the plain 61.54 do-nothing.
    players, projections = spine_with_heroes([10], spike_event=10, hero_value=6.4)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, forced_first_transfers=0,
        available_chips=frozenset({FREE_HIT}),
    )
    held = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, forced_first_transfers=0, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert plan.objective == pytest.approx(held[0].objective, abs=1e-4)
    assert plan.objective == pytest.approx(61.54, abs=1e-4)


def test_a_free_hit_in_hand_but_unused_matches_the_plain_solve():
    # The byte-for-byte guarantee at free hit's own reservation, in eligible
    # weeks so the hold is the value's doing and not the floor's. The spine has
    # nothing worth signing, so the best one-week squad is the spine itself and a
    # free hit gains exactly nothing over holding — it is held, and the objective
    # is the do-nothing solve whether the chip is offered or withheld.
    players, projections = spine([10, 11, 12])

    offered, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({FREE_HIT}),
    )
    withheld, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset(),
    )

    assert path.week1_chip == "none"
    assert all(move.chip == "none" for move in path.moves)
    assert offered.objective == pytest.approx(withheld.objective, abs=1e-4)
    assert offered.objective == pytest.approx(158.31165, abs=1e-4)


def test_a_free_hit_is_played_at_most_once_across_the_horizon():
    # Two lopsided eligible weeks, each with its own fifteen heroes: 16-30 worth
    # 8.0 in GW10 and nothing in GW11, 31-45 worth 8.0 in GW11 and nothing in
    # GW10, over a static spine. Each week clears the bar on its own (99.2 -
    # 61.54 = 37.66, over the 25.0 reservation), so each would take a free hit if
    # it could — the horizon allows one, so it lands on GW10, where the decay
    # bites least, and never twice.
    rows = [
        (pid, position, 50, {10: points, 11: points})
        for pid, position, points in SPINE
    ]
    for pid, position in FIFTEEN_HEROES:
        rows.append((pid, position, 50, {10: 8.0, 11: 0.0}))
    for pid, position in FIFTEEN_HEROES_B:
        rows.append((pid, position, 50, {10: 0.0, 11: 8.0}))
    players, projections = _build(rows)

    plan, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[10, 11],
        decay=DECAY, available_chips=frozenset({FREE_HIT}),
    )

    played = (1 if path.week1_chip == FREE_HIT else 0) + sum(
        1 for move in path.moves if move.chip == FREE_HIT
    )
    assert played == 1
    assert path.week1_chip == FREE_HIT


def test_a_free_hit_and_a_bench_boost_cannot_share_a_gameweek():
    # One eligible gameweek, both chips in hand, and a board where each earns its
    # keep on its own: a current fifteen flat at 10.0 (a bench of 4 x 10.0 =
    # 40.0, boosted 0.9 x 40.0 = 36.0, well over the 20.0 bar) and fifteen heroes
    # worth 30.0 (a one-week squad worth 12.4 x 30.0 = 372.0, a free hit far past
    # its 25.0 bar). Both would be played were there room; the one-chip-a-week
    # rule is the whole of what stops it, and the free hit, worth the most, takes
    # the week. The bench boost is held.
    rows = [(pid, FLAT_POSITIONS[pid - 1], 50, {10: 10.0}) for pid in range(1, 16)]
    for pid, position in FIFTEEN_HEROES:
        rows.append((pid, position, 50, {10: 30.0}))
    players, projections = _build(rows)

    both, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, forced_first_transfers=0,
        available_chips=frozenset({FREE_HIT, BENCH_BOOST}),
    )
    boost_only, boost_path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10],
        decay=DECAY, forced_first_transfers=0,
        available_chips=frozenset({BENCH_BOOST}),
    )

    # The bench boost is worth playing here — it is played when it has the week
    # to itself — so its absence alongside the free hit is the one-chip rule.
    assert boost_path.week1_chip == BENCH_BOOST
    assert path.week1_chip == FREE_HIT
    assert_legal_path(players, SQUAD, 0, 1, [10], both, path)


def test_adding_free_hit_to_the_set_changes_nothing_when_it_is_held():
    # The guard, checked directly, in eligible weeks. On the spine a triple
    # captain is below its bar (a 6.0 captain against an 18.0 reservation) and a
    # free hit gains nothing (the spine cannot improve on itself), so neither is
    # played — and offering free hit alongside the triple captain leaves the
    # objective exactly where the triple-captain-only solve left it: the Task 2
    # model, untouched.
    players, projections = spine([10, 11, 12])

    without_fh, _ = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({TRIPLE_CAPTAIN}),
    )
    with_fh, path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=1, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({TRIPLE_CAPTAIN, FREE_HIT}),
    )

    assert path.week1_chip == "none"
    assert with_fh.objective == pytest.approx(without_fh.objective, abs=1e-4)
    assert with_fh.objective == pytest.approx(158.31165, abs=1e-4)


# --------------------------------------------------------------------------
# The free-hit pricing hoist: priced once for a sweep, byte-identical either way
# --------------------------------------------------------------------------


def test_the_hoisted_prices_are_the_per_week_side_calc():
    # The helper is a faithful factoring of the per-week free-hit price: each
    # week's entry is exactly what _best_one_week_squad returns for that week off
    # the same pool and budget. GW5 is the lopsided one — fifteen heroes at 6.0,
    # a one-week squad worth 12.4 x 6.0 = 74.4 — GW6 and GW7 the flat 4.0 spine,
    # whose best one-week squad is 12.4 x 4.0 = 49.6.
    players, projections = flat_with_heroes(
        [5, 6, 7], spike_event=5, base=4.0, hero_value=6.0
    )
    pool = candidate_pool(players, projections, SQUAD, limit=CANDIDATES_PER_POSITION)
    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)
    budget = sum(players[p].now_cost for p in SQUAD)

    prices = _free_hit_prices(players, projections, SQUAD, 0, [5, 6, 7], None)

    assert set(prices) == {1, 2, 3}
    for w, event in enumerate([5, 6, 7], start=1):
        week_points = {p: projections[p].per_gw.get(event, 0.0) for p in pool}
        expected = _best_one_week_squad(
            pool, players, by_position, by_club, week_points, budget, SOLVER
        )
        assert prices[w] == expected
    assert prices[1][0] == pytest.approx(74.4, abs=1e-4)
    assert prices[2][0] == pytest.approx(49.6, abs=1e-4)


def test_hoisted_free_hit_prices_give_a_byte_identical_plan():
    # The hoist's correctness at the solve. Pricing the free hit once and handing
    # the dict in must land on the very same plan, objective and temp squad as
    # letting the solve price it itself: the prices are the same floats off the
    # same board, so the only thing the lever moves is where the CBC sub-solves
    # run, never the answer. The free hit is played, so the price is load-bearing.
    players, projections = flat_with_heroes(
        [10, 11, 12], spike_event=10, base=4.0, hero_value=8.0
    )
    prices = _free_hit_prices(players, projections, SQUAD, 0, [10, 11, 12], None)

    inline_plan, inline_path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({FREE_HIT}),
    )
    hoisted_plan, hoisted_path = optimize_path(
        players, projections, SQUAD, bank=0, free_transfers=0, events=[10, 11, 12],
        decay=DECAY, available_chips=frozenset({FREE_HIT}), freehit_prices=prices,
    )

    assert inline_path.week1_chip == FREE_HIT
    # Byte-identical, not merely close.
    assert hoisted_plan.objective == inline_plan.objective
    assert hoisted_plan.squad == inline_plan.squad
    assert hoisted_plan.xi == inline_plan.xi
    assert hoisted_path.weekly_xp == inline_path.weekly_xp
    assert hoisted_path.week1_chip == inline_path.week1_chip
    assert hoisted_path.week1_freehit_squad == inline_path.week1_freehit_squad
    assert hoisted_path.week1_freehit_xi == inline_path.week1_freehit_xi


def test_generate_plans_hoists_the_free_hit_without_changing_the_shortlist():
    # The hoist end to end. generate_plans prices the free hit once and threads
    # the one dict through the whole sweep; the shortlist it returns is identical
    # to one built by solving each opening count with the price computed inline —
    # same objectives, same fifteens, same chip on the recommended plan. Same
    # board the free hit actually plays on, so the shared price is load-bearing.
    players, projections = flat_with_heroes(
        [10, 11, 12], spike_event=10, base=4.0, hero_value=8.0
    )
    chips = frozenset({FREE_HIT})

    hoisted = generate_plans(
        players, projections, SQUAD, bank=0, free_transfers=1,
        projections_events=[10, 11, 12], decay=DECAY, available_chips=chips,
    )
    inline_answers = [
        optimize_path(
            players, projections, SQUAD, 0, 1, [10, 11, 12], DECAY,
            forced_first_transfers=count, time_limit=SWEEP_TIME_LIMIT,
            available_chips=chips,
        )
        for count in transfer_counts(1)
    ]
    inline = _shortlist(a[0] if a is not None else None for a in inline_answers)

    assert [p.objective for p in hoisted] == [p.objective for p in inline]
    assert [p.squad for p in hoisted] == [p.squad for p in inline]
    assert [p.path.week1_chip for p in hoisted] == [p.path.week1_chip for p in inline]
    assert hoisted[0].path.week1_chip == FREE_HIT
