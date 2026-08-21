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
    MAX_HITS,
    SOLVER,
    PlannedMove,
    PlannedPath,
    _solver,
    optimize_path,
)
from aigaffer.solver.optimizer import (
    MAX_PER_CLUB,
    SQUAD_QUOTAS,
    SQUAD_SIZE,
    optimize,
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
    """The fifteen, plus one midfielder at ``spare`` nobody would want."""
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
        else:
            move = moves.get(event)
            incoming = move.transfers_in if move else []
            outgoing = move.transfers_out if move else []
            hits = move.hits if move else 0

        assert set(outgoing) <= squad
        assert not set(incoming) & squad
        assert len(incoming) == len(outgoing)
        squad = (squad - set(outgoing)) | set(incoming)
        cash += sum(players[p].now_cost for p in outgoing)
        cash -= sum(players[p].now_cost for p in incoming)
        assert_legal(players, sorted(squad), cash)

        series.append(banked)
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
    # 104.24 + 0.7225 x 104.24 = 225.4574.
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
    assert plan.objective == pytest.approx(225.4574, abs=1e-4)
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
    assert plan.objective == pytest.approx(169.309, abs=1e-4)
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
    # 61.54 + 0.85 x 162.55 - 4 = 195.7075.
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
    assert plan.objective == pytest.approx(195.7075, abs=1e-4)
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
    # 148.55 + 0.85 x 162.55 = 286.7175. Read literally, fifteen free moves
    # would have bought all six in GW5 for nothing: 300.7175.
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
    assert plan.objective == pytest.approx(286.7175, abs=1e-4)
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
    assert plan.objective == pytest.approx(116.745, abs=1e-4)
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
    assert plan.objective == pytest.approx(160.015, abs=1e-4)
    assert plan.xp_total == pytest.approx(172.015, abs=1e-4)
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
    # selling straight back out of it, which is what the plan does.
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
    assert plan.objective == pytest.approx(164.71165, abs=1e-4)
