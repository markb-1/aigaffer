"""The transfer *path* as one mixed-integer program.

:mod:`~aigaffer.solver.optimizer` answers the best question it can about one
gameweek: given the squad, the bank and the free transfers in hand, what is the
best fifteen reachable this week? It cannot answer the question a manager
actually asks in front of a double gameweek — *what do I do this week so that
in three weeks' time I own the players I want?* — because a single-week model
has nowhere to put the three weeks. Money raised has to be spent immediately,
a free transfer rolled is a free transfer wasted, and a plan that needs five
moves and can make three is simply infeasible rather than staged.

So this module solves the window instead of the week. One squad, one XI and one
captain per gameweek; buys and sells that carry the squad from each gameweek to
the next; a bank and a free-transfer count that carry with them. Only the first
gameweek is ever executed — the rest is advice, re-planned from scratch every
run — but the first gameweek's move is chosen knowing what it is for.

Four parts of the formulation are worth stating plainly.

**Free transfers are carried by upper bound alone.** The rule is
``ft[w+1] = min(5, ft[w] − transfers[w] + paid[w] + 1)``, and a linear program
cannot write ``min``. What it can write is the pair of inequalities the ``min``
is made of, with nothing pinning ``ft`` from below at all. That is exact rather
than a relaxation, because more free transfers never hurt: ``ft`` appears
elsewhere only in the pricing of ``paid``, where a larger ``ft`` can only lower
the one quantity the objective is charging four points a unit for, and in the
next gameweek's ceiling, which it raises. The solver therefore has every reason
to push ``ft`` as high as those inequalities allow and none whatever to hold it
down, so at the optimum it sits exactly on the ``min``.

The monotonicity has one more channel than that reads, and it is the channel
that has to be closed by hand. ``paid`` appears in the carry too, so lowering
``ft[w]`` by one raises ``paid[w]`` by one and leaves the carry's right-hand
side exactly where it was — a gameweek could buy back its own ceiling for four
points if ``paid`` were free to rise. It is not, because of the pin below; with
``paid`` held to ``max(0, transfers − ft)`` a lower ``ft`` costs four points and
buys nothing, and the argument above is then the whole of it.

**But the hits have to be pinned from both sides.** ``paid[w] ≥ transfers[w] −
ft[w]`` is the obvious half and, on its own, wrong: ``paid`` also appears in the
carry above, so a plan could take a hit it did not owe in order to leave its
free-transfer bank untouched — buying a free transfer for four points, which
the game does not sell. Left in, the model dodges the two-hit ceiling by
scattering bought transfers across gameweeks. So ``paid[w]`` is held to exactly
``max(0, transfers[w] − ft[w])`` by the usual big-M pair on an indicator: one
binary a gameweek, which on a five-gameweek window is nothing, and without it
the FT dynamics are not the game's.

**Nobody caps the transfers in a gameweek, and nobody has to.** ``paid[w]`` is
capped at :data:`~aigaffer.solver.optimizer.MAX_HITS`, which with
``paid[w] ≥ transfers[w] − ft[w]`` caps the moves at whatever is free plus two
— the single-week solver's doctrine (eight points is as much as a gameweek may
burn) restated a gameweek at a time. That ceiling is the whole reason a plan
ever moves early: five moves do not fit into one gameweek, so the one that only
raises money goes first.

**A wildcard week rewrites all three of those, and the carry is the subtle
one.** A wildcard makes a single gameweek's transfers free and uncapped. Free
is two rows: the hit ceiling ``paid[w] ≤ MAX_HITS·(1 − wc[w])`` forces the
gameweek's hits to zero when it is wildcarded, and the pin's lower half is
relaxed to ``paid[w] ≥ transfers[w] − ft[w] − SQUAD_SIZE·wc[w]`` so that a
gameweek moving more men than it holds free is no longer made to buy hits it
does not owe — ``SQUAD_SIZE`` is a valid big-M because no gameweek ever makes
more than fifteen moves, and at ``wc = 1`` the floor drops to at most zero,
which ``paid ≥ 0`` already gives. Uncapped is one more term on the opening
gameweek's move cap, ``+ SQUAD_SIZE·wc[1]``; the later gameweeks were never
capped except through that same relaxed pin, so relaxing it uncaps them too.

The carry is where the care goes. The game does not spend a wildcard week's
free-transfer bank, and FPL's rules give the week after a wildcard (or a free
hit) no new free transfer either: whatever stood before the gameweek stands
after it, with no usual +1 — ``ft[w+1] = ft[w]`` however many men moved. The
ordinary carry reads ``ft[w+1] ≤ ft[w] − transfers[w] + paid[w] + 1``, and on a
wildcarded gameweek ``paid[w]`` is zero, so that right-hand side is
``ft[w] − transfers[w] + 1`` — short of the truth by exactly the
``transfers[w]`` the week did not really spend, and over it by the +1 it did not
earn. So the carry gains ``+ z[w] − wc[w]``, ``z`` an auxiliary standing for
``transfers[w]·wc[w]``, which adds the spend back and takes the +1 away
precisely when the gameweek is wildcarded and not otherwise. ``z``
is pinned from above alone, ``z ≤ transfers[w]`` and ``z ≤ SQUAD_SIZE·wc[w]``,
and needs no floor for the same reason ``ft`` itself needs none: it appears
only on the raise-``ft`` side of the one carry row, so the solver has every
reason to push it to the smaller of its two ceilings and none to hold it down.
At ``wc = 0`` that smaller ceiling is zero and the carry is the game's
unchanged; at ``wc = 1`` it is ``transfers[w]`` and the carry becomes
``ft[w]`` — exact at both binary points, which is the whole of what the
formulation has to be. A free-hit gameweek takes the same ``− fh[w]`` off the
carry (it moves no one, so nothing needs adding back). Both terms are written
only after a gameweek the chip may be played in, so a window holding neither
chip builds the carry row it always did.

**A free hit is priced beside the model, not inside it, and that is what
makes the revert exact.** A free hit fields a whole one-week squad chosen fresh
from the pool and hands it back: the gameweek after it owns exactly the fifteen
the gameweek before it did, whatever the free-hit week fielded. Two formulations
were on the table — a second set of selection variables for the free-hit week's
own eleven, pinned so they never reach the next gameweek's squad, or a hybrid
that prices the free-hit week to one side and lets a binary choose it. The
hybrid is taken, because it makes the revert *structural* rather than a
constraint the solver has to be trusted to honour: the free-hit squad is never a
variable of this program at all, so it cannot leak into ``squad[w+1]`` however
the branch-and-bound wanders.

The price is a per-gameweek constant, ``best_oneweek[w]`` — the value of the
best legal one-week fifteen the pool holds within the manager's budget, scored
for that gameweek with the armband and the tenth-weighted bench exactly as the
objective scores the standing squad, computed by :func:`_best_one_week_squad` in
its own small program before this one is built — for each gameweek a held free
hit may be played in, and for no other, since nothing reads it there. The
budget it is held to is the one the manager actually has: the bank plus the
sale value of the current squad, which under this module's constant-price
approximation is a constant over the window — a free hit buys a better eleven,
not an unlimited one.

A free-hit gameweek ``fh[w] = 1`` then does two things. It makes no permanent
transfers — ``moves[w] ≤ SQUAD_SIZE·(1 − fh[w])`` pins the gameweek's moves to
zero, which is the game's rule (a free hit is not a transfer window) and which is
what carries the standing squad through untouched: ``squad[w]`` equals
``squad[w−1]``, the hit pin gives ``paid[w] = 0`` off zero moves, and the free-
transfer carry is the ordinary one, less the +1 the week after a free hit does not
earn: ``ft[w+1] = ft[w]``.
So the revert needs no machinery of its own — a free-hit gameweek is a hold
gameweek as far as the squad, the bank and the free transfers are concerned. And
it *replaces that gameweek's score*: the objective already counts
``decay**(w−1)·normal_score[w]`` for the standing squad, and the free hit adds
``decay**(w−1)·(best_oneweek[w]·fh[w] − y[w] − bar[w]·fh[w])`` where ``y[w]``
stands for ``fh[w]·normal_score[w]``. When ``fh[w] = 1`` the standing score and
``y[w]`` cancel and the gameweek is worth ``best_oneweek[w] − bar[w]``; when
``fh[w] = 0`` both new terms vanish and the gameweek is the standing squad's
unchanged.

The linearization of ``y[w]`` is where the tidiness is. ``y[w] ≤ normal_score[w]``,
``y[w] ≤ best_oneweek[w]·fh[w]`` and ``y[w] ≥ normal_score[w] −
best_oneweek[w]·(1 − fh[w])`` with ``y ≥ 0`` — a standard big-M product, and its
big-M is not a guessed bound but ``best_oneweek[w]`` itself, which is genuinely
an upper bound on ``normal_score[w]``: the standing squad is a legal fifteen
inside the same budget drawn from the same pool, so its one-week score cannot
beat the best such squad's. Because ``best_oneweek[w] ≥ normal_score[w]`` always,
the free-hit gain is never negative before its bar, and at ``fh = 1`` the
three rows pin ``y = normal_score`` exactly whatever eleven the model names for
that gameweek — the choice of a free-hit gameweek's standing XI cannot change the
objective, which is right, since that XI is the one that does not play.

**The captain is continuous and lands on an integer anyway.** With the XI
fixed, ``0 ≤ captain ≤ xi`` and ``Σ captain = 1`` describe a simplex whose
vertices are single players, so the solver cannot gain by splitting an armband
and never does. Leaving it continuous keeps a fifth of the binaries out of the
branch-and-bound tree, which on a five-gameweek window is worth having.

**And a transfer nobody wants still costs a hundredth of a point.** Nothing
above prices a pointless move. A gameweek in which two men are projected the
same — a blank week where both are worth nothing at all is the common case —
can swap one for the other for free, so the window has a great many optima
worth exactly the same and comes back with whichever one the search happened
to land on. Some of them buy a player in one gameweek and sell him back in the
next, and a path printed under "the road ahead" saying so is noise wearing the
clothes of advice. So every buy in the window is charged
:data:`CHURN_EPSILON`; see there for why it can never change a decision.

One approximation from the single-week model is inherited unchanged and is
worse here than there, because it is compounded over the window: prices do
not move. Selling prices are real where the caller has them: the squad the
window starts with sells at the purchase ledger's figures
(``selling_prices``), and a player bought inside the window later sells at
his ``now_cost`` — which, under frozen prices, is exactly what he was bought
for, so pricing his resale at ``now_cost`` is the game's own rule and not an
accident. The one residual bias is a starting-squad player sold and later
re-bought inside the window: his second sale is still credited at the
ledger's price when the game would pay his (never lower) re-purchase price
back, so the window can only under-count that corner, never promise money it
does not have.
"""

from dataclasses import dataclass

import pulp

from aigaffer.chips import (
    BENCH_BOOST,
    FREE_HIT,
    TRIPLE_CAPTAIN,
    WILDCARD,
    HeldChip,
    playable_in,
)
from aigaffer.data.free_transfers import MAX_FREE_TRANSFERS
from aigaffer.data.models import Player
from aigaffer.model.xp import PlayerProjection
from aigaffer.solver.optimizer import (
    BENCH_WEIGHT,
    DEFENDER,
    FORWARD,
    GOALKEEPER,
    HIT_POINTS,
    MAX_HITS,
    MAX_PER_CLUB,
    MAX_TRANSFERS,
    MIN_XI_DEFENDERS,
    MIN_XI_FORWARDS,
    SQUAD_QUOTAS,
    SQUAD_SIZE,
    XI_GOALKEEPERS,
    XI_SIZE,
    Plan,
    _chosen,
    _grouped,
    candidate_pool,
)

# The single-week solver looks at forty a position. This one carries every
# column it builds once per gameweek in the window, so it looks at thirty.
CANDIDATES_PER_POSITION = 30

# A minute is longer than the window has ever needed and shorter than the run
# can spare. A solve that hits it comes back not-Optimal, which is None, which
# is the caller falling back on the single-week solver — never an exception.
SOLVE_SECONDS = 60
SOLVER = pulp.PULP_CBC_CMD(msg=0, timeLimit=SOLVE_SECONDS)

# A budget no squad can reach: a squad scoring itself spends nothing.
_UNBOUNDED = 10**9

# What a buy costs over and above what it costs. This is not a model of
# anything — it is a tiebreak, and it is set two orders of magnitude below the
# smallest gain any real transfer is made for. No gameweek can make more than
# MAX_FREE_TRANSFERS + MAX_HITS moves, so the whole of a six-gameweek window
# cannot spend more than forty-two hundredths of a point on this, and the
# closest decision the test suite holds the solver to is three points wide. So
# it never changes a plan that is chasing something; it only decides between
# plans worth exactly the same, and it decides for the one that leaves the
# squad alone.
CHURN_EPSILON = 0.01

# The two chips this solver plans. A bench boost turns the whole fifteen loose
# for one week — the four benched men score in full rather than at
# ``BENCH_WEIGHT`` — and a triple captain adds one more armband multiple to the
# week it is played. Both are free to play, so the only brakes on either are
# that the game hands out one of each a set — a set to play by GW19 and a fresh
# one from GW20, each held chip played at most once inside its own window — and
# the bar it has to clear.
# The strings are the game's own, shared with the rest of the codebase, and
# live in :mod:`aigaffer.chips`; the wildcard and free hit are explained here.
# A wildcard turns one gameweek's transfers free and uncapped: the squad can be
# rebuilt from scratch that week for no hit and with no ceiling on the moves.
# Unlike the bench boost and the triple captain it changes no player's score —
# it changes the transfer rules — so its value is entirely the squad the free
# rebuild reaches and the hits it does not pay, weighed against holding it.
# A free hit fields a whole one-week squad chosen fresh from the pool and then
# reverts: the gameweek after it owns exactly the fifteen the gameweek before it
# did. Unlike the other three it neither scores the standing squad differently
# nor changes its transfer rules — it replaces the standing squad for one week
# with the best legal one the budget can field, and takes it back. See the
# module docstring for how that revert is made structural rather than modelled.
NO_CHIP = "none"
_PLANNABLE_CHIPS = (BENCH_BOOST, TRIPLE_CAPTAIN, WILDCARD, FREE_HIT)

# The bars a chip is held against when no calendar priced it — the constants
# the solver used before the calendar, kept as its failure fallback and as the
# default for callers that pass no bars.
#
# A bar is what stops the model burning a chip in the best week of the next
# six when a far better week waits later in the season the horizon cannot see.
# A chip is free to play, so without a bar the model would spend it at the
# first positive opportunity; the bar is the opportunity cost of not saving it.
# A chip is planned only where its marginal xP for the week clears the bar, and
# held — played nowhere — when it does not. The value is applied decayed, the
# same week factor as the benefit it is weighed against, so within the window
# the model still chooses the best week and the bar only decides play-or-hold.
#
# These are undecayed points, tuned to FPL norms and raised from a first pass
# once the live run showed weeks clearing them on projection noise alone: only
# a genuinely exceptional gameweek should spend a chip the horizon cannot see
# the season's real opportunity for. A bench boost or a triple captain earns
# its keep on a double gameweek, and twenty-odd points is about what a strong
# one clears an ordinary week by. Flat across the season, which is exactly what
# the calendar's per-week bars improve on: these do not know that a first-set
# chip a week from its expiry has nothing left to be saved for. A constant,
# deliberately: there is no env override, so the number lives in one place and
# moves under review.
FALLBACK_BARS: dict[str, float] = {
    BENCH_BOOST: 20.0,
    TRIPLE_CAPTAIN: 18.0,
    # A wildcard is a whole free rebuild, so its bar sits far higher than a
    # boost's: a good one clears an ordinary week by thirty-odd points, and the
    # chip is worth burning only where the free uncapped squad and the hits it
    # spares beat that over the horizon. Undecayed, tuned to FPL norms, refined
    # under review like the others.
    WILDCARD: 35.0,
    # A free hit fields one exceptional week's squad and gives it back, so its
    # bar sits between a boost's and a wildcard's: worth burning on a blank or a
    # lopsided gameweek an ordinary squad cannot cover, and held otherwise.
    # Undecayed, tuned to FPL norms, refined under review like the others.
    FREE_HIT: 25.0,
}


def _solver(time_limit: int | None) -> pulp.LpSolver:
    """The module's solver, or one like it on a shorter leash.

    A caller solving the same window half a dozen times over — which is what
    :func:`~aigaffer.solver.plans.generate_plans` does — cannot give each of
    them the minute a single decision is worth. The command wrapper keeps no
    per-problem state, so a second one costs nothing to build and the silence
    is the only thing it has to inherit.
    """
    if time_limit is None:
        return SOLVER
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)


@dataclass
class PlannedMove:
    """The transfers one future gameweek of the path makes.

    Gameweeks the plan intends to leave alone have no move at all, rather than
    an empty one: a path is the list of things it means to do. ``chip`` is the
    chip the window plans to play that gameweek — ``"none"`` for almost all of
    them — and a gameweek can carry a chip while making no transfers at all, so
    a move with empty ``transfers_in``/``transfers_out`` is not empty if it
    names a chip.
    """

    event: int
    transfers_in: list[int]
    transfers_out: list[int]
    hits: int
    chip: str = "none"


@dataclass
class PlannedPath:
    """The gameweeks after the one being decided.

    ``moves`` covers the window from its second gameweek on — the first is the
    :class:`~aigaffer.solver.optimizer.Plan` itself. ``objective`` is the whole
    window's, the same number the plan carries, and ``weekly_xp`` is each
    gameweek's XI and captain undecayed — with the chip's effect folded in where
    one is played, so a bench-boost week shows its whole fifteen and a triple-
    captain week its third armband — so a reader can see where the points the
    plan is chasing actually are.

    ``week1_chip`` is the chip the window plays in the gameweek being decided,
    or ``"none"``: :class:`~aigaffer.solver.optimizer.Plan` has no chip field of
    its own, so the executed chip is surfaced here, while the advisory chip
    weeks later in the window ride their :class:`PlannedMove`.

    ``week1_freehit_squad`` and ``week1_freehit_xi`` are the temporary fifteen
    and eleven a free hit fields THIS week, when ``week1_chip`` is
    :data:`FREE_HIT` — and None otherwise. The plan's own ``squad``/``xi`` are
    the STANDING team, which is what reverts and what next week owns; on a free-
    hit week that standing team is not the one the owner takes to the deadline:
    the eleven he actually fields is surfaced here for the report to show. It is
    priced by :func:`_best_one_week_squad`, the same one-week squad the free-hit
    value was measured on, so the number in the panel and the team on the sheet
    are the same team.
    """

    moves: list[PlannedMove]
    objective: float
    weekly_xp: dict[int, float]
    week1_chip: str = "none"
    week1_freehit_squad: list[int] | None = None
    week1_freehit_xi: list[int] | None = None


def optimize_path(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    free_transfers: int,
    events: list[int],
    decay: float,
    forced_first_transfers: int | None = None,
    time_limit: int | None = None,
    held_chips: tuple[HeldChip, ...] = (),
    freehit_prices: dict[int, tuple[float, list[int], list[int]]] | None = None,
    selling_prices: dict[int, int] | None = None,
    bars: dict[str, dict[int, float]] | None = None,
) -> tuple[Plan, PlannedPath] | None:
    """The best sequence of squads over ``events``, or None.

    ``events`` are the gameweek ids of the window in order, the first of them
    the one being decided. ``bank`` and prices are in tenths of a million and
    ``free_transfers`` is the bank of free moves standing at the first
    deadline, read as the game reads it and clamped to
    :data:`~aigaffer.data.free_transfers.MAX_FREE_TRANSFERS`: the fifteen with
    which :mod:`aigaffer.solver.lineup` prices a wildcard is not a bank, and a
    window planned as though it were would be a window of illegal gameweeks.
    Drafting from an empty squad is not this solver's business — fifteen
    signings will not fit under a gameweek's move ceiling.
    A player with no projection for a gameweek is worth nothing in it, which is
    what a blank is.

    ``held_chips`` are the chips in hand, each a :class:`~aigaffer.chips.HeldChip`
    — a kind and the window of gameweeks it may still be played in, since FPL
    hands out one set to play by GW19 and a fresh one from GW20 and a first-set
    chip not played by its stop_event is gone. Two of a kind can be held at
    once, the first set's and the second's, so everything here is keyed by the
    held chip's ``id`` and never by its kind. Only the four kinds this solver
    plans — :data:`BENCH_BOOST`, :data:`TRIPLE_CAPTAIN`, :data:`WILDCARD` and
    :data:`FREE_HIT` — are read; any other is ignored. Empty, which is the
    default, is the fallback guarantee: the model built is the pre-chip one to
    the last variable, so a caller who wants chips advisory-only need only
    withhold them. A held chip becomes one binary for each window week inside
    its own window — a chip whose window misses the window entirely builds
    nothing — with at most one chip a gameweek and each held chip played at
    most once.

    ``bars`` is what each of those plays has to clear: keyed by the held
    chip's ``id``, then by gameweek id, the chip calendar's price of the better
    week a chip played here would be spent instead of. A chip is played in a
    week only where its marginal xP there beats that week's bar, and held —
    played nowhere — when no week's does. Because the bar moves week by week,
    it decides not only play-or-hold but which week: a chip a week from expiry
    has nothing beyond the window left to wait for and a bar near nothing, while
    the same chip with a double gameweek ahead of it is held. None, the default,
    reads :data:`FALLBACK_BARS` for every held chip in every week — the flat
    constants the solver used before the calendar — and so does a held chip the
    dict has no entry for, or a week its entry is missing. Even with the
    calendar the window cannot see the season whole: the bars are its only
    window onto what lies past the horizon.

    ``forced_first_transfers`` pins the opening gameweek's moves; left alone,
    the opening gameweek moves at most ``max(MAX_TRANSFERS, ft)`` times, which
    is the single-week solver's ceiling and is here for the same reason it is
    there — past three moves a manager is wildcarding — and for one more: the
    two engines' answers are ranked against each other on one number, and a
    window that outbids the other with moves the other was never allowed to
    consider is not a comparison. A caller asking for a count above the cap is
    asking a question, and gets an answer.

    ``freehit_prices`` is the one lever a caller pulls purely for speed. A free
    hit is priced by a CBC sub-solve a gameweek — the only chip that runs a
    program of its own beside the window's — and that price depends on the board,
    not on ``forced_first_transfers``: the pool, the budget and each week's points
    are the same whatever the opening move is pinned to. So a caller solving the
    same window at several opening counts can price the free hit once with
    :func:`_free_hit_prices` and hand the same dict to every solve, paying for the
    sub-solves once rather than once per count. Left None — the default, and what
    a standalone solve passes — the prices are computed here from the same inputs
    and the same helper, so the answer is identical to the point either way; the
    lever only moves where the work happens, never the result. It is read only
    when a held free hit may be played in some week of the window, and only for
    the weeks one may be played in, and is the week-index → ``(value, fifteen,
    eleven)`` mapping :func:`_free_hit_prices` returns for the same
    ``held_chips``.

    ``selling_prices`` is what each of ``current_squad``'s sales would
    actually raise, from the purchase ledger; a squad member absent from it
    sells at his ``now_cost``, the pre-ledger behaviour. It prices the sale
    side of the cash carry and the free-hit budget, and nothing else: buys
    are always at ``now_cost``, and so are the later resales of players
    bought inside the window — see the module docstring for why that is the
    game's rule under frozen prices rather than a leftover.

    None means no answer, not an error: an infeasible board, a
    ``forced_first_transfers`` the pool or the budget cannot support, or a
    solve that ran out of ``time_limit`` seconds — a minute by default — with
    nothing to show for it. The caller falls back on the single-week solver,
    which is the doctrine everywhere else too.
    """
    if not events:
        return None

    opening_bank = min(max(free_transfers, 0), MAX_FREE_TRANSFERS)
    pool = candidate_pool(
        players, projections, current_squad, limit=CANDIDATES_PER_POSITION
    )
    current = {pid for pid in current_squad if pid in players}
    weeks = list(range(1, len(events) + 1))
    points = {
        (p, w): _projected(projections, p, events[w - 1]) for p in pool for w in weeks
    }
    owned = {p: 1 if p in current else 0 for p in pool}
    # What a sale of each man credits the bank. The starting squad sells at the
    # ledger's prices; everyone else can only be in the squad because the
    # window bought him at now_cost, which is what his later sale pays back.
    sale = selling_prices or {}
    proceeds = {
        p: sale.get(p, players[p].now_cost)
        if p in current
        else players[p].now_cost
        for p in pool
    }

    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)

    # The chips this window may play: the held ones of a kind this solver plans
    # that at least one window week may play, each with the window weeks inside
    # its own window. Two of a kind can be held (each set's), so from here on
    # everything chip-shaped is keyed by the held chip's id; ``chips`` is only
    # the kinds among them, which is what decides the auxiliaries each kind
    # needs. A held chip no window week may play builds nothing, and an empty
    # ``held_chips`` builds exactly the pre-chip model.
    held = [
        chip
        for chip in playable_in(held_chips, events)
        if chip.chip in _PLANNABLE_CHIPS
    ]
    eligible = {
        chip.id: [w for w in weeks if chip.allows(events[w - 1])] for chip in held
    }
    chips = [
        kind for kind in _PLANNABLE_CHIPS if any(chip.chip == kind for chip in held)
    ]
    # The window weeks each kind may be played in — the union of its held
    # chips' eligible weeks, ascending, and empty for a kind not held. A week
    # outside it reads the kind's binary as a constant zero, so whatever the
    # kind would add there — an auxiliary, its pins, its objective term, the
    # free hit's price — could only ever be pinned to nothing, and none of it is
    # built. A window inside one half is every week of it, and builds what it
    # always did; one that straddles GW19/20 holding only one set's chips skips
    # the other half's weeks.
    chip_weeks = {
        kind: sorted(
            {w for chip in held if chip.chip == kind for w in eligible[chip.id]}
        )
        for kind in _PLANNABLE_CHIPS
    }

    # A free hit's value each gameweek is priced to one side, as a constant, so
    # its squad is never a variable of this program and cannot reach the next
    # gameweek's fifteen — the revert is structural, not a constraint. The budget
    # is the one the manager holds: the bank plus the sale value of the current
    # squad, a constant over the window under this module's constant-price
    # approximation. A gameweek whose best legal one-week squad the budget cannot
    # even field is no window at all, which is None like any other infeasibility
    # — but only a gameweek a held free hit may be played in is priced at all, so
    # a week of the other half can neither cost a sub-solve nor sink the window.
    best_oneweek: dict[int, float] = {}
    # The fifteen and eleven each week's free-hit price was measured on, kept so
    # that the week the window actually plays a free hit can field the team it
    # was scored for rather than a re-derivation of it. Only the objective needs
    # the value; only the report needs the squad, and only for the played week.
    best_oneweek_squad: dict[int, tuple[list[int], list[int]]] = {}
    if FREE_HIT in chips:
        # Priced here only when a caller has not priced it already: a sweep hands
        # the same dict to every opening count so the sub-solves run once, and a
        # standalone solve computes it from the very same inputs. Either way None
        # is a free-hit week the budget cannot field, which is no window at all.
        if freehit_prices is None:
            freehit_prices = _free_hit_prices(
                players, projections, current_squad, bank, events, time_limit,
                selling_prices=selling_prices, held_chips=held,
            )
        if freehit_prices is None:
            return None
        for w in chip_weeks[FREE_HIT]:
            value, fh_squad, fh_xi = freehit_prices[w]
            best_oneweek[w] = value
            best_oneweek_squad[w] = (fh_squad, fh_xi)

    problem = pulp.LpProblem("aigaffer_transfer_path", pulp.LpMaximize)

    def per_week(
        name: str, among: list[int] = weeks, **bounds
    ) -> dict[int, dict[int, pulp.LpVariable]]:
        """One variable a player a gameweek — every gameweek, or only those
        ``among`` names — built in a fixed order so that the same board always
        writes the same model file."""
        return {
            w: problem.add_variable_dicts(f"{name}{w}", pool, **bounds) for w in among
        }

    squad = per_week("squad", cat=pulp.LpBinary)
    starting = per_week("xi", cat=pulp.LpBinary)
    # Continuous on purpose — see the module docstring.
    captain = per_week("captain", lowBound=0, upBound=1)
    buy = per_week("buy", cat=pulp.LpBinary)
    sell = per_week("sell", cat=pulp.LpBinary)
    # Integer because hits are whole points off a whole gameweek, and because
    # a handful of variables cost nothing to branch on.
    paid = {
        w: problem.add_variable(
            f"paid{w}", lowBound=0, upBound=MAX_HITS, cat=pulp.LpInteger
        )
        for w in weeks
    }
    # owing[w] = 1 when the gameweek's moves outran its free transfers. It is
    # what pins paid to the max() it stands for; see the module docstring.
    owing = {w: problem.add_variable(f"owing{w}", cat=pulp.LpBinary) for w in weeks}
    # Big enough for both halves of the pin: the middle row has to cover
    # ``banked[w] - moves[w]`` in a gameweek that owes nothing, and the last one
    # ``paid[w]`` in a gameweek that owes something. The first of those is why
    # the opening bank is clamped at the top of this function — a caller's
    # fifteen free transfers, taken at face value, would stand above this M and
    # the pin would quietly stop pinning.
    big_m = MAX_HITS + MAX_FREE_TRANSFERS
    # ft[1] is what the manager holds, a constant; the rest are carried.
    banked: dict[int, float | pulp.LpVariable] = {1: opening_bank}
    banked.update(
        {
            w: problem.add_variable(f"ft{w}", lowBound=0, upBound=MAX_FREE_TRANSFERS)
            for w in weeks[1:]
        }
    )
    # Prices are whole tenths and buy/sell are binary, so the bank lands on an
    # integer without being told to.
    cash = {w: problem.add_variable(f"bank{w}", lowBound=0) for w in weeks}
    moves = {w: pulp.lpSum(buy[w][p] for p in pool) for w in weeks}

    # Chips: one binary per held chip per window week inside its window — two
    # of a kind can be held (each half's), so everything here is keyed by the
    # held chip's id. A held chip no window week may play builds nothing, and
    # an empty ``held_chips`` builds exactly the pre-chip model.
    play = {
        chip.id: {
            w: problem.add_variable(
                f"{chip.chip}{chip.stop_event}_{w}", cat=pulp.LpBinary
            )
            for w in eligible[chip.id]
        }
        for chip in held
    }

    def on(kind: str, w: int):
        """Whether ``kind`` is played in week ``w``: the sum of its held chips'
        binaries there. A kind's windows do not overlap, so at most one term —
        and none at all in a week no held chip of the kind may play, where the
        sum is a constant zero and every row it pins reads as it would without
        the chip."""
        return pulp.lpSum(
            play[chip.id][w]
            for chip in held
            if chip.chip == kind and w in play[chip.id]
        )

    def bar(chip: HeldChip, w: int) -> float:
        """What playing ``chip`` in week ``w`` must clear — the calendar's
        per-week opportunity cost, or the fallback constant where the calendar
        has no word for this chip or this week."""
        priced = (bars or {}).get(chip.id, {})
        return priced.get(events[w - 1], FALLBACK_BARS[chip.chip])

    def bars_paid(kind: str, w: int):
        """The bar ``kind`` pays in week ``w``: its held chip's bar there,
        times the binary that says it was played."""
        return pulp.lpSum(
            bar(chip, w) * play[chip.id][w]
            for chip in held
            if chip.chip == kind and w in play[chip.id]
        )

    # The bilinear terms a chip adds are linearized by an auxiliary pinned to
    # the product with the big-M pair added in the gameweek loop: ``z_bb`` is
    # ``(squad - xi)·bb``, the bench at full weight for the boosted week, and
    # ``z_tc`` is ``captain·tc``, the third armband multiple. Both factors sit in
    # [0, 1], so the bound is exact rather than a relaxation. Each is built only
    # in the weeks its kind may be played in (``chip_weeks``), and so is every
    # auxiliary below: elsewhere the chip's factor is zero and so is the
    # product, with nothing to linearize. A kind not held builds none at all.
    z_bb = per_week("zbb", chip_weeks[BENCH_BOOST], lowBound=0, upBound=1)
    z_tc = per_week("ztc", chip_weeks[TRIPLE_CAPTAIN], lowBound=0, upBound=1)
    # One auxiliary a gameweek, not one a player: ``z_wc[w]`` is ``moves[w]·wc[w]``,
    # the free transfers a wildcarded gameweek did not really spend, added back to
    # the next gameweek's carry. Pinned from above only in the gameweek loop; the
    # module docstring says why that is exact.
    z_wc = {
        w: problem.add_variable(f"zwc{w}", lowBound=0) for w in chip_weeks[WILDCARD]
    }
    # One auxiliary a gameweek: ``y_fh[w]`` is ``fh[w]·normal_score[w]``, the
    # standing squad's own score for the gameweek, which a free hit takes off so
    # that the constant free-hit score can replace it. Pinned by the usual big-M
    # pair in the gameweek loop, its big-M being ``best_oneweek[w]`` itself — a
    # true upper bound on the standing score, so the pin is exact.
    y_fh = {
        w: problem.add_variable(f"yfh{w}", lowBound=0) for w in chip_weeks[FREE_HIT]
    }

    objective = (
        pulp.lpSum(
            decay ** (w - 1)
            * pulp.lpSum(
                points[p, w]
                * (
                    starting[w][p]
                    + captain[w][p]
                    + BENCH_WEIGHT * (squad[w][p] - starting[w][p])
                )
                for p in pool
            )
            for w in weeks
        )
        - HIT_POINTS * pulp.lpSum(paid[w] for w in weeks)
        - CHURN_EPSILON * pulp.lpSum(moves[w] for w in weeks)
    )
    if BENCH_BOOST in chips:
        # The 0.9 the bench was not already scoring, less the bar the boosted
        # week has to clear before the chip is worth playing at all.
        objective += pulp.lpSum(
            decay ** (w - 1)
            * (
                (1 - BENCH_WEIGHT)
                * pulp.lpSum(points[p, w] * z_bb[w][p] for p in pool)
                - bars_paid(BENCH_BOOST, w)
            )
            for w in chip_weeks[BENCH_BOOST]
        )
    if TRIPLE_CAPTAIN in chips:
        # One extra captain multiple, less the same kind of bar.
        objective += pulp.lpSum(
            decay ** (w - 1)
            * (
                pulp.lpSum(points[p, w] * z_tc[w][p] for p in pool)
                - bars_paid(TRIPLE_CAPTAIN, w)
            )
            for w in chip_weeks[TRIPLE_CAPTAIN]
        )
    if WILDCARD in chips:
        # The wildcard adds no scoring term of its own — its gain is the better
        # squad the free uncapped rebuild reaches and the hits it spares, both of
        # which the objective already counts. So only its bar goes in, and the
        # chip is played only where that endogenous gain clears it.
        objective -= pulp.lpSum(
            decay ** (w - 1) * bars_paid(WILDCARD, w) for w in chip_weeks[WILDCARD]
        )
    if FREE_HIT in chips:
        # The free-hit gameweek is worth its best one-week squad in place of the
        # standing one: the objective already counts the standing score, so the
        # chip adds the constant ``best_oneweek``, takes the standing score back
        # off through ``y_fh``, and pays its bar. Where ``fh[w] = 1`` the two
        # standing terms cancel and the gameweek is worth ``best_oneweek − bar``;
        # where ``fh[w] = 0`` both new terms are zero and nothing changes.
        objective += pulp.lpSum(
            decay ** (w - 1)
            * (
                best_oneweek[w] * on(FREE_HIT, w)
                - y_fh[w]
                - bars_paid(FREE_HIT, w)
            )
            for w in chip_weeks[FREE_HIT]
        )
    problem += objective

    for w in weeks:
        before = {p: owned[p] if w == 1 else squad[w - 1][p] for p in pool}

        problem += pulp.lpSum(squad[w][p] for p in pool) == SQUAD_SIZE
        for position, quota in SQUAD_QUOTAS.items():
            problem += pulp.lpSum(squad[w][p] for p in by_position[position]) == quota
        for club_mates in by_club.values():
            problem += pulp.lpSum(squad[w][p] for p in club_mates) <= MAX_PER_CLUB

        problem += pulp.lpSum(starting[w][p] for p in pool) == XI_SIZE
        for p in pool:
            problem += starting[w][p] <= squad[w][p]
            problem += captain[w][p] <= starting[w][p]
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[GOALKEEPER])
            == XI_GOALKEEPERS
        )
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[DEFENDER])
            >= MIN_XI_DEFENDERS
        )
        problem += (
            pulp.lpSum(starting[w][p] for p in by_position[FORWARD]) >= MIN_XI_FORWARDS
        )
        problem += pulp.lpSum(captain[w][p] for p in pool) == 1

        for p in pool:
            problem += squad[w][p] == before[p] + buy[w][p] - sell[w][p]
            # Nobody is bought who is already here and nobody sold who is not,
            # which between them already imply that nobody is both, fractions
            # included — so the third row is redundant and stays anyway: it is
            # worth a third off the solve time, which a five-gameweek window
            # has no business turning down.
            problem += buy[w][p] + sell[w][p] <= 1
            problem += buy[w][p] <= 1 - before[p]
            problem += sell[w][p] <= before[p]

        problem += cash[w] == (bank if w == 1 else cash[w - 1]) + pulp.lpSum(
            proceeds[p] * sell[w][p] - players[p].now_cost * buy[w][p] for p in pool
        )

        # The hit pin, wildcarded. On an ordinary gameweek both rows below read
        # exactly as they did before chips: the floor is ``moves − ft`` and the
        # ceiling is the variable's own ``MAX_HITS``. A wildcarded gameweek drops
        # the floor to zero or below and the ceiling to zero, so it charges no
        # hits however many men it moves — free transfers, made linear. A
        # gameweek no held wildcard may be played in is an ordinary one by
        # construction, and is written as one.
        floor = moves[w] - banked[w]
        if w in z_wc:
            floor = floor - SQUAD_SIZE * on(WILDCARD, w)
            problem += paid[w] <= MAX_HITS * (1 - on(WILDCARD, w))
        problem += paid[w] >= floor
        problem += paid[w] <= moves[w] - banked[w] + big_m * (1 - owing[w])
        problem += paid[w] <= big_m * owing[w]
        if w > 1:
            # The carry gains the wildcard add-back, and loses the week's +1,
            # only after a gameweek a wildcard (or free hit) may be played in:
            # FPL's rules give no new free transfer the week after either chip.
            # Otherwise the row is the pre-chip one, term for term.
            carry = banked[w - 1] - moves[w - 1] + paid[w - 1] + 1
            if w - 1 in z_wc:
                carry = carry + z_wc[w - 1] - on(WILDCARD, w - 1)
            if w - 1 in y_fh:
                carry = carry - on(FREE_HIT, w - 1)
            problem += banked[w] <= carry

        # At most one chip a gameweek, written only where two could clash (a
        # single chip cannot break its own binary).
        here = [play[chip.id][w] for chip in held if w in play[chip.id]]
        if len(here) > 1:
            problem += pulp.lpSum(here) <= 1
        if w in z_bb:
            bb = on(BENCH_BOOST, w)
            for p in pool:
                diff = squad[w][p] - starting[w][p]
                problem += z_bb[w][p] <= diff
                problem += z_bb[w][p] <= bb
                problem += z_bb[w][p] >= diff - (1 - bb)
        if w in z_tc:
            tc = on(TRIPLE_CAPTAIN, w)
            for p in pool:
                problem += z_tc[w][p] <= captain[w][p]
                problem += z_tc[w][p] <= tc
                problem += z_tc[w][p] >= captain[w][p] - (1 - tc)
        if w in z_wc:
            # The two ceilings on the carry's add-back: it cannot exceed the
            # gameweek's moves and it vanishes off a non-wildcarded gameweek. No
            # floor — the carry pushes it to whichever is smaller on its own.
            problem += z_wc[w] <= moves[w]
            problem += z_wc[w] <= SQUAD_SIZE * on(WILDCARD, w)
        if w in y_fh:
            fh = on(FREE_HIT, w)
            # A free hit makes no permanent transfers — that is the game's rule
            # and the whole of the revert: with the gameweek's moves pinned to
            # zero the standing squad carries through untouched, the hit pin
            # gives no hits off zero moves, and the free-transfer carry is the
            # ordinary one. So a free-hit gameweek is a hold gameweek here.
            problem += moves[w] <= SQUAD_SIZE * (1 - fh)
            # The big-M pin on ``y_fh[w] = fh·normal_score``. Its big-M is the
            # gameweek's own ``best_oneweek``, a true upper bound on the standing
            # score, so at ``fh = 1`` the pair forces ``y_fh = normal_score`` and
            # at ``fh = 0`` forces ``y_fh = 0``.
            normal_score = pulp.lpSum(
                points[p, w]
                * (
                    starting[w][p]
                    + captain[w][p]
                    + BENCH_WEIGHT * (squad[w][p] - starting[w][p])
                )
                for p in pool
            )
            problem += y_fh[w] <= normal_score
            problem += y_fh[w] <= best_oneweek[w] * fh
            problem += y_fh[w] >= normal_score - best_oneweek[w] * (1 - fh)

    # Each held chip is the game's once-a-set, so once across the window too
    # (a chip with a single eligible week cannot be played twice there).
    for chip in held:
        if len(play[chip.id]) > 1:
            problem += pulp.lpSum(play[chip.id].values()) <= 1

    # No free hit the gameweek after a free hit — the official rule (use the
    # first in GW19 and the second may not follow in GW20). ``held_chips``
    # already opens a second-set window a week late when the history shows the
    # first played just before it; this is the same rule for the plan's own
    # plays, where both free hits are still in hand. Windows of one kind never
    # overlap, so only the earlier-window chip can come first, but both orders
    # are written so the rule does not lean on that. Rows exist only where
    # they can bind: two free hits held, each with a week beside the other's
    # in consecutive gameweeks. Anything else builds the model it always did.
    free_hits = [chip for chip in held if chip.chip == FREE_HIT]
    for first in free_hits:
        for second in free_hits:
            if first.id == second.id:
                continue
            for w in play[first.id]:
                if w + 1 in play[second.id] and events[w] == events[w - 1] + 1:
                    problem += play[first.id][w] + play[second.id][w + 1] <= 1

    if forced_first_transfers is None:
        # The opening cap, lifted for a wildcarded first gameweek: fifteen is the
        # most any gameweek can move, so the term uncaps it without unbounding it.
        cap = max(MAX_TRANSFERS, opening_bank)
        if 1 in z_wc:
            problem += moves[1] <= cap + SQUAD_SIZE * on(WILDCARD, 1)
        else:
            problem += moves[1] <= cap
    else:
        problem += moves[1] == forced_first_transfers

    status = problem.solve(_solver(time_limit))
    # CBC stopped on its time limit with a squad in hand reports as Optimal
    # here, and that is the answer we want: a plan the solver could not prove
    # best is still a plan, and the alternative is the single-week solver.
    # A timeout with nothing found reports as not-solved, and that is None.
    if pulp.LpStatus[status] != "Optimal":
        return None

    # The objective is added back up from the solution rather than read off the
    # solver, so that the number a report prints is the one its own squads earn
    # — less the churn tiebreak, which is added back up here too, because a
    # plan ought to be ranked on the number it was chosen by.
    objective = 0.0
    hits = 0
    bought = 0
    weekly_xp: dict[int, float] = {}
    path_moves: list[PlannedMove] = []
    opening: list[int] = []
    opening_xi: list[int] = []
    opening_hits = 0
    opening_chip = "none"

    for w in weeks:
        event = events[w - 1]
        chosen = _chosen(squad[w])
        eleven = _chosen(starting[w])
        bench = set(chosen) - set(eleven)
        incoming = _chosen(buy[w])
        outgoing = _chosen(sell[w])

        started = sum(points[p, w] for p in eleven)
        armband = sum(points[p, w] * (captain[w][p].value() or 0.0) for p in pool)
        benched = sum(points[p, w] for p in bench)
        # paid is integral by declaration; round() only clears solver dust.
        taken = round(paid[w].value() or 0.0)

        # Which held chip this gameweek plays, if any — at most one, by
        # construction — and its kind, which is what the path reports.
        played_here = next(
            (
                chip
                for chip in held
                if w in play[chip.id] and (play[chip.id][w].value() or 0) > 0.5
            ),
            None,
        )
        chip = played_here.chip if played_here else "none"
        bb_on, tc_on, wc_on, fh_on = (
            chip == kind for kind in (BENCH_BOOST, TRIPLE_CAPTAIN, WILDCARD, FREE_HIT)
        )

        # weekly_xp is the points the gameweek actually earns, chip and all: a
        # boosted week's whole bench, a tripled week's third armband. A wildcard
        # changes no score, only the transfers, so it touches weekly_xp not at
        # all. The objective carries the same, less the bar the played chip
        # cleared that week — the number the plan was chosen by, exactly as it
        # prices the churn below.
        week_xp = started + armband
        week_score = started + armband + BENCH_WEIGHT * benched
        if bb_on:
            week_xp += benched
            week_score += (1 - BENCH_WEIGHT) * benched - bar(played_here, w)
        if tc_on:
            week_xp += armband
            week_score += armband - bar(played_here, w)
        if wc_on:
            week_score -= bar(played_here, w)
        if fh_on:
            # A free hit fields its best one-week squad in place of the standing
            # one, so the gameweek earns that squad's whole score — the standing
            # squad's own points do not count at all — less the bar. It
            # made no transfers, so nothing is added to the hit or churn totals.
            week_xp = best_oneweek[w]
            week_score = best_oneweek[w] - bar(played_here, w)

        weekly_xp[event] = week_xp
        objective += decay ** (w - 1) * week_score
        hits += taken
        bought += len(incoming)

        if w == 1:
            opening, opening_xi, opening_hits, opening_chip = (
                chosen, eleven, taken, chip
            )
            continue
        # A gameweek with a chip but no transfers is still a gameweek that does
        # something, so it earns a move that names the chip and moves no one.
        if incoming or outgoing or chip != "none":
            path_moves.append(
                PlannedMove(
                    event=event,
                    transfers_in=incoming,
                    transfers_out=outgoing,
                    hits=taken,
                    chip=chip,
                )
            )

    objective -= HIT_POINTS * hits + CHURN_EPSILON * bought
    # A free hit played this week fields a temporary team the plan's standing
    # squad is not: surface the fifteen and eleven it was priced on so the
    # report can show them. Any other opening chip leaves these None — the
    # standing squad is the one that plays.
    freehit_squad, freehit_xi = (
        best_oneweek_squad[1] if opening_chip == FREE_HIT else (None, None)
    )
    path = PlannedPath(
        moves=path_moves,
        objective=objective,
        weekly_xp=weekly_xp,
        week1_chip=opening_chip,
        week1_freehit_squad=freehit_squad,
        week1_freehit_xi=freehit_xi,
    )
    plan = Plan(
        squad=opening,
        xi=opening_xi,
        transfers_in=sorted(set(opening) - current),
        transfers_out=sorted(current - set(opening)),
        hits=opening_hits,
        # The plan is ranked against single-week plans on ``objective``, so that
        # stays the window's own number — hits, churn tiebreak and all;
        # ``xp_total`` is it with the hits added back, which is what the field
        # means everywhere else, to within the hundredth of a point a transfer
        # that no printed figure is quoted closely enough to show.
        xp_total=objective + HIT_POINTS * hits,
        objective=objective,
        path=path,
    )
    return plan, path


def _projected(
    projections: dict[int, PlayerProjection], player_id: int, event: int
) -> float:
    """What a player is projected for one gameweek, undecayed.

    A player with no projection at all, or none for this gameweek, is worth
    nothing in it: a blank gameweek and a missing model are the same fact as
    far as the squad is concerned. The decay belongs to the objective, which is
    why this reads ``per_gw`` and never ``total``.
    """
    projection = projections.get(player_id)
    return projection.per_gw.get(event, 0.0) if projection else 0.0


def _one_week_market(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    selling_prices: dict[int, int] | None,
) -> tuple[list[int], dict[int, list[int]], dict[int, list[int]], int]:
    """What a one-week fifteen is bought from: the pool, it grouped by position
    and by club, and the budget — as ``(pool, by_position, by_club, budget)``.

    Shared by the free hit's pricing and the chip calendar's, which are the
    same sub-solve asked different questions, so that the two can never be
    pricing off different markets. The pool is the window's own, cut by
    :data:`CANDIDATES_PER_POSITION`. The budget is the one a free hit really
    grants: the bank plus what the squad would actually sell for — the
    ledger's prices where the caller has them, the listed price where it does
    not. A riser's paper value is not money.
    """
    pool = candidate_pool(
        players, projections, current_squad, limit=CANDIDATES_PER_POSITION
    )
    current = {pid for pid in current_squad if pid in players}
    by_position = _grouped(pool, lambda p: players[p].element_type)
    by_club = _grouped(pool, lambda p: players[p].team)
    sale = selling_prices or {}
    budget = bank + sum(sale.get(p, players[p].now_cost) for p in current)
    return pool, by_position, by_club, budget


def _free_hit_prices(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    events: list[int],
    time_limit: int | None,
    selling_prices: dict[int, int] | None = None,
    held_chips: tuple[HeldChip, ...] | None = None,
) -> dict[int, tuple[float, list[int], list[int]]] | None:
    """Each window gameweek's free-hit price — value, fifteen and eleven — for
    the gameweeks a held free hit may be played in.

    A free hit's worth in a gameweek is the best legal one-week squad the pool
    holds inside the manager's budget, scored for that week by
    :func:`_best_one_week_squad`. This prices it for the gameweeks in ``events``
    that some free hit in ``held_chips`` allows, keyed by the one-based week
    index the window program uses. Any other gameweek is left out: no free hit
    can be played there, so :func:`optimize_path` builds no free-hit term there
    and has no use for its price, and a sub-solve for it would be time spent on
    nothing — or, were that week the one the budget cannot field, a window
    thrown away over a week it never needed priced. None, the default, prices
    every gameweek, which is what a whole-season free hit would ask for.

    It is factored out because it is the one part of :func:`optimize_path` a
    forced opening-move count never touches: the pool, the budget and each week's
    points are the same whatever the sweep pins the first gameweek to. A caller
    solving the window at several opening counts — which
    :func:`~aigaffer.solver.plans.generate_plans` does across half a dozen — can
    therefore price the free hit once here and hand the same result to every
    solve, paying for the CBC sub-solves once rather than once per count.
    :func:`optimize_path` calls this itself when it is handed nothing, off the
    same pool, budget and held chips, so a standalone solve is unchanged and
    the prices a sweep hoists are identical to the ones each solve would have
    computed.

    None when any priced gameweek's best one-week squad cannot be fielded
    inside the budget — the same infeasibility :func:`optimize_path` returns
    None on.
    """
    pool, by_position, by_club, budget = _one_week_market(
        players, projections, current_squad, bank, selling_prices
    )
    weeks = [
        w
        for w in range(1, len(events) + 1)
        if held_chips is None
        or any(
            chip.chip == FREE_HIT and chip.allows(events[w - 1]) for chip in held_chips
        )
    ]
    # One command wrapper for the lot: it keeps no per-problem state, so pricing
    # every week through the same object is the same solve run several times.
    solver = _solver(time_limit)
    prices: dict[int, tuple[float, list[int], list[int]]] = {}
    for w in weeks:
        week_points = {p: _projected(projections, p, events[w - 1]) for p in pool}
        priced = _best_one_week_squad(
            pool, players, by_position, by_club, week_points, budget, solver
        )
        if priced is None:
            return None
        prices[w] = priced
    return prices


def best_one_week_squads(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    current_squad: list[int],
    bank: int,
    events: list[int],
    *,
    bench_weight: float = BENCH_WEIGHT,
    selling_prices: dict[int, int] | None = None,
    time_limit: int | None = None,
) -> dict[int, tuple[float, list[int], list[int]] | None]:
    """The best legal one-week fifteen for each gameweek in ``events``.

    The chip calendar's pricing, kept beside the free hit's because it is the
    same sub-solve: the same pool, the same budget (the bank plus what the
    squad actually sells for), the same legality — one market,
    :func:`_one_week_market`, for both. Two differences. It is keyed by
    gameweek id, since the calendar works in gameweeks and has no window
    index; and a week the budget cannot field is None rather than the end of
    the answer, since one unpriceable week is a week the calendar skips, not a
    calendar it cannot draw.
    """
    pool, by_position, by_club, budget = _one_week_market(
        players, projections, current_squad, bank, selling_prices
    )
    solver = _solver(time_limit)
    return {
        event: _best_one_week_squad(
            pool,
            players,
            by_position,
            by_club,
            {p: _projected(projections, p, event) for p in pool},
            budget,
            solver,
            bench_weight=bench_weight,
        )
        for event in events
    }


def squad_one_week(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    squad: list[int],
    event: int,
) -> tuple[float, list[int], list[int]]:
    """How ``squad`` lines up in ``event`` on the free-hit basis: its best legal
    eleven, the armband, and the bench at :data:`BENCH_WEIGHT` — as
    ``(value, squad, xi)``.

    The baseline a free hit is measured against when the calendar prices one.
    It goes through :func:`_best_one_week_squad` with the pool cut to the
    squad itself and no budget, so the fifteen is forced and only the eleven
    and the armband are chosen — on the same basis the free-hit side was
    priced on, which :func:`~aigaffer.solver.lineup.pick_lineup` (it captains
    on attacking EV) is not. ``(0.0, squad, [])`` if the squad cannot field a
    legal eleven.
    """
    pool = [pid for pid in squad if pid in players]
    priced = _best_one_week_squad(
        pool,
        players,
        _grouped(pool, lambda p: players[p].element_type),
        _grouped(pool, lambda p: players[p].team),
        {p: _projected(projections, p, event) for p in pool},
        _UNBOUNDED,
        SOLVER,
    )
    return priced if priced is not None else (0.0, list(squad), [])


def squad_one_week_score(
    players: dict[int, Player],
    projections: dict[int, PlayerProjection],
    squad: list[int],
    event: int,
) -> float:
    """:func:`squad_one_week`'s score alone — the baseline a free hit is
    measured against."""
    return squad_one_week(players, projections, squad, event)[0]


def _best_one_week_squad(
    pool: list[int],
    players: dict[int, Player],
    by_position: dict[int, list[int]],
    by_club: dict[int, list[int]],
    week_points: dict[int, float],
    budget: int,
    solver: pulp.LpSolver,
    bench_weight: float = BENCH_WEIGHT,
) -> tuple[float, list[int], list[int]] | None:
    """The best legal one-week fifteen the pool holds within ``budget``.

    This is the price of a free hit for one gameweek, computed to one side of the
    window program and handed in as a constant: the whole point of the hybrid
    free-hit formulation is that the free-hit squad is never a variable of the
    window itself, so it cannot leak into the next gameweek's squad. See the
    module docstring.

    The fifteen is chosen fresh — no current squad, no transfers, no hits, a free
    hit costs none of those — subject to the game's own legality: the 2/5/5/3
    quota, three to a club, an eleven of one keeper and at least three defenders
    and a forward, and a total price inside ``budget``, which is the bank plus the
    sale value of the squad the manager holds. It is scored for the one gameweek
    with the armband doubled and the bench at :data:`BENCH_WEIGHT`, exactly as the
    window scores the standing squad, so the two are comparable to the point.

    ``bench_weight`` is how much a bench point is worth: the window's own
    :data:`BENCH_WEIGHT` for a free hit, and 1 when the chip calendar prices a
    bench boost — a bench that will actually score.

    Returns ``(value, squad, xi)`` — the score, the fifteen and the eleven — or
    None if no legal fifteen fits the budget, which the caller treats as it
    treats any infeasible window: no plan, fall back. ``squad`` and ``xi`` are
    returned for the caller that wants to show the free-hit eleven; the window
    objective needs only ``value``.
    """
    problem = pulp.LpProblem("aigaffer_free_hit_week", pulp.LpMaximize)
    squad = problem.add_variable_dicts("fhsquad", pool, cat=pulp.LpBinary)
    starting = problem.add_variable_dicts("fhxi", pool, cat=pulp.LpBinary)
    # Continuous for the same reason the window's captain is — the XI fixed, the
    # armband lands on a single man without being made a binary.
    captain = problem.add_variable_dicts("fhcap", pool, lowBound=0, upBound=1)

    problem += pulp.lpSum(
        week_points[p]
        * (starting[p] + captain[p] + bench_weight * (squad[p] - starting[p]))
        for p in pool
    )

    problem += pulp.lpSum(squad.values()) == SQUAD_SIZE
    for position, quota in SQUAD_QUOTAS.items():
        problem += pulp.lpSum(squad[p] for p in by_position[position]) == quota
    for club_mates in by_club.values():
        problem += pulp.lpSum(squad[p] for p in club_mates) <= MAX_PER_CLUB
    problem += pulp.lpSum(players[p].now_cost * squad[p] for p in pool) <= budget

    problem += pulp.lpSum(starting.values()) == XI_SIZE
    for p in pool:
        problem += starting[p] <= squad[p]
        problem += captain[p] <= starting[p]
    problem += (
        pulp.lpSum(starting[p] for p in by_position[GOALKEEPER]) == XI_GOALKEEPERS
    )
    problem += (
        pulp.lpSum(starting[p] for p in by_position[DEFENDER]) >= MIN_XI_DEFENDERS
    )
    problem += pulp.lpSum(starting[p] for p in by_position[FORWARD]) >= MIN_XI_FORWARDS
    problem += pulp.lpSum(captain.values()) == 1

    if pulp.LpStatus[problem.solve(solver)] != "Optimal":
        return None

    chosen = _chosen(squad)
    eleven = _chosen(starting)
    bench = set(chosen) - set(eleven)
    armband = sum(week_points[p] * (captain[p].value() or 0.0) for p in pool)
    value = (
        sum(week_points[p] for p in eleven)
        + armband
        + bench_weight * sum(week_points[p] for p in bench)
    )
    return value, chosen, eleven
