"""How likely a player is to play, and for how long.

Expected points start here: nearly every scoring event is proportional to
time on the pitch, so a good minutes estimate matters more than a clever
points model. We take the player's recent minutes as the best guess at his
role and scale it by the chance he features at all.

Recent minutes are the right answer once there are some. Early in a season
there are not, and a mean over one gameweek is not evidence of a role — it is
a single Saturday, and a manager who rotates his squad in August has not
dropped anybody. So under :data:`BLEND_GAMEWEEKS` the history is a floor
rather than the whole answer: what the season's ``starts`` count says about
him wins if it is higher, and by the third gameweek the history has earned the
right to speak for itself.

The history handed in is the gameweeks that have been **played**, and this
module trusts its caller for that. The distinction is not pedantry: the API
adds a history row the moment a deadline goes, so between a Friday deadline
and a Sunday kickoff a fit player's season reads as a 0-minute gameweek he
never had. Averaging those in is what wrote off every premium in GW1 of
2026-27 — the first week the gaffer ran live, the solver benched a fit
Haaland, and the gaffer spent his week overriding seven players by hand. The
cause was rows for matches nobody had played, and the cure is in
:func:`aigaffer.orchestrator._played`, which drops them at the fetch. The
contract here stays what it says: the mean of the entries given.

That leaves the blend the job it is actually good for — a thin history of
played gameweeks — and left, for a while, one limitation. With the phantom
rows gone, the opening weekend has no history at all, so every player fell
back on ``starts``; and ``starts`` is season-to-date like everything else in
the bootstrap, so in GW1 it is zero for everybody. The floor under a £14.5m
striker was then the bench estimate: twenty minutes of a man who will play
ninety.

Last season is the fix, and it *is* in the payload — one row a season in
element-summary's ``history_past``, with the season's minutes on it. Divided
by a season's gameweeks it is what he actually played a week the last time
anybody watched, which is a measurement rather than a guess, and
:func:`season_prior` is where the division happens. So the floor under a thin
August is his own last season where there is one, and the 75/20 guess only
where there is not: a signing from abroad, a promoted club's player, a
teenager. Those are still understated in GW1 and the gaffer's own minute
overrides are still the mitigation for them.

Two things the prior is not. It is not a ceiling — three gameweeks of this
season still speak for themselves, and two good ones still beat it. And it is
not a claim about a role that has changed: a player sold into a smaller part,
or bought into a bigger one, is described by last season for as long as it
takes this one to reach :data:`BLEND_GAMEWEEKS`, which is three weeks.
"""

from aigaffer.data.models import GwHistory, PastSeason, Player

# Statuses that rule a player out: injured, suspended, unavailable (left the
# league), not eligible. 'a' is available and 'd' is doubtful.
RULED_OUT_STATUSES = ("i", "s", "u", "n")

FORM_GAMEWEEKS = 5
STARTER_FALLBACK_MINUTES = 75.0
BENCH_FALLBACK_MINUTES = 20.0

# A match. Nothing this module produces is ever longer than one, whatever a
# season total divided by a season's gameweeks comes out at.
FULL_MATCH = 90.0

# How many gameweeks of history it takes before the history is the whole
# answer. Under this many, the starts-based estimate is a floor beneath it.
# Three is where a run of zeroes stops looking like a rest and starts looking
# like a player who has lost his place — and it is short enough that the floor
# is gone before anybody is planning a season on it.
BLEND_GAMEWEEKS = 3

# What a season's minutes are divided by to become minutes a gameweek. Every
# player is given the full 38 whether or not he was at the club for all of
# them, which understates a January signing — his half-season of minutes
# spread over a whole one. The payload has no games-available column to do
# better with, and the error is in the safe direction: a prior that is too low
# is a floor that does not lift, not a projection that is too high.
SEASON_GAMEWEEKS = 38


def season_prior(past: list[PastSeason]) -> float | None:
    """Last season's minutes a gameweek, or None if there was no last season.

    ``past`` is element-summary's ``history_past``, oldest first, so the
    player's most recent Premier League season is the last row. It is not
    necessarily *last* season — a player who spent a year abroad has a gap —
    and it is still the most recent thing anybody measured about him, which is
    what the floor wants.

    None and 0.0 are different answers and callers must keep them apart. None
    is nothing to read: no Premier League behind him, so the ``starts`` guess
    stands. 0.0 is a season he was registered for and never played, which is
    thin evidence and still evidence — and a player projected at nothing is a
    player the solver will not buy, which is the right way round to be wrong
    about him.
    """
    if not past:
        return None
    minutes = past[-1].minutes / SEASON_GAMEWEEKS
    return min(FULL_MATCH, max(0.0, minutes))


def availability(player: Player) -> float:
    """Probability the player features, in 0.0-1.0.

    FPL publishes a percentage for flagged players; when it is set it is the
    best information available, so it wins over the status code.
    """
    if player.chance_of_playing_next_round is not None:
        return player.chance_of_playing_next_round / 100
    return 0.0 if player.status in RULED_OUT_STATUSES else 1.0


def expected_minutes(
    history: list[GwHistory], player: Player, prior: float | None = None
) -> float:
    """Minutes the player is expected to play in the next gameweek.

    ``history`` is the player's **played** gameweeks in order; only the last
    :data:`FORM_GAMEWEEKS` count, so a lost or won place shows up fast. Every
    entry is taken as a match he was available for, which is why a gameweek
    that has only been entered must never reach here — the caller cuts those
    out (:func:`aigaffer.orchestrator._played`).

    ``prior`` is what he averaged a gameweek in his last Premier League season
    (:func:`season_prior`), and it is the floor beneath a history too thin to
    stand on its own: under :data:`BLEND_GAMEWEEKS` gameweeks, whichever of the
    mean and the floor is higher wins, and from the third gameweek the mean
    stands alone. It replaces a guess with a measurement rather than adding to
    it — where there is a prior the ``starts`` fallback is not consulted at
    all, in either direction. A first-choice player rested on the opening
    weekend is worth his eighty-five minutes and not a substitute's twenty; a
    fringe player is worth his own thirty and not a starter's seventy-five.

    None is the case the prior cannot help: a signing from abroad, a promoted
    club's player, a teenager — nobody with a Premier League season behind him.
    Then the floor is the old one, whether he has started a match *this*
    season, and in GW1 that count is zero for everybody, so those players are
    still projected at a substitute's minutes on the opening weekend. It is
    also the default, so a caller with no opinion about last season — the
    backtest, which is scoring a gameweek in the middle of one — gets exactly
    the model it had before.
    """
    fallback = (
        STARTER_FALLBACK_MINUTES if player.starts > 0 else BENCH_FALLBACK_MINUTES
    )
    floor = fallback if prior is None else prior
    if not history:
        baseline = floor
    else:
        recent = history[-FORM_GAMEWEEKS:]
        mean = sum(gw.minutes for gw in recent) / len(recent)
        baseline = max(mean, floor) if len(history) < BLEND_GAMEWEEKS else mean
    return baseline * availability(player)
