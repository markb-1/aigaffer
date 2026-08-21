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
played gameweeks — and leaves one limitation, which is worth being plain
about. With the phantom rows gone, the opening weekend has no history at all,
so every player falls back on ``starts``; and ``starts`` is season-to-date
like everything else in the payload, so in GW1 it is zero for everybody. The
floor under a £14.5m striker is then the bench estimate: twenty minutes of a
man who will play ninety. From GW2 there is a played gameweek to read and the
blend bites, but the opening weekend is the week nothing here can rescue.
Nothing in the data can: last season is not in the payload, and reading a role
off a price or off ``total_points`` would be a guess wearing the clothes of a
measurement. Prior-season history is the fix, and it is not written yet; the
mitigation until it is is the gaffer, who reads the team news and sets the
minutes by hand.
"""

from aigaffer.data.models import GwHistory, Player

# Statuses that rule a player out: injured, suspended, unavailable (left the
# league), not eligible. 'a' is available and 'd' is doubtful.
RULED_OUT_STATUSES = ("i", "s", "u", "n")

FORM_GAMEWEEKS = 5
STARTER_FALLBACK_MINUTES = 75.0
BENCH_FALLBACK_MINUTES = 20.0

# How many gameweeks of history it takes before the history is the whole
# answer. Under this many, the starts-based estimate is a floor beneath it.
# Three is where a run of zeroes stops looking like a rest and starts looking
# like a player who has lost his place — and it is short enough that the floor
# is gone before anybody is planning a season on it.
BLEND_GAMEWEEKS = 3


def availability(player: Player) -> float:
    """Probability the player features, in 0.0-1.0.

    FPL publishes a percentage for flagged players; when it is set it is the
    best information available, so it wins over the status code.
    """
    if player.chance_of_playing_next_round is not None:
        return player.chance_of_playing_next_round / 100
    return 0.0 if player.status in RULED_OUT_STATUSES else 1.0


def expected_minutes(history: list[GwHistory], player: Player) -> float:
    """Minutes the player is expected to play in the next gameweek.

    ``history`` is the player's **played** gameweeks in order; only the last
    :data:`FORM_GAMEWEEKS` count, so a lost or won place shows up fast. Every
    entry is taken as a match he was available for, which is why a gameweek
    that has only been entered must never reach here — the caller cuts those
    out (:func:`aigaffer.orchestrator._played`). With no history at all —
    pre-season, a new signing, or a deadline that has gone with nothing played
    behind it — we fall back on whether he has started a match this season.

    That fallback is also a floor for the first :data:`BLEND_GAMEWEEKS` weeks,
    and the higher of the two wins: one rotated gameweek should not overrule
    the fact that he has started, and a full ninety should not be dragged down
    to 75 by it. From the third gameweek on the mean stands alone.

    The floor is only as good as the count it is read off. Under three
    gameweeks the season's ``starts`` is at most two, and in GW1 it is zero for
    everybody — so the opening weekend lands on the bench estimate, which is
    the limitation the module docstring sets out and the gaffer's own minute
    overrides exist to cover.
    """
    fallback = (
        STARTER_FALLBACK_MINUTES if player.starts > 0 else BENCH_FALLBACK_MINUTES
    )
    if not history:
        baseline = fallback
    else:
        recent = history[-FORM_GAMEWEEKS:]
        mean = sum(gw.minutes for gw in recent) / len(recent)
        baseline = max(mean, fallback) if len(history) < BLEND_GAMEWEEKS else mean
    return baseline * availability(player)
