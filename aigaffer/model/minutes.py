"""How likely a player is to play, and for how long.

Expected points start here: nearly every scoring event is proportional to
time on the pitch, so a good minutes estimate matters more than a clever
points model. We take the player's recent minutes as the best guess at his
role and scale it by the chance he features at all.

Recent minutes are the right answer once there are some. Early in a season
there are not, and a mean over one gameweek is not evidence of a role — it is
a single Saturday, and the thing that most often makes it zero is the manager
resting somebody he intends to play every week thereafter. So under
:data:`BLEND_GAMEWEEKS` the history is a floor rather than the whole answer:
what the season's ``starts`` count says about him wins if it is higher, and by
the third gameweek the history has earned the right to speak for itself.

The case that bought this: GW1 of 2026-27, the first week the gaffer ran live.
The World Cup had just finished, the internationals were rested for the opening
weekend, and one gameweek of 0 minutes was the entire season history. Every one
of them projected zero minutes and zero points; the solver benched a fit
Haaland; and the gaffer spent his week overriding seven players by hand to
undo it. A mean of one number was never a role estimate, and this is the
smallest change that stops it being read as one.
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

    ``history`` is the player's season so far in gameweek order; only the
    last :data:`FORM_GAMEWEEKS` count, so a lost or won place shows up fast.
    With no history at all — pre-season, or a new signing — we fall back on
    whether he has started a match this season.

    That fallback is also a floor for the first :data:`BLEND_GAMEWEEKS` weeks,
    and the higher of the two wins: one rested gameweek should not overrule a
    season's worth of starts, and a full ninety should not be dragged down to
    75 by them. From the third gameweek on the mean stands alone.
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
