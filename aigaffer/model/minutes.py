"""How likely a player is to play, and for how long.

Expected points start here: nearly every scoring event is proportional to
time on the pitch, so a good minutes estimate matters more than a clever
points model. We take the player's recent minutes as the best guess at his
role and scale it by the chance he features at all.
"""

from aigaffer.data.models import GwHistory, Player

# Statuses that rule a player out: injured, suspended, unavailable (left the
# league), not eligible. 'a' is available and 'd' is doubtful.
RULED_OUT_STATUSES = ("i", "s", "u", "n")

FORM_GAMEWEEKS = 5
STARTER_FALLBACK_MINUTES = 75.0
BENCH_FALLBACK_MINUTES = 20.0


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
    """
    if history:
        recent = history[-FORM_GAMEWEEKS:]
        baseline = sum(gw.minutes for gw in recent) / len(recent)
    elif player.starts > 0:
        baseline = STARTER_FALLBACK_MINUTES
    else:
        baseline = BENCH_FALLBACK_MINUTES
    return baseline * availability(player)
