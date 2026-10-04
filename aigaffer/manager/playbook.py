"""How to think about each chip — the judgement the numbers cannot make.

The solver and the chip calendar do the arithmetic: which week a chip is
saved for, what it must clear this week, which bench a boost would field.
This is what a manager checks before agreeing with them or not. It rides in
the cached system prompt, so it costs nothing per run.
"""

CHIP_PLAYBOOK = """\
## Chips

Chips come in two sets. The first set — bench boost, triple captain, wildcard \
and free hit — expires after GW19: a first-set chip not played by the GW19 \
deadline is lost. A second set, the same four, is playable from GW20 to GW38. \
At most one chip a gameweek. The briefing's chip calendar says which week the \
solver is saving each chip for, what it is worth there, and the bar a chip \
has to clear this week to be worth playing now rather than then.

Bench boost. Worth playing when all four bench players start. Check team news \
for the bench as carefully as for the eleven: a bench boost with a doubtful \
substitute keeper is a wasted chip. The solver builds the bench in the weeks \
before the boost it plans; if a bench player's minutes are in doubt, adjust \
them and resolve rather than playing the boost on hope.

Triple captain. Minutes certainty first: never on a doubt, and be wary of a \
player with a midweek European tie or a rotation-prone manager. Then the \
fixture: at home against a weak defence. The extra multiple is the captain's \
whole score, so a nailed attacker in the week's best fixture is the case for it.

Wildcard. For a fixture swing that needs four or more moves, or a squad whose \
structure is wrong — not to fix one injury, which a free transfer or a hit \
does more cheaply. Played the week before a bench boost, it builds the bench \
the boost will field.

Free hit. For a gameweek most of the squad cannot play — a blank, or mass \
absences — fielding a one-week team and reverting. Not for an ordinary week \
that merely looks a few points better.

Expiry. In a chip's final two eligible gameweeks it is free points: finalize \
the solver's chip unless team news makes the week a dud, and name the week \
you would play it instead. If you finalize 'none' over a chip the solver \
plans in its final eligible gameweek, your rationale must say so and why.
"""
