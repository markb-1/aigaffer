"""The manager: the part of the bot that reads the news and decides.

The solver says what is optimal under the model. This package is what asks
whether the model knows what happened at Friday's press conference — a Claude
agent handed the week's position (:mod:`aigaffer.manager.briefing`), a web
search, the ability to overrule the minutes model and re-solve, and a
guardrailed way to commit to one of the solver's own legal plans.

Nothing here may be imported by the orchestrator at module scope: the
orchestrator is what builds the briefing's inputs, so the dependency runs one
way only and the manager annotates its side of the seam under
``TYPE_CHECKING``.
"""
