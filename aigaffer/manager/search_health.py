"""What one response says about its web searches, for the journal and the loop.

The search tool runs on Anthropic's servers and fails inside a successful
response: a search that was refused is a result block with an error in it, and
a search refused while code was running it is a code run that ended badly. The
loop read neither, so a run whose searches were capped and a run whose searches
were broken left the same record, which was none.

:func:`examine` reads one response and says two kinds of thing. ``lines`` are
for the journal: an error code the API defines, or a count. ``spent``,
``fresh_next`` and ``transient`` are for the loop, which tells the gaffer.

Nothing anybody else wrote is ever in a line. Queries, titles, addresses, page
text, a run's output and its error text all stay where they are, and a code
that does not look like one is printed as ``unknown``.

Pure: no printing, no network, and no import from the rest of the project.
"""

import re
from dataclasses import dataclass, field
from typing import Any

SEARCH = "web_search_tool_result"
MAX_USES = "max_uses_exceeded"

# Server-side and worth one more try. The rest are the query's own fault, or
# the cap, which is not a fault at all.
TRANSIENT = {"too_many_requests", "unavailable", "execution_time_exceeded"}

# The tools a search's code can run in, and what their results contain. The
# encrypted one is what a search filtered by code comes back as.
RUNS = ("code_execution", "bash_code_execution", "text_editor_code_execution")
RUN_RESULTS = {
    "code_execution_result",
    "encrypted_code_execution_result",
    "bash_code_execution_result",
}

CODE = re.compile(r"[a-z_]{1,40}")
UNKNOWN = "unknown"
FROM_CODE = "code_execution"
DIRECT = "direct"


@dataclass
class SearchHealth:
    """One response's searches.

    ``spent`` is the per-request cap reached. ``fresh_next`` is whether the
    next request, which carries only tool results and so continues the same
    turn, really starts a new allowance: the docs say it does for searches run
    from code and does not for direct ones, so it is claimed for the first
    alone.
    """

    lines: list[str] = field(default_factory=list)
    spent: bool = False
    fresh_next: bool = False
    transient: bool = False


def examine(response: Any, cap: int) -> SearchHealth:
    """Read one response. ``cap`` is the request's ``max_uses``.

    A failed search counts against the cap, so every result block is counted.
    A block of a type this does not know is ignored.
    """
    health = SearchHealth()
    callers: list[str] = []
    refused = False
    for block in _get(response, "content") or []:
        kind = _get(block, "type")
        content = _get(block, "content")
        if kind == SEARCH:
            callers.append(_caller(block))
            # A list is results, and an empty one is a search with no hits.
            if _get(content, "type") == f"{SEARCH}_error":
                code = _code(content)
                health.lines.append(f"search error: {code}")
                health.transient |= code in TRANSIENT
                refused |= code == MAX_USES
        else:
            for run in RUNS:
                if kind == f"{run}_tool_result":
                    _examine_run(health, run, content)

    if refused or len(callers) >= cap:
        named = _named(callers)
        health.spent = True
        health.fresh_next = named == FROM_CODE
        health.lines.append(
            f"search note: allowance spent ({len(callers)} in one request, {named})"
        )
    return health


def _examine_run(health: SearchHealth, run: str, content: Any) -> None:
    """One code run. An error code is the server's; an exit code or a stderr
    is usually the gaffer's own filter code, so it is noted and no more."""
    kind = _get(content, "type")
    if kind == f"{run}_tool_result_error":
        code = _code(content)
        health.lines.append(f"search error: {run} {code}")
        health.transient |= code in TRANSIENT
    elif isinstance(kind, str) and kind in RUN_RESULTS:
        status = _get(content, "return_code")
        if isinstance(status, int) and not isinstance(status, bool) and status:
            health.lines.append(f"search note: {run} exit {status}")
        elif _get(content, "stderr"):
            health.lines.append(f"search note: {run} stderr")


def _caller(block: Any) -> str:
    kind = _get(_get(block, "caller"), "type")
    if kind == DIRECT:
        return DIRECT
    if isinstance(kind, str) and kind.startswith(FROM_CODE):
        return FROM_CODE
    return UNKNOWN


def _named(callers: list[str]) -> str:
    """Who ran them all: one direct call is enough to lose the promise."""
    if DIRECT in callers:
        return DIRECT
    if callers and all(caller == FROM_CODE for caller in callers):
        return FROM_CODE
    return UNKNOWN


def _code(content: Any) -> str:
    code = _get(content, "error_code")
    # fullmatch: a ``$`` would let a trailing newline through to the journal.
    return code if isinstance(code, str) and CODE.fullmatch(code) else UNKNOWN


def _get(thing: Any, name: str) -> Any:
    """A field of an SDK object or of a dict, or None."""
    if isinstance(thing, dict):
        return thing.get(name)
    return getattr(thing, name, None)
