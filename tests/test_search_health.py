"""Tests for what one response says about its searches.

Blocks come in the two shapes the loop can meet — the SDK's objects and plain
dicts — and every block type the docs give a search that runs from code. What
is never allowed out is anything somebody else wrote: queries, titles, pages,
output and error text all stay where they are.
"""

from types import SimpleNamespace
from typing import Any

import anthropic
import pytest

from aigaffer.manager.search_health import examine

CODE = {"type": "code_execution_20260120", "tool_id": "srv_0"}
DIRECT = {"type": "direct"}


def hit(caller: Any = CODE) -> dict:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": "srv_1",
        "caller": caller,
        "content": [{"type": "web_search_result", "title": "Gale out", "url": "https://x"}],
    }


def miss(code: Any, caller: Any = CODE) -> dict:
    return {
        "type": "web_search_tool_result",
        "tool_use_id": "srv_1",
        "caller": caller,
        "content": {"type": "web_search_tool_result_error", "error_code": code},
    }


def ran(
    content_type: str = "encrypted_code_execution_result",
    block_type: str = "code_execution_tool_result",
    **fields: Any,
) -> dict:
    content = {"type": content_type, "return_code": 0, "stderr": "", **fields}
    return {"type": block_type, "tool_use_id": "srv_0", "content": content}


def response(*blocks: Any) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks))


def objects(value: Any) -> Any:
    """The same block as the SDK hands it over: attributes, not keys."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: objects(item) for key, item in value.items()})
    if isinstance(value, list):
        return [objects(item) for item in value]
    return value


# --- clean turns -------------------------------------------------------------


def test_a_turn_that_searched_cleanly_has_nothing_to_report():
    health = examine(response(hit(), ran()), cap=8)

    assert health.lines == []
    assert not health.spent and not health.transient


def test_a_search_that_matched_nothing_is_not_an_error():
    empty = {**hit(), "content": []}

    assert examine(response(empty), cap=8).lines == []


def test_a_turn_with_no_content_has_nothing_to_report():
    assert examine(SimpleNamespace(content=None), cap=8).lines == []
    assert examine(SimpleNamespace(), cap=8).lines == []


# --- web-search errors -------------------------------------------------------


@pytest.mark.parametrize(
    "code, transient",
    [
        ("too_many_requests", True),
        ("unavailable", True),
        ("invalid_tool_input", False),
        ("query_too_long", False),
        ("request_too_large", False),
    ],
)
@pytest.mark.parametrize("shape", [lambda block: block, objects])
def test_a_search_error_is_one_line_with_its_code(code, transient, shape):
    health = examine(response(shape(miss(code)), shape(hit())), cap=8)

    assert health.lines == [f"search error: {code}"]
    assert health.transient is transient
    assert not health.spent


def test_every_error_gets_its_own_line():
    health = examine(response(miss("unavailable"), miss("unavailable")), cap=8)

    assert health.lines == ["search error: unavailable", "search error: unavailable"]


# --- the cap -----------------------------------------------------------------


@pytest.mark.parametrize("shape", [lambda block: block, objects])
def test_as_many_results_as_the_cap_is_a_spent_allowance(shape):
    health = examine(response(*[shape(hit())] * 8), cap=8)

    assert health.spent and health.fresh_next
    assert health.lines == ["search note: allowance spent (8 in one request, code_execution)"]


def test_one_short_of_the_cap_is_not():
    assert not examine(response(*[hit()] * 7), cap=8).spent


def test_failed_searches_count_towards_the_cap():
    health = examine(response(*[hit()] * 7, miss("unavailable")), cap=8)

    assert health.spent and health.transient


def test_max_uses_exceeded_is_a_spent_allowance_however_few_came_back():
    health = examine(response(hit(), miss("max_uses_exceeded")), cap=8)

    assert health.spent and not health.transient
    assert health.lines == [
        "search error: max_uses_exceeded",
        "search note: allowance spent (2 in one request, code_execution)",
    ]


@pytest.mark.parametrize(
    "callers, named",
    [
        ([DIRECT] * 8, "direct"),
        ([CODE] * 7 + [DIRECT], "direct"),
        ([None] * 8, "unknown"),
        ([CODE] * 7 + [None], "unknown"),
        ([{"type": "something_new"}] * 8, "unknown"),
    ],
)
def test_only_searches_run_from_code_are_promised_a_fresh_allowance(callers, named):
    # A request of nothing but tool results continues the same turn, and direct
    # searches from earlier in it still count against the cap.
    health = examine(response(*[hit(caller) for caller in callers]), cap=8)

    assert health.spent and not health.fresh_next
    assert health.lines == [f"search note: allowance spent (8 in one request, {named})"]


# --- the code the searches run in --------------------------------------------


@pytest.mark.parametrize(
    "prefix", ["code_execution", "bash_code_execution", "text_editor_code_execution"]
)
@pytest.mark.parametrize(
    "code, transient",
    [
        ("unavailable", True),
        ("too_many_requests", True),
        ("execution_time_exceeded", True),
        ("invalid_tool_input", False),
    ],
)
def test_a_code_execution_error_is_one_line_with_its_code(prefix, code, transient):
    block = {
        "type": f"{prefix}_tool_result",
        "tool_use_id": "srv_0",
        "content": {"type": f"{prefix}_tool_result_error", "error_code": code},
    }
    health = examine(response(block), cap=8)

    assert health.lines == [f"search error: {prefix} {code}"]
    assert health.transient is transient


@pytest.mark.parametrize(
    "content_type, block_type, prefix",
    [
        ("code_execution_result", "code_execution_tool_result", "code_execution"),
        ("encrypted_code_execution_result", "code_execution_tool_result", "code_execution"),
        ("bash_code_execution_result", "bash_code_execution_tool_result", "bash_code_execution"),
    ],
)
def test_a_run_that_failed_is_a_note_and_not_a_search_error(content_type, block_type, prefix):
    # Usually his own filter code, so it is evidence and never a retry.
    failed = examine(response(ran(content_type, block_type, return_code=1, stderr="x")), cap=8)
    warned = examine(response(ran(content_type, block_type, stderr="x")), cap=8)

    assert failed.lines == [f"search note: {prefix} exit 1"]
    assert warned.lines == [f"search note: {prefix} stderr"]
    assert not failed.transient and not failed.spent


# --- what must never get out, and what must never break ----------------------


SECRETS = ["q-secret", "t-secret", "u-secret", "o-secret", "e-secret", "s-secret"]


def test_nothing_anybody_else_wrote_reaches_a_line():
    asked = {"type": "server_tool_use", "name": "web_search", "input": {"query": "q-secret"}}
    found = {
        **hit(),
        "content": [{"type": "web_search_result", "title": "t-secret", "url": "u-secret"}],
    }
    failed = ran(
        return_code=1, stdout="o-secret", stderr="e-secret", encrypted_stdout="s-secret"
    )
    health = examine(response(asked, found, miss("unavailable"), failed), cap=2)

    assert len(health.lines) == 3
    assert not any(secret in line for line in health.lines for secret in SECRETS)


@pytest.mark.parametrize(
    "code",
    [
        "Guéhi injury update",
        "unavailable\n",
        "unavailable\x1b[31m",
        "x" * 41,
        "",
        None,
        7,
        ["unavailable"],
        {"code": "unavailable"},
    ],
)
def test_a_code_that_is_not_a_code_is_unknown(code):
    health = examine(response(miss(code)), cap=8)

    assert health.lines == ["search error: unknown"]
    assert not health.transient and not health.spent


@pytest.mark.parametrize("exit_code", ["1; rm", None, True, [1]])
def test_an_exit_code_that_is_not_a_number_is_not_printed(exit_code):
    assert examine(response(ran(return_code=exit_code)), cap=8).lines == []


@pytest.mark.parametrize(
    "block",
    [
        None,
        7,
        "web_search_tool_result",
        {},
        {"type": None},
        {"type": ["web_search_tool_result"]},
        {"type": "something_new_tool_result", "content": {"error_code": "unavailable"}},
        {"type": "text", "text": "unavailable"},
        {"type": "web_search_tool_result"},
        {"type": "web_search_tool_result", "content": "oops"},
        {"type": "code_execution_tool_result", "content": None},
        {"type": "code_execution_tool_result", "content": {"type": "something_new"}},
        {
            "type": "text_editor_code_execution_tool_result",
            "content": {"type": "text_editor_code_execution_view_result"},
        },
    ],
)
def test_a_block_it_does_not_know_is_ignored(block):
    health = examine(response(block), cap=8)

    assert health.lines == []
    assert not health.spent and not health.transient


def test_the_sdks_own_message_reads_the_same():
    # The fakes above are ours; this is the type the loop is really handed.
    message = anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [
                {"type": "server_tool_use", "id": "srv_0", "name": "code_execution", "input": {}},
                {
                    **hit(),
                    "content": [
                        {
                            "type": "web_search_result",
                            "title": "t-secret",
                            "url": "u-secret",
                            "encrypted_content": "e",
                            "page_age": None,
                        }
                    ],
                },
                miss("unavailable"),
                ran(return_code=1, stderr="e-secret", encrypted_stdout="s", content=[]),
            ],
        }
    )
    health = examine(message, cap=2)

    assert health.lines == [
        "search error: unavailable",
        "search note: code_execution exit 1",
        "search note: allowance spent (2 in one request, code_execution)",
    ]
    assert health.spent and health.fresh_next and health.transient
