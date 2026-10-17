"""The review writer's conversation, against a scripted client."""

import pytest

from aigaffer.config import Config
from aigaffer.manager.review_writer import (
    ANSWER,
    REVIEW_MAX_WORDS,
    review_briefing,
    validate_review,
    write_review,
)
from aigaffer.manager.tools import ToolError
from aigaffer.review import ReviewUnavailable
from tests.test_manager_agent import FakeClient, reply, text, use
from tests.test_review import facts

CFG = Config(team_id=42, anthropic_api_key="sk-test", manager_model="claude-opus-5-5")
GOOD = {"went_well": ["Captaincy paid: Five returned ten."],
        "went_wrong": ["The bench outscored two starters."],
        "lessons": ["Start the in-form defender over the rotation risk."]}


def test_a_valid_answer_first_time_is_one_request():
    client = FakeClient([reply(use(ANSWER, GOOD))])
    written = write_review(CFG, facts(), client=client)
    assert written.lessons == ("Start the in-form defender over the rotation risk.",)
    assert written.turns == 1 and len(client.requests) == 1


def test_the_request_forces_nothing_and_searches_nothing():
    client = FakeClient([reply(use(ANSWER, GOOD))])
    write_review(CFG, facts(), client=client)
    request = client.requests[0]
    assert "tool_choice" not in request and "thinking" not in request
    assert [tool["name"] for tool in request["tools"]] == [ANSWER]
    assert request["output_config"] == {"effort": "medium"}
    assert request["max_tokens"] == 16000
    assert request["model"] == "claude-opus-5-5"


def test_an_invalid_answer_gets_its_error_back_as_the_tool_result():
    bad = {**GOOD, "went_well": [], "went_wrong": []}
    client = FakeClient([reply(use(ANSWER, bad, "tu_1")), reply(use(ANSWER, GOOD, "tu_2"))])
    write_review(CFG, facts(), client=client)
    assert len(client.requests) == 2
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user"
    result = last["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "tu_1" and result["is_error"] is True


def test_talking_instead_of_answering_gets_a_plain_ask():
    client = FakeClient([reply(text("Let me think."), stop="end_turn"), reply(use(ANSWER, GOOD))])
    write_review(CFG, facts(), client=client)
    messages = client.requests[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert ANSWER in messages[-1]["content"][-1]["text"]


def test_a_cut_off_turn_never_happened_and_the_ask_joins_the_briefing():
    client = FakeClient([reply(text("…"), stop="max_tokens"), reply(use(ANSWER, GOOD))])
    write_review(CFG, facts(), client=client)
    messages = client.requests[1]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert ANSWER in messages[0]["content"][-1]["text"]


def test_two_failures_are_unavailable():
    bad = {**GOOD, "lessons": ["x"] * 4}
    client = FakeClient([reply(use(ANSWER, bad)), reply(use(ANSWER, bad, "tu_2"))])
    with pytest.raises(ReviewUnavailable):
        write_review(CFG, facts(), client=client)
    assert len(client.requests) == 2


def test_an_api_error_is_unavailable():
    import anthropic, httpx

    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    with pytest.raises(ReviewUnavailable, match="APIConnectionError"):
        write_review(CFG, facts(), client=FakeClient([error]))


def test_three_points_are_too_many_and_the_schema_and_prompt_say_two():
    # The first live review ran to ~300 words; two points a list keeps it near 180.
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": ["a", "b", "c"]})
    assert validate_review({**GOOD, "lessons": ["a", "b"]})[2] == ("a", "b")
    client = FakeClient([reply(use(ANSWER, GOOD))])
    write_review(CFG, facts(), client=client)
    request = client.requests[0]
    properties = request["tools"][0]["input_schema"]["properties"]
    assert [properties[k]["maxItems"] for k in ("went_well", "went_wrong", "lessons")] == [2, 2, 2]
    assert "up to two points each" in request["system"][0]["text"]
    assert "100 to 180 words in total" in request["system"][0]["text"]


def test_the_word_cap_is_250_and_the_error_says_to_aim_for_180():
    assert REVIEW_MAX_WORDS == 250
    with pytest.raises(ToolError, match="keep it near 180, and 250 at most"):
        validate_review({**GOOD, "went_well": ["word " * 251]})


def test_validation_caps_counts_and_words_and_flattens_lines():
    assert validate_review({**GOOD, "lessons": ["  Two\nlines  "]})[2] == ("Two lines",)
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": ["a", "b", "c", "d"]})
    with pytest.raises(ToolError):
        validate_review({**GOOD, "went_well": ["word " * (REVIEW_MAX_WORDS + 1)]})
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": ["## A heading"]})
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": [""]})
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": "not a list"})
    # A forced re-review cuts the file at the first marker, so a point that
    # held the marker would cut it in the wrong place.
    with pytest.raises(ToolError):
        validate_review({**GOOD, "lessons": ["see <!-- looking-back gw2 -->"]})
    assert validate_review({**GOOD, "lessons": ["- A bullet"]})[2] == ("A bullet",)


def test_the_briefing_carries_the_facts_and_says_whose_verdict_it_was():
    from tests.test_review import verdict

    gaffer = review_briefing(facts())
    assert "GW2 final:" in gaffer and "The gaffer's verdict (deadline report)." in gaffer
    assert "Rationale: Hold; captain Five." in gaffer
    solver = review_briefing(facts(verdict=verdict(decision_source="solver-fallback: x")))
    assert "The solver's verdict — the gaffer did not decide this week" in solver
    assert "Verdict followed" not in solver
    none = review_briefing(facts(verdict=None))
    assert "No verdict on record" in none


def test_a_valid_answer_after_an_unknown_tool_in_one_turn_is_taken():
    client = FakeClient([reply(use("other", {}, "tu_a"), use(ANSWER, GOOD, "tu_b"))])
    written = write_review(CFG, facts(), client=client)
    assert written.lessons == tuple(GOOD["lessons"]) and len(client.requests) == 1


def test_every_tool_call_in_a_failed_turn_gets_its_own_result():
    bad = {**GOOD, "lessons": ["x"] * 4}
    client = FakeClient([
        reply(use(ANSWER, bad, "tu_a"), use(ANSWER, bad, "tu_b")),
        reply(use(ANSWER, GOOD, "tu_c")),
    ])
    write_review(CFG, facts(), client=client)
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "user"
    assert [r["tool_use_id"] for r in last["content"]] == ["tu_a", "tu_b"]
    assert all(r["is_error"] is True for r in last["content"])


def test_an_empty_end_turn_joins_the_ask_to_the_briefing():
    client = FakeClient([reply(stop="end_turn"), reply(use(ANSWER, GOOD))])
    write_review(CFG, facts(), client=client)
    messages = client.requests[1]["messages"]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    assert ANSWER in messages[0]["content"][-1]["text"]


def test_the_prompt_leaves_the_team_that_played_to_the_owner():
    from aigaffer.manager.review_writer import SYSTEM_PROMPT

    assert "that was the owner's call" in SYSTEM_PROMPT
    assert "never that you ignored or overrode your own verdict" in SYSTEM_PROMPT
