"""The Anthropic client every conversation with the gaffer is built with."""

import anthropic

from aigaffer.manager.client import CLIENT_RETRIES, CLIENT_TIMEOUT_SECONDS, build_client
from tests.fixtures import config


def test_the_client_is_built_against_the_clock(monkeypatch):
    # The SDK's own defaults are ten minutes and two retries a request — half
    # an hour of one turn. Five minutes and one retry is what the hourly
    # manager has run on since GW3, and the chip opinion inherits it.
    built: list[dict] = []

    class Recorder:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)

    client = build_client(config(anthropic_api_key="sk-test"))

    assert isinstance(client, Recorder)
    assert built == [{"api_key": "sk-test", "timeout": 300.0, "max_retries": 1}]
    assert (CLIENT_TIMEOUT_SECONDS, CLIENT_RETRIES) == (300.0, 1)
