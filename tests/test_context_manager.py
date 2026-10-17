"""Unit tests for backend/context_manager.py (pure trimming logic)."""
from __future__ import annotations

import pytest

from backend import context_manager
from backend.context_manager import split_to_fit, trim_messages_to_context
from backend.token_utils import count_single_message_tokens, count_tokens


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


@pytest.fixture()
def word_tokens(monkeypatch):
    """Deterministic token accounting: 3 tokens of overhead + one per word."""

    def fake_count(message: dict) -> int:
        return 3 + len(str(message.get("content", "")).split())

    monkeypatch.setattr(context_manager, "count_single_message_tokens", fake_count)


def test_split_to_fit_empty_input():
    assert split_to_fit([]) == ([], [])


def test_split_to_fit_is_noop_under_budget(word_tokens):
    messages = [_msg("system", "sys"), _msg("user", "one"), _msg("assistant", "two")]
    kept, removed = split_to_fit(messages, max_tokens=1000)
    assert kept == messages
    assert removed == []


def test_split_to_fit_keeps_system_and_last_and_drops_oldest(word_tokens):
    messages = [
        _msg("system", "sys"),
        _msg("user", "one"),
        _msg("assistant", "two"),
        _msg("user", "three"),
        _msg("assistant", "four"),
    ]
    kept, removed = split_to_fit(messages, max_tokens=12)
    assert kept == [messages[0], messages[3], messages[4]]
    assert removed == [messages[1], messages[2]]  # oldest first, in order


def test_split_to_fit_never_leaves_assistant_message_first(word_tokens):
    messages = [
        _msg("user", "one"),
        _msg("assistant", "two"),
        _msg("user", "three"),
        _msg("assistant", "four"),
    ]
    kept, removed = split_to_fit(messages, max_tokens=10)
    # Dropping only the first message would leave the assistant reply orphaned,
    # so the assistant message goes with it.
    assert [m["content"] for m in kept] == ["three", "four"]
    assert kept[0]["role"] == "user"
    assert removed == [messages[0], messages[1]]


def test_split_to_fit_always_keeps_last_message_even_over_budget(word_tokens):
    messages = [
        _msg("system", "sys"),
        _msg("user", "one two three four"),
        _msg("assistant", "five"),
    ]
    kept, removed = split_to_fit(messages, max_tokens=1)
    assert kept[0] == messages[0]
    assert kept[-1] == messages[-1]
    assert messages[1] in removed


def test_split_to_fit_keeps_system_even_when_over_budget(word_tokens):
    messages = [_msg("system", "a b c d e f g h"), _msg("user", "hi")]
    kept, removed = split_to_fit(messages, max_tokens=2)
    assert kept == messages
    assert removed == []


def test_trim_messages_to_context_returns_kept_list(word_tokens):
    messages = [
        _msg("system", "sys"),
        _msg("user", "one"),
        _msg("assistant", "two"),
        _msg("user", "three"),
        _msg("assistant", "four"),
    ]
    snapshot = [dict(m) for m in messages]

    trimmed = trim_messages_to_context(messages, max_tokens=12)

    assert trimmed == [messages[0], messages[3], messages[4]]
    assert messages == snapshot  # input list is not mutated


def test_count_single_message_tokens_adds_role_overhead():
    """Holds for any tokenizer (gpt2, cache fallback or whitespace fallback)."""
    content = "hello world, this is a slightly longer test message"
    assert count_single_message_tokens({"role": "user", "content": content}) == 3 + count_tokens(content)
    assert count_single_message_tokens({"role": "user", "content": content}) > (
        count_single_message_tokens({"role": "user", "content": "hi"})
    )


def test_context_budget_uses_the_scratch_config():
    # conftest.py points METIS_CONFIG at a config with n_ctx=2048, max_tokens=256.
    assert context_manager.context_budget() == 2048 - 256
