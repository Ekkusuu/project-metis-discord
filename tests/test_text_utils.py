"""Unit tests for backend/text_utils.py (pure text helpers, no models or network)."""
from __future__ import annotations

import random

import pytest

from backend.text_utils import (
    ThinkFilter,
    clean_for_speech,
    find_sentence_boundary,
    is_sentence_boundary,
    pop_sentences,
    split_message,
    strip_name_prefix,
    strip_thinking,
)

# ---------------------------------------------------------------------------
# strip_thinking
# ---------------------------------------------------------------------------


def test_strip_thinking_removes_complete_block():
    assert strip_thinking("Hello <think>secret reasoning</think>world") == "Hello world"
    assert strip_thinking("<think>only reasoning</think>") == ""


def test_strip_thinking_multiline_block_and_whitespace():
    assert strip_thinking("Answer<think>\nstep 1\nstep 2\n</think>  tail  ") == "Answer  tail"


def test_strip_thinking_is_case_insensitive():
    assert strip_thinking("<THINK>upper</THINK>visible") == "visible"
    assert strip_thinking("<Think>mixed</Think>visible") == "visible"


def test_strip_thinking_stray_closing_tag_drops_everything_before_it():
    assert strip_thinking("secret stuff</think>visible") == "visible"
    assert strip_thinking("</think>hello") == "hello"


def test_strip_thinking_unclosed_opening_tag_drops_everything_after_it():
    assert strip_thinking("visible<think>never finished") == "visible"


def test_strip_thinking_no_tags_and_empty_input():
    assert strip_thinking("plain text, no tags") == "plain text, no tags"
    assert strip_thinking("") == ""


# ---------------------------------------------------------------------------
# ThinkFilter (streaming variant)
# ---------------------------------------------------------------------------

# Inputs where incrementally filtering must match strip_thinking(). Deliberately
# excluded: a stray "</think>" that arrives *after* the text before it was
# already emitted (see the xfail test below).
_EQUIVALENT_INPUTS = [
    "Hello <think>reasoning</think>world",
    "</think>hello",  # stray closing tag at the very start
    "visible<think>nope",  # unclosed opening tag
    "plain text with no tags at all.",
    "Multi<think>\nline one\nline two\n</think>  visible tail  ",
    "one <think>two</think> three",
    "<THINK>UPPER</THINK>after",
    "trailing partial <thi",
    "</think>second part",
]


def _feed_in_chunks(text: str, chunks) -> str:
    think_filter = ThinkFilter()
    out = [think_filter.feed(chunk) for chunk in chunks]
    out.append(think_filter.flush())
    return "".join(out)


@pytest.mark.parametrize("text", _EQUIVALENT_INPUTS)
def test_think_filter_char_by_char_matches_strip_thinking(text):
    visible = _feed_in_chunks(text, list(text))
    assert visible.strip() == strip_thinking(text)


@pytest.mark.parametrize("text", _EQUIVALENT_INPUTS)
def test_think_filter_random_chunks_match_strip_thinking(text):
    rng = random.Random(1234)
    chunks = []
    rest = text
    while rest:
        n = rng.randint(1, 5)
        chunks.append(rest[:n])
        rest = rest[n:]
    visible = _feed_in_chunks(text, chunks)
    assert visible.strip() == strip_thinking(text)


@pytest.mark.parametrize(
    "chunks",
    [
        ["Hello <thi", "nk>secret</thi", "nk> world"],  # tag split across chunks
        ["<think>", "hidden", "</think>", "visible"],
        ["visible<thi", "nk>nope"],  # unclosed tag split across chunks
        ["</thi", "nk>hello"],  # stray closing tag split across chunks
    ],
)
def test_think_filter_handles_tags_split_across_chunks(chunks):
    text = "".join(chunks)
    visible = _feed_in_chunks(text, chunks)
    assert visible.strip() == strip_thinking(text)


def test_think_filter_flush_drops_unclosed_reasoning():
    think_filter = ThinkFilter()
    assert think_filter.feed("<think>never closed") == ""
    assert think_filter.flush() == ""


@pytest.mark.xfail(
    reason=(
        "ThinkFilter diverges from strip_thinking for a stray '</think>' that arrives in a later "
        "chunk: it emits text eagerly while outside a think block, and out.clear() in "
        "backend/text_utils.py:60-67 can only retract output produced by the same feed() call. "
        "Real bug (streaming), not a test artifact. Repro: feed('secret ') then feed('</think>ok')."
    ),
    strict=False,
)
def test_think_filter_stray_close_after_emitted_text_matches_strip_thinking():
    text = "secret reasoning</think>visible answer"
    # Feed it the way the LLM streams it: text first, stray closing tag later.
    visible = _feed_in_chunks(text, ["secret reasoning", "</think>visible answer"])
    assert visible.strip() == strip_thinking(text)


# ---------------------------------------------------------------------------
# strip_name_prefix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Metis: hi", "hi"),
        ("[Metis]: hi", "hi"),
        ("metis : hi", "hi"),
        ("  Metis: hi", "hi"),
        ("METIS: hi", "hi"),
        ("Metis is cool", "Metis is cool"),
        ("hello Metis: hi", "hello Metis: hi"),
    ],
)
def test_strip_name_prefix(raw, expected):
    assert strip_name_prefix(raw) == expected


# ---------------------------------------------------------------------------
# Sentence boundaries
# ---------------------------------------------------------------------------


def test_is_sentence_boundary_ignores_abbreviations():
    text = "Dr. Smith went home"
    assert is_sentence_boundary(text, 2) is False  # "Dr."


def test_is_sentence_boundary_ignores_decimals():
    text = "It cost 12.5 dollars"
    assert is_sentence_boundary(text, 10) is False  # "12.5"


def test_is_sentence_boundary_ignores_initialisms():
    text = "The U.S. army moved"
    assert is_sentence_boundary(text, 7) is False  # "U.S."


def test_is_sentence_boundary_waits_for_full_question_marks():
    text = "Wait?! Really"
    assert is_sentence_boundary(text, 4) is False  # "?" is followed by "!"
    assert is_sentence_boundary(text, 5) is True  # "?!" run is complete


def test_is_sentence_boundary_false_at_end_of_buffer():
    assert is_sentence_boundary("done.", 4) is False
    assert is_sentence_boundary("done.", -1) is False
    assert is_sentence_boundary("done.", 99) is False


def test_find_sentence_boundary_skips_abbreviation():
    text = "Dr. Smith went home. Then he left."
    idx = find_sentence_boundary(text)
    assert text[: idx + 1].endswith("home.")


def test_find_sentence_boundary_skips_decimal():
    text = "It cost 12.5 dollars. OK"
    idx = find_sentence_boundary(text)
    assert text[: idx + 1].endswith("dollars.")


def test_find_sentence_boundary_newline():
    assert find_sentence_boundary("line one\nline two") == 8


def test_find_sentence_boundary_returns_minus_one_for_incomplete_sentence():
    # A period at the very end of the buffer might be a decimal or an ellipsis.
    assert find_sentence_boundary("Just one sentence.") == -1


def test_pop_sentences_splits_complete_sentences():
    sentences, rest = pop_sentences("First sentence. Second one!\nThird")
    assert sentences == ["First sentence.", "Second one!"]
    assert rest == "Third"


def test_pop_sentences_keeps_incomplete_tail():
    assert pop_sentences("Just one sentence.") == ([], "Just one sentence.")


def test_pop_sentences_ignores_decimals_and_abbreviations():
    assert pop_sentences("Pi is 3.14 roughly. Yes") == (["Pi is 3.14 roughly."], "Yes")
    assert pop_sentences("Dr. Smith arrived. Then") == (["Dr. Smith arrived."], "Then")


def test_pop_sentences_newline_boundary():
    sentences, rest = pop_sentences("First line\nsecond line here")
    assert sentences == ["First line"]
    assert rest == "second line here"


# ---------------------------------------------------------------------------
# clean_for_speech
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("*sighs* fine", "fine"),
        ("**bold** text", "text"),
        ("see https://example.com/x now", "see now"),
        ("hello \U0001f600 world", "hello world"),
        ("hello there friend", "hello there friend"),
        (
            "*waves* Hello **there** friend! https://x.co/y \U0001f44b",
            "Hello friend!",
        ),
    ],
)
def test_clean_for_speech(raw, expected):
    assert clean_for_speech(raw) == expected


# ---------------------------------------------------------------------------
# split_message
# ---------------------------------------------------------------------------


def test_split_message_short_text_is_one_chunk():
    assert split_message("hello world") == ["hello world"]


def test_split_message_empty_input():
    assert split_message("") == []
    assert split_message("   \n  ") == []


def test_split_message_no_spaces_never_exceeds_limit():
    text = "x" * 5000
    chunks = split_message(text, limit=2000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert "".join(chunks) == text  # nothing lost on hard character cuts


def test_split_message_spaces_never_exceeds_limit_and_keeps_words():
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    chunks = split_message(text, limit=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert [w for chunk in chunks for w in chunk.split()] == words


def test_split_message_prefers_newline_boundaries():
    lines = [f"line {i}" for i in range(20)]
    text = "\n".join(lines)
    chunks = split_message(text, limit=30)
    assert len(chunks) > 1
    assert all(len(chunk) <= 30 for chunk in chunks)
    for chunk in chunks:
        for line in chunk.split("\n"):
            assert line in lines  # no line was cut in half


def test_split_message_code_block_fences_stay_balanced():
    code = "```python\n" + ("x = 1\n" * 200) + "```"
    chunks = split_message(code, limit=200)
    assert len(chunks) > 2
    assert all(len(chunk) <= 200 for chunk in chunks)
    for chunk in chunks:
        fence_lines = [line for line in chunk.split("\n") if line.startswith("```")]
        assert len(fence_lines) % 2 == 0
        assert len(fence_lines) >= 2  # opened and closed inside the same chunk
    # Every chunk after the first reopens the fence with the same language tag.
    assert chunks[0].startswith("```python\n")
    assert all(chunk.startswith("```python\n") for chunk in chunks[1:])
