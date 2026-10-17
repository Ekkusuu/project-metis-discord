"""Unit tests for the pure merge/parse helpers in backend/rag_engine.py.

`retrieve_context` and the tokenizer helpers are monkeypatched, so no vector
store client is created, no embedding/reranker model is loaded and no HTTP call
is made.
"""
from __future__ import annotations

import os

import pytest

from backend import rag_engine


def _ctx(source="a.md", index=0, text="some text", distance=0.1, rerank_score=None, **extra):
    ctx = {
        "text": text,
        "metadata": {"source_file": source, "chunk_index": index},
        "distance": distance,
        "rerank_score": rerank_score,
    }
    ctx.update(extra)
    return ctx


def _result(accepted=(), rejected_by_distance=(), rejected_by_score=()):
    return {
        "accepted": list(accepted),
        "rejected_by_distance": list(rejected_by_distance),
        "rejected_by_score": list(rejected_by_score),
    }


@pytest.fixture()
def fake_retrieve(monkeypatch):
    """Returns (calls, responses): `responses[query]` is what retrieve_context returns."""
    calls = []
    responses = {}

    def fake(query, top_k=None):
        calls.append((query, top_k))
        return responses.get(query, _result())

    monkeypatch.setattr(rag_engine, "retrieve_context", fake)
    return calls, responses


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_deduplicates_the_same_chunk_across_queries(fake_retrieve):
    calls, responses = fake_retrieve
    responses["q1"] = _result(accepted=[_ctx("a.md", 0, rerank_score=0.5)])
    responses["q2"] = _result(accepted=[_ctx("a.md", 0, rerank_score=0.5)])

    out = rag_engine.search(["q1", "q2"])

    assert len(out["accepted"]) == 1
    assert out["accepted"][0]["metadata"] == {"source_file": "a.md", "chunk_index": 0}


def test_search_keeps_the_best_score_for_a_duplicate_chunk(fake_retrieve):
    _, responses = fake_retrieve
    responses["q1"] = _result(accepted=[_ctx("a.md", 0, rerank_score=0.3)])
    responses["q2"] = _result(accepted=[_ctx("a.md", 0, rerank_score=0.9)])

    out = rag_engine.search(["q1", "q2"])

    assert len(out["accepted"]) == 1
    assert out["accepted"][0]["rerank_score"] == 0.9


def test_search_accepted_by_one_query_beats_rejected_by_another(fake_retrieve):
    _, responses = fake_retrieve
    responses["q1"] = _result(accepted=[_ctx("a.md", 1, distance=0.2)])
    responses["q2"] = _result(rejected_by_distance=[_ctx("a.md", 1, distance=0.9)])

    out = rag_engine.search(["q1", "q2"])

    assert [c["text"] for c in out["accepted"]] == ["some text"]
    assert out["rejected"] == []


def test_search_sorts_accepted_by_rerank_score_desc(fake_retrieve):
    _, responses = fake_retrieve
    responses["q1"] = _result(
        accepted=[
            _ctx("a.md", 0, rerank_score=0.2),
            _ctx("b.md", 0, rerank_score=0.9),
            _ctx("c.md", 0, rerank_score=0.5),
        ]
    )

    out = rag_engine.search(["q1"])

    assert [c["metadata"]["source_file"] for c in out["accepted"]] == ["b.md", "c.md", "a.md"]


def test_search_sorts_rejected_by_distance(fake_retrieve):
    _, responses = fake_retrieve
    responses["q1"] = _result(
        rejected_by_distance=[_ctx("far.md", 0, distance=0.9)],
        rejected_by_score=[_ctx("near.md", 0, distance=0.2, rejection_reason="score")],
    )

    out = rag_engine.search(["q1"])

    assert [c["metadata"]["source_file"] for c in out["rejected"]] == ["near.md", "far.md"]


def test_search_skips_duplicate_and_blank_queries(fake_retrieve):
    calls, responses = fake_retrieve
    responses["q1"] = _result(accepted=[_ctx("a.md", 0)])

    out = rag_engine.search(["q1", "q1", "   ", ""])

    assert [c[0] for c in calls] == ["q1"]  # only one real retrieval
    assert out["queries"] == ["q1", "   ", ""]  # reported as given, de-duplicated


def test_search_passes_top_k_through(fake_retrieve):
    calls, responses = fake_retrieve
    responses["q1"] = _result()

    rag_engine.search(["q1"], top_k=7)

    assert calls == [("q1", 7)]


def test_search_with_no_queries_returns_empty(fake_retrieve):
    calls, _ = fake_retrieve
    out = rag_engine.search(["", "   "])
    assert calls == []
    assert out["accepted"] == []
    assert out["rejected"] == []


# ---------------------------------------------------------------------------
# _chunk_key()
# ---------------------------------------------------------------------------


def test_chunk_key_uses_source_file_and_chunk_index():
    assert rag_engine._chunk_key(_ctx("docs/a.md", 3)) == ("docs/a.md", 3)


def test_chunk_key_falls_back_to_text_prefix():
    key = rag_engine._chunk_key({"text": "hello there", "metadata": {}})
    assert key == (None, "hello there")


# ---------------------------------------------------------------------------
# results_for_ui()
# ---------------------------------------------------------------------------


def test_results_for_ui_marks_used_and_unused():
    search_result = {
        "queries": ["q1"],
        "accepted": [_ctx("a.md", 0, text="kept text", distance=0.2, rerank_score=0.9)],
        "rejected": [_ctx("b.md", 1, text="dropped", distance=0.8, rejection_reason="distance")],
    }

    out = rag_engine.results_for_ui(search_result, "the original query")

    assert out["queries"] == ["q1"]
    assert out["original_query"] == "the original query"
    used, unused = out["results"]
    assert used["used"] is True
    assert "rejection_reason" not in used
    assert used["source_file"] == "a.md"
    assert used["chunk_index"] == 0
    assert used["rerank_score"] == 0.9
    assert unused["used"] is False
    assert unused["rejection_reason"] == "distance"
    assert unused["text_preview"] == "dropped"


def test_results_for_ui_defaults_and_truncation():
    long_text = "x" * 500
    search_result = {
        "queries": [],
        "accepted": [{"text": long_text, "metadata": None, "distance": 0.0, "rerank_score": None}],
        "rejected": [{"text": "no reason given", "metadata": {}, "distance": 0.5}],
    }

    out = rag_engine.results_for_ui(search_result, "q", extra_field=1)

    assert out["extra_field"] == 1
    used, unused = out["results"]
    assert used["source_file"] == "unknown"
    assert used["chunk_index"] == 0
    assert used["text_preview"] == long_text[:200]
    assert unused["rejection_reason"] == "distance"  # fallback when not set


# ---------------------------------------------------------------------------
# _parse_query_lines()
# ---------------------------------------------------------------------------


def test_parse_query_lines_strips_numbering_bullets_and_quotes():
    text = "\n".join(
        [
            "1. first query",
            "2) second query",
            "- third query",
            "\u2022 fourth query",
            "* fifth query",
            '"sixth query"',
            "'seventh query'",
        ]
    )
    assert rag_engine._parse_query_lines(text, 10) == [
        "first query",
        "second query",
        "third query",
        "fourth query",
        "fifth query",
        "sixth query",
        "seventh query",
    ]


def test_parse_query_lines_respects_limit():
    assert rag_engine._parse_query_lines("alpha\nbeta\ngamma", 2) == ["alpha", "beta"]


def test_parse_query_lines_skips_blanks_prompt_leakage_and_overlong_lines():
    text = "\n".join(["", "   ", "RULES: ignore this", "x" * 201, "kept query"])
    assert rag_engine._parse_query_lines(text, 10) == ["kept query"]


# ---------------------------------------------------------------------------
# has_file_changed()
# ---------------------------------------------------------------------------


def test_has_file_changed_true_for_unknown_file(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    assert rag_engine.has_file_changed(target, {}) is True


def test_has_file_changed_false_when_size_and_mtime_match(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    metadata = {str(target): rag_engine.get_file_info(target)}
    assert rag_engine.has_file_changed(target, metadata) is False


def test_has_file_changed_detects_size_change(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    metadata = {str(target): rag_engine.get_file_info(target)}

    target.write_text("hello, world", encoding="utf-8")

    assert rag_engine.has_file_changed(target, metadata) is True


def test_has_file_changed_detects_mtime_change(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("hello", encoding="utf-8")
    info = rag_engine.get_file_info(target)
    metadata = {str(target): info}

    os.utime(target, (info["mtime"] + 1000, info["mtime"] + 1000))  # same size, new mtime

    assert rag_engine.has_file_changed(target, metadata) is True


# ---------------------------------------------------------------------------
# chunk_text()
# ---------------------------------------------------------------------------


def test_chunk_text_uses_the_tokenizer(monkeypatch):
    monkeypatch.setattr("backend.token_utils.encode_text", lambda text: text.split())
    monkeypatch.setattr("backend.token_utils.decode_tokens", lambda ids: " ".join(ids))

    text = " ".join(f"w{i}" for i in range(10))
    chunks = rag_engine.chunk_text(text, chunk_size=4, overlap=1)

    # step = chunk_size - overlap = 3; the loop stops once a window reaches the end.
    assert chunks == [
        "w0 w1 w2 w3",
        "w3 w4 w5 w6",
        "w6 w7 w8 w9",
    ]


def test_chunk_text_falls_back_to_characters_without_a_tokenizer(monkeypatch):
    def broken_encode(text):
        raise RuntimeError("no tokenizer available")

    monkeypatch.setattr("backend.token_utils.encode_text", broken_encode)

    chunks = rag_engine.chunk_text("abcdefghij", chunk_size=2, overlap=1)

    # size = chunk_size * 4, step = (chunk_size - overlap) * 4
    assert chunks == ["abcdefgh", "efghij", "ij"]
