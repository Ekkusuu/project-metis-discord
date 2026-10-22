"""The web researcher (backend/research.py), page reading, and the spoken-reply sentence limit. No network or LLM."""
from __future__ import annotations

import asyncio

import pytest

import backend.research as research_mod
from backend.research import format_brief, research, site_asked_for
from backend.routers.chat import ChatRequest, Message, _sentence_end, _sentence_limit
from backend.tool_manager import _CORRECTION
from backend.tools import web_page_tool


@pytest.mark.parametrize("message, site", [
    ("search uh the web for in the site u.g what is the best jungler right now", "u.gg"),
    ("look it up on op gg", "op.gg"),
    ("check on reddit.com what people think", "reddit.com"),
    ("what does it say on imdb.com", "imdb.com"),
    ("search the web for the best jungler", None),
    ("what is 3.5 times 2", None),
])
def test_site_asked_for(message, site):
    assert site_asked_for(message) == site


def _fake_results(*domains):
    return [{"title": f"Page on {d}", "url": f"https://{d}/page", "content": f"snippet from {d}"} for d in domains]


def _run(monkeypatch, steps, results):
    """research() with canned search results and researcher answers; returns (result, searches, notifications)."""
    searches, notes, steps = [], [], list(steps)

    async def fake_search(query, count=6):
        searches.append(query)
        return results.pop(0) if results else []

    async def fake_read(url, question, max_chars):
        return f"text of {url}"

    monkeypatch.setattr(research_mod, "_search", fake_search)
    monkeypatch.setattr(research_mod, "_read", fake_read)
    monkeypatch.setattr(research_mod, "_rank", lambda results, question, site: results)
    monkeypatch.setattr(research_mod, "json_completion", lambda *args: steps.pop(0))
    found = asyncio.run(research("what is the best jungler on u.gg", "best jungler", rounds=3, notify=notes.append))
    return found, searches, notes


def test_research_stops_when_found(monkeypatch):
    found, searches, notes = _run(
        monkeypatch, [{"answer": "Master Yi, 51.2% win rate (u.gg)", "found": True, "search_again": ""}],
        [_fake_results("u.gg", "metasrc.com", "jungler.gg")])
    assert found["found"] and found["answer"].startswith("Master Yi")
    assert searches == ["best jungler site:u.gg"]  # the site they asked for is searched
    assert found["sources"] == ["u.gg", "metasrc.com"]  # two pages read
    assert notes == []


def test_research_searches_again_and_says_so(monkeypatch):
    found, searches, notes = _run(monkeypatch, [
        {"answer": "", "found": False, "search_again": "jungle tier list patch 26.21 site:u.gg"},
        {"answer": "Master Yi", "found": True, "search_again": ""},
    ], [_fake_results("u.gg"), _fake_results("u.gg")])
    assert searches == ["best jungler site:u.gg", "jungle tier list patch 26.21 site:u.gg"]
    assert notes == ["web_search_more"]  # the voice bot says "still looking" between searches
    assert found["found"]


def test_research_gives_up_after_the_rounds(monkeypatch):
    not_found = {"answer": "", "found": False, "search_again": "another try"}
    found, searches, _ = _run(monkeypatch, [not_found, {**not_found, "search_again": "third try"}, not_found],
                              [_fake_results("a.com"), _fake_results("b.com"), _fake_results("c.com")])
    assert len(searches) == 3 and not found["found"]
    assert "couldn't find" in format_brief(found, "q")


def test_research_drops_site_when_it_has_no_results(monkeypatch):
    found, searches, _ = _run(monkeypatch, [{"answer": "x", "found": True, "search_again": ""}],
                              [[], _fake_results("metasrc.com")])
    assert searches == ["best jungler site:u.gg", "best jungler"]


def test_visible_text_used_for_data_pages(monkeypatch):
    rows = "".join(f"<div><span>Champion {i}</span><span>5{i}.0% WR</span></div>" for i in range(40))
    page = f"<html><head><title>Tier List</title></head><body><p>The only tier list you need.</p>{rows}</body></html>"
    monkeypatch.setattr(web_page_tool, "_download", lambda url: (url, "text/html", page))
    got = web_page_tool.fetch_page_text("https://u.gg/x", 10_000, full=True)
    assert got["title"] == "Tier List"
    assert "Champion 7 57.0% WR" in got["text"]


def test_download_refuses_private_addresses():
    with pytest.raises(web_page_tool.UnsafeURLError):
        web_page_tool._download("http://127.0.0.1:8730/status")


def test_sentence_end_skips_voice_tags():
    text = "[teasing] Heehee. You again? Fine. I'll tell you"
    assert text[:_sentence_end(text, 2)] == "[teasing] Heehee. You again?"
    assert _sentence_end(text, 4) == -1


def test_sentence_limit_depends_on_the_question():
    def req(text, voice=True):
        return ChatRequest(messages=[Message(role="user", content=text)], voice_mode=voice)

    assert _sentence_limit(req("[ekkusu]: hello")) == 3
    assert _sentence_limit(req("[ekkusu]: explain how black holes form")) == 6
    assert _sentence_limit(req("[ekkusu]: hello", voice=False)) == 0


def test_correction_after_a_search():
    assert _CORRECTION.search("No, I meant the best jungle champion right now.")
    assert _CORRECTION.search("nah I mean the one from last year")
    assert not _CORRECTION.search("no way, that's crazy")
