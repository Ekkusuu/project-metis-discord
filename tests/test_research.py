"""The web researcher (backend/research.py), page reading, and the spoken-reply sentence limit. No network or LLM."""
from __future__ import annotations

import asyncio

import pytest

import backend.research as research_mod
from backend.research import format_brief, research, site_in
from backend.routers.chat import ChatRequest, Message, _sentence_end, _sentence_limit
from backend.tool_manager import _CORRECTION
from backend.tools import web_page_tool


@pytest.mark.parametrize("query, site", [
    ("best budget laptop site:notebookcheck.net", "notebookcheck.net"),
    ("dune reviews site:www.rottentomatoes.com", "rottentomatoes.com"),
    ("best budget laptop", None),
])
def test_site_in(query, site):
    assert site_in(query) == site


def _fake_results(*domains):
    return [{"title": f"Page on {d}", "url": f"https://{d}/page", "content": f"snippet from {d}"} for d in domains]


def _step(action, arg=""):
    return {"thought": "", "action": action, "input": arg}


def _run(monkeypatch, steps, results, max_steps=5):
    """research() with canned searches, pages and model steps; returns (result, searches, pages read, notifications)."""
    searches, opened, notes, steps, prompts = [], [], [], list(steps), []

    async def fake_search(query, kind="text", count=6):
        searches.append((kind, query))
        return results.pop(0) if results else []

    async def fake_read(url, question, max_chars):
        opened.append(url)
        return f"text of {url}"

    def fake_model(messages, schema, *args):
        prompts.append((messages[-1]["content"], schema["properties"]["action"]["enum"]))
        return steps.pop(0)

    monkeypatch.setattr(research_mod, "_search", fake_search)
    monkeypatch.setattr(research_mod, "_read", fake_read)
    monkeypatch.setattr(research_mod, "json_completion", fake_model)
    found = asyncio.run(research("what is the best budget laptop on notebookcheck",
                                 "best budget laptop site:notebookcheck.net", max_steps=max_steps, notify=notes.append))
    return found, searches, opened, notes, prompts


def test_search_open_answer(monkeypatch):
    found, searches, opened, notes, prompts = _run(
        monkeypatch, [_step("open", "1,2"), _step("answer", "The Acer Swift Go 14 (notebookcheck.net)")],
        [_fake_results("notebookcheck.net", "rtings.com", "pcmag.com")])
    assert searches == [("text", "best budget laptop site:notebookcheck.net")]  # first search needs no model step
    assert opened == ["https://notebookcheck.net/page", "https://rtings.com/page"]  # two pages at once
    assert "[3] Page on pcmag.com" in prompts[0][0]  # the model sees numbered results
    assert "text of https://rtings.com/page" in prompts[1][0]  # and what it read
    assert found["found"] and found["sources"] == ["notebookcheck.net", "rtings.com"]
    assert notes == ["web_read"]  # the voice bot says "reading a couple of pages"


def test_searches_again_and_says_so(monkeypatch):
    found, searches, opened, notes, _ = _run(monkeypatch, [
        _step("search", "budget laptop review"), _step("news", "budget laptop launch"), _step("open", "3"),
        _step("answer", "x")], [_fake_results("a.com"), _fake_results("b.com"), _fake_results("c.com")])
    assert [k for k, _ in searches] == ["text", "text", "news"]
    assert opened == ["https://c.com/page"]  # result numbers continue across searches
    assert notes == ["web_search_more", "web_search_more", "web_read"]  # "still looking" in voice


def test_last_step_must_answer(monkeypatch):
    found, _, _, _, prompts = _run(monkeypatch, [_step("search", "a"), _step("search", "b"), _step("give_up", "")],
                                   [_fake_results("a.com"), [], []], max_steps=3)
    assert prompts[-1][1] == ["answer", "give_up"]
    assert not found["found"] and "couldn't find" in format_brief(found, "q")


def test_repeated_search_and_bad_open_are_explained(monkeypatch):
    _, searches, opened, _, prompts = _run(monkeypatch, [
        _step("search", "best budget laptop site:notebookcheck.net"), _step("open", "9"), _step("answer", "x")],
        [_fake_results("a.com")])
    assert len(searches) == 1 and opened == []
    assert "almost the same as an earlier search" in prompts[1][0] and "give result numbers" in prompts[2][0]


def test_visible_text_used_for_data_pages(monkeypatch):
    rows = "".join(f"<div><span>Champion {i}</span><span>5{i}.0% WR</span></div>" for i in range(40))
    page = f"<html><head><title>Tier List</title></head><body><p>The only tier list you need.</p>{rows}</body></html>"
    monkeypatch.setattr(web_page_tool, "_download", lambda url: (url, "text/html", page))
    got = web_page_tool.fetch_page_text("https://stats.example.com/x", 10_000, full=True)
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


def test_search_races_engines_and_retries_a_misheard_site(monkeypatch):
    from backend.tools import web_search_tool

    tried = []

    def fake_sync(self, query, count, kind, backend="auto"):
        tried.append((query, backend))
        if "site:" in query or backend == "yahoo":
            return []
        return [{"title": "ad", "url": "https://www.bing.com/aclick?x", "content": ""}] + _fake_results("u.gg")

    monkeypatch.setattr(web_search_tool.WebSearchTool, "_search_sync", fake_sync)
    got = asyncio.run(research_mod._search("best jungle champion site:u.g"))
    assert [r["url"] for r in got] == ["https://u.gg/page"]  # the ad is dropped
    assert ("best jungle champion u.g", "yandex") in tried  # the site name becomes a keyword
    assert {b for q, b in tried if "site:" in q} == {"yandex", "yahoo", "auto"}


def test_read_page_renders_javascript_pages(monkeypatch):
    web_page_tool._page_cache.clear()
    calls = []
    monkeypatch.setattr(web_page_tool, "fetch_page_text",
                        lambda url, *a, **k: {"url": url, "title": "", "text": "Loading..."})
    monkeypatch.setattr(web_page_tool, "_check_public", lambda url: None)
    monkeypatch.setattr(web_page_tool, "_rendered_text", lambda url: calls.append(url) or "Rendered table " * 20)
    assert web_page_tool.read_page("https://app.example.com/x", "table", 5000).startswith("Rendered table")
    web_page_tool.read_page("https://app.example.com/x", "table", 5000)
    assert calls == ["https://app.example.com/x"]  # the second read comes from the cache


def test_opening_a_page_again_is_refused(monkeypatch):
    _, _, opened, _, prompts = _run(monkeypatch, [_step("open", "1"), _step("open", "1"), _step("answer", "x")],
                                    [_fake_results("a.com", "b.com")])
    assert opened == ["https://a.com/page"]
    assert "already read above" in prompts[2][0]


@pytest.mark.parametrize("query, message, kept", [
    ("Dune reviews site:rottentomatoes.com", "check on the site rotten tomatoes how good dune is", True),
    ("best jungler site:u.gg", "search the web in the site u.g what is the best jungler", True),
    ("laptop review site:notebookcheck.net", "what does notebookcheck say about it", True),
    ("cheapest camera phone site:rottentomatoes.com", "No, I meant the cheapest phone with a really good camera.", False),
    ("cheapest camera phone", "no site here", True),
])
def test_site_kept_only_when_named(query, message, kept):
    from backend.tool_manager import keep_named_site

    assert (keep_named_site(query, message) == query) is kept


def test_clip_shortens_old_steps_and_always_finishes():
    from backend.research import _clip

    parts = ["a" * 3000, "b" * 3000, "c" * 3000, "last " * 400]
    clipped = _clip(parts, 7000)  # used to loop forever once the steps passed the budget
    assert clipped.startswith("a" * 400 + " …(shortened)")
    assert clipped.endswith("last " * 399 + "last ")  # the newest step is kept whole
    assert _clip(["x" * 5000, "y"], 100).count("…(shortened)") == 1


def test_site_search_runs_with_the_plain_name_at_once(monkeypatch):
    tried = []

    async def fake_once(query, kind, count):
        tried.append(query)
        return [] if "site:" in query else _fake_results("u.gg")

    monkeypatch.setattr(research_mod, "_search_once", fake_once)
    assert asyncio.run(research_mod._search("best jungle champion site:u.g"))[0]["url"] == "https://u.gg/page"
    assert sorted(tried) == ["best jungle champion site:u.g", "best jungle champion u.g"]


def test_video_pages_are_not_opened(monkeypatch):
    _, _, opened, _, prompts = _run(monkeypatch, [_step("open", "1,2"), _step("answer", "x")],
                                    [_fake_results("youtube.com", "a.com")])
    assert opened == ["https://a.com/page"]
    assert "(video/social, can't be read)" in prompts[0][0]


def test_near_duplicate_search_is_refused(monkeypatch):
    """"... Patch 26.20" then "... Patch 26.20 tier list" gave the same results and used up the steps."""
    _, searches, _, _, prompts = _run(monkeypatch, [
        _step("search", "best budget laptop notebookcheck"), _step("search", "best chess opening"),
        _step("answer", "x")], [_fake_results("a.com"), _fake_results("b.com")])
    assert len(searches) == 2 and "almost the same" in prompts[1][0] and prompts[2][0].count("almost the same") == 1
