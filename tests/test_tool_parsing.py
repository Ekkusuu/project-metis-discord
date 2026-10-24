"""Unit tests for tool-call parsing/formatting in backend/tool_manager.py.

No LLM is involved: parse_tool_calls() works on the raw selector output string.
"""
from __future__ import annotations

import json

import pytest

from backend.tool_manager import ToolResult, format_tool_results_for_context

VALID_TOOLS = ["knowledge_search", "web_search"]
FALLBACK = "fallback query"


def _parse(response: str, valid_tools=None):
    return parse_tool_calls(response, FALLBACK, valid_tools or VALID_TOOLS)


def test_format_tool_results_string_result():
    results = [ToolResult("knowledge_search", True, "found some notes")]
    assert format_tool_results_for_context(results) == "[knowledge_search]\nfound some notes"


def test_format_tool_results_dict_result():
    payload = {"answer": 42, "sources": ["a.md"]}
    results = [ToolResult("knowledge_search", True, payload)]
    expected = "[knowledge_search]\n" + json.dumps(payload, indent=2, ensure_ascii=False)
    assert format_tool_results_for_context(results) == expected


def test_format_tool_results_failure():
    results = [ToolResult("web_search", False, None, "connection refused")]
    assert format_tool_results_for_context(results) == "[web_search] failed: connection refused"


def test_format_tool_results_joins_multiple_and_skips_empty():
    results = [
        ToolResult("a", True, "first"),
        ToolResult("b", True, None),  # empty result: nothing to inject
        ToolResult("c", True, ""),
        ToolResult("d", False, None, "boom"),
    ]
    assert format_tool_results_for_context(results) == "[a]\nfirst\n\n[d] failed: boom"


def test_format_tool_results_empty_list():
    assert format_tool_results_for_context([]) == ""


# --- select_tools (model calls mocked) -------------------------------------------------------
@pytest.fixture
def planner(monkeypatch):
    """The planning model call, replaced: returns state["plan"] and records what it was shown."""
    import backend.tool_manager as tm

    seen = {"calls": []}
    state = {"plan": {"standalone": "", "action": "none", "query": "", "recent": False}}

    def fake_json(messages, schema, *args, **kwargs):
        seen["calls"].append({"messages": messages, "schema": schema})
        return dict(state["plan"])

    monkeypatch.setattr(tm, "json_completion", fake_json)
    monkeypatch.setattr(tm, "_last_search", {})  # no search from an earlier test
    return state, seen


def plan(state, action, query="", standalone="", recent=False):
    state["plan"] = {"standalone": standalone, "action": action, "query": query, "recent": recent}


async def test_backchannel_skips_the_planner(planner):
    from backend.tool_manager import select_tools

    state, seen = planner
    result = await select_tools([], "[Sava]: lol nice", 2)
    assert [(c.tool_name, c.arguments) for c in result] == [("knowledge_search", {"query": "lol nice"})]
    assert seen["calls"] == []
    await select_tools([], "[Sava]: who is the best support?", 2)
    assert seen["calls"][0]["schema"]["properties"]["action"] == {"enum": ["none", "remember", "web_search"]}


async def test_web_search_plan_becomes_a_search(planner):
    import backend.tool_manager as tm
    from backend.tool_manager import select_tools

    state, _ = planner
    plan(state, "web_search", "f1 race winner", "Who won the F1 race yesterday?", recent=True)
    result = await select_tools([], "[Sava]: who won the f1 race yesterday?", 2, )
    assert ("web_search", {"query": "f1 race winner", "search_type": "news"}) in [(c.tool_name, c.arguments) for c in result]
    assert tm._last_search[""]["query"] == "f1 race winner"


async def test_none_plan_runs_no_tool(planner):
    from backend.tool_manager import decide_tools

    state, _ = planner
    plan(state, "none", "", "How would Metis go about the heist we're imagining?")
    assert await decide_tools([], "[Sava]: so how would you go about it? what's the best way to do it?") == []


async def test_links_are_always_fetched(planner):
    from backend.tool_manager import select_tools

    result = await select_tools([], "[Sava]: summarize https://example.com/post please", 2)
    assert ("fetch_webpage", {"url": "https://example.com/post"}) in [(c.tool_name, c.arguments) for c in result]


async def test_remember_gets_the_self_contained_question(planner):
    from backend.tool_manager import decide_tools

    state, _ = planner
    plan(state, "remember", "", "What subject was Max's exam last week?")
    calls = await decide_tools([], "[Max]: do you remember what subject it was?")
    assert [(c.tool_name, c.arguments) for c in calls] == [("remember", {"query": "What subject was Max's exam last week?"})]


async def test_planner_sees_the_conversation_and_the_last_search(planner, monkeypatch):
    """Follow-ups ("this time for Emerald Plus", "then do it") are resolved by the planner from this context."""
    import time
    import backend.tool_manager as tm

    state, seen = planner
    monkeypatch.setitem(tm._last_search, "voice-1", {"time": time.time(), "query": "best champion site:u.gg"})
    messages = [{"role": "user", "content": "[Sava]: best champion on u.gg?"},
                {"role": "assistant", "content": "Braum, 52.1%."},
                {"role": "user", "content": "[Sava]: could you double-check, but this time for Emerald Plus?"}]
    await tm.decide_tools(messages, messages[-1]["content"], conversation_id="voice-1")
    shown = seen["calls"][0]["messages"][-1]["content"]
    assert "Last web search: best champion site:u.gg" in shown and "Metis: Braum, 52.1%." in shown
    assert shown.endswith("Latest message: [Sava]: could you double-check, but this time for Emerald Plus?")
    await tm.decide_tools(messages, messages[-1]["content"], conversation_id="voice-2")
    assert "Last web search" not in seen["calls"][1]["messages"][-1]["content"]  # per conversation


async def test_site_is_kept_only_when_someone_names_it(planner):
    from backend.tool_manager import decide_tools

    state, _ = planner
    plan(state, "web_search", "best support site:u.gg", "Who is the best support right now?")
    calls = await decide_tools([], "[Sava]: who is the best support right now?")
    assert calls[0].arguments["query"] == "best support"  # carried over from an older question
    plan(state, "web_search", "best support site:u.gg", "Who is the best support on u.gg?")
    calls = await decide_tools([], "[Sava]: who is the best support on u.g?")
    assert calls[0].arguments["query"] == "best support site:u.gg"


async def test_query_made_from_metis_words_is_replaced(planner):
    """A query built from Metis's own made-up reply searched for 'roadster' when asked about France."""
    from backend.tool_manager import decide_tools

    state, _ = planner
    plan(state, "web_search", "roadster date France news", "What happened in France recently?", recent=True)
    messages = [{"role": "user", "content": "[ekkusu]: hello"},
                {"role": "assistant", "content": "Are you here for the roadster date today?"},
                {"role": "user", "content": "[ekkusu]: What happened in France recently that everyone is talking about?"}]
    calls = await decide_tools(messages, messages[-1]["content"])
    assert calls[0].arguments == {"query": "What happened in France recently?", "search_type": "news"}


def test_quick_search_keeps_only_close_matches(monkeypatch):
    import backend.rag_engine as rag

    class FakeCollection:
        def count(self):
            return 4

        def query(self, query_texts, n_results):
            return {"documents": [["a", "b", "c", "d"]], "metadatas": [[{}, {}, {}, {}]],
                    "distances": [[0.9, 1.1, 1.15, 1.45]]}

    monkeypatch.setattr(rag, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(rag, "_rag_cfg", lambda: {"enabled": True})
    found = rag.quick_search("how do I stop doomscrolling", max_distance=1.2, limit=2)
    assert [c["text"] for c in found["accepted"]] == ["a", "b"]
    assert {c["text"]: c["rejection_reason"] for c in found["rejected"]} == {"d": "distance", "c": "rank"}


async def test_stream_announces_slow_tools_before_the_reply(monkeypatch):
    import json as _json

    import backend.routers.chat as chat

    async def fake_prepare(req, notify=None):
        notify("remember")
        return [{"role": "user", "content": "hi"}], {}

    async def fake_stream(final, stats=None, **kw):
        yield "Mochi!"

    monkeypatch.setattr(chat, "_prepare", fake_prepare)
    monkeypatch.setattr(chat, "astream_chat", fake_stream)
    req = chat.ChatRequest(messages=[{"role": "user", "content": "[a]: what's my cat called?"}], voice_mode=True)
    response = await chat.chat_stream(req, None)
    lines = [_json.loads(chunk) async for chunk in response.body_iterator]
    assert lines[0] == {"thinking": "remember"}
    assert lines[1] == {"delta": "Mochi!"}
    assert lines[-1]["done"]


async def test_stream_keeps_showing_signs_of_life(monkeypatch):
    import asyncio as _asyncio
    import json as _json

    import backend.routers.chat as chat

    async def slow_prepare(req, notify=None):
        notify("web_search")
        await _asyncio.sleep(0.35)  # a long lookup with nothing else said
        return [{"role": "user", "content": "hi"}], {}

    async def fake_stream(final, stats=None, **kw):
        yield "Found it."

    real_config = chat.get_config()
    monkeypatch.setattr(chat, "get_config", lambda: {**real_config, "voice": {"thinking_heartbeat": 0.1}})
    monkeypatch.setattr(chat, "_prepare", slow_prepare)
    monkeypatch.setattr(chat, "astream_chat", fake_stream)
    req = chat.ChatRequest(messages=[{"role": "user", "content": "[a]: who won?"}], voice_mode=True)
    response = await chat.chat_stream(req, None)
    lines = [_json.loads(chunk) async for chunk in response.body_iterator]
    thinking = [line["thinking"] for line in lines if "thinking" in line]
    assert thinking[0] == "web_search" and 2 <= thinking.count("still_thinking") <= 4
    assert {"delta": "Found it."} in lines


def test_found_answer_is_the_last_instruction():
    from backend.routers.chat import reply_guidance

    messages = [{"role": "user", "content": "[a]: who won the race?"}]
    guidance = reply_guidance(messages, voice=True, has_context=True, searched_web=True,
                              found_answer="Verstappen won the Singapore GP (f1.com)")
    assert guidance.rstrip().splitlines()[-1].startswith("- Your first sentence tells them what you found")
    assert "Verstappen" in guidance.splitlines()[-1]
    assert "\"Verstappen" not in guidance  # facts to retell, not a quote to read out


def test_recent_lookups_are_remembered_for_a_while(monkeypatch):
    import backend.routers.chat as chat

    monkeypatch.setattr(chat, "_recent_lookups", {})
    chat.remember_lookup("best jungler?", "Master Yi (u.gg)", "voice-1")
    assert "best jungler? -> Master Yi (u.gg)" in chat.recent_lookups_text("voice-1")
    assert chat.recent_lookups_text("voice-2") == ""  # another channel doesn't see it
    monkeypatch.setattr(chat, "_LOOKUP_MEMORY_SECONDS", -1)
    assert chat.recent_lookups_text("voice-1") == ""


async def test_hanging_up_during_a_lookup_cancels_it(monkeypatch):
    """"No, don't look that up" while she was searching: the bot drops the request, and the lookup must stop
    too instead of running on for nobody."""
    import asyncio as _asyncio

    import backend.routers.chat as chat

    state = {"cancelled": False}

    async def endless_prepare(req, notify=None):
        notify("web_search")
        try:
            await _asyncio.sleep(30)
        except _asyncio.CancelledError:
            state["cancelled"] = True
            raise

    monkeypatch.setattr(chat, "_prepare", endless_prepare)
    req = chat.ChatRequest(messages=[{"role": "user", "content": "[a]: how do i ...?"}], voice_mode=True)
    response = await chat.chat_stream(req, None)
    body = response.body_iterator
    await body.__anext__()  # {"thinking": "web_search"}
    await body.aclose()     # the bot hung up
    await _asyncio.sleep(0.05)
    assert state["cancelled"]
