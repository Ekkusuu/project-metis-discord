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
def fake_model(monkeypatch):
    import backend.tool_manager as tm

    calls = {"decide": [], "query": []}
    state = {"choice": "none", "query": "f1 race winner"}

    def fake_json(messages, schema, *args, **kwargs):
        calls["decide"].append(schema)
        return state["choice"]

    def fake_chat(messages, *args, **kwargs):
        calls["query"].append(messages)
        return state["query"]

    monkeypatch.setattr(tm, "json_completion", fake_json)
    monkeypatch.setattr(tm, "chat_completion", fake_chat)
    monkeypatch.setattr(tm, "_last_search", {})  # no search from an earlier test
    return state, calls


async def test_small_talk_runs_only_knowledge_search(fake_model):
    from backend.tool_manager import select_tools

    state, calls = fake_model
    result = await select_tools([], "[Sava]: lol nice", 2)
    assert [(c.tool_name, c.arguments) for c in result] == [("knowledge_search", {"query": "lol nice"})]
    assert calls["decide"] == []  # a plain statement doesn't even ask the model (it cost 0.2-0.7 s per line)
    assert calls["query"] == []
    await select_tools([], "[Sava]: who is the best support?", 2)
    assert calls["decide"][0] == {"enum": ["none", "remember", "web_search"]}


async def test_web_search_choice_gets_a_generated_query(fake_model):
    from backend.tool_manager import select_tools

    state, _ = fake_model
    state["choice"] = "web_search"
    result = await select_tools([], "[Sava]: who won the f1 race yesterday?", 2)
    assert ("web_search", {"query": "f1 race winner", "search_type": "news"}) in [(c.tool_name, c.arguments) for c in result]


async def test_links_are_always_fetched(fake_model):
    from backend.tool_manager import select_tools

    result = await select_tools([], "[Sava]: summarize https://example.com/post please", 2)
    assert ("fetch_webpage", {"url": "https://example.com/post"}) in [(c.tool_name, c.arguments) for c in result]


async def test_remember_uses_the_message_without_writing_a_web_query(fake_model):
    from backend.tool_manager import select_tools

    state, calls = fake_model
    state["choice"] = "remember"
    result = await select_tools([], "[Sava]: what did I tell you about my exam?", 2)
    assert [(c.tool_name, c.arguments) for c in result] == [
        ("knowledge_search", {"query": "what did I tell you about my exam?"}),
        ("remember", {"query": "what did I tell you about my exam?"}),
    ]
    assert calls["query"] == []  # the remember tool writes its own memory queries


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
    assert guidance.rstrip().splitlines()[-1].startswith("- Your first sentence gives them what you found")
    assert "Verstappen" in guidance.splitlines()[-1]


def test_recent_lookups_are_remembered_for_a_while(monkeypatch):
    import backend.routers.chat as chat

    monkeypatch.setattr(chat, "_recent_lookups", {})
    chat.remember_lookup("best jungler?", "Master Yi (u.gg)", "voice-1")
    assert "best jungler? -> Master Yi (u.gg)" in chat.recent_lookups_text("voice-1")
    assert chat.recent_lookups_text("voice-2") == ""  # another channel doesn't see it
    monkeypatch.setattr(chat, "_LOOKUP_MEMORY_SECONDS", -1)
    assert chat.recent_lookups_text("voice-1") == ""


@pytest.mark.parametrize("message", [
    "Could you tell me by win rate what is the best champion",
    "what's the strongest support right now",
    "is the new iphone worth it",
])
async def test_fact_questions_search_without_being_asked_to(fake_model, message):
    """People don't say "search": "what is the best champion by win rate" got no search (the model said none)."""
    from backend.tool_manager import select_tools

    result = await select_tools([], f"[Sava]: {message}", 2)
    assert "web_search" in [c.tool_name for c in result]


@pytest.mark.parametrize("message", ["You're kinda mean. Why are you so mean?", "No, I know about that. I think I'm pretty kind.",
                                     "Fair enough.", "how are we doing?"])
async def test_chat_stays_without_a_search(fake_model, message):
    from backend.tool_manager import select_tools

    result = await select_tools([], f"[Sava]: {message}", 2)
    assert "web_search" not in [c.tool_name for c in result]


@pytest.mark.parametrize("message", ["Uh could you double-check, but this time for Emerald Plus?",
                                     "Uh could you use u.g as reference?"])
async def test_follow_ups_redo_the_last_search_with_its_site(fake_model, monkeypatch, message):
    """The model said "none" to these and Metis kept saying "I'm pulling up u.gg" without searching."""
    import time
    import backend.tool_manager as tm
    from backend.tool_manager import select_tools

    state, calls = fake_model
    state["query"] = "best champion win rate emerald plus site:u.gg"
    monkeypatch.setitem(tm._last_search, "", {"time": time.time(), "query": "best champion win rate site:u.gg"})
    result = await select_tools([], f"[Sava]: {message}", 2)
    searches = [c.arguments["query"] for c in result if c.tool_name == "web_search"]
    assert searches == ["best champion win rate emerald plus site:u.gg"]
    assert "Last search: best champion win rate site:u.gg" in calls["query"][0][-1]["content"]


async def test_no_follow_up_search_without_an_earlier_one(fake_model, monkeypatch):
    import backend.tool_manager as tm
    from backend.tool_manager import select_tools

    monkeypatch.setitem(tm._last_search, "", {"time": 0.0, "query": "old"})
    result = await select_tools([], "[Sava]: what about you?", 2)
    assert "web_search" not in [c.tool_name for c in result]


@pytest.mark.parametrize("metis, searches", [("I'll check u.gg for Emerald Plus right now.", True),
                                             ("Thresh has the best win rate in Emerald Plus.", False)])
async def test_nudges_search_only_after_a_promise(fake_model, metis, searches):
    """"So are we checking?" after she said she'd check runs the search; after she answered, it doesn't."""
    from backend.tool_manager import select_tools

    messages = [{"role": "user", "content": "[Sava]: best champion for emerald plus on u.gg?"},
                {"role": "assistant", "content": metis}, {"role": "user", "content": "[Sava]: So are we checking?"}]
    result = await select_tools(messages, messages[-1]["content"], 2)
    assert ("web_search" in [c.tool_name for c in result]) == searches


async def test_last_search_is_per_conversation(fake_model, monkeypatch):
    """A search in one channel isn't a "last search" for a follow-up in another."""
    import time
    import backend.tool_manager as tm

    monkeypatch.setitem(tm._last_search, "voice-1", {"time": time.time(), "query": "best champion site:u.gg"})
    calls = await tm.decide_tools([], "[Sava]: could you double-check, but this time for Emerald Plus?", 2,
                                  conversation_id="voice-2")
    assert "web_search" not in [c.tool_name for c in calls]
