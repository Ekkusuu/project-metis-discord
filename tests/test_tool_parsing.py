"""Unit tests for tool-call parsing/formatting in backend/tool_manager.py.

No LLM is involved: parse_tool_calls() works on the raw selector output string.
"""
from __future__ import annotations

import json

import pytest

from backend.tool_manager import ToolResult, format_tool_results_for_context, parse_tool_calls

VALID_TOOLS = ["knowledge_search", "web_search"]
FALLBACK = "fallback query"


def _parse(response: str, valid_tools=None):
    return parse_tool_calls(response, FALLBACK, valid_tools or VALID_TOOLS)


def test_parse_clean_json_array():
    response = '[{"tool": "knowledge_search", "arguments": {"query": "favorite games"}}]'
    calls = _parse(response)
    assert len(calls) == 1
    assert calls[0].tool_name == "knowledge_search"
    assert calls[0].arguments == {"query": "favorite games"}
    assert calls[0].raw_output == response


def test_parse_single_object():
    calls = _parse('{"tool": "web_search", "arguments": {"query": "weather berlin"}}')
    assert [c.tool_name for c in calls] == ["web_search"]
    assert calls[0].arguments == {"query": "weather berlin"}


def test_parse_markdown_fenced_json():
    response = '```json\n[{"tool": "knowledge_search", "arguments": {"query": "hello world"}}]\n```'
    calls = _parse(response)
    assert len(calls) == 1
    assert calls[0].tool_name == "knowledge_search"


def test_parse_json_preceded_by_prose():
    response = 'Sure, here is my pick: [{"tool": "knowledge_search", "arguments": {"query": "dark souls lore"}}]'
    calls = _parse(response)
    assert [c.tool_name for c in calls] == ["knowledge_search"]
    assert calls[0].arguments == {"query": "dark souls lore"}


def test_parse_empty_array():
    assert _parse("[]") == []


def test_parse_truncated_json_recovers_tool_name_with_fallback_query():
    response = '[{"tool": "knowledge_search", "arguments": {"query": "sava fav'
    calls = _parse(response)
    assert [c.tool_name for c in calls] == ["knowledge_search"]
    assert calls[0].arguments == {"query": FALLBACK}


def test_parse_truncated_json_recovers_query_when_present():
    response = '[{"tool": "knowledge_search", "query": "dark souls lore", "arguments": {"qu'
    calls = _parse(response)
    assert [c.tool_name for c in calls] == ["knowledge_search"]
    assert calls[0].arguments == {"query": "dark souls lore"}


def test_parse_unknown_tool_is_filtered_out():
    response = '[{"tool": "delete_everything", "arguments": {"query": "all the things"}}]'
    assert _parse(response) == []


def test_parse_duplicate_calls_are_deduplicated():
    response = (
        '[{"tool": "knowledge_search", "arguments": {"query": "same query"}},'
        ' {"tool": "knowledge_search", "arguments": {"query": "same query"}}]'
    )
    calls = _parse(response)
    assert len(calls) == 1


def test_parse_keeps_same_tool_with_different_arguments():
    response = (
        '[{"tool": "knowledge_search", "arguments": {"query": "first topic"}},'
        ' {"tool": "knowledge_search", "arguments": {"query": "second topic"}}]'
    )
    calls = _parse(response)
    assert [c.arguments["query"] for c in calls] == ["first topic", "second topic"]


def test_parse_missing_query_is_replaced_with_fallback():
    calls = _parse('[{"tool": "knowledge_search", "arguments": {}}]')
    assert calls[0].arguments == {"query": FALLBACK}


def test_parse_blank_query_is_replaced_with_fallback():
    calls = _parse('[{"tool": "web_search", "arguments": {"query": "   "}}]')
    assert calls[0].arguments == {"query": FALLBACK}


def test_parse_non_dict_arguments_become_fallback_query():
    calls = _parse('[{"tool": "knowledge_search", "arguments": "not a dict"}]')
    assert calls[0].arguments == {"query": FALLBACK}


def test_parse_accepts_params_alias():
    calls = _parse('[{"tool": "knowledge_search", "params": {"query": "alias works"}}]')
    assert calls[0].arguments == {"query": "alias works"}


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
