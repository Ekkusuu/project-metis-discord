"""
Tool Manager for Project Metis - MCP-style tool system.

- Tools declare a JSON-Schema definition and an async execute().
- select_tools() asks the LLM which tools (if any) a message needs.
- execute_tools() runs the selected tools concurrently.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config import get_config
from backend.llama_engine import chat_completion, json_completion


@dataclass
class ToolParameter:
    """Defines a single parameter for a tool."""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None
    items: Optional[Dict[str, Any]] = None  # For array types


@dataclass
class ToolDefinition:
    """MCP-style tool definition with JSON Schema."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_json_schema(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in self.parameters:
            prop: Dict[str, Any] = {"type": param.type, "description": param.description}
            if param.enum:
                prop["enum"] = param.enum
            if param.items:
                prop["items"] = param.items
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        }


class BaseTool(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition with JSON schema."""

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Run the tool. Must not block the event loop (use asyncio.to_thread for
        blocking work). Returns a dict with "success", "result" and optional "error".
        """


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]
    raw_output: str = ""


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None  # Full tool response for the UI


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}
        self._enabled: set = set()

    def register(self, tool: BaseTool, enabled: bool = True) -> None:
        name = tool.definition.name
        self._tools[name] = tool
        if enabled:
            self._enabled.add(name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name) if name in self._enabled else None

    def get_all_enabled(self) -> Dict[str, BaseTool]:
        return {name: self._tools[name] for name in sorted(self._enabled)}

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        return [tool.definition.to_json_schema() for tool in self.get_all_enabled().values()]


_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        from backend.tools import get_all_tools

        registry = ToolRegistry()
        tools_cfg = get_config().get("tools", {})
        for tool in get_all_tools():
            tool_cfg = tools_cfg.get(tool.definition.name) or {}
            registry.register(tool, enabled=tool_cfg.get("enabled", True))
        _tool_registry = registry
    return _tool_registry


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
# Selection is built for small local models:
# - fetch_webpage runs for any link in the message; no decision needed.
# - knowledge_search runs on every message (tools.knowledge_search.always): it is cheap, and
#   the distance/rerank filters drop irrelevant chunks, so there's nothing for the model to get wrong.
# - For the remaining tools the model makes ONE constrained choice from an enum ("none" or a tool
#   name) and then writes a short query. A 2B model picked the right option 6/8 times this way vs
#   3/8 when asked for a JSON array of calls.
_DECIDE_PROMPT = """Decide whether an AI companion needs a tool before replying to the latest Discord message.
{options}
- none: greetings, small talk, feelings, jokes, or things you can answer from general knowledge
Answer with one option."""

_QUERY_PROMPT = ("Write a short web search query (2-6 keywords) for what the latest message is asking. Use the "
                 "earlier messages only to work out what words like 'it' or 'that' refer to. Output only the query.")

# Recent-events questions go to the news index, which is sorted by date; a text search for
# "what happened in France" returns encyclopedia pages.
_NEWS = re.compile(r"\b(news|happen(ed|ing)?|going on|latest|recent(ly)?|today|tonight|yesterday|this (week|month))\b",
                   re.IGNORECASE)


# Questions that are obviously about fresh information get a web search even if the small model says
# "none" (it missed e.g. "who won the f1 race yesterday?").
_FRESH_INFO = re.compile(
    r"\b(today|tonight|yesterday|tomorrow|right now|currently|latest|newest|recent(ly)?|this (week|month|year)|"
    r"news|score|scores|won|winner|results?|price|prices|cost|costs|stock|weather|forecast|"
    r"release(d|s)?|release date|launch(ed|es)?|came out|come out|announced|when (was|is|did|does|will)|"
    r"what (year|date|day)|exact date|how (much|many))\b",
    re.IGNORECASE,
)
# Someone explicitly asking Metis to look something up.
_SEARCH_REQUEST = re.compile(
    r"\b(search|google|look (it |that |this )?up|look for|check online|find out|browse)\b", re.IGNORECASE
)
_QUESTION_WORDS = re.compile(r"^\W*(who|what|when|where|which|why|how|is|are|was|were|did|does|do|can|could|tell me|give me)\b",
                             re.IGNORECASE)


def is_question_like(text: str) -> bool:
    """Questions, including spoken ones that the transcriber didn't end with '?'."""
    return "?" in text or bool(_QUESTION_WORDS.search(text)) or bool(re.search(r"\b(tell|give) me\b", text, re.IGNORECASE))


def _message_text(message: str) -> str:
    """Strip the '[Name]:' speaker prefix."""
    return re.sub(r"^\[[^\]]+\]:?\s*", "", message).strip()


def _words(text: str) -> set:
    """Distinctive words (short ones like 'date' or 'news' are too common to tell anything), by stem."""
    return {w[:5] for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 5}


def grounded_query(query: str, said_by_people: str, said_by_metis: str, fallback: str) -> str:
    """
    The query, or `fallback` if it picked up words only Metis used. Her replies can be made up, and
    a topic she invented must not become the search. Generic additions ("date", "winner") are fine.
    """
    if not query:
        return fallback
    borrowed = (_words(query) & _words(said_by_metis)) - _words(said_by_people)
    return fallback if borrowed else query


def _first_string_param(tool: BaseTool) -> Optional[str]:
    for p in tool.definition.parameters:
        if p.required and p.type == "string":
            return p.name
    return None


async def select_tools(
    messages: List[Dict[str, str]],
    last_user_message: str,
    max_tools: int = 3,
) -> List[ToolCall]:
    """Decide which tools to run for the latest message (see the notes above)."""
    calls = immediate_calls(last_user_message) + await decide_tools(messages, last_user_message, max_tools)
    print(f"Tool selection: {[c.tool_name for c in calls] or 'none'}")
    return calls


def _knowledge_always() -> bool:
    return (get_config().get("tools", {}).get("knowledge_search") or {}).get("always", True)


def immediate_calls(last_user_message: str) -> List[ToolCall]:
    """Calls that need no model decision (links, always-on knowledge search). Can start right away."""
    enabled = get_tool_registry().get_all_enabled()
    text = _message_text(last_user_message)
    calls: List[ToolCall] = []
    if "fetch_webpage" in enabled:
        from backend.tools.web_page_tool import find_urls
        for url in find_urls(text)[:2]:
            calls.append(ToolCall("fetch_webpage", {"url": url}, "link in message"))
    if "knowledge_search" in enabled and _knowledge_always() and len(text) > 2:
        calls.append(ToolCall("knowledge_search", {"query": text}, "always"))
    return calls


async def decide_tools(
    messages: List[Dict[str, str]],
    last_user_message: str,
    max_tools: int = 3,
) -> List[ToolCall]:
    """Tools the model chooses for this message (one constrained choice, then a query)."""
    enabled = get_tool_registry().get_all_enabled()
    tools_cfg = get_config().get("tools", {})
    text = _message_text(last_user_message)
    calls: List[ToolCall] = []
    decidable = {
        name: tool for name, tool in enabled.items()
        if name != "fetch_webpage" and not (name == "knowledge_search" and _knowledge_always())
        and _first_string_param(tool)
    }
    if not decidable or max_tools <= 0 or len(text) < 3:
        return calls

    context_count = int(tools_cfg.get("context_messages", 3))
    recent = [m for m in messages if m.get("role") in ("user", "assistant")][-context_count - 1:-1] if context_count > 0 else []
    context = "\n".join(f"{'Metis' if m['role'] == 'assistant' else 'User'}: {m['content'][:200]}" for m in recent)
    options = "\n".join(f"- {name}: {tool.definition.description}" for name, tool in decidable.items())
    prompt = (f"Recent conversation:\n{context}\n\n" if context else "") + f"Latest message: {last_user_message}"
    # The query is written from what people said, not from Metis's replies: if she made something up,
    # a query built from it searches for her mistake ("roadster" when asked about news in France).
    said = "\n".join(m["content"][:200] for m in recent if m["role"] == "user")
    query_prompt = (f"Earlier messages:\n{said}\n\n" if said else "") + f"Latest message: {last_user_message}"

    try:
        choice = await asyncio.to_thread(
            json_completion,
            [{"role": "system", "content": _DECIDE_PROMPT.format(options=options)}, {"role": "user", "content": prompt}],
            {"enum": ["none", *decidable]},
            0.0,
            12,
        )
        if choice == "none" and "web_search" in decidable and (
                _SEARCH_REQUEST.search(text) or (is_question_like(text) and _FRESH_INFO.search(text))):
            choice = "web_search"
        if choice in decidable:
            query = await asyncio.to_thread(
                chat_completion,
                [{"role": "system", "content": _QUERY_PROMPT}, {"role": "user", "content": query_prompt}],
                0.0, 0.9, 24,
            )
            query = query.strip().strip('"').splitlines()[0].strip() if query.strip() else ""
            metis_said = " ".join(m["content"][:200] for m in recent if m["role"] == "assistant")
            query = grounded_query(query, f"{said} {text}", metis_said, text)
            arguments: Dict[str, Any] = {_first_string_param(decidable[choice]): query}
            if choice == "web_search" and _NEWS.search(text):
                arguments["search_type"] = "news"
            print(f"🔎 {choice}: {arguments}")
            calls.append(ToolCall(choice, arguments, f"chose {choice}"))
    except Exception as e:
        print(f"Tool decision failed: {e}")
    return calls


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
async def _execute_one(call: ToolCall, extra_context: Dict[str, Any]) -> ToolResult:
    tool = get_tool_registry().get(call.tool_name)
    if tool is None:
        return ToolResult(call.tool_name, False, None, f"Tool '{call.tool_name}' not found or disabled")
    start = time.perf_counter()
    try:
        result = await tool.execute(**{**extra_context, **call.arguments})
        return ToolResult(
            tool_name=call.tool_name,
            success=result.get("success", True),
            result=result.get("result"),
            error=result.get("error"),
            execution_time=time.perf_counter() - start,
            raw_response=result,
        )
    except Exception as e:
        return ToolResult(call.tool_name, False, None, str(e), time.perf_counter() - start)


async def execute_tools(tool_calls: List[ToolCall], extra_context: Optional[Dict[str, Any]] = None) -> List[ToolResult]:
    """Execute tool calls concurrently (results keep the call order)."""
    extra_context = extra_context or {}
    return list(await asyncio.gather(*(_execute_one(c, extra_context) for c in tool_calls)))


def format_tool_results_for_context(results: List[ToolResult]) -> str:
    """Render tool output for injection into the prompt."""
    parts = []
    for r in results:
        if r.success and r.result:
            body = r.result if isinstance(r.result, str) else json.dumps(r.result, indent=2, ensure_ascii=False)
            parts.append(f"[{r.tool_name}]\n{body}")
        elif not r.success:
            parts.append(f"[{r.tool_name}] failed: {r.error}")
    return "\n\n".join(parts)
