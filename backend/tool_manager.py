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
# - knowledge_search runs on every message (tools.knowledge_search.always): a quick lookup (~50 ms) that
#   only keeps close matches, so there's nothing for the model to get wrong. The thorough search is
#   `remember`, which the model chooses like the other tools.
# - For the remaining tools the model fills in ONE small plan (JSON constrained by a schema), the way
#   Perplexica's classifier does: the latest message rewritten so it stands on its own ("do it", "this time
#   for Emerald Plus", "what about him" resolved from the conversation), the action, and the search query.
#   Rewriting first is what lets a small model see what a follow-up is about; regexes for follow-ups,
#   nudges and "facts" kept missing new phrasings and misfiring on others ("what's the best way to do it",
#   about something else entirely, searched an older topic).
_PLAN_PROMPT = """You plan the next step for Metis, a girl hanging out with friends in a Discord voice chat, \
before she replies to the latest message.
- standalone: the latest message rewritten so it makes sense on its own. Use the conversation and the last \
web search to resolve "it", "that", "do it", "check again", "this time for ...", "what about ...".
- action: what Metis needs before replying:
{options}
  - none: chatting, jokes, feelings, opinions, roleplay or hypotheticals, and questions about Metis herself \
or the people in the chat
- query: for web_search, 2-8 search keywords for the standalone message (add site:domain.com if they name a \
website; speech-to-text mangles names, so write the real domain). Otherwise empty.
- recent: true if it's about news or very recent events.
Metis's own knowledge is old and often wrong about specifics. When the message wants facts about a game, \
product, person, team, show, place or event, or asks to look something up or check again, choose web_search. \
If you are unsure whether a factual question needs the web, choose web_search.
But choose none when:
- they ask Metis herself: what she thinks, likes, would do, or imagines, hypotheticals and "pretend" games, \
even if they mention real things. She answers those in her own words, like a person would.
- they said not to look it up, or to just tell them what she thinks."""


# The last web search per conversation, shown to the planner so "check again, but for X" keeps its topic and site.
_SEARCH_MEMORY_SECONDS = 600
_last_search: Dict[str, Dict[str, Any]] = {}  # conversation -> {"time", "query"}


def _plan_schema(actions: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "standalone": {"type": "string", "maxLength": 240},
            "action": {"enum": ["none", *actions]},
            "query": {"type": "string", "maxLength": 120},
            "recent": {"type": "boolean"},
        },
        "required": ["standalone", "action", "query", "recent"],
    }


def _example(context: str, latest: str, standalone: str, action: str, query: str = "", recent: bool = False):
    question = (f"Conversation:\n{context}\n\n" if context else "") + f"Latest message: {latest}"
    answer = json.dumps({"standalone": standalone, "action": action, "query": query, "recent": recent})
    return [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]


# Worked examples, taught as earlier turns (small models follow examples far better than rules).
_PLAN_EXAMPLES = [
    *_example("[Robin]: when was the tesla roadster released?\nMetis: I think 2008?",
              "[Robin]: can't you just search for the date?",
              "When was the Tesla Roadster released?", "web_search", "Tesla Roadster release date"),
    *_example("[Sam]: who won the f1 race yesterday\nMetis: Want me to look it up?", "[Sam]: then do it",
              "Who won the F1 race yesterday?", "web_search", "F1 race winner", True),
    *_example("", "[Lee]: check on rotten tomatoes how good the new dune movie is",
              "How good is the new Dune movie on Rotten Tomatoes?", "web_search",
              "Dune movie reviews site:rottentomatoes.com"),
    *_example("Last web search: best phone camera site:gsmarena.com\n[Kai]: what's the best phone camera on "
              "gsmarena\nMetis: The Pixel, by their score.", "[Kai]: could you double-check, but this time under "
              "300 dollars?", "What's the best phone camera under 300 dollars on GSMArena?", "web_search",
              "best phone camera under 300 dollars site:gsmarena.com"),
    *_example("[Mia]: what's a good counter to Yasuo\nMetis: Malzahar shuts him down.",
              "[Mia]: thanks! you're the best", "Thanks, you're the best.", "none"),
    *_example("[Jo]: let's plan a heist on a casino\nMetis: Ooh, I'm in.",
              "[Jo]: so how would you go about it? what's the best way to do it?",
              "How would Metis go about the casino heist we're imagining?", "none"),
    *_example("[Ana]: hi Metis", "[Ana]: who are you anyway? are you a girl?",
              "Who is Metis, and is she a girl?", "none"),
    *_example("[Lee]: imagine we were villains in a movie", "[Lee]: so how would you set a building on fire, "
              "in real life? what's the most effective way?",
              "How would Metis set a building on fire in real life?", "none"),
    *_example("[Lee]: what's the best champion in league?\nMetis: Let me check.",
              "[Lee]: no, don't look it up, just tell me what you think",
              "What does Metis personally think is the best champion in League?", "none"),
    *_example("[Kim]: i just got a new keyboard", "[Kim]: if you could have any superpower what would it be?",
              "What superpower would Metis want?", "none"),
    *_example("[Max]: I told you about my exam last week", "[Max]: do you remember what subject it was?",
              "What subject was Max's exam last week?", "remember"),
]


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


def keep_named_site(query: str, message: str) -> str:
    """
    The query without its "site:" if the latest message doesn't name that site. The query writer carried a
    site from an earlier question into an unrelated one. Names are matched loosely: speech-to-text writes
    "rotten tomatoes" or "u.g" for rottentomatoes.com and u.gg.
    """
    from difflib import SequenceMatcher

    match = re.search(r"\s*\bsite:(\S+)", query)
    if not match:
        return query
    labels = match.group(1).lower().removeprefix("www.").split(".")
    letters = re.sub(r"[^a-z0-9]", "", message.lower())
    for target in {"".join(labels), "".join(labels[:-1]) or labels[0]}:
        if len(target) < 2:
            continue
        for size in (len(target), len(target) - 1):
            for i in range(max(1, len(letters) - size + 1)):
                if SequenceMatcher(None, target, letters[i:i + size]).ratio() >= 0.75:
                    return query
    return (query[:match.start()] + query[match.end():]).strip()


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
    conversation_id: str = "",
) -> List[ToolCall]:
    """Tools the model chooses for this message: one planning call (see _PLAN_PROMPT)."""
    from backend.turn_taking import is_acknowledgement

    enabled = get_tool_registry().get_all_enabled()
    tools_cfg = get_config().get("tools", {})
    text = _message_text(last_user_message)
    calls: List[ToolCall] = []
    decidable = {
        name: tool for name, tool in enabled.items()
        if name != "fetch_webpage" and not (name == "knowledge_search" and _knowledge_always())
        and _first_string_param(tool)
    }
    # "Yeah", "ok", "lol": nothing to plan.
    if not decidable or max_tools <= 0 or len(text) < 3 or is_acknowledgement(text):
        return calls

    context_count = int(tools_cfg.get("context_messages", 6))
    recent = [m for m in messages[:-1] if m.get("role") in ("user", "assistant")][-context_count:]
    lines = [m["content"][:240] if m["role"] == "user" else f"Metis: {m['content'][:240]}" for m in recent]
    previous = _last_search.get(conversation_id) or {"time": 0.0, "query": ""}
    last_query = previous["query"] if time.time() - previous["time"] < _SEARCH_MEMORY_SECONDS else ""
    if last_query:
        lines.insert(0, f"Last web search: {last_query}")
    context = "\n".join(lines)
    options = "\n".join(f"  - {name}: {tool.definition.description}" for name, tool in decidable.items())
    # Her personality (the first system message), so "who's Mochi?" is answered from it, not looked up.
    persona = next((m["content"] for m in messages if m.get("role") == "system"), "")
    question = ((f"About Metis: {persona[:700]}\n\n" if persona else "")
                + (f"Conversation:\n{context}\n\n" if context else "") + f"Latest message: {last_user_message}")

    try:
        plan = await asyncio.to_thread(
            json_completion,
            [{"role": "system", "content": _PLAN_PROMPT.format(options=options)}, *_PLAN_EXAMPLES,
             {"role": "user", "content": question}],
            _plan_schema(list(decidable)), 0.0, 200,
        )
    except Exception as e:
        print(f"Tool decision failed: {e}")
        return calls
    action, standalone = plan.get("action"), str(plan.get("standalone") or text).strip()
    print(f"🧭 plan: {action} | {standalone!r}" + (f" | query {plan.get('query')!r}" if plan.get("query") else ""))
    if action == "remember":
        # The tool writes its own memory queries; the self-contained question is the best starting point.
        calls.append(ToolCall("remember", {"query": standalone}, "chose remember"))
    elif action in decidable:
        written = str(plan.get("query") or "").strip().strip('"')
        # Evidence checks, not word lists: the query may not be built from things only Metis said (her replies
        # can be made up), and a site stays only if this message or the follow-up names it.
        said = " ".join(m["content"][:240] for m in recent if m["role"] == "user")
        metis_said = " ".join(m["content"][:240] for m in recent if m["role"] == "assistant")
        grounded = grounded_query(written, f"{said} {text} {standalone} {last_query}", metis_said, standalone)
        query = keep_named_site(grounded, f"{text} {standalone}")
        arguments: Dict[str, Any] = {_first_string_param(decidable[action]): query}
        if action == "web_search":
            _last_search[conversation_id] = {"time": time.time(), "query": query}
            if plan.get("recent"):
                arguments["search_type"] = "news"
        print(f"🔎 {action} query {query!r}" + (f" ({arguments['search_type']})" if "search_type" in arguments else "")
              + f"\n   made from: {last_user_message!r}"
              + (f"\n   model wrote {written!r}, replaced: it used words only Metis said" if written and grounded != written else "")
              + (f"\n   dropped the site: from {grounded!r}, nobody named that site" if query != grounded else ""))
        calls.append(ToolCall(action, arguments, f"chose {action}"))
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
