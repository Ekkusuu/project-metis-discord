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
_SELECTION_PROMPT = """You decide which tools an AI assistant needs before replying to the latest chat message.

Available tools:
{tools}

Rules:
- Greetings, small talk and simple reactions need NO tools: reply with [].
- Use knowledge_search when a user talks about themselves, other people, their past, or anything that may be
  in stored notes or memories of earlier conversations.
- Use web_search for current events, news, prices, weather, sports, releases, or facts you are unsure about.
- Use fetch_webpage only with a full URL the user gave.
- Queries must be short keyword phrases (2-6 words).

Respond with a JSON array of tool calls, for example:
[{{"tool":"knowledge_search","arguments":{{"query":"sava favorite games"}}}}]
[{{"tool":"web_search","arguments":{{"query":"weather Berlin today"}}}}]
[]"""


def _selection_schema(registry: "ToolRegistry", max_tools: int) -> Dict[str, Any]:
    """JSON schema that only admits calls to enabled tools with their required arguments."""
    variants = []
    for tool in registry.get_all_enabled().values():
        props = {
            p.name: {"type": "number" if p.type == "number" else "string"}
            for p in tool.definition.parameters
            if p.required and p.type in ("string", "number")
        }
        variants.append({
            "type": "object",
            "properties": {"tool": {"const": tool.definition.name}, "arguments": {"type": "object", "properties": props}},
        })
    return {"type": "array", "maxItems": max(1, max_tools), "items": {"oneOf": variants}}


def _extract_json_array(text: str) -> Optional[List[Any]]:
    """Find and parse the first JSON array (or single object) in model output."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = json.loads(text[start:i + 1])
                        except json.JSONDecodeError:
                            break
                        return parsed if isinstance(parsed, list) else [parsed]
            start = text.find(opener, start + 1)
    return None


def parse_tool_calls(response: str, fallback_query: str, valid_tools: List[str]) -> List[ToolCall]:
    """
    Turn the selector's output into ToolCalls. Falls back to regex recovery of
    `"tool": "name"` mentions when the JSON is truncated or malformed.
    """
    calls: List[ToolCall] = []
    parsed = _extract_json_array(response)
    if parsed is not None:
        for item in parsed:
            if isinstance(item, dict) and isinstance(item.get("tool"), str):
                args = item.get("arguments") or item.get("params") or {}
                calls.append(ToolCall(item["tool"], args if isinstance(args, dict) else {}, response))
    else:
        for name in dict.fromkeys(re.findall(r'"tool"\s*:\s*"(\w+)"', response)):
            match = re.search(rf'"tool"\s*:\s*"{name}"[^{{}}]*?"query"\s*:\s*"([^"]+)"', response)
            query = match.group(1) if match and len(match.group(1)) >= 3 else fallback_query
            calls.append(ToolCall(name, {"query": query}, response))

    valid: List[ToolCall] = []
    seen = set()
    for call in calls:
        if call.tool_name not in valid_tools:
            print(f"Warning: LLM selected unknown tool '{call.tool_name}'")
            continue
        if call.tool_name == "fetch_webpage":
            if not str(call.arguments.get("url", "")).lower().startswith(("http://", "https://")):
                continue  # the model invented a non-URL; nothing to open
        elif not str(call.arguments.get("query", "")).strip():
            call.arguments["query"] = fallback_query
        key = (call.tool_name, json.dumps(call.arguments, sort_keys=True))
        if key not in seen:
            seen.add(key)
            valid.append(call)
    return valid


async def select_tools(
    messages: List[Dict[str, str]],
    last_user_message: str,
    max_tools: int = 3,
) -> List[ToolCall]:
    """Ask the LLM which tools to use for the latest message."""
    registry = get_tool_registry()
    schemas = registry.get_tools_schema()
    if not schemas:
        return []

    tools_text = "\n".join(f"- {s['function']['name']}: {s['function']['description']}" for s in schemas)
    context_count = int(get_config().get("tools", {}).get("context_messages", 3))
    recent = [m for m in messages if m.get("role") in ("user", "assistant")][-context_count - 1:-1] if context_count > 0 else []
    context = "\n".join(f"{'User' if m['role'] == 'user' else 'AI'}: {m['content'][:300]}" for m in recent)

    user_content = (f"Recent conversation:\n{context}\n\n" if context else "") + f"Latest message: {last_user_message}\n\nTools?"
    llm_messages = [
        {"role": "system", "content": _SELECTION_PROMPT.format(tools=tools_text)},
        {"role": "user", "content": user_content},
    ]

    enabled = list(registry.get_all_enabled())
    calls: List[ToolCall] = []

    # A shared link is always read; no need to ask the model.
    if "fetch_webpage" in enabled:
        from backend.tools.web_page_tool import find_urls
        for url in find_urls(last_user_message)[:2]:
            calls.append(ToolCall("fetch_webpage", {"url": url}, "url in message"))

    try:
        selected = await asyncio.to_thread(
            json_completion, llm_messages, _selection_schema(registry, max_tools), 0.1, 300
        )
        response = json.dumps(selected)
    except Exception as e:
        print(f"Grammar-constrained tool selection failed ({e}); falling back to free text")
        try:
            response = await asyncio.to_thread(chat_completion, llm_messages, 0.1, 0.9, 200)
        except Exception as e2:
            print(f"Error selecting tools: {e2}")
            return calls

    for call in parse_tool_calls(response, last_user_message, enabled):
        if call.tool_name == "fetch_webpage" and any(c.tool_name == "fetch_webpage" for c in calls):
            continue
        calls.append(call)
    calls = calls[:max(max_tools, len([c for c in calls if c.raw_output == "url in message"]))]
    print(f"Tool selection: {[c.tool_name for c in calls] or 'none'}")
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
