"""
Tool Manager for Project Metis - MCP-style tool system.

This module provides:
- Tool registration with JSON Schema definitions
- Tool selection via LLM
- Tool execution with result handling
- Easy extensibility for new tools
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from backend.llama_engine import chat_completion, get_config
from backend.prompt_loader import resolve_prompt


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
        """Convert to MCP/OpenAI-style JSON schema format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
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
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_prompt_format(self) -> str:
        """Convert to a human-readable prompt format for the LLM."""
        params_str = ""
        for param in self.parameters:
            req = "(required)" if param.required else "(optional)"
            params_str += f"\n    - {param.name} ({param.type}) {req}: {param.description}"
            if param.enum:
                params_str += f" [options: {', '.join(param.enum)}]"
            if param.default is not None:
                params_str += f" [default: {param.default}]"
        
        no_params = "\n    (no parameters)"
        params_section = params_str if params_str else no_params
        return f"- **{self.name}**: {self.description}{params_section}"


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition with JSON schema."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with the given parameters.
        
        Returns:
            Dict with at least:
            - "success": bool
            - "result": Any (the tool's output)
            - "error": Optional[str] (if success is False)
        """
        pass


@dataclass
class ToolCall:
    """Represents a parsed tool call from LLM output."""
    tool_name: str
    arguments: Dict[str, Any]
    raw_output: str = ""


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None  # Full tool response for UI


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._enabled_tools: set = set()
    
    def register(self, tool: BaseTool, enabled: bool = True) -> None:
        """Register a tool in the registry."""
        name = tool.definition.name
        self._tools[name] = tool
        if enabled:
            self._enabled_tools.add(name)
    
    def unregister(self, name: str) -> None:
        """Remove a tool from the registry."""
        self._tools.pop(name, None)
        self._enabled_tools.discard(name)
    
    def enable(self, name: str) -> None:
        """Enable a registered tool."""
        if name in self._tools:
            self._enabled_tools.add(name)
    
    def disable(self, name: str) -> None:
        """Disable a tool (keeps it registered but unavailable)."""
        self._enabled_tools.discard(name)
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name (only if enabled)."""
        if name in self._enabled_tools:
            return self._tools.get(name)
        return None
    
    def get_all_enabled(self) -> Dict[str, BaseTool]:
        """Get all enabled tools."""
        return {name: self._tools[name] for name in self._enabled_tools if name in self._tools}
    
    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """Get JSON schemas for all enabled tools."""
        return [
            self._tools[name].definition.to_json_schema()
            for name in self._enabled_tools
            if name in self._tools
        ]
    
    def get_tools_prompt(self) -> str:
        """Get a formatted prompt describing all enabled tools."""
        if not self._enabled_tools:
            return "No tools available."
        
        tool_descriptions = []
        for name in sorted(self._enabled_tools):
            if name in self._tools:
                tool_descriptions.append(self._tools[name].definition.to_prompt_format())
        
        return "\n\n".join(tool_descriptions)


# Global tool registry
_tool_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
        _register_default_tools()
    return _tool_registry


def _register_default_tools():
    """Register the default set of tools."""
    from backend.tools import get_all_tools
    
    registry = get_tool_registry()
    config = get_config()
    tools_cfg = config.get("tools", {})
    
    for tool in get_all_tools():
        tool_name = tool.definition.name
        # Check if tool is enabled in config (default to True)
        enabled = tools_cfg.get(tool_name, {}).get("enabled", True)
        registry.register(tool, enabled=enabled)


def parse_tool_calls(llm_output: str) -> List[ToolCall]:
    """
    Parse tool calls from LLM output.
    
    Supports multiple formats:
    1. JSON format: {"tool": "name", "arguments": {...}}
    2. Array format: [{"tool": "name", "arguments": {...}}, ...]
    3. Function call format: tool_name(arg1="value1", arg2="value2")
    4. Simple format: TOOL: name
                     ARGUMENTS: {"key": "value"}
    
    Returns:
        List of ToolCall objects
    """
    tool_calls = []
    
    # Try JSON object/array format first
    try:
        # Look for JSON in the output
        json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', llm_output)
        if json_match:
            parsed = json.loads(json_match.group(1))
            
            # Handle array of tool calls
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "tool" in item:
                        tool_calls.append(ToolCall(
                            tool_name=item["tool"],
                            arguments=item.get("arguments", item.get("params", {})),
                            raw_output=llm_output
                        ))
            # Handle single tool call
            elif isinstance(parsed, dict):
                if "tool" in parsed:
                    tool_calls.append(ToolCall(
                        tool_name=parsed["tool"],
                        arguments=parsed.get("arguments", parsed.get("params", {})),
                        raw_output=llm_output
                    ))
                # Handle {"tools": [...]} format
                elif "tools" in parsed and isinstance(parsed["tools"], list):
                    for item in parsed["tools"]:
                        if isinstance(item, dict) and "tool" in item:
                            tool_calls.append(ToolCall(
                                tool_name=item["tool"],
                                arguments=item.get("arguments", item.get("params", {})),
                                raw_output=llm_output
                            ))
            
            if tool_calls:
                return tool_calls
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    
    # Try simple TOOL: format
    tool_match = re.search(r'TOOL:\s*(\w+)', llm_output, re.IGNORECASE)
    if tool_match:
        tool_name = tool_match.group(1)
        arguments = {}
        
        # Try to find ARGUMENTS: section
        args_match = re.search(r'ARGUMENTS:\s*(\{.*?\})', llm_output, re.DOTALL | re.IGNORECASE)
        if args_match:
            try:
                arguments = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Also try QUERY: for backward compatibility with RAG
        query_match = re.search(r'QUERY:\s*(.+?)(?:\n|$)', llm_output, re.IGNORECASE)
        if query_match and not arguments:
            arguments = {"query": query_match.group(1).strip()}
        
        tool_calls.append(ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            raw_output=llm_output
        ))
        return tool_calls
    
    # Try NONE/NO_TOOL format
    if re.search(r'\b(NONE|NO_TOOL|NO TOOL)\b', llm_output, re.IGNORECASE):
        return []  # Explicitly no tools needed
    
    return tool_calls


def _extract_partial_tool_call(response: str, user_message: str) -> List[ToolCall]:
    """
    Extract tool calls from truncated/malformed JSON by finding tool names
    and using the user's message as the query argument.
    
    This is a fallback when JSON parsing fails due to truncation.
    """
    tool_calls = []
    registry = get_tool_registry()
    available_tools = list(registry._enabled_tools)
    
    # Find ALL tool mentions in the response
    tool_matches = re.findall(r'"tool"\s*:\s*"(\w+)"', response)
    seen_tools = set()
    
    for tool_name in tool_matches:
        # Skip duplicates and invalid tools
        if tool_name in seen_tools or tool_name not in available_tools:
            continue
        seen_tools.add(tool_name)
        
        # Try to extract the query for this specific tool
        # Look for pattern: "tool": "name"..."query": "value"
        pattern = rf'"tool"\s*:\s*"{tool_name}"[^{{}}]*?"query"\s*:\s*"([^"]+)"'
        query_match = re.search(pattern, response)
        
        if query_match and len(query_match.group(1)) >= 5:
            query = query_match.group(1)
        else:
            # Fall back to user's message as query
            query = user_message
        
        tool_calls.append(ToolCall(
            tool_name=tool_name,
            arguments={"query": query},
            raw_output=response
        ))
        print(f"Recovered tool call: {tool_name} with query: {query[:50]}")
    
    return tool_calls


async def select_tools(
    messages: List[Dict[str, str]],
    last_user_message: str,
    max_tools: int = 3
) -> List[ToolCall]:
    """
    Use LLM to select which tools to use based on conversation context.
    Uses MCP-style JSON schemas for reliable tool selection.
    
    Args:
        messages: Conversation history
        last_user_message: The latest user message
        max_tools: Maximum number of tools to select
    
    Returns:
        List of ToolCall objects with selected tools and their arguments
    """
    config = get_config()
    tools_cfg = config.get("tools", {})
    
    registry = get_tool_registry()
    tools_schema = registry.get_tools_schema()
    
    if not tools_schema:
        return []
    
    # Build conversation context
    query_context_messages = tools_cfg.get("context_messages", 3)
    
    recent_messages = []
    if query_context_messages > 0 and messages and len(messages) > 1:
        recent_messages = [
            msg for msg in messages[-query_context_messages-1:]
            if msg.get("role") in ["user", "assistant"]
        ]
    
    if len(recent_messages) > 1:
        conversation_context = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'AI'}: {msg['content']}"
            for msg in recent_messages[:-1]
        ])
    else:
        conversation_context = ""
    
    # Build compact tool list for prompt
    tool_names = [t["function"]["name"] for t in tools_schema]
    tool_descriptions = "\n".join([
        f"- {t['function']['name']}: {t['function']['description'][:100]}"
        for t in tools_schema
    ])
    
    # Compact system prompt for reliable, short responses
    # Note: Double braces {{ }} escape literal braces in f-strings
    system_prompt = f"""Select tools for the user's request. Available tools:
{tool_descriptions}

Respond with JSON array. Keep queries SHORT (2-5 words). Examples:
- [{{"tool":"knowledge_search","arguments":{{"query":"user preferences"}}}}]
- [{{"tool":"web_search","arguments":{{"query":"current weather NYC"}}}}]
- [] (no tools needed)

ONLY output the JSON array, nothing else."""

    # Build user message with context
    if conversation_context:
        user_content = f"""Context: {conversation_context[:200]}
Message: {last_user_message}
Tools?"""
    else:
        user_content = f"""Message: {last_user_message}
Tools?"""
    
    llm_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    try:
        # Generate tool selection with low temperature for reliability
        # Use generous max_tokens to avoid truncation
        response = chat_completion(llm_messages, temperature=0.1, top_p=0.9, max_tokens=1024)
        
        # Strip thinking tags if present (for reasoning models)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r'^.*?</think>\s*', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean response - extract JSON from markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # Parse JSON response directly
        tool_calls = []
        try:
            parsed = json.loads(response)
            
            # Handle empty array (no tools needed)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "tool" in item:
                        tool_calls.append(ToolCall(
                            tool_name=item["tool"],
                            arguments=item.get("arguments", {}),
                            raw_output=response
                        ))
        except json.JSONDecodeError:
            # Fallback: try to extract tool name and reconstruct with user's query
            print(f"Tool selection JSON parse failed, trying extraction: {response[:150]}")
            tool_calls = _extract_partial_tool_call(response, last_user_message)
        
        # Validate tool calls - only return tools that exist
        validated_calls = []
        for call in tool_calls[:max_tools]:
            if registry.get(call.tool_name):
                validated_calls.append(call)
            else:
                print(f"Warning: LLM selected unknown tool '{call.tool_name}'")
        
        print(f"Tool selection: {[c.tool_name for c in validated_calls]} from response: {response[:100]}")
        return validated_calls
        
    except Exception as e:
        print(f"Error selecting tools: {e}")
        import traceback
        traceback.print_exc()
        return []


async def execute_tools(
    tool_calls: List[ToolCall],
    extra_context: Optional[Dict[str, Any]] = None
) -> List[ToolResult]:
    """
    Execute a list of tool calls and return their results.
    
    Args:
        tool_calls: List of ToolCall objects to execute
        extra_context: Optional extra context to pass to tools (e.g., conversation_context)
    
    Returns:
        List of ToolResult objects
    """
    import time
    
    registry = get_tool_registry()
    results = []
    extra_context = extra_context or {}
    
    for call in tool_calls:
        tool = registry.get(call.tool_name)
        if not tool:
            results.append(ToolResult(
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=f"Tool '{call.tool_name}' not found or disabled"
            ))
            continue
        
        try:
            start_time = time.perf_counter()
            # Merge extra_context with call arguments (call args take precedence)
            merged_args = {**extra_context, **call.arguments}
            result = await tool.execute(**merged_args)
            execution_time = time.perf_counter() - start_time
            
            results.append(ToolResult(
                tool_name=call.tool_name,
                success=result.get("success", True),
                result=result.get("result"),
                error=result.get("error"),
                execution_time=execution_time,
                raw_response=result  # Store full response for UI access
            ))
        except Exception as e:
            results.append(ToolResult(
                tool_name=call.tool_name,
                success=False,
                result=None,
                error=str(e)
            ))
    
    return results


def format_tool_results_for_context(results: List[ToolResult]) -> str:
    """
    Format tool results to be injected into the conversation context.
    
    Args:
        results: List of ToolResult objects
    
    Returns:
        Formatted string for context injection
    """
    if not results:
        return ""
    
    formatted_parts = []
    
    for result in results:
        if result.success and result.result:
            if isinstance(result.result, str):
                formatted_parts.append(f"[{result.tool_name}]:\n{result.result}")
            elif isinstance(result.result, dict):
                # For structured results, format nicely
                formatted_parts.append(f"[{result.tool_name}]:\n{json.dumps(result.result, indent=2)}")
            elif isinstance(result.result, list):
                # For list results
                formatted_parts.append(f"[{result.tool_name}]:\n{json.dumps(result.result, indent=2)}")
        elif not result.success:
            formatted_parts.append(f"[{result.tool_name} error]: {result.error}")
    
    return "\n\n".join(formatted_parts)
