# Tool System Guide

## Overview

The tool system allows Metis to dynamically select and use tools based on the conversation context. This is implemented using MCP-style (Model Context Protocol) JSON schemas, making it easy to add new tools.

## Architecture

```
backend/
├── tool_manager.py      # Core tool management (registry, selection, execution)
└── tools/
    ├── __init__.py      # Tool registration
    ├── rag_tool.py      # knowledge_search: notes + long-term memories
    ├── web_search_tool.py  # web_search: DuckDuckGo, no API key
    └── (future tools)   # Add new tools here
```

## How It Works

1. **Tool Selection**: the LLM sees each enabled tool's name and description and answers with a JSON array of calls (or `[]`). Malformed/truncated JSON is recovered where possible; unknown tools are dropped.
2. **Tool Execution**: selected tools run **concurrently**. `execute()` must not block the event loop — wrap blocking work in `asyncio.to_thread`.
3. **Context Injection**: results are inserted as a system message *just before the latest user message* (with the current date/time). The system prompt and earlier history stay byte-identical between turns, which lets the LLM service reuse its KV cache.
4. **Response Generation**: the reply is streamed back to Discord.

## Configuration

In `config.yaml`:

```yaml
tools:
  enabled: true                    # Enable/disable tool system
  context_messages: 3              # Messages to include for context
  max_tools: 3                     # Max tools per request (voice: always 1)
  
  # Individual tool settings (every tool reads tools.<name>.enabled)
  knowledge_search:
    enabled: true
    expand_queries: true           # extra LLM-generated queries in text chat
  web_search:
    enabled: true
```

## Adding a New Tool

### 1. Create the Tool File

Create a new file in `backend/tools/`, e.g., `web_search_tool.py`:

```python
from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter
from typing import Any, Dict, Optional

class WebSearchTool(BaseTool):
    """Web search tool for finding information online."""
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Use for recent events, news, or information not in the knowledge base.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return",
                    required=False,
                    default=5
                )
            ]
        )
    
    async def execute(self, query: str, num_results: int = 5, **kwargs) -> Dict[str, Any]:
        """Execute the web search."""
        try:
            # Your implementation here
            results = await perform_web_search(query, num_results)
            
            return {
                "success": True,
                "result": format_results(results),
                "metadata": {
                    "query": query,
                    "num_results": len(results)
                }
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
```

### 2. Register the Tool

Add your tool to `backend/tools/__init__.py`:

```python
def get_all_tools() -> List[BaseTool]:
    from backend.tools.rag_tool import RAGTool
    from backend.tools.web_search_tool import WebSearchTool  # Add import
    
    return [
        RAGTool(),
        WebSearchTool(),  # Add tool instance
    ]
```

### 3. Add Configuration (Optional)

In `config.yaml`, add tool-specific settings:

```yaml
tools:
  web_search:
    enabled: true
    api_key: "your-api-key"
```

### 4. Describe When to Use It

The selector decides from your tool's `description`, so state plainly *when* the tool should
be used. If it still needs a nudge, add a rule to `_SELECTION_PROMPT` in `backend/tool_manager.py`.

## Tool Definition Components

### ToolParameter

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Parameter name |
| `type` | str | "string", "number", "boolean", "array", "object" |
| `description` | str | What the parameter is for |
| `required` | bool | Is this parameter required? |
| `default` | Any | Default value if not provided |
| `enum` | List[str] | Allowed values (optional) |
| `items` | Dict | For array types, describes item schema |

### ToolDefinition

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Unique tool identifier |
| `description` | str | What the tool does (shown to LLM) |
| `parameters` | List[ToolParameter] | Tool parameters |

### Return Format

Tools must return a dict with:

```python
{
    "success": bool,           # Did the tool succeed?
    "result": Any,             # The tool's output (string for context injection)
    "error": Optional[str],    # Error message if success=False
    "metadata": Optional[dict] # Additional info (for logging/UI)
}
```

## JSON Schema Output

Tools are converted to MCP/OpenAI-style JSON schemas:

```json
{
    "type": "function",
    "function": {
        "name": "knowledge_search",
        "description": "Search the knowledge base...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query..."
                }
            },
            "required": ["query"]
        }
    }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tools/available` | GET | List all available tools with schemas |
| `/tools/last-execution` | GET | Get last tool execution results |

## Debugging

The dashboard's **Tools** tab shows every selection live (arguments, success, execution time),
and the backend console logs `Tool selection: [...]`. Check for:
- Which tools were selected
- What arguments were passed
- Execution times
- Any errors

## Voice Mode Optimization

Voice requests send `voice_mode: true`, which:
- limits selection to a single tool (`max_tools: 1`)
- makes `knowledge_search` use the selector's query directly instead of asking the LLM for
  extra queries (saves a full LLM round-trip, ~0.5 s)

## Fallback Behavior

If the tool system fails:
1. Logs the error
2. Falls back to legacy RAG mode (if RAG is enabled)
3. Continues with no tool context if both fail
