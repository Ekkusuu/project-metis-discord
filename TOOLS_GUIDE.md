# Tool System Guide

## Overview

The tool system allows Metis to dynamically select and use tools based on the conversation context. This is implemented using MCP-style (Model Context Protocol) JSON schemas, making it easy to add new tools.

## Architecture

```
backend/
├── tool_manager.py      # Core tool management (registry, selection, execution)
└── tools/
    ├── __init__.py      # Tool registration
    ├── rag_tool.py      # Knowledge base search tool
    └── (future tools)   # Add new tools here
```

## How It Works

1. **Tool Selection**: When a message comes in, the LLM is asked to select which tools (if any) to use
2. **Tool Execution**: Selected tools are executed with their arguments
3. **Context Injection**: Tool results are injected into the system prompt
4. **Response Generation**: The LLM generates a response using the tool results

## Configuration

In `config.yaml`:

```yaml
tools:
  enabled: true                    # Enable/disable tool system
  context_messages: 3              # Messages to include for context
  max_tools: 3                     # Max tools per request
  
  # Tool selection prompts
  selection_system_prompt: "@tools.selection_system"
  selection_user_prompt: "@tools.selection_user"
  
  # Voice-optimized prompts
  selection_voice_system_prompt: "@tools.selection_voice_system"
  selection_voice_user_prompt: "@tools.selection_voice_user"
  
  # Individual tool settings
  knowledge_search:
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

### 4. Update Prompts (Optional)

If your tool needs special guidance, update `system_prompts.yaml`:

```yaml
tools:
  selection_system: >
    ...
    WHEN TO USE TOOLS:
    - Use web_search when: user asks about current events, news, or real-time information
    ...
```

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

Enable debug logging to see tool selection and execution:

```python
# In tool_manager.py
print(f"Tool selection result: {tool_calls}")
print(f"Tool execution result: {results}")
```

Check the console output for:
- Which tools were selected
- What arguments were passed
- Execution times
- Any errors

## Voice Mode Optimization

Voice mode uses optimized prompts for faster tool selection:
- `@tools.selection_voice_system` - Simplified system prompt
- `@tools.selection_voice_user` - Shorter user prompt
- `max_tools: 1` - Limit to single tool

## Fallback Behavior

If the tool system fails:
1. Logs the error
2. Falls back to legacy RAG mode (if RAG is enabled)
3. Continues with no tool context if both fail
