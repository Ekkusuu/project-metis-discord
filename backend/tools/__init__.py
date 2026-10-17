"""
Tools package for Project Metis.

This package contains all available tools that can be used by the AI.
Each tool follows the BaseTool interface and provides:
- A ToolDefinition with JSON schema
- An execute() method for running the tool

To add a new tool:
1. Create a new file in this directory (e.g., my_tool.py)
2. Create a class that inherits from BaseTool
3. Add it to the get_all_tools() function below
"""
from __future__ import annotations

from typing import List
from backend.tool_manager import BaseTool


def get_all_tools() -> List[BaseTool]:
    """
    Get all available tool instances.
    
    Add new tools here as they are implemented.
    """
    from backend.tools.rag_tool import RAGTool
    from backend.tools.web_search_tool import WebSearchTool
    
    return [
        RAGTool(),
        WebSearchTool(),
    ]


__all__ = ["get_all_tools"]
