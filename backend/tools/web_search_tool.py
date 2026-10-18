"""
Web Search Tool - Search the web using DuckDuckGo.

Uses the `ddgs` package - no server, API key or Docker required.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from backend.config import get_config
from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter


class WebSearchTool(BaseTool):
    """
    Web search tool using DuckDuckGo.
    
    Searches the web for current information. No server setup required,
    works out of the box with the duckduckgo-search package.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_search",
            description="Search the web for current information. Use this for recent events, news, facts not in the knowledge base, or when the user asks about something that requires up-to-date information from the internet.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The search query. Be specific and include relevant keywords.",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Maximum number of results to return (default: 5, max: 10)",
                    required=False,
                    default=5
                ),
                ToolParameter(
                    name="search_type",
                    type="string",
                    description="Type of search: 'text' for general web search, 'news' for recent news articles (default: text)",
                    required=False,
                    default="text",
                    enum=["text", "news"]
                )
            ]
        )
    
    def _search_sync(
        self, 
        query: str, 
        num_results: int = 5,
        search_type: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Synchronous search function to run in thread pool.
        """
        # Try new package name first, fall back to old
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                raise ImportError(
                    "DuckDuckGo search package not installed. "
                    "Run: pip install ddgs"
                )
        
        results = []
        
        with DDGS() as ddgs:
            if search_type == "news":
                # News search
                for r in ddgs.news(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", "No title"),
                        "url": r.get("url", ""),
                        "content": r.get("body", "No description"),
                        "source": r.get("source", "unknown"),
                        "date": r.get("date", "")
                    })
            else:
                # Regular text search
                for r in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": r.get("title", "No title"),
                        "url": r.get("href", ""),
                        "content": r.get("body", "No description"),
                        "source": "web"
                    })
        
        return results
    
    async def _read_top_result(self, results: List[Dict[str, Any]], voice_mode: bool) -> str:
        cfg = get_config().get("tools", {}).get("web_search") or {}
        enabled = cfg.get("read_top_result_voice", False) if voice_mode else cfg.get("read_top_result", True)
        if not enabled:
            return ""
        from backend.tools.web_page_tool import fetch_page_text

        for r in results[:3]:
            url = r.get("url")
            if not url:
                continue
            try:
                page = await asyncio.wait_for(
                    asyncio.to_thread(fetch_page_text, url, int(cfg.get("top_result_chars", 2500))), timeout=8
                )
            except Exception:
                continue  # paywalls, bot walls, timeouts: try the next result
            if len(page["text"]) > 200:
                return f"({page['url']})\n{page['text']}"
        return ""

    async def execute(
        self, 
        query: str, 
        num_results: int = 5,
        search_type: str = "text",
        voice_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the web search.
        
        Args:
            query: The search query
            num_results: Maximum number of results (default: 5, max: 10)
            search_type: 'text' for web search, 'news' for news (default: text)
        
        Returns:
            Dict with success status and search results
        """
        try:
            num_results = min(max(1, int(num_results or 5)), 10)
            if search_type not in ("text", "news"):
                search_type = "text"
            
            # ddgs is synchronous; run it off the event loop and don't let a slow
            # search hold up the reply forever.
            results = await asyncio.wait_for(
                asyncio.to_thread(self._search_sync, query, num_results, search_type),
                timeout=15,
            )
            
            if not results:
                return {
                    "success": True,
                    "result": f"No results found for: {query}",
                    "metadata": {
                        "query": query,
                        "num_results": 0,
                        "search_type": search_type
                    }
                }
            
            # Format results for context
            formatted_results = []
            
            for i, r in enumerate(results, 1):
                title = r.get("title", "No title")
                url = r.get("url", "")
                content = r.get("content", "No description")
                
                # Clean up content
                content = content.strip()
                if len(content) > 500:
                    content = content[:500] + "..."
                
                # Add date for news results
                date_str = ""
                if r.get("date"):
                    date_str = f"\n    Date: {r['date']}"
                
                formatted_results.append(
                    f"[{i}] {title}\n    URL: {url}{date_str}\n    {content}"
                )
            
            search_label = "News" if search_type == "news" else "Web"
            formatted_text = f"{search_label} search results for: {query}\n\n" + "\n\n".join(formatted_results)

            # Snippets are often too thin to answer from; read the best result's page too.
            page_text = await self._read_top_result(results, voice_mode)
            if page_text:
                formatted_text += f"\n\nContent of the top result:\n{page_text}"
            
            return {
                "success": True,
                "result": formatted_text,
                "structured_results": results,
                "metadata": {
                    "query": query,
                    "num_results": len(results),
                    "search_type": search_type
                }
            }
            
        except asyncio.TimeoutError:
            return {"success": False, "result": None, "error": "Search timed out"}
        except ImportError as e:
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
        except Exception as e:
            import traceback
            print(f"Web search error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "result": None,
                "error": f"Search failed: {str(e)}"
            }
