"""
Web Search Tool - Search the web using DuckDuckGo.

Uses the `ddgs` package - no server, API key or Docker required.
"""
from __future__ import annotations

import asyncio
import re
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
            description="any question about facts: games (best picks, tier lists, builds, stats), products, people, shows, places, news, sports, weather, prices, releases, or anything that changes over time",
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
        search_type: str = "text",
        backend: str = "auto",
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
                for r in ddgs.text(query, max_results=num_results, backend=backend):
                    results.append({
                        "title": r.get("title", "No title"),
                        "url": r.get("href", ""),
                        "content": r.get("body", "No description"),
                        "source": "web"
                    })
        
        return results
    
    async def _weather(self, query: str, message: str) -> str:
        """Live weather for weather questions (search snippets don't have it); "" if not one or no place found."""
        from backend.tools import weather

        if not (weather.WEATHER_QUESTION.search(query) or weather.WEATHER_QUESTION.search(message)):
            return ""
        # The person's own words first: the model's query sometimes swaps the town for the country.
        for text in dict.fromkeys(t for t in (re.sub(r"^\[[^\]]+\]:?\s*", "", message), query) if t):
            try:
                found = await asyncio.wait_for(asyncio.to_thread(weather.lookup, text), timeout=12)
            except Exception as e:
                print(f"Weather lookup failed: {e}")
                return ""
            if found:
                return found
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
        weather = await self._weather(query, kwargs.get("message", ""))
        if weather:
            print(f"🌦️ {weather.splitlines()[0]}")
            return {"success": True, "result": weather,
                    "metadata": {"query": query, "num_results": 1, "search_type": "weather"}}
        cfg = get_config().get("tools", {}).get("web_search") or {}
        from backend.research import format_brief, research

        question = re.sub(r"^\[[^\]]+\]:?\s*", "", kwargs.get("message", "")).strip() or query
        budget = float(cfg.get("research_seconds_voice", 20) if voice_mode else cfg.get("research_seconds", 60))
        try:
            # Each step has its own limit too; this is the backstop so a reply can never wait forever.
            found = await asyncio.wait_for(research(
                question, query, kwargs.get("conversation_context", ""),
                max_steps=int(cfg.get("research_steps_voice", 5) if voice_mode else cfg.get("research_steps", 8)),
                time_budget=budget, first_kind="news" if search_type == "news" else "text",
                auto_open=int(cfg.get("research_auto_open", 2)),
                notify=kwargs.get("notify")), timeout=budget + 20)
        except asyncio.TimeoutError:
            print(f"🔬 research gave up after {budget + 20:.0f}s")
            found = {"answer": "", "found": False, "sources": [], "searches": [query], "steps": 0}
        return {"success": True, "result": format_brief(found, question),
                "metadata": {"query": query, "searches": found["searches"], "found": found["found"],
                             "search_type": "research"}}
