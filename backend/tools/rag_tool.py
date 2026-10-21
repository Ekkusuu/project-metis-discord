"""
Knowledge base (notes + long-term memories), searched two ways:

- knowledge_search: a quick lookup on every message (one vector search with the message itself, close
  matches only, ~50 ms), so replies aren't held up.
- remember: the thorough search (the model writes queries from the conversation, results are reranked;
  1-2 s). The model chooses it when someone asks about something to think back on; in voice chat Metis
  says something like "hmm, let me think" while it runs.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.config import get_config
from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter


class RAGTool(BaseTool):
    deep = False

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_search",
            description=(
                "Quick lookup in the knowledge base (notes, memories about users) for the latest message."
            ),
            parameters=[
                ToolParameter(name="query", type="string", description="What to look for.", required=True),
            ],
        )

    async def execute(
        self,
        query: str,
        num_results: Optional[int] = None,
        conversation_context: str = "",
        voice_mode: bool = False,
        message: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._run, query, num_results, conversation_context, voice_mode, message)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "result": None, "error": str(e)}

    def _queries(self, query: str, conversation_context: str, voice_mode: bool, message: str) -> List[str]:
        rag_cfg = get_config().get("rag", {})
        tool_cfg = get_config().get("tools", {}).get("remember") or {}
        if not tool_cfg.get("expand_queries", True):
            return [query]
        from backend.rag_engine import generate_optimized_queries

        count = int(tool_cfg.get("voice_query_count", 2)) if voice_mode else int(rag_cfg.get("query_generation_count", 3))
        generated = generate_optimized_queries(
            message or query, conversation_context=conversation_context, num_queries=count, voice_mode=voice_mode,
        )
        # The message itself stays as a query too: it still finds notes that use its exact words.
        return list(dict.fromkeys([*generated, query]))

    def _search(self, query: str, num_results: Optional[int], conversation_context: str, voice_mode: bool,
                message: str) -> Dict[str, Any]:
        from backend.rag_engine import quick_search

        cfg = get_config().get("tools", {}).get("knowledge_search") or {}
        found = quick_search(query, float(cfg.get("quick_max_distance", 1.2)), int(cfg.get("quick_results", 2)))
        print(f"📚 Quick knowledge lookup: {query!r}")
        return found

    def _run(self, query: str, num_results: Optional[int], conversation_context: str, voice_mode: bool,
             message: str = "") -> Dict[str, Any]:
        from backend.rag_engine import format_context_for_prompt

        found = self._search(query, num_results, conversation_context, voice_mode, message)
        accepted = found["accepted"]
        sources = sorted({Path(str((c.get("metadata") or {}).get("source_file", "?"))).name for c in accepted})
        print(f"📚 Knowledge ({self.definition.name}): {len(accepted)} result(s)"
              + (f" from {', '.join(sources)}" if sources else ""))
        return {
            "success": True,
            "result": format_context_for_prompt(accepted) or None,
            "message": None if accepted else "No relevant information found in the knowledge base.",
            "metadata": {
                "original_query": query,
                "optimized_queries": found["queries"],
                "num_results": len(accepted),
                "num_rejected": len(found["rejected"]),
            },
            "search": found,
        }


class RememberTool(RAGTool):
    deep = True

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="remember",
            description=(
                "think back: they ask about something from before (what someone told you, a past conversation) "
                "or want advice or knowledge from your notes (psychology, habits, relationships, self-improvement)"
            ),
            parameters=[
                ToolParameter(name="query", type="string", description="What to remember.", required=True),
                ToolParameter(name="num_results", type="number",
                              description="Vector-search candidates per query (default: rag.top_k).", required=False),
            ],
        )

    def _search(self, query: str, num_results: Optional[int], conversation_context: str, voice_mode: bool,
                message: str) -> Dict[str, Any]:
        from backend.rag_engine import search

        queries = self._queries(query, conversation_context, voice_mode, message)
        print(f"🧠 Remember queries: {queries}" + (f"\n   made from: {message!r}" if message else ""))
        return search(queries, top_k=int(num_results) if num_results else None)
