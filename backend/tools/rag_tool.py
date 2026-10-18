"""
knowledge_search - retrieval from the indexed knowledge base (notes + long-term memories).
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from backend.config import get_config
from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter


class RAGTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_search",
            description=(
                "Search the knowledge base: stored notes, memories about users, and facts from past "
                "conversations. Use for questions about the users, their lives and preferences, or stored knowledge."
            ),
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="What to look for (short keywords).",
                    required=True,
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Maximum number of vector-search candidates per query (default: rag.top_k).",
                    required=False,
                ),
            ],
        )

    async def execute(
        self,
        query: str,
        num_results: Optional[int] = None,
        conversation_context: str = "",
        voice_mode: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            return await asyncio.to_thread(self._run, query, num_results, conversation_context, voice_mode)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"success": False, "result": None, "error": str(e)}

    def _queries(self, query: str, conversation_context: str, voice_mode: bool) -> List[str]:
        rag_cfg = get_config().get("rag", {})
        tool_cfg = get_config().get("tools", {}).get("knowledge_search") or {}
        # The selector already wrote a focused query. In voice mode latency matters more
        # than recall, so skip the extra LLM round-trip for query expansion.
        if voice_mode or not tool_cfg.get("expand_queries", False):
            return [query]
        from backend.rag_engine import generate_optimized_queries

        expanded = generate_optimized_queries(
            query,
            conversation_context=conversation_context,
            num_queries=int(rag_cfg.get("query_generation_count", 3)),
        )
        return list(dict.fromkeys([query, *expanded]))

    def _run(self, query: str, num_results: Optional[int], conversation_context: str, voice_mode: bool) -> Dict[str, Any]:
        from backend.rag_engine import format_context_for_prompt, search

        queries = self._queries(query, conversation_context, voice_mode)
        print(f"📚 Knowledge queries: {queries}")
        found = search(queries, top_k=int(num_results) if num_results else None)
        accepted = found["accepted"]
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
