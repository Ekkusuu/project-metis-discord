"""
RAG Tool - Knowledge base retrieval tool for Project Metis.

This tool allows the AI to search the indexed knowledge base
for relevant information based on a query.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter


def generate_optimized_queries(
    last_user_message: str,
    conversation_context: str = "",
    num_queries: int = 3,
    voice_mode: bool = False
) -> List[str]:
    """
    Generate optimized RAG queries using the system prompts.
    
    This uses the same query generation as the legacy RAG system
    to produce better search queries.
    """
    from backend.llama_engine import chat_completion, get_config
    from backend.prompt_loader import resolve_prompt
    
    config = get_config()
    rag_cfg = config.get("rag", {})
    
    # Use voice-optimized prompts for faster response, or full prompts for text
    if voice_mode:
        system_prompt = resolve_prompt(rag_cfg.get("query_generation_voice_system", "@rag.query_generation_voice_system"))
        user_prompt_template = resolve_prompt(rag_cfg.get("query_generation_voice_user", "@rag.query_generation_voice_user"))
        num_queries = 1  # Voice mode always uses 1 query
    else:
        system_prompt = resolve_prompt(rag_cfg.get("query_generation_system", "@rag.query_generation_system"))
        user_prompt_template = resolve_prompt(rag_cfg.get("query_generation_user", "@rag.query_generation_user"))
    
    # Build user prompt with context
    user_prompt = user_prompt_template.format(
        conversation_context=conversation_context or "No previous context.",
        last_user_message=last_user_message
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        response = chat_completion(messages, temperature=0.3, max_tokens=256)
        
        # Parse queries from response (one per line)
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]
        
        # Limit to num_queries
        queries = queries[:num_queries]
        
        if not queries:
            # Fallback to original message
            queries = [last_user_message]
        
        return queries
    except Exception as e:
        print(f"Query generation error: {e}")
        return [last_user_message]


class RAGTool(BaseTool):
    """
    RAG (Retrieval Augmented Generation) tool.
    
    Searches the indexed knowledge base for relevant context
    based on a provided query. Uses optimized query generation
    from system prompts for better results.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_search",
            description="Search the knowledge base for relevant information. Use this when you need to find specific facts, memories about users, stored notes, or any information that might be in the indexed documents. Good for questions about personal information, past conversations, stored knowledge, or facts.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="The user's question or topic to search for. This will be optimized into proper search queries automatically.",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Maximum number of results to return (default: uses config value)",
                    required=False,
                    default=None
                )
            ]
        )
    
    async def execute(
        self, 
        query: str, 
        num_results: Optional[int] = None,
        conversation_context: str = "",
        voice_mode: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the RAG search with optimized query generation.
        
        Args:
            query: The user's question/topic (will be optimized)
            num_results: Optional override for number of results
            conversation_context: Previous conversation for context
            voice_mode: If True, use faster single-query mode
        
        Returns:
            Dict with success status and retrieved contexts
        """
        try:
            from backend.rag_engine import retrieve_context, format_context_for_prompt, get_config
            
            config = get_config()
            rag_cfg = config.get("rag", {})
            
            # Use provided num_results or fall back to config
            top_k = num_results if num_results is not None else rag_cfg.get("top_k", 6)
            
            # Generate optimized queries using system prompts
            num_queries = 1 if voice_mode else rag_cfg.get("query_generation_count", 3)
            optimized_queries = generate_optimized_queries(
                query,
                conversation_context=conversation_context,
                num_queries=num_queries,
                voice_mode=voice_mode
            )
            
            print(f"📚 RAG queries: {optimized_queries}")
            
            # Execute searches for all queries and combine results
            all_accepted = []
            all_rejected_distance = []
            all_rejected_score = []
            seen_keys = set()
            
            for opt_query in optimized_queries:
                results = retrieve_context(opt_query, top_k=top_k)
                
                # Handle both old list format and new dict format
                if isinstance(results, dict):
                    accepted = results.get("accepted", [])
                    rejected_distance = results.get("rejected_by_distance", [])
                    rejected_score = results.get("rejected_by_score", [])
                else:
                    accepted = results if results else []
                    rejected_distance = []
                    rejected_score = []
                
                # Deduplicate results
                for ctx in accepted:
                    md = ctx.get("metadata", {})
                    key = (md.get("source_file"), md.get("chunk_index"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_accepted.append(ctx)
                
                for ctx in rejected_distance:
                    md = ctx.get("metadata", {})
                    key = (md.get("source_file"), md.get("chunk_index"))
                    if key not in seen_keys:
                        all_rejected_distance.append(ctx)
                
                for ctx in rejected_score:
                    md = ctx.get("metadata", {})
                    key = (md.get("source_file"), md.get("chunk_index"))
                    if key not in seen_keys:
                        all_rejected_score.append(ctx)
            
            if not all_accepted:
                return {
                    "success": True,
                    "result": None,
                    "message": "No relevant information found in the knowledge base.",
                    "metadata": {
                        "original_query": query,
                        "optimized_queries": optimized_queries,
                        "num_results": 0,
                        "rejected_by_distance": len(all_rejected_distance),
                        "rejected_by_score": len(all_rejected_score)
                    },
                    "raw_contexts": {
                        "accepted": [],
                        "rejected_by_distance": all_rejected_distance,
                        "rejected_by_score": all_rejected_score
                    }
                }
            
            # Format contexts for the response
            formatted_context = format_context_for_prompt(all_accepted)
            
            # Also return structured data for the UI
            structured_results = []
            for ctx in all_accepted:
                structured_results.append({
                    "text": ctx.get("text", ""),
                    "source": ctx.get("metadata", {}).get("source_file", "unknown"),
                    "distance": ctx.get("distance", 0.0),
                    "rerank_score": ctx.get("rerank_score")
                })
            
            return {
                "success": True,
                "result": formatted_context,
                "structured_results": structured_results,
                "metadata": {
                    "original_query": query,
                    "optimized_queries": optimized_queries,
                    "num_results": len(all_accepted),
                    "rejected_by_distance": len(all_rejected_distance),
                    "rejected_by_score": len(all_rejected_score)
                },
                # Raw data for UI panel
                "raw_contexts": {
                    "accepted": all_accepted,
                    "rejected_by_distance": all_rejected_distance,
                    "rejected_by_score": all_rejected_score
                }
            }
            
        except Exception as e:
            import traceback
            print(f"RAG tool error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }


class MultiQueryRAGTool(BaseTool):
    """
    Multi-query RAG tool that generates multiple search queries
    to improve retrieval coverage.
    
    This is an advanced version that can generate multiple queries
    from a single user request to capture different facets.
    """
    
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_search_multi",
            description="Advanced knowledge base search that uses multiple queries to find comprehensive information. Use this for complex questions that might need information from different perspectives or topics.",
            parameters=[
                ToolParameter(
                    name="queries",
                    type="array",
                    description="List of search queries to execute. Each query should focus on a different aspect or facet of the information needed.",
                    required=True,
                    items={"type": "string"}
                ),
                ToolParameter(
                    name="num_results_per_query",
                    type="number",
                    description="Maximum number of results per query (default: 3)",
                    required=False,
                    default=3
                )
            ]
        )
    
    async def execute(self, queries: List[str], num_results_per_query: int = 3, **kwargs) -> Dict[str, Any]:
        """
        Execute multiple RAG searches and combine results.
        
        Args:
            queries: List of search queries
            num_results_per_query: Results per query
        
        Returns:
            Dict with combined results
        """
        try:
            from backend.rag_engine import retrieve_context, format_context_for_prompt
            
            all_accepted = []
            all_rejected = []
            seen_keys = set()
            per_query_results = []
            
            for query in queries:
                results = retrieve_context(query, top_k=num_results_per_query)
                
                if isinstance(results, dict):
                    accepted = results.get("accepted", [])
                    rejected = results.get("rejected_by_distance", []) + results.get("rejected_by_score", [])
                else:
                    accepted = results if results else []
                    rejected = []
                
                # Deduplicate
                unique_accepted = []
                for ctx in accepted:
                    md = ctx.get("metadata", {})
                    key = (md.get("source_file"), md.get("chunk_index"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        unique_accepted.append(ctx)
                        all_accepted.append(ctx)
                
                per_query_results.append({
                    "query": query,
                    "num_results": len(unique_accepted)
                })
                
                for ctx in rejected:
                    md = ctx.get("metadata", {})
                    key = (md.get("source_file"), md.get("chunk_index"))
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_rejected.append(ctx)
            
            if not all_accepted:
                return {
                    "success": True,
                    "result": None,
                    "message": "No relevant information found for any of the queries.",
                    "metadata": {
                        "queries": queries,
                        "total_results": 0,
                        "per_query": per_query_results
                    }
                }
            
            formatted_context = format_context_for_prompt(all_accepted)
            
            return {
                "success": True,
                "result": formatted_context,
                "metadata": {
                    "queries": queries,
                    "total_results": len(all_accepted),
                    "per_query": per_query_results
                },
                "raw_contexts": {
                    "accepted": all_accepted,
                    "rejected": all_rejected
                }
            }
            
        except Exception as e:
            import traceback
            print(f"Multi-query RAG tool error: {e}")
            traceback.print_exc()
            return {
                "success": False,
                "result": None,
                "error": str(e)
            }
