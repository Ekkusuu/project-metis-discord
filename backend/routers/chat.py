from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llama_engine import chat_completion, chat_completion_stream, get_config
from backend.rag_engine import (
    retrieve_context,
    format_context_for_prompt,
    index_all_folders,
    generate_rag_query,
    generate_rag_queries,
)
from backend.context_manager import trim_messages_to_context

router = APIRouter(prefix="", tags=["chat"])

# Store last RAG retrieval results
_last_rag_results: List[Dict[str, Any]] = []

# Store last tool execution results
_last_tool_results: Dict[str, Any] = {}

# Store current chat context (replaces on each request)
_current_context: List[Dict[str, Any]] = []


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = Field(
        default=None, description="OpenAI-style chat messages array"
    )
    prompt: Optional[str] = Field(
        default=None, description="Simple prompt; used if messages not provided"
    )
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Tool system options (uses MCP JSON schemas)
    use_tools: Optional[bool] = Field(
        default=None, description="Override tool system usage (None = use config)"
    )
    max_tools: Optional[int] = Field(
        default=None, description="Maximum tools to select (default: 3)"
    )
    voice_mode: Optional[bool] = Field(
        default=False, description="Voice mode uses single query for faster response"
    )
    # Legacy RAG options (for backward compatibility when tools disabled)
    rag_query_count: Optional[int] = Field(
        default=None, description="Override number of RAG queries to generate (legacy mode)"
    )
    rag_system_prompt: Optional[str] = Field(
        default=None, description="Override RAG query generation system prompt (legacy mode)"
    )
    rag_user_prompt: Optional[str] = Field(
        default=None, description="Override RAG query generation user prompt (legacy mode)"
    )


class ChatResponse(BaseModel):
    reply: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if (not req.messages or len(req.messages) == 0) and (not req.prompt):
        raise HTTPException(status_code=400, detail="Provide `messages` or `prompt`.")

    config = get_config()
    chat_cfg = config["chat"]
    system_prompt = chat_cfg.get("system_prompt", "You are Metis, a helpful AI assistant.")

    messages: List[Message]
    if req.messages and len(req.messages) > 0:
        # Prepend system prompt if not already present
        if not any(m.role == "system" for m in req.messages):
            messages = [Message(role="system", content=system_prompt)] + req.messages
        else:
            messages = req.messages
    else:
        # Build minimal chat with system prompt from config
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=req.prompt or ""),
        ]

    # Use request params or fall back to config defaults
    temperature = req.temperature if req.temperature is not None else chat_cfg.get("temperature", 0.7)
    top_p = req.top_p if req.top_p is not None else chat_cfg.get("top_p", 0.95)
    max_tokens = req.max_tokens if req.max_tokens is not None else chat_cfg.get("max_tokens", 512)

    # Trim messages to fit within context limit
    messages_dict = [m.model_dump() for m in messages]
    messages_dict = trim_messages_to_context(messages_dict)

    reply = chat_completion(
        messages_dict,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return ChatResponse(reply=reply)


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """
    Streaming version of /chat endpoint. Returns newline-delimited JSON chunks.
    Each line is a JSON object: {"delta": "token text"} or {"done": true}
    
    Supports both the new tool system and legacy RAG mode.
    """
    if (not req.messages or len(req.messages) == 0) and (not req.prompt):
        raise HTTPException(status_code=400, detail="Provide `messages` or `prompt`.")

    config = get_config()
    chat_cfg = config["chat"]
    rag_cfg = config.get("rag", {})
    tools_cfg = config.get("tools", {})
    system_prompt = chat_cfg.get("system_prompt", "You are Metis, a helpful AI assistant.")

    messages: List[Message]
    if req.messages and len(req.messages) > 0:
        # Prepend system prompt if not already present
        if not any(m.role == "system" for m in req.messages):
            messages = [Message(role="system", content=system_prompt)] + req.messages
        else:
            messages = req.messages
    else:
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=req.prompt or ""),
        ]

    # Determine whether to use tool system or legacy RAG
    global _last_rag_results, _last_tool_results
    import time as time_module
    tool_timings = {}  # Store timing data for tool operations
    
    # Use tool system if enabled in config (can be overridden by request)
    use_tools = req.use_tools if req.use_tools is not None else tools_cfg.get("enabled", False)
    rag_enabled = rag_cfg.get("enabled", False)
    
    # Get the last user message
    user_messages = [m for m in messages if m.role == "user"]
    last_user_query = user_messages[-1].content if user_messages else ""
    
    tool_results_for_context = ""
    
    if use_tools and last_user_query:
        # === NEW TOOL SYSTEM ===
        try:
            from backend.tool_manager import select_tools, execute_tools, format_tool_results_for_context, get_tool_registry
            
            # Initialize registry if needed
            get_tool_registry()
            
            # === TOOL SELECTION (uses MCP JSON schemas) ===
            t_start = time_module.perf_counter()
            messages_for_tools = [{"role": m.role, "content": m.content} for m in messages]
            
            # Build conversation context string for RAG query generation
            context_messages = tools_cfg.get("context_messages", 3)
            conversation_context = ""
            if context_messages > 0 and len(messages_for_tools) > 1:
                recent = messages_for_tools[-context_messages-1:-1] if len(messages_for_tools) > context_messages else messages_for_tools[:-1]
                conversation_context = "\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in recent
                ])
            
            # Run tool selection synchronously (wrap async)
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            tool_calls = loop.run_until_complete(
                select_tools(
                    messages_for_tools,
                    last_user_query,
                    max_tools=req.max_tools or tools_cfg.get("max_tools", 3)
                )
            )
            tool_timings['tool_selection'] = time_module.perf_counter() - t_start
            
            # === TOOL EXECUTION ===
            if tool_calls:
                t_start = time_module.perf_counter()
                
                # Pass conversation context to tools (especially for RAG query generation)
                extra_context = {
                    "conversation_context": conversation_context,
                    "voice_mode": req.voice_mode or False  # Voice mode uses single query
                }
                
                tool_results = loop.run_until_complete(execute_tools(tool_calls, extra_context))
                tool_timings['tool_execution'] = time_module.perf_counter() - t_start
                
                # Format results for context injection
                tool_results_for_context = format_tool_results_for_context(tool_results)
                
                # Store results for UI
                _last_tool_results = {
                    "tool_calls": [
                        {"tool": tc.tool_name, "arguments": tc.arguments}
                        for tc in tool_calls
                    ],
                    "results": [
                        {
                            "tool": tr.tool_name,
                            "success": tr.success,
                            "execution_time": tr.execution_time,
                            "error": tr.error
                        }
                        for tr in tool_results
                    ]
                }
                
                # Also update RAG results for backward compatibility with UI
                for tr in tool_results:
                    if tr.tool_name == "knowledge_search" and tr.success:
                        # Extract RAG-specific data for the panel from raw_response
                        raw_response = tr.raw_response or {}
                        raw_contexts = raw_response.get("raw_contexts", {})
                        metadata = raw_response.get("metadata", {})
                        
                        # Get the optimized queries (generated by RAG tool)
                        optimized_queries = metadata.get("optimized_queries", [])
                        original_query = metadata.get("original_query", "")
                        
                        # Fallback to tool call query if no optimized queries
                        if not optimized_queries:
                            for tc in tool_calls:
                                if tc.tool_name == "knowledge_search":
                                    optimized_queries = [tc.arguments.get("query", "")]
                                    break
                        
                        accepted = raw_contexts.get("accepted", []) if raw_contexts else []
                        rejected_distance = raw_contexts.get("rejected_by_distance", []) if raw_contexts else []
                        rejected_score = raw_contexts.get("rejected_by_score", []) if raw_contexts else []
                        
                        results_list = []
                        for ctx in accepted:
                            md = ctx.get("metadata", {})
                            results_list.append({
                                "source_file": md.get("source_file", "unknown"),
                                "distance": ctx.get("distance", 0),
                                "rerank_score": ctx.get("rerank_score"),
                                "text_preview": ctx.get("text", "")[:200],
                                "chunk_index": md.get("chunk_index", 0),
                                "used": True,
                            })
                        for ctx in rejected_distance + rejected_score:
                            md = ctx.get("metadata", {})
                            results_list.append({
                                "source_file": md.get("source_file", "unknown"),
                                "distance": ctx.get("distance", 0),
                                "rerank_score": ctx.get("rerank_score"),
                                "text_preview": ctx.get("text", "")[:200],
                                "chunk_index": md.get("chunk_index", 0),
                                "used": False,
                                "rejection_reason": ctx.get("rejection_reason", "unknown")
                            })
                        
                        _last_rag_results = {
                            "queries": optimized_queries,
                            "original_query": original_query or last_user_query,
                            "results": results_list,
                            "tool_mode": True
                        }
            else:
                _last_tool_results = {
                    "tool_calls": [],
                    "results": [],
                    "message": "No tools selected"
                }
                _last_rag_results = {
                    "queries": [],
                    "original_query": last_user_query,
                    "results": [],
                    "tool_mode": True,
                    "message": "No tools needed"
                }
                
        except Exception as e:
            import traceback
            print(f"Tool system error: {e}")
            traceback.print_exc()
            # Fall back to legacy RAG if tool system fails
            use_tools = False
    
    # Legacy RAG mode (if tools disabled or failed)
    rag_timings = {}
    if not use_tools and rag_enabled and last_user_query:
        # === RAG QUERY GENERATION ===
        t_start = time_module.perf_counter()
        messages_for_query = [{"role": m.role, "content": m.content} for m in messages]
        rag_queries = generate_rag_queries(
            messages_for_query, 
            last_user_query,
            num_queries=req.rag_query_count,
            system_prompt_override=req.rag_system_prompt,
            user_prompt_override=req.rag_user_prompt
        )
        rag_timings['query_generation'] = time_module.perf_counter() - t_start
        
        # Record generated queries
        _last_rag_results = {
            "queries": rag_queries,
            "original_query": last_user_query,
            "results": []
        }
        
        # === RAG RETRIEVAL ===
        t_start = time_module.perf_counter()
        combined_contexts: List[Dict[str, Any]] = []
        seen = set()
        per_query_results: List[Dict[str, Any]] = []
        
        for q in rag_queries:
            try:
                results = retrieve_context(q)
            except Exception:
                results = {"accepted": [], "rejected_by_distance": []}

            rejected_distance = results.get("rejected_by_distance", []) or []
            rejected_score = results.get("rejected_by_score", []) or []
            per_query_results.append({
                "query": q,
                "accepted": results.get("accepted", []),
                "rejected": rejected_distance + rejected_score,
            })
        rag_timings['retrieval'] = time_module.perf_counter() - t_start

        # Combine accepted chunks
        combined_accepted: List[Dict[str, Any]] = []
        seen = set()
        for bucket in per_query_results:
            for ctx in bucket["accepted"]:
                md = ctx.get("metadata", {})
                src = md.get("source_file")
                idx = md.get("chunk_index")
                if src is None or idx is None:
                    key = (None, ctx.get("text", "")[:160])
                else:
                    key = (src, idx)

                if key in seen:
                    continue
                seen.add(key)
                combined_accepted.append(ctx)

        # Collect rejected chunks
        combined_rejected: List[Dict[str, Any]] = []
        for bucket in per_query_results:
            for ctx in bucket["rejected"]:
                md = ctx.get("metadata", {})
                src = md.get("source_file")
                idx = md.get("chunk_index")
                if src is None or idx is None:
                    key = (None, ctx.get("text", "")[:160])
                else:
                    key = (src, idx)

                if key in seen:
                    continue
                seen.add(key)
                if "rejection_reason" not in ctx:
                    ctx["rejection_reason"] = "distance"
                combined_rejected.append(ctx)

        combined_contexts = combined_accepted + combined_rejected
        contexts = combined_contexts
        relevant_contexts = [c for c in contexts if "rejection_reason" not in c]
        
        # Build results list for UI
        accepted_list = [c for c in combined_contexts if "rejection_reason" not in c]
        rejected_list = [c for c in combined_contexts if "rejection_reason" in c]

        for bucket in per_query_results:
            acc = bucket.get("accepted", [])
            rej = bucket.get("rejected", [])
            if any(x.get("rerank_score") is not None for x in acc):
                acc.sort(key=lambda x: float(x.get("rerank_score") or -1e9), reverse=True)
            else:
                acc.sort(key=lambda x: float(x.get("distance", 0)))
            rej.sort(key=lambda x: float(x.get("distance", 0)))
            bucket["accepted"] = acc
            bucket["rejected"] = rej

        if any(x.get("rerank_score") is not None for x in accepted_list):
            accepted_list.sort(key=lambda x: float(x.get("rerank_score") or -1e9), reverse=True)
        else:
            accepted_list.sort(key=lambda x: float(x.get("distance", 0)))

        rejected_list.sort(key=lambda x: float(x.get("distance", 0)))

        results_list = []
        for ctx in accepted_list + rejected_list:
            md = ctx.get("metadata", {})
            is_used = "rejection_reason" not in ctx
            item = {
                "source_file": md.get("source_file", "unknown"),
                "distance": ctx.get("distance", 0),
                "rerank_score": ctx.get("rerank_score"),
                "text_preview": ctx.get("text", "")[:200],
                "chunk_index": md.get("chunk_index", 0),
                "used": is_used,
            }
            if not is_used:
                item["rejection_reason"] = ctx.get("rejection_reason")
            results_list.append(item)

        _last_rag_results = {
            "queries": rag_queries,
            "original_query": last_user_query,
            "per_query_results": per_query_results,
            "results": results_list,
        }
        
        if relevant_contexts:
            context_text = format_context_for_prompt(relevant_contexts)
            tool_results_for_context = context_text
    
    # Inject tool/RAG results into system prompt
    if tool_results_for_context:
        for i, msg in enumerate(messages):
            if msg.role == "system":
                enhanced_system = f"{msg.content}\n\n---\nRelevant information from your knowledge base:\n\n{tool_results_for_context}\n\nUse this information to answer the user's question."
                messages[i] = Message(role="system", content=enhanced_system)
                break

    # Use request params or fall back to config defaults
    temperature = req.temperature if req.temperature is not None else chat_cfg.get("temperature", 0.7)
    top_p = req.top_p if req.top_p is not None else chat_cfg.get("top_p", 0.95)
    max_tokens = req.max_tokens if req.max_tokens is not None else chat_cfg.get("max_tokens", 512)

    # Trim messages to fit within context limit
    messages_dict = [m.model_dump() for m in messages]
    messages_dict = trim_messages_to_context(messages_dict)

    # Store current context (replace, not append)
    global _current_context
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    
    # Replace context with current messages (using trimmed messages)
    _current_context = [
        {
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": timestamp
        }
        for msg in messages_dict
    ]

    def generate():
        import json
        import time
        
        global _current_context
        
        start_time = time.time()
        token_count = 0
        assistant_response = ""
        
        for token in chat_completion_stream(
            messages_dict,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            token_count += 1
            assistant_response += token
            # Send each token as NDJSON (newline-delimited JSON)
            yield json.dumps({"delta": token}) + "\n"
        
        # Calculate tokens per second
        elapsed = time.time() - start_time
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        llm_generation_time = elapsed
        
        # Update context with assistant response
        _current_context.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Final chunk with performance stats and trimmed message count
        final_chunk = {
            "done": True,
            "tokens_per_second": round(tokens_per_second, 2),
            "rag_timings": rag_timings,
            "tool_timings": tool_timings,
            "llm_generation_time": round(llm_generation_time, 3),
            "trimmed_messages": [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages_dict
            ]
        }
        yield json.dumps(final_chunk) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/rag/reindex")
def reindex_knowledge_base(clear_existing: bool = True):
    """
    Trigger re-indexing of all folders specified in config.yaml.
    
    - clear_existing: if True, removes old chunks before re-indexing
    """
    try:
        results = index_all_folders(clear_existing=clear_existing)
        total = sum(results.values())
        return {
            "status": "success",
            "total_chunks": total,
            "folders": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/rag/stats")
def get_rag_stats():
    """Get RAG system statistics."""
    try:
        from backend.rag_engine import get_collection, get_file_metadata_path
        import json
        
        config = get_config()
        rag_cfg = config.get("rag", {})
        
        collection = get_collection()
        total_chunks = collection.count()
        
        # Count unique files from metadata
        metadata_path = get_file_metadata_path()
        total_files = 0
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                file_metadata = json.load(f)
                total_files = len(file_metadata)
        
        return {
            "total_chunks": total_chunks,
            "total_files": total_files,
            "collection_name": rag_cfg.get("collection_name", "metis_knowledge"),
            "enabled": rag_cfg.get("enabled", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.post("/rag/clear")
def clear_rag_database():
    """Clear all embeddings from the RAG database."""
    try:
        from backend.rag_engine import get_collection, get_file_metadata_path, save_file_metadata
        
        collection = get_collection()
        count_before = collection.count()
        
        # Delete all items in the collection
        if count_before > 0:
            all_items = collection.get()
            if all_items and all_items.get("ids"):
                collection.delete(ids=all_items["ids"])
        
        # Clear file metadata
        save_file_metadata({})
        
        return {
            "status": "success",
            "deleted_count": count_before
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@router.get("/rag/last-retrieval")
def get_last_rag_retrieval():
    """Get the last RAG retrieval results."""
    global _last_rag_results
    return {
        "results": _last_rag_results
    }


@router.get("/tools/last-execution")
def get_last_tool_execution():
    """Get the last tool execution results."""
    global _last_tool_results
    return {
        "results": _last_tool_results
    }


@router.get("/tools/available")
def get_available_tools():
    """Get list of available tools and their schemas."""
    try:
        from backend.tool_manager import get_tool_registry
        
        registry = get_tool_registry()
        tools = registry.get_all_enabled()
        
        return {
            "tools": [
                {
                    "name": tool.definition.name,
                    "description": tool.definition.description,
                    "schema": tool.definition.to_json_schema()
                }
                for tool in tools.values()
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@router.get("/rag/stream")
async def stream_rag_updates():
    """Stream RAG updates in real-time via SSE."""
    import asyncio
    import json
    
    async def event_generator():
        global _last_rag_results
        last_sent = None
        
        while True:
            # Check if RAG results changed
            current = json.dumps(_last_rag_results, sort_keys=True)
            
            if current != last_sent:
                # Send new RAG data
                data = json.dumps({"results": _last_rag_results})
                yield f"data: {data}\n\n"
                last_sent = current
            
            # Heartbeat
            yield f": heartbeat\n\n"
            await asyncio.sleep(1)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/chat/history")
def get_chat_history():
    """Get the current chat context (legacy endpoint - use /chat/context instead)."""
    global _current_context
    return {
        "messages": _current_context
    }


@router.get("/chat/context")
def get_chat_context():
    """Get the current chat context."""
    global _current_context
    return {
        "messages": _current_context
    }
