from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llama_engine import chat_completion, chat_completion_stream, get_config
from backend.rag_engine import retrieve_context, format_context_for_prompt, index_all_folders, generate_rag_query
from backend.context_manager import trim_messages_to_context

router = APIRouter(prefix="", tags=["chat"])

# Store last RAG retrieval results
_last_rag_results: List[Dict[str, Any]] = []

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
    """
    if (not req.messages or len(req.messages) == 0) and (not req.prompt):
        raise HTTPException(status_code=400, detail="Provide `messages` or `prompt`.")

    config = get_config()
    chat_cfg = config["chat"]
    rag_cfg = config.get("rag", {})
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

    # RAG: Retrieve context if enabled
    global _last_rag_results
    if rag_cfg.get("enabled", False):
        # Get the last user message as the query
        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            last_user_query = user_messages[-1].content
            
            # Generate contextual RAG query using conversation history
            messages_for_query = [{"role": m.role, "content": m.content} for m in messages]
            rag_query = generate_rag_query(messages_for_query, last_user_query)
            
            # Retrieve contexts using the generated query
            contexts = retrieve_context(rag_query)
            
            # Filter contexts by thresholds
            max_distance = rag_cfg.get("max_distance", 1.5)
            reranker_min_score = rag_cfg.get("reranker_min_score", -1)
            use_reranker = rag_cfg.get("use_reranker", False)
            
            relevant_contexts = []
            for ctx in contexts:
                # Check distance threshold (if not disabled with -1)
                if max_distance != -1 and ctx.get("distance", 0) > max_distance:
                    continue
                
                # Check rerank score threshold (if reranker is enabled and threshold is not disabled with -1)
                if use_reranker and reranker_min_score != -1:
                    if ctx.get("rerank_score") is not None and ctx.get("rerank_score") < reranker_min_score:
                        continue
                
                relevant_contexts.append(ctx)
            
            # Store results for the frontend to display, including the generated query
            _last_rag_results = {
                "query": rag_query,  # Include the generated query
                "original_query": last_user_query,  # Include the original user message
                "results": [
                    {
                        "source_file": ctx["metadata"].get("source_file", "unknown"),
                        "distance": ctx.get("distance", 0),
                        "rerank_score": ctx.get("rerank_score"),  # Include rerank score if available
                        "text_preview": ctx["text"][:200],
                        "chunk_index": ctx["metadata"].get("chunk_index", 0),
                        "used": True
                    }
                    for ctx in relevant_contexts
                ] + [
                    {
                        "source_file": ctx["metadata"].get("source_file", "unknown"),
                        "distance": ctx.get("distance", 0),
                        "rerank_score": ctx.get("rerank_score"),  # Include rerank score if available
                        "text_preview": ctx["text"][:200],
                        "chunk_index": ctx["metadata"].get("chunk_index", 0),
                        "used": False
                    }
                    for ctx in contexts if ctx not in relevant_contexts
                ]
            }
            
            if relevant_contexts:
                # Inject context into the system prompt
                context_text = format_context_for_prompt(relevant_contexts)
                
                # Find the system message and append context to it
                for i, msg in enumerate(messages):
                    if msg.role == "system":
                        enhanced_system = f"{msg.content}\n\n---\nRelevant information from your knowledge base:\n\n{context_text}\n\nUse this information to answer the user's question."
                        messages[i] = Message(role="system", content=enhanced_system)
                        break
            else:
                # No relevant contexts, but still include the query info
                if not contexts:
                    _last_rag_results = {
                        "query": rag_query,
                        "original_query": last_user_query,
                        "results": []
                    }

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
