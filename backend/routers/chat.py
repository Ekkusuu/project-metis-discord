from __future__ import annotations

from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llama_engine import chat_completion, chat_completion_stream, get_config
from backend.rag_engine import retrieve_context, format_context_for_prompt, index_all_folders

router = APIRouter(prefix="", tags=["chat"])


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

    reply = chat_completion(
        [m.model_dump() for m in messages],
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
    if rag_cfg.get("enabled", False):
        # Get the last user message as the query
        user_messages = [m for m in messages if m.role == "user"]
        if user_messages:
            last_user_query = user_messages[-1].content
            print(f"\n[RAG] Query: {last_user_query}")
            contexts = retrieve_context(last_user_query)
            
            # Filter contexts by distance threshold
            max_distance = rag_cfg.get("max_distance", 1.5)
            relevant_contexts = [ctx for ctx in contexts if ctx.get("distance", 0) <= max_distance]
            
            if relevant_contexts:
                print(f"[RAG] Retrieved {len(contexts)} contexts, {len(relevant_contexts)} within threshold (max_distance: {max_distance}):")
                for i, ctx in enumerate(relevant_contexts, 1):
                    source = ctx["metadata"].get("source_file", "unknown")
                    distance = ctx.get("distance", 0)
                    print(f"  [{i}] {source} (distance: {distance:.4f})")
                    print(f"      Preview: {ctx['text'][:100]}...")
                
                # Inject context as part of the user's message
                context_text = format_context_for_prompt(relevant_contexts)
                print(f"\n[RAG] Formatted context being injected:")
                print(f"{context_text[:500]}...\n")
                
                # Prepend the context to the last user message
                last_user_msg = messages[-1]
                enhanced_query = f"""Context from your knowledge base:

{context_text}

---

Based on the context above, please answer: {last_user_msg.content}

Remember: Use ONLY the provided context to answer. Do not use your general knowledge if it conflicts with this context."""
                
                messages[-1] = Message(role="user", content=enhanced_query)
                
                print(f"[RAG] Final message count: {len(messages)}")
                print(f"[RAG] Message roles: {[m.role for m in messages]}")
            else:
                if contexts:
                    print(f"[RAG] Retrieved {len(contexts)} contexts, but none within distance threshold ({max_distance}). Skipping RAG.")
                    for i, ctx in enumerate(contexts, 1):
                        distance = ctx.get("distance", 0)
                        source = ctx["metadata"].get("source_file", "unknown")
                        print(f"  [{i}] {source} (distance: {distance:.4f}) - REJECTED")
                else:
                    print(f"[RAG] No contexts retrieved for query")

    # Use request params or fall back to config defaults
    temperature = req.temperature if req.temperature is not None else chat_cfg.get("temperature", 0.7)
    top_p = req.top_p if req.top_p is not None else chat_cfg.get("top_p", 0.95)
    max_tokens = req.max_tokens if req.max_tokens is not None else chat_cfg.get("max_tokens", 512)

    def generate():
        import json
        import time
        
        start_time = time.time()
        token_count = 0
        
        for token in chat_completion_stream(
            [m.model_dump() for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            token_count += 1
            # Send each token as NDJSON (newline-delimited JSON)
            yield json.dumps({"delta": token}) + "\n"
        
        # Calculate tokens per second
        elapsed = time.time() - start_time
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        
        # Final chunk with performance stats
        yield json.dumps({"done": True, "tokens_per_second": round(tokens_per_second, 2)}) + "\n"

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
