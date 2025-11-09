from __future__ import annotations

from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.llama_engine import chat_completion, chat_completion_stream, get_config

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
