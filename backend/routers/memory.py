"""
Memory management endpoints.
"""
import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import get_config
from backend.memory_manager import (
    get_temp_memory_content,
    get_temp_memory_token_count,
    summarize_and_archive_temp_memory,
)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryStatus(BaseModel):
    temp_memory_tokens: int
    temp_memory_content_preview: str
    threshold: int


@router.get("/status", response_model=MemoryStatus)
async def get_memory_status():
    threshold = int(get_config().get("memory", {}).get("temp_memory_token_limit", 1000))
    content = get_temp_memory_content()
    tokens = await asyncio.to_thread(get_temp_memory_token_count)
    preview = content[:500] + "..." if len(content) > 500 else content
    return MemoryStatus(temp_memory_tokens=tokens, temp_memory_content_preview=preview, threshold=threshold)


@router.post("/summarize")
async def trigger_summarization():
    """Summarize temp memory into long-term memory now, regardless of the token limit."""
    tokens = await asyncio.to_thread(get_temp_memory_token_count)
    if tokens == 0:
        return {"status": "skipped", "message": "Temp memory is empty, nothing to summarize"}
    try:
        written = await asyncio.to_thread(summarize_and_archive_temp_memory)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {e}")
    return {"status": "success", "message": f"Summarized {tokens} tokens into {written}"}
