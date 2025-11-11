"""
Memory management endpoints.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.memory_manager import (
    get_temp_memory_content,
    get_temp_memory_token_count,
    summarize_and_archive_temp_memory,
)
from backend.llama_engine import get_config

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryStatus(BaseModel):
    temp_memory_tokens: int
    temp_memory_content_preview: str
    threshold: int


@router.get("/status", response_model=MemoryStatus)
async def get_memory_status():
    """Get current memory system status."""
    try:
        config = get_config()
        memory_cfg = config.get("memory", {})
        threshold = memory_cfg.get("temp_memory_token_limit", 1000)
        
        content = get_temp_memory_content()
        token_count = get_temp_memory_token_count()
        
        # Preview first 500 characters
        preview = content[:500] + "..." if len(content) > 500 else content
        
        return MemoryStatus(
            temp_memory_tokens=token_count,
            temp_memory_content_preview=preview,
            threshold=threshold
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory status: {str(e)}")


@router.post("/summarize")
async def trigger_summarization():
    """Manually trigger temp_memory summarization."""
    try:
        token_count = get_temp_memory_token_count()
        
        if token_count == 0:
            return {
                "status": "skipped",
                "message": "Temp memory is empty, nothing to summarize"
            }
        
        summarize_and_archive_temp_memory()
        
        return {
            "status": "success",
            "message": f"Summarized {token_count} tokens and archived to long_term memory"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to summarize: {str(e)}")
