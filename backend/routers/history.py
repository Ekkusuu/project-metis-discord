"""
Chat history management endpoints.
Stores chat history as JSON in the memory folder.
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.memory_manager import save_messages_before_reset

router = APIRouter(prefix="/history", tags=["history"])

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
CHAT_HISTORY_FILE = MEMORY_DIR / "chat_history.json"

# Ensure memory directory exists
MEMORY_DIR.mkdir(exist_ok=True)

# Ensure chat history file exists with a sensible default
if not CHAT_HISTORY_FILE.exists():
    try:
        from datetime import datetime as _dt
        default = {
            "messages": [
                {
                    "id": "1",
                    "text": "Hello! How can I assist you today?",
                    "sender": "ai",
                    "timestamp": _dt.now().isoformat(),
                    "tokensPerSecond": None,
                }
            ],
            "lastUpdated": _dt.now().isoformat(),
        }
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as _f:
            json.dump(default, _f, indent=2, ensure_ascii=False)
    except Exception as _e:
        # Non-fatal: just log the issue and continue
        print(f"Warning: Failed to create default chat history file: {_e}")


class Message(BaseModel):
    id: str
    text: str
    sender: str  # 'user' or 'ai'
    timestamp: str
    tokensPerSecond: float | None = None


class ChatHistory(BaseModel):
    messages: List[Message]
    lastUpdated: str


@router.get("/load", response_model=ChatHistory)
async def load_chat_history():
    """Load chat history from disk."""
    try:
        if not CHAT_HISTORY_FILE.exists():
            # Return default initial message if no history exists
            return ChatHistory(
                messages=[
                    Message(
                        id="1",
                        text="Hello! How can I assist you today?",
                        sender="ai",
                        timestamp=datetime.now().isoformat(),
                        tokensPerSecond=None,
                    )
                ],
                lastUpdated=datetime.now().isoformat(),
            )
        
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return ChatHistory(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load chat history: {str(e)}")


@router.post("/save")
async def save_chat_history(history: ChatHistory):
    """Save chat history to disk."""
    try:
        # Update the lastUpdated timestamp
        history.lastUpdated = datetime.now().isoformat()
        
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history.dict(), f, indent=2, ensure_ascii=False)
        
        return {"status": "success", "message": "Chat history saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chat history: {str(e)}")


@router.post("/reset")
async def reset_chat_history():
    """Reset chat history to initial state."""
    try:
        # Load existing history and save messages to temp_memory in background
        if CHAT_HISTORY_FILE.exists():
            try:
                with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert to format expected by memory_manager
                    messages = [
                        {"role": "user" if msg["sender"] == "user" else "assistant", "content": msg["text"]}
                        for msg in data.get("messages", [])
                    ]
                    
                    # Run memory saving in background thread to avoid lag
                    def save_in_background():
                        try:
                            save_messages_before_reset(messages)
                        except Exception as e:
                            print(f"Warning: Failed to save messages to temp_memory in background: {e}")
                    
                    thread = threading.Thread(target=save_in_background, daemon=True)
                    thread.start()
                    
            except Exception as e:
                print(f"Warning: Failed to load messages for temp_memory: {e}")
            
            # Delete the history file immediately (don't wait for background thread)
            CHAT_HISTORY_FILE.unlink()
        
        # Return the default initial message immediately
        initial_history = ChatHistory(
            messages=[
                Message(
                    id="1",
                    text="Hello! How can I assist you today?",
                    sender="ai",
                    timestamp=datetime.now().isoformat(),
                    tokensPerSecond=None,
                )
            ],
            lastUpdated=datetime.now().isoformat(),
        )
        
        return initial_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset chat history: {str(e)}")
