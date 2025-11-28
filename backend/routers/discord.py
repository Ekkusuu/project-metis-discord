"""
Discord integration endpoints.
Provides SSE (Server-Sent Events) for real-time message broadcasting to UI.
"""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
from typing import List, Dict, Any

router = APIRouter(prefix="/discord", tags=["discord"])

# Import message queue from discord bot
# This will be populated by the Discord bot
discord_messages: List[Dict] = []

# Flag to track if startup reset was done
_startup_reset_done = False


def get_discord_messages() -> List[Dict]:
    """Get current Discord messages (called from Discord bot)."""
    return discord_messages


def add_discord_message(message_data: Dict):
    """Add a message from Discord bot."""
    discord_messages.append(message_data)
    # Keep only last 200 messages
    if len(discord_messages) > 200:
        discord_messages.pop(0)


@router.get("/messages")
async def get_messages():
    """Get all Discord messages."""
    return {"messages": discord_messages}


@router.post("/add-message")
async def add_message(message: Dict[str, Any]):
    """Add a message from Discord bot."""
    add_discord_message(message)
    return {"status": "success"}


@router.get("/stream")
async def stream_messages():
    """
    SSE endpoint for streaming Discord messages to frontend.
    Clients connect to this and receive real-time updates.
    """
    async def event_generator():
        last_sent_count = 0
        
        while True:
            # Check if there are new messages
            current_count = len(discord_messages)
            
            if current_count > last_sent_count:
                # Send new messages
                new_messages = discord_messages[last_sent_count:]
                for msg in new_messages:
                    data = json.dumps(msg)
                    yield f"data: {data}\n\n"
                
                last_sent_count = current_count
            
            # Send heartbeat every 15 seconds to keep connection alive
            yield f": heartbeat\n\n"
            
            # Wait a bit before checking again
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


@router.post("/clear")
async def clear_messages():
    """Clear all Discord messages."""
    discord_messages.clear()
    return {"status": "success", "message": "Discord messages cleared"}


def initialize_discord_history():
    """Initialize Discord history on startup - reset and save to temp_memory."""
    global _startup_reset_done
    
    if _startup_reset_done:
        return
    
    _startup_reset_done = True
    
    from pathlib import Path
    import json
    from backend.memory_manager import append_to_temp_memory
    
    # Path to chat history
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    CHAT_HISTORY_FILE = PROJECT_ROOT / "memory" / "chat_history.json"
    
    if not CHAT_HISTORY_FILE.exists():
        print("No existing chat history to migrate.")
        return
    
    try:
        # Load existing history
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        messages = data.get("messages", [])
        
        if messages:
            # Convert to format with usernames preserved
            formatted_messages = []
            for msg in messages:
                # Extract username if present (Discord format: [username]: message)
                text = msg.get("text", "")
                
                if msg["sender"] == "user":
                    # Keep username prefix if present, otherwise mark as "User"
                    if text.startswith("[") and "]:" in text:
                        # Already has username format
                        formatted_messages.append({
                            "role": "user",
                            "content": text
                        })
                    else:
                        formatted_messages.append({
                            "role": "user",
                            "content": f"[User]: {text}"
                        })
                else:
                    formatted_messages.append({
                        "role": "assistant",
                        "content": text
                    })
            
            # Save to temp_memory
            append_to_temp_memory(formatted_messages)
            print(f"Saved {len(formatted_messages)} messages from previous session to temp_memory")
        
        # Clear the history file for fresh start
        CHAT_HISTORY_FILE.unlink()
        print("Chat history reset for Discord bot session")
        
    except Exception as e:
        print(f"Warning: Failed to migrate chat history: {e}")
