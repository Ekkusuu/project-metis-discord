"""
Append-only transcript of the current session's Discord conversation
(memory/chat_history.jsonl). On startup the previous session's transcript is
moved to memory/sessions/ — learning already happened while it was live.

A legacy memory/chat_history.json (from before continuous learning existed)
is learned from once and then moved to memory/sessions/ for safekeeping.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.config import MEMORY_DIR

CHAT_LOG_FILE = MEMORY_DIR / "chat_history.jsonl"
SESSIONS_DIR = MEMORY_DIR / "sessions"
LEGACY_CHAT_HISTORY_FILE = MEMORY_DIR / "chat_history.json"

_lock = threading.Lock()


def append(entry: Dict[str, Any]) -> None:
    """Append one message (thread-safe, O(1))."""
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        MEMORY_DIR.mkdir(exist_ok=True)
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _legacy_messages() -> List[Dict[str, Any]]:
    try:
        data = json.loads(LEGACY_CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: could not read {LEGACY_CHAT_HISTORY_FILE.name}: {e}")
        return []
    messages = []
    for entry in data.get("messages", []):
        text = (entry.get("text") or "").strip()
        if not text or text == "Hello! How can I assist you today?":
            continue
        role = "user" if entry.get("sender") == "user" else "assistant"
        if role == "user" and not text.startswith("["):
            text = f"[{entry.get('discord_user') or 'User'}]: {text}"
        messages.append({"role": role, "content": text})
    return messages


def rotate_previous_session() -> None:
    """Move last session's transcript to memory/sessions/."""
    with _lock:
        if CHAT_LOG_FILE.exists() and CHAT_LOG_FILE.stat().st_size > 0:
            SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.fromtimestamp(CHAT_LOG_FILE.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
            CHAT_LOG_FILE.replace(SESSIONS_DIR / f"session_{stamp}.jsonl")


def learn_from_legacy_history() -> Optional[int]:
    """Learn once from a pre-learning chat_history.json (needs the LLM). Returns messages queued."""
    with _lock:
        if not LEGACY_CHAT_HISTORY_FILE.exists():
            return None
        messages = _legacy_messages()
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        LEGACY_CHAT_HISTORY_FILE.replace(SESSIONS_DIR / "legacy_chat_history.json")
    if messages:
        from backend.memory_manager import learn_from
        learn_from(messages)
    return len(messages)
