"""
Append-only log of the current session's Discord conversation (memory/chat_history.jsonl).

On the next startup the previous session's log is moved into temp memory so
nothing said to Metis is lost between restarts.
"""
from __future__ import annotations

import json
import threading
from typing import Any, Dict, List

from backend.config import MEMORY_DIR

CHAT_LOG_FILE = MEMORY_DIR / "chat_history.jsonl"
LEGACY_CHAT_HISTORY_FILE = MEMORY_DIR / "chat_history.json"

_lock = threading.Lock()


def append(entry: Dict[str, Any]) -> None:
    """Append one message (thread-safe, O(1) — the old JSON file was rewritten on every message)."""
    line = json.dumps(entry, ensure_ascii=False)
    with _lock:
        MEMORY_DIR.mkdir(exist_ok=True)
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _read_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if LEGACY_CHAT_HISTORY_FILE.exists():
        try:
            data = json.loads(LEGACY_CHAT_HISTORY_FILE.read_text(encoding="utf-8"))
            entries.extend(data.get("messages", []))
        except Exception as e:
            print(f"Warning: could not read {LEGACY_CHAT_HISTORY_FILE.name}: {e}")
    if CHAT_LOG_FILE.exists():
        for line in CHAT_LOG_FILE.read_text(encoding="utf-8").splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _to_memory_message(entry: Dict[str, Any]) -> Dict[str, str] | None:
    text = (entry.get("text") or entry.get("content") or "").strip()
    if not text:
        return None
    role = entry.get("role") or ("user" if entry.get("sender") == "user" else "assistant")
    if role == "user" and not text.startswith("["):
        text = f"[{entry.get('discord_user') or entry.get('username') or 'User'}]: {text}"
    return {"role": role, "content": text}


def migrate_previous_session() -> int:
    """Move the previous session's log into temp memory. Returns the number of messages moved."""
    from backend.memory_manager import append_to_temp_memory

    with _lock:
        entries = _read_entries()
        messages = [m for m in (_to_memory_message(e) for e in entries) if m]
        if messages:
            # Summarization (an LLM call) is deferred; the LLM service may not be up yet.
            append_to_temp_memory(messages, summarize=False)
        CHAT_LOG_FILE.unlink(missing_ok=True)
        LEGACY_CHAT_HISTORY_FILE.unlink(missing_ok=True)
    return len(messages)
