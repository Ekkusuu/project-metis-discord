"""
Memory orchestration.

Metis learns continuously: every few messages (and when a conversation is
reset, compacted or the bot shuts down) the new part of the conversation is
handed to backend.learning, which stores facts per person in memory/people.json
and mirrors them to memory/long_term/ for the knowledge base.

All learning runs on one background worker so replies never wait for it and
two learning passes never race on the same files.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List

from backend.config import MEMORY_DIR

LEGACY_TEMP_MEMORY_FILE = MEMORY_DIR / "temp_memory.txt"

MEMORY_DIR.mkdir(exist_ok=True)
(MEMORY_DIR / "long_term").mkdir(exist_ok=True)

_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")
_stats_lock = threading.Lock()
stats: Dict[str, Any] = {"facts_learned": 0, "notes_learned": 0, "passes": 0, "pending": 0}


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        import traceback
        print(f"Memory task failed: {e}")
        traceback.print_exc()
        return None


def _learn(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    from backend.learning import learn

    try:
        result = learn(messages)
    finally:
        with _stats_lock:
            stats["pending"] -= 1
    with _stats_lock:
        stats["passes"] += 1
        stats["facts_learned"] += result["facts"]
        stats["notes_learned"] += result["notes"]
    return result


def learn_from(messages: List[Dict[str, Any]]) -> Future:
    """Queue messages for learning without blocking the caller."""
    messages = [m for m in messages if m.get("role") in ("user", "assistant") and m.get("content")]
    with _stats_lock:
        stats["pending"] += 1
    return _worker.submit(_safe, _learn, messages)


def migrate_legacy_memory() -> Future | None:
    """
    One-time migration (needs the LLM): conversation left in the old
    temp_memory.txt is learned from, and the file is kept in memory/sessions/.
    """
    if not LEGACY_TEMP_MEMORY_FILE.exists():
        return None
    text = LEGACY_TEMP_MEMORY_FILE.read_text(encoding="utf-8")
    from backend.learning import parse_legacy_transcript

    messages = parse_legacy_transcript(text)
    (MEMORY_DIR / "sessions").mkdir(exist_ok=True)
    LEGACY_TEMP_MEMORY_FILE.replace(MEMORY_DIR / "sessions" / "legacy_temp_memory.txt")
    if not messages:
        return None
    print(f"Learning from {len(messages)} message(s) left in the old temp memory")
    return learn_from(messages)
