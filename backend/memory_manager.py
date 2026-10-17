"""
Memory system for Project Metis.

Messages that fall out of the context window are appended to
memory/temp_memory.txt. When that file grows past `memory.temp_memory_token_limit`
the LLM extracts user facts from it into memory/long_term/*.md, which the RAG
index picks up so Metis can "remember" things later.

All file work happens on a single background worker so the chat path never
waits for a summary and two summaries can't race on the same files.
"""
from __future__ import annotations

import datetime
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, Iterable, List, Optional, Union

from backend.config import MEMORY_DIR, get_config
from backend.llama_engine import chat_completion
from backend.prompt_loader import resolve_prompt
from backend.text_utils import strip_thinking
from backend.token_utils import count_tokens

TEMP_MEMORY_FILE = MEMORY_DIR / "temp_memory.txt"
LONG_TERM_DIR = MEMORY_DIR / "long_term"

MEMORY_DIR.mkdir(exist_ok=True)
LONG_TERM_DIR.mkdir(exist_ok=True)
TEMP_MEMORY_FILE.touch(exist_ok=True)

_lock = threading.RLock()
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="memory")


# ---------------------------------------------------------------------------
# Temp memory
# ---------------------------------------------------------------------------
def _format_line(msg: Union[Dict[str, str], str]) -> Optional[str]:
    if isinstance(msg, dict):
        role = msg.get("role")
        content = strip_thinking(str(msg.get("content", "")))
        if not content:
            return None
        if role == "user":
            return f"User: {content}"
        if role == "assistant":
            return f"AI: {content}"
        return None
    content = strip_thinking(str(msg))
    return f"User: {content}" if content else None


def append_to_temp_memory(messages: Iterable[Union[Dict[str, str], str]], summarize: bool = True) -> None:
    """Append messages to temp memory (synchronous). Optionally summarize if over the limit."""
    lines = [line for line in (_format_line(m) for m in messages) if line]
    if not lines:
        return
    with _lock:
        with open(TEMP_MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n\n")
        print(f"Appended {len(lines)} message(s) to temp memory")
        if summarize:
            check_and_summarize_temp_memory()


def archive_messages(messages: List[Dict[str, str]]) -> Future:
    """Queue messages for temp memory + summarization without blocking the caller."""
    return _worker.submit(_safe, append_to_temp_memory, list(messages))


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        import traceback
        print(f"Memory task failed: {e}")
        traceback.print_exc()


def get_temp_memory_content() -> str:
    try:
        return TEMP_MEMORY_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def get_temp_memory_token_count() -> int:
    return count_tokens(get_temp_memory_content())


def check_and_summarize_temp_memory() -> None:
    threshold = int(get_config().get("memory", {}).get("temp_memory_token_limit", 1000))
    tokens = get_temp_memory_token_count()
    if tokens >= threshold:
        print(f"Temp memory at {tokens} tokens (limit {threshold}); summarizing...")
        summarize_and_archive_temp_memory()


# ---------------------------------------------------------------------------
# Long-term memory
# ---------------------------------------------------------------------------
def format_summary_with_prefix(summary: str) -> str:
    """Prefix each bullet with 'user personal information:' so RAG queries can target them."""
    out = []
    for line in strip_thinking(summary).split("\n"):
        stripped = line.strip()
        if stripped[:2] in ("- ", "* ", "• "):
            rest = stripped[2:].strip()
            if not rest.lower().startswith("user personal information:"):
                indent = len(line) - len(line.lstrip())
                line = " " * indent + f"{stripped[0]} user personal information: {rest}"
        out.append(line)
    return "\n".join(out).strip()


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def summarize_and_archive_temp_memory() -> Optional[str]:
    """
    Summarize temp memory into the newest long-term file (or a new one when it
    would exceed `memory.long_term_memory_token_limit`), then clear temp memory.
    Returns the long-term file name written, or None.
    """
    with _lock:
        content = get_temp_memory_content()
        if not content.strip():
            return None

        config = get_config()
        prompts_cfg = config.get("prompts", {})
        memory_cfg = config.get("memory", {})
        system_prompt = resolve_prompt(prompts_cfg.get(
            "memory_summarization_system",
            "You are a memory extraction assistant. Extract only explicitly stated facts about the user.",
        ))
        user_template = resolve_prompt(prompts_cfg.get(
            "memory_summarization_user",
            "Extract the key facts about the user from this conversation:\n\n```\n{content}\n```",
        ))

        summary = chat_completion(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_template.format(content=content)}],
            temperature=0.1,
            max_tokens=1000,
        )
        formatted = format_summary_with_prefix(summary)
        if not formatted:
            print("Memory summary was empty; keeping temp memory for next time")
            return None

        limit = int(memory_cfg.get("long_term_memory_token_limit", 1000))
        existing = sorted(LONG_TERM_DIR.glob("memory_*.md"))
        latest = existing[-1] if existing else None
        if latest and count_tokens(latest.read_text(encoding="utf-8")) + count_tokens(formatted) < limit:
            with open(latest, "a", encoding="utf-8") as f:
                f.write(f"\n\n---\n**Updated:** {_now()}\n\n{formatted}\n")
            target = latest
        else:
            target = LONG_TERM_DIR / f"memory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            target.write_text(f"# Memory Summary\n**Created:** {_now()}\n\n---\n\n{formatted}\n", encoding="utf-8")

        TEMP_MEMORY_FILE.write_text("", encoding="utf-8")
        print(f"✓ Archived memory summary to {target.name}")

    # Make the new memories searchable right away instead of after the next restart.
    try:
        from backend.rag_engine import reindex_path_if_tracked
        reindex_path_if_tracked(target)
    except Exception as e:
        print(f"Warning: could not reindex {target.name}: {e}")
    return target.name


def save_messages_before_reset(messages: List[Dict[str, str]]) -> Future:
    """Archive a conversation that is being reset."""
    return archive_messages([m for m in messages if m.get("role") in ("user", "assistant")])
