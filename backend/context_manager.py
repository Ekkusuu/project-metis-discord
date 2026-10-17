"""
Keep conversations within the model's context window.

Trimming is a pure operation: it returns the kept messages and the removed ones.
Whoever owns the conversation (the Discord bot) decides what to do with the
removed messages — previously the HTTP endpoint archived them on *every*
request, so the same old messages were written to memory again and again.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from backend.config import get_config
from backend.token_utils import count_single_message_tokens

Message = Dict[str, str]


def context_budget() -> int:
    """Tokens available for the prompt: n_ctx minus room reserved for the reply."""
    config = get_config()
    n_ctx = int(config.get("model", {}).get("n_ctx", 8192))
    reply_reserve = int(config.get("chat", {}).get("max_tokens", 512))
    return max(512, n_ctx - reply_reserve)


def split_to_fit(messages: List[Message], max_tokens: Optional[int] = None) -> Tuple[List[Message], List[Message]]:
    """
    Drop the oldest non-system messages until the conversation fits.

    Returns (kept, removed). Leading system messages and the final message are
    always kept. Removal never leaves an assistant message first in the history.
    """
    if not messages:
        return [], []
    budget = max_tokens if max_tokens is not None else context_budget()

    n_system = 0
    while n_system < len(messages) and messages[n_system].get("role") == "system":
        n_system += 1
    head, body = messages[:n_system], messages[n_system:]

    sizes = [count_single_message_tokens(m) for m in body]
    total = sum(count_single_message_tokens(m) for m in head) + sum(sizes)
    cut = 0
    while total > budget and cut < len(body) - 1:
        total -= sizes[cut]
        cut += 1
    # Don't start the history with an orphaned assistant reply.
    while cut < len(body) - 1 and body[cut].get("role") == "assistant":
        cut += 1

    return head + body[cut:], body[:cut]


def trim_messages_to_context(messages: List[Message], max_tokens: Optional[int] = None) -> List[Message]:
    """Return a copy of `messages` that fits the context window (removed messages are discarded)."""
    kept, removed = split_to_fit(messages, max_tokens)
    if removed:
        print(f"Context trimmed: dropped {len(removed)} oldest message(s) for this request")
    return kept
