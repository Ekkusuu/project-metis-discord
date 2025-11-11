"""
Token counting utilities.
Simple token estimation for context management.
"""


def estimate_token_count(text: str) -> int:
    """
    Simple token estimation: ~4 characters per token on average.
    This is a rough approximation but works well enough for context management.
    """
    return len(text) // 4


def count_message_tokens(messages: list) -> int:
    """Count approximate total tokens in all messages."""
    total = 0
    for msg in messages:
        # Add role overhead (~3 tokens per message for role markers)
        total += 3
        # Add content tokens
        total += estimate_token_count(msg.get("content", ""))
    return total
