"""
Token counting utilities.
Simple token estimation for context management.
"""

from typing import List, Optional
from transformers import AutoTokenizer

# Global tokenizer instance
_tokenizer: Optional[AutoTokenizer] = None


def get_tokenizer() -> AutoTokenizer:
    """Get or load the tokenizer for accurate token counting."""
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    
    try:
        # Use a standard tokenizer (GPT-2 is good for general text)
        # This gives accurate token counts for chunking
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        return _tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer: {e}")
        raise


def count_tokens(text: str) -> int:
    """
    Count actual tokens in text using a tokenizer.
    This is more accurate than character-based estimation.
    """
    try:
        tokenizer = get_tokenizer()
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        # Fallback to estimation
        return estimate_token_count(text)


def encode_text(text: str) -> List[int]:
    """
    Encode text to token IDs.
    Used for precise chunking by token count.
    """
    tokenizer = get_tokenizer()
    return tokenizer.encode(text, add_special_tokens=False)


def decode_tokens(token_ids: List[int]) -> str:
    """
    Decode token IDs back to text.
    Used for precise chunking by token count.
    """
    tokenizer = get_tokenizer()
    return tokenizer.decode(token_ids, skip_special_tokens=True)


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
        total += count_tokens(msg.get("content", ""))
    return total
