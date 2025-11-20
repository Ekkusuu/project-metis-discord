"""
Token counting utilities.
Simple token estimation for context management.
"""

from typing import List, Optional, Any
from transformers import AutoTokenizer
from backend.llama_engine import get_config

# Global tokenizer instance
_tokenizer: Optional[Any] = None


def get_tokenizer() -> Any:
    """Get or load the tokenizer for accurate token counting.

    This prefers a local tokenizer path defined in `config.yaml` under
    `model.tokenizer_path` and uses `local_files_only=True` to avoid
    contacting the Hugging Face Hub. If the local tokenizer is missing,
    a very small whitespace-based fallback tokenizer is used so the
    rest of the app can continue to run (counts will be approximate).
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer

    config = get_config()
    model_cfg = config.get("model", {})
    tokenizer_path = model_cfg.get("tokenizer_path", "gpt2")
    cache_dir = model_cfg.get("cache_dir") if isinstance(model_cfg, dict) else None

    try:
        kwargs = {"local_files_only": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, **kwargs)
        return _tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from local path '{tokenizer_path}': {e}")
        # Try to load a local gpt2 tokenizer as a fallback (still local-only)
        try:
            _tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True, local_files_only=True)
            return _tokenizer
        except Exception:
            print("Warning: Local 'gpt2' tokenizer not available. Using simple whitespace fallback tokenizer.")

            class _SimpleWhitespaceTokenizer:
                def encode(self, text: str, add_special_tokens: bool = False):
                    # Represent tokens as incremental ids per whitespace token
                    parts = text.split()
                    return list(range(len(parts)))

                def decode(self, token_ids: List[int], skip_special_tokens: bool = True):
                    # Decoding is not meaningful for the fallback; return placeholder text
                    return " ".join(["<tok>" for _ in token_ids])

            _tokenizer = _SimpleWhitespaceTokenizer()
            return _tokenizer


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
