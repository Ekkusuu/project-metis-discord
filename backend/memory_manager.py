"""
Memory management system for Project Metis.
Handles temporary memory storage and long-term memory summarization.
"""
from pathlib import Path
from typing import List, Dict
import datetime
from backend.llama_engine import chat_completion, get_config
from backend.token_utils import estimate_token_count

# Memory paths
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
TEMP_MEMORY_FILE = MEMORY_DIR / "temp_memory.txt"
LONG_TERM_DIR = MEMORY_DIR / "long_term"

# Ensure directories exist
MEMORY_DIR.mkdir(exist_ok=True)
LONG_TERM_DIR.mkdir(exist_ok=True)


def append_to_temp_memory(messages) -> None:
    """
    Append messages to temp_memory.txt.
    
    Args:
        messages: List of message dicts with role and content, or list of strings
    """
    if not messages:
        return
    
    with open(TEMP_MEMORY_FILE, "a", encoding="utf-8") as f:
        for msg in messages:
            # Handle both dict messages and plain strings
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    f.write(f"User: {content}\n")
                elif role == "assistant":
                    f.write(f"AI: {content}\n")
            else:
                # Plain string (legacy support)
                f.write(f"User: {msg}\n")
        f.write("\n")  # Add blank line between conversation chunks
    
    print(f"Appended {len(messages)} message(s) to temp_memory")
    
    # Check if we need to summarize
    check_and_summarize_temp_memory()


def get_temp_memory_content() -> str:
    """Get the current content of temp_memory.txt."""
    if not TEMP_MEMORY_FILE.exists():
        return ""
    
    with open(TEMP_MEMORY_FILE, "r", encoding="utf-8") as f:
        return f.read()


def get_temp_memory_token_count() -> int:
    """Get the approximate token count of temp_memory."""
    content = get_temp_memory_content()
    return estimate_token_count(content)


def check_and_summarize_temp_memory() -> None:
    """
    Check if temp_memory has reached the token threshold from config.
    If so, summarize it and move to long_term storage.
    """
    config = get_config()
    memory_cfg = config.get("memory", {})
    threshold = memory_cfg.get("temp_memory_token_limit", 1000)
    
    token_count = get_temp_memory_token_count()
    
    if token_count >= threshold:
        print(f"Temp memory reached {token_count} tokens (threshold: {threshold}), creating summary...")
        summarize_and_archive_temp_memory()


def summarize_and_archive_temp_memory() -> None:
    """
    Use AI to summarize temp_memory content and save to long_term folder.
    Clears temp_memory after archiving.
    """
    content = get_temp_memory_content()
    
    if not content.strip():
        print("Temp memory is empty, nothing to summarize")
        return
    
    print(f"Starting summarization of {len(content)} characters...")
    
    # Create summary using AI
    try:
        system_prompt = """You are a memory extraction assistant. Extract ONLY explicitly stated facts about the user from conversations.

CRITICAL RULES:
- Only include information the user DIRECTLY stated
- DO NOT infer, assume, or elaborate beyond what was said
- DO NOT add interpretations or explanations
- DO NOT include goals/interests unless explicitly mentioned
- Skip any section that has no explicit information
- Be extremely literal and conservative

Format as brief bullet points under relevant headings. Only create headings if you have actual facts."""

        user_prompt = f"""Extract ONLY the explicitly stated facts about the user from this conversation. Do not infer or elaborate:

```
{content}
```

List only what the user directly said about themselves. Be brief and literal."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        print("Calling LLM service for summarization...")
        summary = chat_completion(messages, temperature=0.1, max_tokens=1000)
        print(f"Summary generated: {len(summary)} characters")
        
        # Save to long_term with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = LONG_TERM_DIR / f"memory_{timestamp}.md"
        
        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(f"# Memory Summary\n")
            f.write(f"**Created:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"---\n\n")
            f.write(summary)
        
        print(f"✓ Created long-term memory: {summary_file.name}")
        
        # Clear temp_memory
        TEMP_MEMORY_FILE.write_text("", encoding="utf-8")
        print("✓ Cleared temp_memory.txt")
        
    except Exception as e:
        print(f"ERROR summarizing temp memory: {e}")
        import traceback
        traceback.print_exc()
        # Don't re-raise to avoid blocking the main flow


def save_messages_before_reset(messages: List[Dict[str, str]]) -> None:
    """
    Save all messages (user and assistant) to temp_memory before chat reset.
    
    Args:
        messages: List of message dicts with role and content
    """
    # Save all messages for context, not just user messages
    conversation_messages = [
        msg for msg in messages 
        if msg.get("role") in ["user", "assistant"]
    ]
    
    if conversation_messages:
        append_to_temp_memory(conversation_messages)
        print(f"Saved {len(conversation_messages)} message(s) before reset")
