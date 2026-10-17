"""
Utility module for loading system prompts from system_prompts.yaml.
Prompts in config.yaml can reference prompts using @ notation:
  system_prompt: "@text_chat.default"
  system_prompt: "@judgment.voice_active"

Auto-personality mode:
  If config has a 'personality' setting, prompts are automatically selected:
  personality: "casual" will use @text_chat.casual, @voice_chat.casual, etc.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from backend.config import PROJECT_ROOT, get_config

# Cache for loaded prompts
_prompts_cache: Optional[Dict[str, Any]] = None
_prompts_file = PROJECT_ROOT / "system_prompts.yaml"


def load_prompts() -> Dict[str, Any]:
    """Load system prompts from system_prompts.yaml file."""
    global _prompts_cache
    
    if _prompts_cache is not None:
        return _prompts_cache
    
    try:
        if _prompts_file.exists():
            with open(_prompts_file, 'r', encoding='utf-8') as f:
                _prompts_cache = yaml.safe_load(f) or {}
                print(f"✓ Loaded system prompts from {_prompts_file}")
        else:
            print(f"⚠️  system_prompts.yaml not found at {_prompts_file}")
            _prompts_cache = {}
    except Exception as e:
        print(f"❌ Failed to load system_prompts.yaml: {e}")
        _prompts_cache = {}
    
    return _prompts_cache


def resolve_prompt(prompt_value: Any, context: Optional[str] = None) -> str:
    """
    Resolve a prompt value. If it starts with @, load from system_prompts.yaml.
    If personality is set in config, automatically selects the right prompt variant.
    Otherwise, return the value as-is.
    
    Examples:
        "@text_chat" + personality="casual" -> loads text_chat.casual
        "@text_chat.therapist" -> loads text_chat.therapist (explicit)
        "You are a bot" -> returns "You are a bot"
    
    To add new personalities:
        1. Add sections in system_prompts.yaml (e.g., text_chat.friendly, voice_chat.friendly)
        2. Set personality: "friendly" in config.yaml
        That's it! No code changes needed.
    
    Args:
        prompt_value: Either a prompt reference (@category or @category.tag) or direct prompt text
        context: Optional context hint (e.g., "text_chat", "voice_chat", "judgment_text")
        
    Returns:
        The resolved prompt text
    """
    if not isinstance(prompt_value, str):
        return str(prompt_value)
    
    # Check if it's a reference to system_prompts.yaml
    if prompt_value.startswith("@"):
        # Remove @ and split by dots
        ref_path = prompt_value[1:].split(".")
        
        # Load prompts and config
        prompts = load_prompts()
        config = get_config()
        personality = config.get("personality", "casual")  # Default to casual if not set
        
        # If only category is specified (e.g., "@text_chat"), append personality
        if len(ref_path) == 1:
            ref_path.append(personality)
        
        # Navigate through the nested structure
        result = prompts
        for key in ref_path:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                print(f"⚠️  Prompt reference '@{'.'.join(ref_path)}' not found in system_prompts.yaml")
                print(f"    Available personalities: {list(prompts.get(ref_path[0], {}).keys()) if ref_path else 'N/A'}")
                return prompt_value  # Return the reference itself if not found
        
        if isinstance(result, str):
            return result.strip()
        else:
            print(f"⚠️  Prompt reference '@{'.'.join(ref_path)}' did not resolve to a string")
            return prompt_value
    
    # Return as-is if not a reference
    return prompt_value


def reload_prompts():
    """Force reload of system prompts (useful for development)."""
    global _prompts_cache
    _prompts_cache = None
    load_prompts()
