from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any

from llama_cpp import Llama

# Global singleton for the model to avoid re-loading between requests
_llama_model: Optional[Llama] = None
_config: Optional[Dict[str, Any]] = None

# Default relative model path
DEFAULT_MODEL_NAME = "dolphin-2.6-mistral-7b.Q5_K_M.gguf"
DEFAULT_MODEL_PATH = (Path(__file__).resolve().parents[1] / "Model" / DEFAULT_MODEL_NAME)
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml with fallback defaults."""
    global _config
    if _config is not None:
        return _config

    default_config = {
        "model": {
            "path": str(DEFAULT_MODEL_PATH),
            "n_ctx": 4096,
            "n_gpu_layers": -1,
            "n_threads": None,
            "use_flash_attn": True,
            "use_mlock": True,
        },
        "chat": {
            "system_prompt": "You are Metis, a helpful AI assistant.",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
        },
    }

    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f) or {}
            # Merge with defaults for any missing keys
            for section in default_config:
                if section not in _config:
                    _config[section] = default_config[section]
                else:
                    for key, value in default_config[section].items():
                        if key not in _config[section]:
                            _config[section][key] = value
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}. Using defaults.")
            _config = default_config
    else:
        _config = default_config

    return _config


def get_config() -> Dict[str, Any]:
    """Get the loaded configuration."""
    return load_config()


def get_model(model_path: Optional[str | os.PathLike[str]] = None) -> Llama:
    """
    Lazily load and return the global Llama model instance.

    - model_path: optional explicit path; falls back to config.yaml or default
    """
    global _llama_model
    if _llama_model is not None:
        return _llama_model

    config = get_config()
    model_cfg = config["model"]

    # Resolve model path: explicit arg > config.yaml > default
    if model_path:
        resolved = Path(model_path)
    else:
        configured_path = model_cfg.get("path", str(DEFAULT_MODEL_PATH))
        # If relative path in config, resolve from project root
        resolved = Path(configured_path)
        if not resolved.is_absolute():
            resolved = (Path(__file__).resolve().parents[1] / configured_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"LLM model not found at: {resolved}")

    # Load model settings from config with env var overrides
    n_ctx = int(os.environ.get("LLAMA_CTX", str(model_cfg.get("n_ctx", 4096))))

    # Auto threads: config (null = auto) > env > auto-detect
    config_threads = model_cfg.get("n_threads")
    if config_threads is None:
        auto_threads = os.cpu_count() or 4
    else:
        auto_threads = config_threads
    n_threads = int(os.environ.get("LLAMA_THREADS", str(auto_threads)))

    # GPU layers and other flags
    n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", str(model_cfg.get("n_gpu_layers", -1))))
    
    use_mlock_cfg = model_cfg.get("use_mlock", True)
    use_mlock = os.environ.get("LLAMA_USE_MLOCK", str(use_mlock_cfg)).lower() in ("1", "true", "yes")
    
    use_flash_attn_cfg = model_cfg.get("use_flash_attn", True)
    use_flash_attn = os.environ.get("LLAMA_USE_FLASH_ATTN", str(use_flash_attn_cfg)).lower() in ("1", "true", "yes")

    _llama_model = Llama(
        model_path=str(resolved),
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        use_mlock=use_mlock,
        use_flash_attn=use_flash_attn,
        verbose=False,
    )
    return _llama_model


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> str:
    """
    Run a chat completion using llama.cpp's chat API; messages format:
    [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
    Returns the assistant reply content as a string.
    """
    model = get_model()

    # llama-cpp-python uses create_chat_completion with OpenAI-like schema
    output = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    try:
        # OpenAI-like response structure
        return output["choices"][0]["message"]["content"].strip()
    except Exception:
        # Fallback: attempt legacy format
        return str(output)


def chat_completion_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
) -> Iterator[str]:
    """
    Stream chat completion tokens one by one using llama.cpp's streaming API.
    Yields delta content strings (individual tokens or partial text).
    """
    model = get_model()

    # Set stream=True to get an iterator of delta chunks
    stream = model.create_chat_completion(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stream=True,
    )

    for chunk in stream:
        # OpenAI streaming format: chunk["choices"][0]["delta"]["content"]
        try:
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]
        except (KeyError, IndexError):
            # Skip malformed or final chunks
            continue
