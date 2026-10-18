"""
Single source of truth for configuration and project paths.

Every module reads config.yaml through `get_config()` so there is exactly one
cached copy (previously llama_engine, prompt_loader and voice_service each
parsed the file separately).
"""
from __future__ import annotations

import copy
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Both can be overridden (e.g. to run tests against a scratch config and memory folder).
CONFIG_PATH = Path(os.getenv("METIS_CONFIG") or PROJECT_ROOT / "config.yaml")
MEMORY_DIR = Path(os.getenv("METIS_MEMORY_DIR") or PROJECT_ROOT / "memory")

_DEFAULTS: Dict[str, Any] = {
    "personality": "casual",
    "model": {
        "path": "model/model.gguf",
        "n_ctx": 8192,
        "n_gpu_layers": -1,
    },
    "llm_service": {
        "host": "127.0.0.1",
        "port": 3730,
    },
    "backend": {
        "host": "127.0.0.1",
        "port": 8730,
    },
    "chat": {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    },
}

_config: Optional[Dict[str, Any]] = None
_lock = threading.Lock()


def _merge_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in defaults.items():
        if key not in target or target[key] is None:
            target[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(target[key], dict):
            _merge_defaults(target[key], value)
    return target


def load_config(force: bool = False) -> Dict[str, Any]:
    """Load (or reload) config.yaml, filling in defaults for missing keys."""
    global _config
    with _lock:
        if _config is not None and not force:
            return _config
        data: Dict[str, Any] = {}
        if CONFIG_PATH.exists():
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: failed to parse config.yaml ({e}); using defaults.")
        _config = _merge_defaults(data, _DEFAULTS)
        return _config


def get_config() -> Dict[str, Any]:
    return load_config()


def resolve_path(path: str | os.PathLike) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def llm_service_url() -> str:
    cfg = get_config()["llm_service"]
    port = os.getenv("LLM_SERVICE_PORT") or cfg.get("port", 3730)
    return f"http://{cfg.get('host', 'localhost')}:{port}"


def port_in_use(host: str, port: int) -> bool:
    """True if something is already listening on host:port."""
    import socket

    host = "127.0.0.1" if host in ("0.0.0.0", "localhost", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, int(port))) == 0


def backend_url() -> str:
    cfg = get_config()["backend"]
    host = cfg.get("host", "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    port = os.getenv("BACKEND_PORT") or cfg.get("port", 8730)
    return f"http://{host}:{port}"
