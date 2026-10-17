"""
Test bootstrap.

This module is imported by pytest *before* any test module, and it must run
before any `backend` import: `backend/config.py` reads the env vars
METIS_CONFIG / METIS_MEMORY_DIR at import time, and `backend/token_utils.py`
resolves its tokenizer through that config.

Both env vars point at a scratch directory inside the OS temp folder, so the
suite never touches the real `config.yaml` or `memory/` folder, never needs the
network and never loads an embedding/reranker/STT/TTS model.

`model.tokenizer_path` intentionally points at a non-existent file: token_utils
then tries a locally cached gpt2 tokenizer and finally falls back to its tiny
whitespace tokenizer, so the tests stay independent of which one is available.
Tests that care about exact token counts monkeypatch the counter instead.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

_SESSION_TMP = Path(tempfile.mkdtemp(prefix="metis-tests-"))

(_SESSION_TMP / "memory").mkdir(parents=True, exist_ok=True)

_CONFIG_PATH = _SESSION_TMP / "config.yaml"
_CONFIG_PATH.write_text(
    "\n".join(
        [
            "model:",
            "  n_ctx: 2048",
            "  tokenizer_path: does-not-exist",  # forces the local fallback tokenizer
            "chat:",
            "  max_tokens: 256",
            "personality: casual",
            "tools: {}",
            "rag:",
            "  enabled: false",
            "",
        ]
    ),
    encoding="utf-8",
)

os.environ["METIS_CONFIG"] = str(_CONFIG_PATH)
os.environ["METIS_MEMORY_DIR"] = str(_SESSION_TMP / "memory")

# Belt and braces: make Hugging Face libraries refuse any network access, so a
# cold tokenizer cache degrades to token_utils' whitespace fallback instead of
# reaching for the internet.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest hook signature
    shutil.rmtree(_SESSION_TMP, ignore_errors=True)
