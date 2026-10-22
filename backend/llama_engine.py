"""
Thin HTTP client for the Node.js LLM service (backend/llm_service/server.js).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from backend.config import get_config, llm_service_url
from backend.text_utils import strip_thinking

__all__ = [
    "get_config",
    "get_llm_service_url",
    "get_model",
    "chat_completion",
    "chat_completion_stream",
    "json_completion",
    "astream_chat",
]

_http_session: Optional[requests.Session] = None


def get_llm_service_url() -> str:
    return llm_service_url()


def get_http_session() -> requests.Session:
    """Shared keep-alive session for all LLM service requests."""
    global _http_session
    if _http_session is None:
        session = requests.Session()
        # Only retry when the service is unreachable/restarting, never re-run a generation.
        retries = Retry(total=3, connect=3, read=0, backoff_factor=0.3, status_forcelist=(502, 503, 504),
                        allowed_methods=frozenset({"GET"}))
        adapter = HTTPAdapter(pool_connections=16, pool_maxsize=16, max_retries=retries)
        session.mount("http://", adapter)
        _http_session = session
    return _http_session


def get_model() -> bool:
    """Return True if the LLM service is reachable and has a model loaded."""
    try:
        response = get_http_session().get(f"{get_llm_service_url()}/health", timeout=5)
        return response.ok and bool(response.json().get("model_loaded"))
    except Exception:
        return False


def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    timeout: float = 180,
    json_schema: Optional[Dict[str, Any]] = None,
    priority: str = "interactive",
) -> str:
    """Run a non-streaming completion and return the reply text (reasoning removed)."""
    payload: Dict[str, Any] = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if json_schema is not None:
        payload["json_schema"] = json_schema
    if priority != "interactive":
        payload["priority"] = priority  # "background" work yields to anyone waiting for a reply
    try:
        response = get_http_session().post(f"{get_llm_service_url()}/chat/completion", json=payload, timeout=timeout)
        response.raise_for_status()
        content: Any = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"LLM service error: {e}") from e
    return strip_thinking(content if isinstance(content, str) else str(content))


def json_completion(
    messages: List[Dict[str, str]],
    schema: Dict[str, Any],
    temperature: float = 0.2,
    max_tokens: int = 512,
    timeout: float = 180,
) -> Any:
    """
    Completion whose output is grammar-constrained to `schema` by the LLM service,
    parsed into Python objects. Small local models can't produce malformed JSON this way.
    """
    text = chat_completion(messages, temperature=temperature, top_p=0.9, max_tokens=max_tokens, json_schema=schema,
                           timeout=timeout)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Only possible if generation hit max_tokens mid-object.
        raise RuntimeError(f"LLM returned incomplete JSON ({e}): {text[:200]}") from e


def chat_completion_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stats: Optional[Dict[str, Any]] = None,
) -> Iterator[str]:
    """
    Stream a completion, yielding text deltas.
    If `stats` is given, the service's final stats (tokens_per_second) are stored in it.
    """
    payload = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    try:
        response = get_http_session().post(
            f"{get_llm_service_url()}/chat/stream", json=payload, stream=True, timeout=(10, 300)
        )
        response.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"LLM service streaming error: {e}") from e

    with response:
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(f"LLM service streaming error: {chunk['error']}")
            if chunk.get("delta"):
                yield chunk["delta"]
            if chunk.get("done"):
                if stats is not None:
                    stats["tokens_per_second"] = chunk.get("tokens_per_second", 0)
                break


# ---------------------------------------------------------------------------
# Async streaming (used by the FastAPI chat endpoint)
# ---------------------------------------------------------------------------
async def astream_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 512,
    stats: Optional[Dict[str, Any]] = None,
):
    """
    Async version of chat_completion_stream. If the consumer stops iterating
    (e.g. the HTTP client disconnected), the upstream connection is closed and
    the LLM service aborts the generation instead of finishing it for nobody.
    """
    import aiohttp

    payload = {"messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{get_llm_service_url()}/chat/stream", json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"LLM service error {response.status}: {(await response.text())[:300]}")
            async for raw in response.content:
                line = raw.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(f"LLM service streaming error: {chunk['error']}")
                if chunk.get("delta"):
                    yield chunk["delta"]
                if chunk.get("done"):
                    if stats is not None:
                        stats["tokens_per_second"] = chunk.get("tokens_per_second", 0)
                    return
