"""
fetch_webpage - read the main text of a web page (articles, docs, wikis, posts).

Pages are fetched with httpx and reduced to their readable content with
trafilatura (boilerplate, menus and ads removed). Because anyone on Discord
can make Metis open a URL, private/loopback addresses are refused so the bot
can't be used to read services on your machine or network.
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from backend.config import get_config
from backend.tool_manager import BaseTool, ToolDefinition, ToolParameter

URL_PATTERN = re.compile(r"https?://[^\s<>\"'`]+", re.IGNORECASE)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
_MAX_BYTES = 3_000_000


class UnsafeURLError(ValueError):
    pass


def find_urls(text: str) -> list[str]:
    return [u.rstrip(".,!?;:)]}") for u in URL_PATTERN.findall(text or "")]


def _check_public(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise UnsafeURLError("Only http(s) URLs can be opened")
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UnsafeURLError(f"Could not resolve {parsed.hostname}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            raise UnsafeURLError("That address points to a private network")


def fetch_page_text(url: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """Fetch `url` and return {"url", "title", "text"} with the readable content (synchronous)."""
    import httpx
    import trafilatura

    if max_chars is None:
        max_chars = int((get_config().get("tools", {}).get("fetch_webpage") or {}).get("max_chars", 4000))

    # Checked for every request, including each redirect hop, before it is sent.
    hooks = {"request": [lambda request: _check_public(str(request.url))]}
    with httpx.Client(follow_redirects=True, max_redirects=5, timeout=12, event_hooks=hooks,
                      headers={"User-Agent": _USER_AGENT}) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            ctype = response.headers.get("content-type", "")
            body = b""
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > _MAX_BYTES:
                    break
            final_url = str(response.url)

    raw = body.decode(response.encoding or "utf-8", errors="replace")
    if "html" in ctype or raw.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        text = trafilatura.extract(raw, url=final_url, include_comments=False, include_tables=False, favor_precision=True) or ""
        meta = trafilatura.extract_metadata(raw, default_url=final_url)
        title = (meta.title if meta and meta.title else "") or ""
    elif ctype.startswith("text/") or "json" in ctype:
        text, title = raw, ""
    else:
        raise ValueError(f"Can't read this kind of content ({ctype or 'unknown'})")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return {"url": final_url, "title": title, "text": text}


class WebPageTool(BaseTool):
    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="fetch_webpage",
            description=(
                "Open a web page and read its main text. Use when the user shares a link or asks about "
                "the contents of a specific URL."
            ),
            parameters=[
                ToolParameter(name="url", type="string", description="Full http(s) URL to read.", required=True),
            ],
        )

    async def execute(self, url: str = "", **kwargs: Any) -> Dict[str, Any]:
        url = (find_urls(url) or [url.strip()])[0]
        try:
            page = await asyncio.wait_for(asyncio.to_thread(fetch_page_text, url), timeout=20)
        except UnsafeURLError as e:
            return {"success": False, "result": None, "error": str(e)}
        except asyncio.TimeoutError:
            return {"success": False, "result": None, "error": "The page took too long to load"}
        except Exception as e:
            return {"success": False, "result": None, "error": f"Could not read the page: {e}"}

        if not page["text"]:
            return {"success": True, "result": f"The page at {page['url']} has no readable text.", "metadata": page}
        header = f"Page: {page['title']}\nURL: {page['url']}" if page["title"] else f"URL: {page['url']}"
        return {"success": True, "result": f"{header}\n\n{page['text']}", "metadata": {"url": page["url"], "title": page["title"]}}
