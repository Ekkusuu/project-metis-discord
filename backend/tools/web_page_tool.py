"""
fetch_webpage - read the main text of a web page (articles, docs, wikis, posts).

Pages are fetched with httpx and reduced to their readable content with
trafilatura (boilerplate, menus and ads removed). Because anyone on Discord
can make Metis open a URL, private/loopback addresses are refused so the bot
can't be used to read services on your machine or network.
"""
from __future__ import annotations

import asyncio
import html
import ipaddress
import re
import socket
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

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


def _download(url: str) -> Tuple[str, str, str]:
    """(final url, content type, body text). Every redirect hop is checked before it is requested."""
    try:
        import primp
    except ImportError:
        primp = None
    if primp is None:
        import httpx

        hooks = {"request": [lambda request: _check_public(str(request.url))]}
        with httpx.Client(follow_redirects=True, max_redirects=5, timeout=12, event_hooks=hooks,
                          headers={"User-Agent": _USER_AGENT}) as client:
            response = client.get(url)
            response.raise_for_status()
            return str(response.url), response.headers.get("content-type", ""), response.text[:_MAX_BYTES]

    # primp (installed with ddgs) looks like a real Chrome to bot walls; sites like u.gg, metasrc and
    # mobalytics answered plain httpx with 403.
    client = primp.Client(impersonate="chrome_133", impersonate_os="windows", follow_redirects=False, timeout=12)
    for _ in range(6):
        _check_public(url)
        response = client.get(url)
        location = response.headers.get("location")
        if response.status_code in (301, 302, 303, 307, 308) and location:
            url = urljoin(url, location)
            continue
        if response.status_code >= 400:
            raise ValueError(f"HTTP {response.status_code}")
        return url, response.headers.get("content-type", ""), response.text[:_MAX_BYTES]
    raise ValueError("Too many redirects")


def _visible_text(html: str) -> str:
    """All text a visitor would see, for pages whose content isn't an article (tier lists, tables, stats)."""
    import lxml.html

    doc = lxml.html.fromstring(html)
    for node in doc.xpath("//script|//style|//noscript|//svg|//template|//head"):
        node.drop_tree()
    lines = []
    for block in doc.iter():
        if not isinstance(block.tag, str):
            continue
        for piece in (block.text, block.tail):
            if piece and piece.strip():
                lines.append(piece.strip())
    return re.sub(r"[ \t]+", " ", " ".join(lines))


def fetch_page_text(url: str, max_chars: Optional[int] = None, *, full: bool = False) -> Dict[str, Any]:
    """
    Fetch `url` and return {"url", "title", "text"} with the readable content (synchronous).
    full=True returns the whole text (for picking the relevant parts with relevant_passages).
    """
    import trafilatura

    if max_chars is None:
        max_chars = int((get_config().get("tools", {}).get("fetch_webpage") or {}).get("max_chars", 4000))

    final_url, ctype, raw = _download(url)
    if "html" in ctype or raw.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        # The article extractor keeps prose and drops the rest; on data pages that's everything that
        # matters (u.gg's tier list: 278 characters of intro vs 2,000 of champions and win rates), and it
        # glues table cells together ("Champion 7" + "57.0%" -> "Champion 757.0%"). Full text is read by
        # relevant_passages, which skips menus by itself.
        visible = _visible_text(raw)
        text = "" if full else trafilatura.extract(raw, url=final_url, include_comments=False, include_tables=True) or ""
        if len(text) < len(visible) / 3:
            text = visible
        found = re.search(r"<title[^>]*>(.*?)</title>", raw[:200_000], re.IGNORECASE | re.DOTALL)
        title = html.unescape(found.group(1)).strip() if found else ""
    elif ctype.startswith("text/") or "json" in ctype:
        text, title = raw, ""
    else:
        raise ValueError(f"Can't read this kind of content ({ctype or 'unknown'})")

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not full and len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " …"
    return {"url": final_url, "title": title, "text": text}


def relevant_passages(text: str, question: str, max_chars: int = 1500, size: int = 500) -> str:
    """The parts of a long page that are about `question` (embedding similarity), in page order."""
    if len(text) <= max_chars:
        return text
    import numpy as np

    from backend.rag_engine import get_embedder

    pieces = [text[i:i + size] for i in range(0, len(text), size)][:200]
    embedder = get_embedder()
    q = np.array(embedder.embed_query(question)[0])
    docs = np.array(embedder.embed_documents(pieces))
    scores = docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q) + 1e-9)
    keep = sorted(np.argsort(-scores)[: max(1, max_chars // size)])
    out = ""
    for n, i in enumerate(keep):
        out += pieces[i] if n and keep[n - 1] == i - 1 else (" … " if out else "") + pieces[i]
    return out.strip()


_JINA_READER = "https://r.jina.ai/"
_page_cache: Dict[str, Tuple[float, str]] = {}  # url -> (fetched at, full text); a page is often opened twice


def _rendered_text(url: str) -> str:
    """The page as a browser shows it, through Jina Reader (free, no key; it runs the page's JavaScript).
    For apps that send an empty shell and build the page in the browser."""
    import primp

    client = primp.Client(impersonate="chrome_133", timeout=12)
    response = client.get(_JINA_READER + url, headers={"X-Return-Format": "text"})
    if response.status_code >= 400:
        raise ValueError(f"reader returned HTTP {response.status_code}")
    return response.text.strip()


def read_page(url: str, question: str, max_chars: int = 1500) -> str:
    """
    The parts of a page about `question`, like a coding agent's web fetch (synchronous). Pages with almost no
    text in their HTML (built by JavaScript) are read through a renderer instead (tools.fetch_webpage.renderer:
    "jina" or "none"; the URL is sent to r.jina.ai).
    """
    import time

    cached = _page_cache.get(url)
    if cached and time.time() - cached[0] < 600:
        text = cached[1]
    else:
        text, error = "", None
        try:
            text = fetch_page_text(url, full=True)["text"]
        except UnsafeURLError:
            raise
        except Exception as e:
            error = e
        renderer = (get_config().get("tools", {}).get("fetch_webpage") or {}).get("renderer", "jina")
        if len(text) < 500 and renderer == "jina":
            _check_public(url)
            try:
                text = _rendered_text(url) or text
            except Exception as e:
                print(f"   renderer couldn't read {url}: {e}")
        if not text:
            raise ValueError(f"no readable text ({error or 'empty page'})")
        if len(_page_cache) > 30:
            _page_cache.clear()
        _page_cache[url] = (time.time(), text)
    return relevant_passages(text, question, max_chars)


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
