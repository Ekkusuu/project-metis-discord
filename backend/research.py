"""
Looking something up on the web the way a person would: search, open the most promising pages, read the
parts about the question, and search again with better words if the answer isn't there yet.

The reading is done by a neutral "researcher" prompt, not by Metis: asked to read raw search results and
stay in character at the same time, a small model invented numbers ("Cho'Gath, 19% win rate" from a page
title). The researcher writes a short factual answer with its source; Metis then says it her way.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from backend.llama_engine import json_completion

_SYSTEM = """You research questions for a voice assistant. You get web search results and excerpts of the pages \
they link to. Using only that material, write down what answers the question.
- answer: one to three plain sentences with the exact names, numbers and dates that answer the question, and \
which site says so. Empty if the material doesn't answer it.
- found: true only if the answer is in the material.
- search_again: if not found, a better web search (2-8 words; add site:domain.com if they asked for a site). \
Otherwise empty."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "found": {"type": "boolean"},
        "search_again": {"type": "string"},
    },
    "required": ["answer", "found", "search_again"],
}

# Sites people name in speech, as the transcriber writes them ("u.g", "u g g").
_SPOKEN_SITES = {r"\bu\.? ?g\.? ?g?\b": "u.gg", r"\bop\.? ?gg?\b": "op.gg", r"\breddit\b": "reddit.com",
                 r"\bwiki(pedia)?\b": "wikipedia.org", r"\bmobalytics\b": "mobalytics.gg",
                 r"\bmetasrc\b": "metasrc.com", r"\bimdb\b": "imdb.com", r"\byoutube\b": "youtube.com"}
_SITE_ASKED = re.compile(r"\b(on|in|from|at|use|using|check)\s+(the\s+)?(site|website|page)?\s*([a-z0-9-]+(\.[a-z0-9-]+)+)",
                         re.IGNORECASE)


def site_asked_for(message: str) -> Optional[str]:
    """The website the person asked to use ("search on u.gg ..."), if any."""
    if re.search(r"\b(site|website|on|from)\b", message, re.IGNORECASE):
        for pattern, site in _SPOKEN_SITES.items():
            if re.search(pattern, message, re.IGNORECASE):
                return site
    match = _SITE_ASKED.search(message)
    if match and not match.group(4)[0].isdigit() and len(match.group(4).rsplit(".", 1)[1]) >= 2:
        return match.group(4).lower()
    return None


# Video and social pages have no text to read.
_UNREADABLE = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com")


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


async def _search(query: str, count: int = 6) -> List[Dict[str, Any]]:
    """Web results; one engine first (Bing answered in ~0.9 s, ddgs's "auto" mix took 2-5 s), then the mix."""
    from backend.config import get_config
    from backend.tools.web_search_tool import WebSearchTool

    engine = (get_config().get("tools", {}).get("web_search") or {}).get("engine", "bing")
    for backend in dict.fromkeys([engine, "auto"]):
        try:
            found = await asyncio.wait_for(
                asyncio.to_thread(WebSearchTool()._search_sync, query, count, "text", backend), timeout=8)
            if found:
                return found
        except Exception as e:
            print(f"   search ({backend}) failed for {query!r}: {e}")
    return []


def _rank(results: List[Dict[str, Any]], question: str, site: Optional[str]) -> List[Dict[str, Any]]:
    """Results by how well their title and snippet match the question; the site they asked for first."""
    if not results:
        return []
    try:
        import numpy as np

        from backend.rag_engine import get_embedder

        embedder = get_embedder()
        q = np.array(embedder.embed_query(question)[0])
        docs = np.array(embedder.embed_documents([f"{r['title']}. {r['content']}" for r in results]))
        scores = list(docs @ q / (np.linalg.norm(docs, axis=1) * np.linalg.norm(q) + 1e-9))
    except Exception:
        scores = [-i for i in range(len(results))]  # keep the search engine's order
    for i, r in enumerate(results):
        if site and _domain(r["url"]).endswith(site):
            scores[i] += 1.0
        if _domain(r["url"]).endswith(_UNREADABLE):
            scores[i] -= 1.0
    return [r for _, r in sorted(zip(scores, results), key=lambda pair: -pair[0])]


async def _read(url: str, question: str, max_chars: int) -> str:
    from backend.tools.web_page_tool import fetch_page_text, relevant_passages

    try:
        page = await asyncio.wait_for(asyncio.to_thread(fetch_page_text, url, None, full=True), timeout=5)
        return await asyncio.to_thread(relevant_passages, page["text"], question, max_chars)
    except Exception as e:
        print(f"   couldn't read {url}: {e}")
        return ""


async def research(question: str, query: str, earlier: str = "", *, rounds: int = 2, pages: int = 2,
                   page_chars: int = 1500, notify: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Search and read until the question is answered or `rounds` searches were made.
    Returns {"answer", "found", "sources", "searches", "material"}; material is the last round's reading.
    """
    site = site_asked_for(question)
    if site and "site:" not in query:
        query = f"{query} site:{site}"
    searches: List[str] = []
    seen: set = set()
    answer, found, sources, material = "", False, [], ""
    for round_no in range(rounds):
        t0 = time.perf_counter()
        searches.append(query)
        results = await _search(query)
        t_search = time.perf_counter() - t0
        if not results and site and "site:" in query:
            # Some sites aren't indexed well; look in the whole web instead.
            query = query.split("site:")[0].strip()
            results = await _search(query)
        ranked = _rank([r for r in results if r.get("url") not in seen], question, site)
        chosen = ranked[:pages]
        seen.update(r["url"] for r in chosen)
        texts = await asyncio.gather(*(_read(r["url"], question, page_chars) for r in chosen))
        t_read = time.perf_counter() - t0
        listing = "\n".join(f"[{i}] {r['title']} ({_domain(r['url'])})\n    {r['content'][:300]}"
                            for i, r in enumerate(ranked[:5], 1))
        pages_text = "\n\n".join(f"Page {_domain(r['url'])} ({r['title']}):\n{t}" for r, t in zip(chosen, texts) if t)
        material = f"Search results for {query!r}:\n{listing}" + (f"\n\n{pages_text}" if pages_text else "")
        prompt = (f"Question: {question}\n" + (f"Said before: {earlier}\n" if earlier else "")
                  + (f"Notes from the earlier search: {answer}\n" if answer else "") + f"\n{material}")
        try:
            step = await asyncio.to_thread(
                json_completion, [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                _SCHEMA, 0.1, 220)
        except Exception as e:
            print(f"   researcher failed: {e}")
            break
        answer = str(step.get("answer") or answer).strip()
        found = bool(step.get("found")) and bool(answer)
        sources = [_domain(r["url"]) for r, t in zip(chosen, texts) if t] or [_domain(r["url"]) for r in ranked[:2]]
        print(f"🔬 search {round_no + 1} {query!r}: read {', '.join(_domain(r['url']) for r in chosen) or 'nothing'} "
              f"in {time.perf_counter() - t0:.1f}s (search {t_search:.1f}, read {t_read - t_search:.1f}) -> {'found' if found else 'not found'}: {answer[:160]!r}")
        next_query = str(step.get("search_again") or "").strip()
        if found or not next_query or next_query in searches or round_no + 1 >= rounds:
            break
        query = next_query
        if notify:
            notify("web_search_more")
    return {"answer": answer, "found": found, "sources": sources, "searches": searches, "material": material}


def format_brief(result: Dict[str, Any], question: str) -> str:
    """
    What Metis gets to see: only the researched answer, or that nothing was found. Given the pages too, the
    9B model mixed in details from other results (an older patch, a different site) and built "fun facts" on them.
    """
    if result["found"]:
        return f"You looked this up ({', '.join(result['sources'])}). What you found: {result['answer']}"
    return ("You looked this up but couldn't find a clear answer"
            + (f" (closest: {result['answer']})" if result["answer"] else "") + ". Say so; don't guess.")
