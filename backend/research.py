"""
A web research agent, built like the web tools of coding agents: the model works step by step with
`search` (web results with snippets), `news` (dated news results), `open` (read pages, keeping only the parts
about the question; several at once) and finally `answer` or `give_up`. Each step is one constrained JSON
choice, so even a small local model can't produce a malformed tool call.

The research is done by a neutral prompt, not by Metis: asked to read raw search results and stay in
character at the same time, a small model invented numbers from page titles. The agent writes down the
answer with its source; Metis then says it her way.
"""
from __future__ import annotations

import asyncio
import datetime
import re
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from backend.llama_engine import json_completion

_SYSTEM = """You find information on the web for a voice assistant, one step at a time. Today is {today}.
Tools:
- search: web search. input = 2-8 keywords (add site:domain.com to search one website)
- news: news search, newest first, for recent events. input = keywords
- open: read pages. input = result numbers like "1" or "1,3", or a URL. You get the parts about the question.
- answer: input = one to three plain sentences that answer the question with the exact names, numbers and \
dates, and which site says so
- give_up: the answer can't be found. input = the closest thing you found, or empty
Rules:
- Snippets are short and often outdated. Open pages before answering with numbers, prices, rankings, \
dates or lists, unless a snippet states the answer plainly.
- If the pages don't have it, search again with different words or open other results.
- Use only what you read. Never guess or fill in from memory.
Write `thought` as one short sentence about what to do next."""

_ACTIONS = ["search", "news", "open", "answer", "give_up"]


def _schema(actions: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {"thought": {"type": "string"}, "action": {"enum": actions}, "input": {"type": "string"}},
        "required": ["thought", "action", "input"],
    }


# Video and social pages have no text to read.
_UNREADABLE = ("youtube.com", "youtu.be", "tiktok.com", "instagram.com", "facebook.com", "x.com", "twitter.com")


def site_in(query: str) -> Optional[str]:
    """The site a query is limited to ("... site:example.com"), if any. The query writer adds it when someone
    names a website; nothing here knows particular sites."""
    match = re.search(r"\bsite:(\S+)", query)
    return match.group(1).lower().removeprefix("www.") if match else None


def _domain(url: str) -> str:
    host = urlparse(url).hostname or ""
    return host[4:] if host.startswith("www.") else host


async def _search(query: str, kind: str = "text", count: int = 6) -> List[Dict[str, Any]]:
    """Results from DuckDuckGo's ddgs: one engine first (Bing answered in ~0.9 s, the "auto" mix in 2-5 s),
    then the mix. A site: search with no results is retried on the whole web (the name may be misheard)."""
    from backend.config import get_config
    from backend.tools.web_search_tool import WebSearchTool

    engine = (get_config().get("tools", {}).get("web_search") or {}).get("engine", "bing")
    attempts = [(query, b) for b in dict.fromkeys([engine, "auto"])] if kind == "text" else [(query, "auto")]
    if site_in(query):
        attempts.append((re.sub(r"\s*\bsite:\S+", "", query).strip(), "auto"))
    for q, backend in attempts:
        try:
            found = await asyncio.wait_for(
                asyncio.to_thread(WebSearchTool()._search_sync, q, count, kind, backend), timeout=8)
            if found:
                return found
        except Exception as e:
            print(f"   {kind} search ({backend}) failed for {q!r}: {e}")
    return []


async def _read(url: str, question: str, max_chars: int) -> str:
    from backend.tools.web_page_tool import read_page

    try:
        return await asyncio.wait_for(asyncio.to_thread(read_page, url, question, max_chars), timeout=12)
    except Exception as e:
        print(f"   couldn't read {url}: {e}")
        return ""


def _listing(results: List[Dict[str, Any]], start: int) -> str:
    lines = []
    for i, r in enumerate(results, start):
        date = f", {r['date'][:10]}" if r.get("date") else ""
        note = " (video/social, can't be read)" if _domain(r["url"]).endswith(_UNREADABLE) else ""
        lines.append(f"[{i}] {r['title']} ({_domain(r['url'])}{date}){note}\n    {r['content'][:280]}")
    return "\n".join(lines)


def _clip(transcript: List[str], budget: int) -> str:
    """The steps so far, oldest observations shortened first so the prompt fits the utility context."""
    parts = list(transcript)
    while sum(map(len, parts)) > budget and any(len(p) > 400 for p in parts[:-1]):
        i = next(i for i, p in enumerate(parts[:-1]) if len(p) > 400)
        parts[i] = parts[i][:400] + " …(shortened)"
    return "\n\n".join(parts)


async def research(question: str, query: str, earlier: str = "", *, max_steps: int = 5, page_chars: int = 1500,
                   time_budget: float = 25.0, first_kind: str = "text",
                   notify: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    """
    Research `question`, starting with a `first_kind` ("text" or "news") search for `query` (no model step
    needed for that).
    Returns {"answer", "found", "sources", "searches", "steps"}.
    """
    start = time.perf_counter()
    results: List[Dict[str, Any]] = []  # every result seen; the model refers to them by number
    searches: List[str] = []
    sources: List[str] = []
    transcript: List[str] = []
    opened: set = set()
    system = _SYSTEM.format(today=datetime.date.today().strftime("%d %B %Y"))

    async def run_search(q: str, kind: str) -> str:
        if q in searches:
            return f"{kind} {q!r}: already searched; try different words or open a result."
        if searches and notify:
            notify("web_search_more")  # "hmm, not there, let me check somewhere else"
        searches.append(q)
        found = [r for r in await _search(q, kind) if r.get("url") not in {x["url"] for x in results}]
        first = len(results) + 1
        results.extend(found)
        return f"{kind} {q!r}:\n" + (_listing(found, first) if found else "no results")

    async def run_open(target: str) -> str:
        urls = []
        for token in re.split(r"[,\s]+", target.strip()):
            if token.isdigit() and 1 <= int(token) <= len(results):
                urls.append(results[int(token) - 1]["url"])
            elif token.startswith("http"):
                urls.append(token)
        urls = list(dict.fromkeys(urls))[:3]
        if not urls:
            return f"open {target!r}: give result numbers from the lists above, or a URL."
        # The 4B model opened the same result again hoping for more; reading it twice gives the same text.
        again = [u for u in urls if u in opened]
        urls = [u for u in urls if u not in opened]
        if not urls:
            return f"open {target}: already read above. Open a different result, search again, or answer."
        opened.update(urls)
        texts = await asyncio.gather(*(_read(u, question, page_chars) for u in urls))
        out = []
        for url, text in zip(urls, texts):
            if text:
                sources.append(_domain(url))
            out.append(f"Page {_domain(url)}:\n{text or '(could not be read)'}")
        skipped = f" (already read: {', '.join(_domain(u) for u in again)})" if again else ""
        return f"open {target}{skipped}:\n" + "\n\n".join(out)

    transcript.append(await run_search(query, first_kind))
    answer, found, steps = "", False, 0
    while True:
        steps += 1
        last_step = steps >= max_steps or time.perf_counter() - start > time_budget
        actions = ["answer", "give_up"] if last_step else _ACTIONS
        prompt = (f"Question: {question}\n" + (f"Said before: {earlier}\n" if earlier else "")
                  + f"\nSteps so far:\n{_clip(transcript, 7000)}\n\n"
                  + ("No more steps: answer from what you read, or give up." if last_step else "Next step?"))
        try:
            step = await asyncio.to_thread(
                json_completion, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                _schema(actions), 0.2, 260)
        except Exception as e:
            print(f"   researcher failed: {e}")
            break
        action, arg = step.get("action"), str(step.get("input") or "").strip()
        print(f"🔬 step {steps}: {action} {arg[:120]!r}  ({step.get('thought', '')[:100]})")
        if action in ("answer", "give_up"):
            answer, found = arg, action == "answer" and bool(arg)
            break
        if action in ("search", "news"):
            transcript.append(await run_search(arg, "news" if action == "news" else "text"))
        elif action == "open":
            transcript.append(await run_open(arg))
    print(f"🔬 {'found' if found else 'not found'} in {time.perf_counter() - start:.1f}s, {steps} step(s): {answer[:160]!r}")
    return {"answer": answer, "found": found, "sources": list(dict.fromkeys(sources)) or [_domain(r["url"]) for r in results[:1]],
            "searches": searches, "steps": steps}


def format_brief(result: Dict[str, Any], question: str) -> str:
    """
    What Metis gets to see: only the researched answer, or that nothing was found. Given the pages too, the
    9B model mixed in details from other results (an older patch, a different site) and built "fun facts" on them.
    """
    if result["found"]:
        return f"You looked this up ({', '.join(result['sources'])}). What you found: {result['answer']}"
    return ("You looked this up but couldn't find a clear answer"
            + (f" (closest: {result['answer']})" if result["answer"] else "") + ". Say so; don't guess.")
