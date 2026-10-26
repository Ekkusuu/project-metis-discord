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
- answer: input = the facts only, as short notes: the exact names, numbers and dates, and which site says so \
(e.g. "u.gg, patch 26.21: Thresh 52.1% win rate, Braum 52.0%"). No "based on the search results".
- give_up: the answer can't be found. input = the closest thing you found, or empty
Rules:
- As soon as a page you read answers the question, answer. Don't open more sites to double-check.
- If the question is vague (e.g. "best champion" without a role), answer with the main options from the page \
(the top pick per role) instead of searching more.
- Snippets are short and often outdated. Open pages before answering with numbers, prices, rankings, \
dates or lists, unless a snippet states the answer plainly.
- If a page could not be read, open a different result; don't answer numbers or rankings from snippets.
- If the pages don't have it, search again with different words or open other results. After 2-3 tries \
that find nothing, give up with the closest thing you found.
- Don't open a result you already read; it gives the same text.
- If they asked about a particular website, answer with what that website says.
- Use only what you read. Never guess or fill in from memory.{preferred}
Write `thought` as one short sentence about what to do next."""

_ACTIONS = ["search", "news", "open", "answer", "give_up"]


def _schema(actions: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        # Lengths capped: with no limit a long "thought" ran past the token limit, the JSON was cut off
        # mid-string and the whole lookup failed ("incomplete JSON").
        "properties": {"thought": {"type": "string", "maxLength": 200}, "action": {"enum": actions},
                       "input": {"type": "string", "maxLength": 600}},
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


# Bing and Google are disabled in ddgs 9.9 (asking for them silently falls back to the slow mix); DuckDuckGo,
# Brave and Mojeek returned nothing. Yandex (~0.7 s) and Yahoo (~1 s) worked, so both run at once.
_ENGINES = ("yandex", "yahoo")
_AD = re.compile(r"bing\.com/aclick|/aclk\?|doubleclick\.net", re.IGNORECASE)


async def _search_once(query: str, kind: str, count: int) -> List[Dict[str, Any]]:
    """The first engine with results wins (ddgs's "auto" mix for news, or if those fail)."""
    from backend.config import get_config
    from backend.tools.web_search_tool import WebSearchTool

    def run(backend: str) -> List[Dict[str, Any]]:
        found = WebSearchTool()._search_sync(query, count, kind, backend)
        return [r for r in found if r.get("url") and not _AD.search(r["url"])]

    engines = (get_config().get("tools", {}).get("web_search") or {}).get("engines") or _ENGINES
    t0 = time.perf_counter()

    async def attempt(engine: str, delay: float = 0.0) -> tuple:
        await asyncio.sleep(delay)
        return engine, await asyncio.wait_for(asyncio.to_thread(run, engine), timeout=6)

    # The mix joins the race if the fast engines haven't answered in 2.5 s. Started only after they failed,
    # a search took 13 s when Yandex and Yahoo were slow.
    fast = list(engines) if kind == "text" else []
    tasks = [asyncio.create_task(attempt(e)) for e in fast]
    tasks.append(asyncio.create_task(attempt("auto", 2.5 if fast else 0.0)))
    try:
        for next_done in asyncio.as_completed(tasks, timeout=9):
            try:
                engine, found = await next_done
            except Exception:
                continue
            if found:
                where = "the mix" if engine == "auto" else engine
                print(f"   {kind} search {query!r}: {len(found)} from {where} in {time.perf_counter() - t0:.1f}s")
                return found
    finally:
        for t in tasks:
            t.cancel()
    print(f"   {kind} search failed for {query!r} after {time.perf_counter() - t0:.1f}s")
    return []


async def _search(query: str, kind: str = "text", count: int = 6) -> List[Dict[str, Any]]:
    """
    Web or news results. A site: search with no results is retried with the site's name as a plain keyword:
    a misheard name ("site:u.g") still finds the right site that way.
    """
    site = site_in(query)
    if not site:
        return await _search_once(query, kind, count)
    # Both at once: waiting for a misheard site: search to fail first cost 3 s.
    exact, loose = await asyncio.gather(_search_once(query, kind, count),
                                        _search_once(re.sub(r"\bsite:\S+", site, query).strip(), kind, count))
    if exact:
        return exact
    # The misheard site is probably the closest domain among the plain results ("u.g" -> u.gg): put its pages first.
    from difflib import SequenceMatcher

    def closeness(r: Dict[str, Any]) -> float:
        return SequenceMatcher(None, site, _domain(r["url"])).ratio()

    return sorted(loose, key=lambda r: -closeness(r) if closeness(r) >= 0.75 else 0)


async def _read(url: str, question: str, max_chars: int) -> str:
    from backend.tools.web_page_tool import read_page

    t0 = time.perf_counter()
    try:
        text = await asyncio.wait_for(asyncio.to_thread(read_page, url, question, max_chars), timeout=8)
        print(f"   read {_domain(url)} in {time.perf_counter() - t0:.1f}s")
        return text
    except Exception as e:
        print(f"   couldn't read {url} ({time.perf_counter() - t0:.1f}s): {e!r}")
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
    # Each older part is shortened at most once. (Shortening to 400 + a marker left it over 400, so the same
    # part was picked again forever: the loop froze the whole backend mid-search.)
    parts = list(transcript)
    for i in range(len(parts) - 1):
        if sum(map(len, parts)) <= budget:
            break
        if len(parts[i]) > 400:
            parts[i] = parts[i][:400] + " …(shortened)"
    return "\n\n".join(parts)


async def research(question: str, query: str, earlier: str = "", *, max_steps: int = 5, page_chars: int = 1500,
                   time_budget: float = 25.0, first_kind: str = "text", auto_open: int = 0,
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
    from backend.tool_manager import preferred_sites_text

    system = _SYSTEM.format(today=datetime.date.today().strftime("%d %B %Y"), preferred=preferred_sites_text(
        "\n- The user prefers these websites for these topics; search and read them first:"))

    async def run_search(q: str, kind: str) -> str:
        # Adding a word to the last search ("... Patch 26.20" then "... Patch 26.20 tier list") gave the same
        # results twice and used up the steps.
        words = set(re.findall(r"\w+", q.lower()))
        if any(len(words & set(re.findall(r"\w+", s.lower()))) >= 0.75 * len(words) for s in searches):
            return (f"{kind} {q!r}: almost the same as an earlier search, it gives the same results. "
                    "Open a result you haven't read, search for something different, or answer.")
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
        # Video and social pages have no text to read (the model opened TikTok and YouTube anyway).
        urls = [u for u in dict.fromkeys(urls) if not _domain(u).endswith(_UNREADABLE)][:3]
        if not urls:
            return f"open {target!r}: give result numbers from the lists above, or a URL."
        # The 4B model opened the same result again hoping for more; reading it twice gives the same text.
        again = [u for u in urls if u in opened]
        urls = [u for u in urls if u not in opened]
        if not urls:
            return f"open {target}: already read above. Open a different result, search again, or answer."
        opened.update(urls)
        if notify:
            notify("web_read")  # "okay, reading a couple of pages"
        texts = await asyncio.gather(*(_read(u, question, page_chars) for u in urls))
        out = []
        for url, text in zip(urls, texts):
            if text:
                sources.append(_domain(url))
            out.append(f"Page {_domain(url)}:\n{text or '(could not be read: the site blocks readers. Open a different result.)'}")
        skipped = f" (already read: {', '.join(_domain(u) for u in again)})" if again else ""
        return f"open {target}{skipped}:\n" + "\n\n".join(out)

    transcript.append(await run_search(query, first_kind))
    if auto_open and results:
        # Read the top results right away, like coding agents' web tools: having the model pick them cost a
        # step (1-2 s) every time, and it often opened one it had already read. The named site goes first.
        site = site_in(query)
        readable = [i for i, r in enumerate(results, 1) if not _domain(r["url"]).endswith(_UNREADABLE)]
        readable.sort(key=lambda i: not (site and _domain(results[i - 1]["url"]).endswith(site.split("/")[0])))
        if readable:
            transcript.append(await run_open(",".join(str(i) for i in readable[:auto_open])))
    answer, found, steps = "", False, 0
    while True:
        steps += 1
        left = time_budget - (time.perf_counter() - start)
        last_step = steps >= max_steps or left < 4
        actions = ["answer", "give_up"] if last_step else _ACTIONS
        # The 4B model kept re-opening pages it had read and searching on with no sense of the budget.
        read = [str(i) for i, r in enumerate(results, 1) if r["url"] in opened]
        status = (f"Step {steps} of {max_steps}. Results already read: {', '.join(read) or 'none'}. "
                  + ("No more steps: answer from what you read, or give up." if last_step else "Next step?"))
        prompt = (f"Question: {question}\n" + (f"Said before: {earlier}\n" if earlier else "")
                  + f"\nSteps so far:\n{_clip(transcript, 7000)}\n\n{status}")
        t = time.perf_counter()
        try:
            # A step normally takes 1-2 s. With no limit, one call never returned and Metis went silent.
            step = await asyncio.to_thread(
                json_completion, [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                _schema(actions), 0.2, 320, max(6.0, min(20.0, left + 6)))
        except Exception as e:
            print(f"   researcher step {steps} failed after {time.perf_counter() - t:.1f}s: {e}")
            break
        action, arg = step.get("action"), str(step.get("input") or "").strip()
        print(f"🔬 step {steps} ({time.perf_counter() - t:.1f}s): {action} {arg[:120]!r}  ({step.get('thought', '')[:100]})")
        if action in ("answer", "give_up"):
            answer, found = arg, action == "answer" and bool(arg)
            if found and not adds_information(arg, question):
                # "Kazik win rate vs Kindred on Yudor.gg" came back as the answer: the question with no facts.
                # Passed on as found, Metis made up "a 32% win rate".
                print("🔬 the answer only repeats the question; treating it as not found")
                found = False
            break
        if action in ("search", "news"):
            transcript.append(await run_search(arg, "news" if action == "news" else "text"))
        elif action == "open":
            transcript.append(await run_open(arg))
    print(f"🔬 {'found' if found else 'not found'} in {time.perf_counter() - start:.1f}s, {steps} step(s): {answer[:160]!r}")
    return {"answer": answer, "found": found, "sources": list(dict.fromkeys(sources)) or [_domain(r["url"]) for r in results[:1]],
            "searches": searches, "steps": steps}


def adds_information(answer: str, question: str) -> bool:
    """False if the answer is just the question's own words (no number, no new name or fact)."""
    words = [w for w in re.findall(r"[a-z0-9]+", answer.lower()) if len(w) >= 3 or w.isdigit()]
    if not words or any(c.isdigit() for c in answer):
        return bool(words)
    asked = {w[:5] for w in re.findall(r"[a-z0-9]+", question.lower())}
    return sum(w[:5] in asked for w in words) / len(words) < 0.6


def format_brief(result: Dict[str, Any], question: str) -> str:
    """
    What Metis gets to see: only the researched answer, or that nothing was found. Given the pages too, the
    9B model mixed in details from other results (an older patch, a different site) and built "fun facts" on them.
    """
    if result["found"]:
        return f"You looked this up ({', '.join(result['sources'])}). What you found: {result['answer']}"
    return ("You looked this up but couldn't find a clear answer"
            + (f" (closest: {result['answer']})" if result["answer"] else "") + ". Say so; don't guess.")
