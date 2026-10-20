"""
Chat pipeline: (tools | legacy RAG) -> prompt assembly -> streamed LLM reply.

Also exposes the "what did the bot just do" state the dashboard shows:
the last retrieval, the last tool execution and the exact prompt sent to the LLM.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.config import get_config
from backend.context_manager import trim_messages_to_context
from backend.llama_engine import astream_chat
from backend.prompt_loader import resolve_prompt

router = APIRouter(prefix="", tags=["chat"])


# ---------------------------------------------------------------------------
# Dashboard state
# ---------------------------------------------------------------------------
class _DashboardState:
    def __init__(self) -> None:
        self.rag: Dict[str, Any] = {}
        self.tools: Dict[str, Any] = {}
        self.context: List[Dict[str, Any]] = []
        self.version = 0

    def update(self, *, rag: Optional[Dict[str, Any]] = None, tools: Optional[Dict[str, Any]] = None) -> None:
        if rag is not None:
            self.rag = rag
        if tools is not None:
            self.tools = tools
        self.version += 1


state = _DashboardState()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Participant(BaseModel):
    id: str
    name: str


class ChatRequest(BaseModel):
    messages: Optional[List[Message]] = Field(default=None, description="OpenAI-style chat messages")
    prompt: Optional[str] = Field(default=None, description="Simple prompt; used if messages not provided")
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    use_tools: Optional[bool] = Field(default=None, description="Override tools.enabled from config")
    max_tools: Optional[int] = Field(default=None, description="Maximum tools to select")
    voice_mode: bool = Field(default=False, description="Favour latency over recall (voice chat)")
    participants: List[Participant] = Field(
        default_factory=list, description="Recent speakers; what Metis knows about them is added to the prompt"
    )
    present: List[str] = Field(default_factory=list, description="People currently in the voice channel")
    # Legacy RAG options (used when the tool system is disabled)
    rag_query_count: Optional[int] = None
    rag_system_prompt: Optional[str] = None
    rag_user_prompt: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def _base_messages(req: ChatRequest) -> List[Dict[str, str]]:
    if not req.messages and not req.prompt:
        raise HTTPException(status_code=400, detail="Provide `messages` or `prompt`.")
    if req.messages:
        messages = [m.model_dump() for m in req.messages]
    else:
        messages = [{"role": "user", "content": req.prompt or ""}]
    if not any(m["role"] == "system" for m in messages):
        prompt_ref = get_config().get("chat", {}).get("system_prompt", "@text_chat")
        messages.insert(0, {"role": "system", "content": resolve_prompt(prompt_ref)})
    if messages[-1]["role"] != "user":
        raise HTTPException(status_code=400, detail="The last message must be from the user.")
    return messages


def _conversation_text(messages: List[Dict[str, str]], count: int) -> str:
    """
    What people said in the last `count` messages before the latest one (for query generation).
    Metis's replies are left out: when she makes something up, queries built from it search for it.
    """
    if count <= 0:
        return ""
    history = [m for m in messages[:-1] if m["role"] in ("user", "assistant")][-count:]
    return "\n".join(m["content"][:300] for m in history if m["role"] == "user")


async def _run_tools(messages: List[Dict[str, str]], req: ChatRequest, timings: Dict[str, float]) -> str:
    from backend.tool_manager import decide_tools, execute_tools, format_tool_results_for_context, immediate_calls

    tools_cfg = get_config().get("tools", {})
    last_user = messages[-1]["content"]
    extra = {
        "conversation_context": _conversation_text(messages, int(tools_cfg.get("context_messages", 3))),
        "voice_mode": req.voice_mode,
        "message": last_user,  # for the log: which message a query was made from
    }

    # Calls that need no decision (knowledge search, links) run while the model decides about the web.
    t0 = time.perf_counter()
    immediate = immediate_calls(last_user)
    immediate_task = asyncio.create_task(execute_tools(immediate, extra)) if immediate else None
    decided = await decide_tools(messages, last_user, max_tools=req.max_tools or int(tools_cfg.get("max_tools", 3)))
    timings["tool_selection"] = round(time.perf_counter() - t0, 3)
    calls = immediate + decided
    print(f"Tool selection: {[c.tool_name for c in calls] or 'none'}")

    if not calls:
        state.update(
            tools={"tool_calls": [], "results": [], "message": "No tools needed", "timings": dict(timings)},
            rag={"queries": [], "original_query": last_user, "results": [], "message": "No tools needed"},
        )
        return ""

    t1 = time.perf_counter()
    decided_results = await execute_tools(decided, extra) if decided else []
    results = (await immediate_task if immediate_task else []) + decided_results
    timings["tool_execution"] = round(time.perf_counter() - t1, 3)

    rag_view: Optional[Dict[str, Any]] = None
    for r in results:
        if r.tool_name == "knowledge_search" and r.success and r.raw_response:
            from backend.rag_engine import results_for_ui
            rag_view = results_for_ui(r.raw_response["search"], last_user, tool_mode=True)
    state.update(
        tools={
            "tool_calls": [{"tool": c.tool_name, "arguments": c.arguments} for c in calls],
            "results": [
                {"tool": r.tool_name, "success": r.success, "execution_time": round(r.execution_time, 3), "error": r.error}
                for r in results
            ],
            "timings": dict(timings),
        },
        rag=rag_view,
    )
    return format_tool_results_for_context(results)


def _run_legacy_rag(messages: List[Dict[str, str]], req: ChatRequest, timings: Dict[str, float]) -> str:
    from backend.rag_engine import format_context_for_prompt, generate_rag_queries, results_for_ui, search

    last_user = messages[-1]["content"]
    t0 = time.perf_counter()
    queries = generate_rag_queries(
        messages, last_user,
        num_queries=req.rag_query_count,
        system_prompt_override=req.rag_system_prompt,
        user_prompt_override=req.rag_user_prompt,
    )
    timings["query_generation"] = round(time.perf_counter() - t0, 3)
    print(f"📚 Knowledge queries: {queries}\n   made from: {last_user!r}")

    t0 = time.perf_counter()
    found = search(queries)
    timings["retrieval"] = round(time.perf_counter() - t0, 3)

    state.update(rag=results_for_ui(found, last_user))
    return format_context_for_prompt(found["accepted"])


async def gather_context(messages: List[Dict[str, str]], req: ChatRequest) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """Run tools (or legacy RAG). Returns (context_text, tool_timings, rag_timings)."""
    config = get_config()
    tool_timings: Dict[str, float] = {}
    rag_timings: Dict[str, float] = {}
    use_tools = req.use_tools if req.use_tools is not None else config.get("tools", {}).get("enabled", False)

    if use_tools:
        try:
            return await _run_tools(messages, req, tool_timings), tool_timings, rag_timings
        except Exception as e:
            import traceback
            print(f"Tool system error, falling back to plain RAG: {e}")
            traceback.print_exc()

    if config.get("rag", {}).get("enabled", False):
        try:
            text = await asyncio.to_thread(_run_legacy_rag, messages, req, rag_timings)
            return text, tool_timings, rag_timings
        except Exception as e:
            print(f"RAG error: {e}")
    return "", tool_timings, rag_timings


def people_context(req: ChatRequest) -> str:
    """What Metis knows about the people in this conversation, plus who is in the voice channel."""
    from backend.people import get_people

    parts = []
    if req.participants:
        parts.append(
            "Each message starts with the speaker's name in brackets; different names are different people. "
            "Keep track of who said what. You are Metis: reply only as yourself, never write lines for anyone "
            "else, and don't start your reply with a name in brackets."
            # In voice an @ is read out (and she started pinging herself: "@Metis it's 13 degrees").
            + ("" if req.voice_mode else " To ping someone, write @ and their name.")
        )
    if req.present:
        parts.append("In the voice channel right now: " + ", ".join(req.present))
    profiles = get_people().profile_block([p.id for p in req.participants])
    if profiles:
        parts.append(
            "What you remember about the people in this conversation (use it naturally, like a friend would; "
            "don't recite it). These are your friends on Discord, not celebrities or historical figures with the "
            "same name: when asked about them, use only this, and say so if you don't know something:\n" + profiles
        )
    return "\n\n".join(parts)


def reply_guidance(messages: List[Dict[str, str]], voice: bool, has_context: bool, searched_web: bool = False) -> str:
    """
    A short reminder placed right before the latest message. Small models follow the most recent
    instructions far more than a long persona prompt at the top, so without this the personality
    takes over and questions go unanswered. It works with any persona and doesn't change it.
    """
    match = re.match(r"^\[([^\]]+)\]", messages[-1]["content"])
    who = match.group(1) if match else "the person"
    rules = [
        f"Reply to {who}'s last message. If they asked a question or asked you to do something, "
        "answer or do exactly that in your first sentence.",
        "Your personality is flavour, never a replacement for answering.",
        ("Facts, dates and numbers must come from the information above; if it isn't there, say you don't know "
         "instead of guessing." if has_context else
         "Never invent facts, dates or numbers; if you don't know, say so."),
    ]
    if searched_web:
        # Without this a small model replied "I could search that if you want" with the results right there.
        rules.append("You can search the web, and you already did for this message: the results are above. "
                     "Say what they found (the date, number, name or event). Don't offer to search, and don't "
                     "say you can't look things up. If the results don't answer it, say you couldn't find it.")
    if voice:
        # Personas that roleplay replied with only "*checks phone, rolls her eyes*", which the voice
        # can't say, and the mean one wrapped every reply in quotes.
        rules.append("This is spoken aloud: one or two short sentences of what you say, nothing else. No "
                     "*actions* or descriptions, no quotation marks around your reply, no emoji, no lists.")
    return "How to reply:\n- " + "\n- ".join(rules)


def build_prompt(messages: List[Dict[str, str]], context_text: str, people_text: str = "",
                 guidance: str = "") -> List[Dict[str, str]]:
    """
    Insert people memory, gathered information and reply guidance as a system message right
    before the latest user message. Keeping the system prompt and earlier history
    byte-identical between turns lets the LLM service reuse its KV cache.
    """
    if not context_text and not people_text and not guidance:
        return list(messages)
    now = datetime.datetime.now().strftime("%A, %d %B %Y, %H:%M")
    sections = [f"Current date and time: {now}"]
    if people_text:
        sections.append(people_text)
    if context_text:
        sections.append(
            "Information retrieved for the next reply (from notes, memories, or the web). "
            "Use it when it is relevant; do not mention that you looked it up.\n\n" + context_text
        )
    if guidance:
        sections.append(guidance)
    return messages[:-1] + [{"role": "system", "content": "\n\n".join(sections)}, messages[-1]]


def _sampling(req: ChatRequest) -> Dict[str, Any]:
    chat_cfg = get_config().get("chat", {})
    return {
        "temperature": req.temperature if req.temperature is not None else chat_cfg.get("temperature", 0.7),
        "top_p": req.top_p if req.top_p is not None else chat_cfg.get("top_p", 0.95),
        "max_tokens": req.max_tokens if req.max_tokens is not None else chat_cfg.get("max_tokens", 512),
    }


async def _prepare(req: ChatRequest) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    messages = _base_messages(req)
    context_text, tool_timings, rag_timings = await gather_context(messages, req)
    guidance = reply_guidance(messages, req.voice_mode, bool(context_text), "[web_search]" in context_text)
    final = trim_messages_to_context(build_prompt(messages, context_text, people_context(req), guidance))
    stamp = datetime.datetime.now().isoformat()
    state.context = [{**m, "timestamp": stamp} for m in final]
    return final, {"tool_timings": tool_timings, "rag_timings": rag_timings}


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """
    Stream a reply as NDJSON: {"delta": "..."} lines, then a final
    {"done": true, "tokens_per_second", "llm_generation_time", "ttft", "tool_timings", "rag_timings"}.
    """
    final, meta = await _prepare(req)
    sampling = _sampling(req)

    async def generate():
        stats: Dict[str, Any] = {}
        start = time.perf_counter()
        first_token: Optional[float] = None
        reply = ""
        try:
            async for delta in astream_chat(final, stats=stats, **sampling):
                if first_token is None:
                    first_token = time.perf_counter() - start
                reply += delta
                yield json.dumps({"delta": delta}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e), "done": True}) + "\n"
            return
        state.context.append({"role": "assistant", "content": reply, "timestamp": datetime.datetime.now().isoformat()})
        state.update()
        yield json.dumps({
            "done": True,
            "tokens_per_second": stats.get("tokens_per_second", 0),
            "llm_generation_time": round(time.perf_counter() - start, 3),
            "ttft": round(first_token or 0, 3),
            **meta,
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Non-streaming variant of /chat/stream."""
    final, _ = await _prepare(req)
    parts = [delta async for delta in astream_chat(final, **_sampling(req))]
    return ChatResponse(reply="".join(parts).strip())


# ---------------------------------------------------------------------------
# Knowledge base management
# ---------------------------------------------------------------------------
@router.post("/rag/reindex")
async def reindex_knowledge_base(clear_existing: bool = False):
    """Re-index the configured folders. clear_existing=true rebuilds everything from scratch."""
    from backend.rag_engine import index_all_folders

    try:
        results = await asyncio.to_thread(index_all_folders, clear_existing)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    return {"status": "success", "total_chunks": sum(results.values()), "folders": results}


@router.get("/rag/stats")
async def get_rag_stats():
    from backend.rag_engine import get_collection, load_file_metadata

    rag_cfg = get_config().get("rag", {})
    try:
        collection = await asyncio.to_thread(get_collection)
        total_chunks = collection.count()
        total_files = len(load_file_metadata())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")
    return {
        "total_chunks": total_chunks,
        "total_files": total_files,
        "collection_name": rag_cfg.get("collection_name", "metis_knowledge"),
        "enabled": rag_cfg.get("enabled", False),
    }


@router.post("/rag/clear")
async def clear_rag_database():
    from backend.rag_engine import clear_collection

    try:
        deleted = await asyncio.to_thread(clear_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {e}")
    return {"status": "success", "deleted_count": deleted}


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------
@router.get("/rag/last-retrieval")
async def get_last_rag_retrieval():
    return {"results": state.rag}


@router.get("/tools/last-execution")
async def get_last_tool_execution():
    return {"results": state.tools}


@router.get("/tools/available")
async def get_available_tools():
    from backend.tool_manager import get_tool_registry

    tools = get_tool_registry().get_all_enabled()
    return {
        "tools": [
            {"name": t.definition.name, "description": t.definition.description, "schema": t.definition.to_json_schema()}
            for t in tools.values()
        ]
    }


@router.get("/rag/stream")
async def stream_rag_updates(request: Request):
    """SSE feed of {"results": <last retrieval>, "tools": <last tool execution>} whenever either changes."""
    async def events():
        sent_version = -1
        idle = 0.0
        while not await request.is_disconnected():
            if state.version != sent_version:
                sent_version = state.version
                idle = 0.0
                yield f"data: {json.dumps({'results': state.rag, 'tools': state.tools})}\n\n"
            elif idle >= 15:
                idle = 0.0
                yield ": heartbeat\n\n"
            await asyncio.sleep(0.5)
            idle += 0.5

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/chat/context")
async def get_chat_context():
    """The exact prompt most recently sent to the LLM (plus the reply)."""
    return {"messages": state.context}
