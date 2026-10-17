"""
Project Metis backend: chat pipeline API, dashboard feeds, and the Discord bot.

Startup order matters: the knowledge base is indexed *before* the bot connects,
so the first Discord message never hits a half-initialised backend.
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_config
from backend.routers.chat import router as chat_router
from backend.routers.discord import router as discord_router
from backend.routers.memory import router as memory_router

# Windows consoles/redirects default to cp1252, which can't print the emoji in our logs.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _banner(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


async def _startup() -> None:
    config = get_config()

    from backend.chat_log import migrate_previous_session
    moved = await asyncio.to_thread(migrate_previous_session)
    if moved:
        print(f"Moved {moved} message(s) from the previous session into temp memory")

    _banner("Checking LLM service...")
    from backend.llama_engine import get_llm_service_url, get_model
    if await asyncio.to_thread(get_model):
        print(f"✓ LLM service ready at {get_llm_service_url()}")
    else:
        print(f"⚠ LLM service not reachable at {get_llm_service_url()}")
        print("  Start it with: cd backend/llm_service && npm install && npm start")

    if config.get("rag", {}).get("enabled", False):
        _banner("Indexing knowledge base (new / changed / deleted files)...")
        try:
            from backend.rag_engine import get_collection, index_all_folders
            start = time.perf_counter()
            results = await asyncio.to_thread(index_all_folders, False)
            total = (await asyncio.to_thread(get_collection)).count()
            print(f"✓ {sum(results.values())} chunk(s) updated, {total} in index ({time.perf_counter() - start:.1f}s)")
        except Exception as e:
            print(f"✗ RAG indexing failed: {e}")

    # Summarize leftovers from last session in the background if they're over the limit.
    from backend.memory_manager import _safe, _worker, check_and_summarize_temp_memory
    _worker.submit(_safe, check_and_summarize_temp_memory)

    from backend import discord_bot
    discord_bot.start_in_background()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _startup()
    yield
    from backend import discord_bot
    discord_bot.stop()


app = FastAPI(title="Project Metis Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_config().get("backend", {}).get("cors_origins") or [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(discord_router)


@app.get("/", tags=["root"])
async def read_root():
    return {"message": "Project Metis backend is running"}


@app.get("/health", tags=["health"])
async def health():
    return {"status": "ok"}


_llm_status = {"ready": False, "checked": 0.0}


@app.get("/status", tags=["health"])
async def status():
    """Overall status for the dashboard header."""
    config = get_config()
    if time.monotonic() - _llm_status["checked"] > 5:
        from backend.llama_engine import get_model
        _llm_status["ready"] = await asyncio.to_thread(get_model)
        _llm_status["checked"] = time.monotonic()

    from backend import discord_bot
    bot = discord_bot.status()
    return {
        "llm_ready": _llm_status["ready"],
        "discord_connected": bot["connected"],
        "bot_user": bot["user"],
        "personality": config.get("personality"),
        "rag_enabled": bool(config.get("rag", {}).get("enabled", False)),
        "tools_enabled": bool(config.get("tools", {}).get("enabled", False)),
        "voice_channels": bot["voice_channels"],
    }


if __name__ == "__main__":
    import uvicorn

    backend_cfg = get_config()["backend"]
    uvicorn.run("backend.main:app", host=backend_cfg.get("host", "127.0.0.1"), port=int(backend_cfg.get("port", 8000)))
