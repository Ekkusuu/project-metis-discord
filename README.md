# Project Metis

Metis is a fully local AI companion for Discord. She chats in text channels, joins voice
channels to talk out loud, remembers what people tell her, can look things up in your notes
or on the web, and comes with a live dashboard that shows exactly what she's thinking with.

Everything runs on your own machine: a GGUF model through `node-llama-cpp`, local embeddings
and reranking for retrieval, NVIDIA Parakeet for speech-to-text and Kokoro for text-to-speech.

## How it fits together

```
 Discord ──► discord_bot.py ──HTTP──► FastAPI backend (backend/main.py) ──HTTP──► LLM service (Node)
   ▲  text: judgment → reply           │  /chat/stream: tools → prompt → stream     server.js
   │  voice: STT → judgment → TTS      │  ├─ knowledge_search ─► ChromaDB + reranker  node-llama-cpp
   └───────────── replies ◄────────────┘  ├─ web_search ───────► DuckDuckGo           (GPU)
                                          ├─ memory: temp → summaries → long_term/*.md
                                          └─ SSE feeds ─► dashboard (frontend/, Vite + React)
```

| Piece | What it does |
|---|---|
| `backend/llm_service/server.js` | Loads the model once, serves `/chat/stream`, `/chat/completion` and a grammar-constrained `/judgment`. Uses the model's own prompt format, queues requests, and stops generating when a client disconnects. |
| `backend/main.py` | FastAPI app. On startup it indexes the knowledge base, then starts the Discord bot in a background thread. |
| `backend/routers/chat.py` | The reply pipeline: pick tools → run them concurrently → insert results before the latest message → stream the reply. |
| `backend/discord_bot.py` | Text + voice front-end, commands, per-channel conversations that compact themselves into memory. |
| `backend/rag_engine.py` | Indexing (incremental, size + mtime) and retrieval (vector search → distance filter → cross-encoder rerank). |
| `backend/memory_manager.py` | Messages that leave the context window go to `memory/temp_memory.txt`; past a token limit they're summarized into `memory/long_term/` and indexed. |
| `frontend/` | Read-only dashboard: live Discord feed, retrieved knowledge, tool calls and the exact prompt sent to the model. |

## Setup (Windows)

Requirements: Python 3.10+, Node.js 18+, an NVIDIA GPU is strongly recommended.

1. **Python environment**
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```
2. **Models** (paths are configured in `config.yaml`)
   - Chat model: any instruct GGUF in `model/` (e.g. a Llama 3.1 8B Q4 build) → `model.path`
   - Embeddings: [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) in `rag-models/embeddinggemma-300m`
   - Reranker: [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) in `rag-models/bge-reranker-base`
   - Voice (optional): Parakeet downloads itself on first `>join`; Kokoro needs
     [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) installed.
3. **Discord bot**: create an application at <https://discord.com/developers/applications>,
   add a bot, enable the **Message Content** intent, invite it with *Send Messages*,
   *Read Message History*, *Connect* and *Speak* permissions. Then:
   ```cmd
   copy .env.example .env
   ```
   and paste the token into `DISCORD_BOT_TOKEN`.
4. **Run everything**: `start.bat`. It installs Node dependencies on first run, waits for the
   model to load, then starts the backend (and bot) and the dashboard at <http://localhost:5173>.

Running the pieces by hand, each in its own terminal:

```cmd
cd backend\llm_service && npm start
venv\Scripts\python -m uvicorn backend.main:app --port 8000
cd frontend && npm run dev
```

(LLM service on :3000, backend + bot on :8000, dashboard on :5173.)

Set `discord.enabled: false` to run the backend and dashboard without the bot.

## Using Metis on Discord

Metis reads every message in channels she can see and decides whether to join in.
She always answers when you **@mention** her, **reply** to her, say **"Metis"**, or **DM** her.

| Command | |
|---|---|
| `>join` | Join your voice channel and start listening. Text replies pause while she's in voice. |
| `>leave` | Leave voice (she also leaves by herself when everyone else does). |
| `>reset` | Start the channel's conversation fresh; the old one is saved to memory first. |
| `>status` | Personality, context usage, tools and voice state. |
| `>personality [name]` | List personalities or switch (needs *Manage Server*; lasts until restart). |
| `>help` | All commands. `!` works as a prefix too. |

## Configuration highlights

Everything lives in `config.yaml` (prompts in `system_prompts.yaml`, see
[SYSTEM_PROMPTS_GUIDE.md](SYSTEM_PROMPTS_GUIDE.md)).

- `personality` — picks `text_chat.<name>` / `voice_chat.<name>` prompts.
- `model.chat_wrapper` — `auto` reads the prompt format from the GGUF. Only override it for a model with broken metadata.
- `llm_service.aux_context_size` — a second small context for judgment/tool calls so the chat cache survives between turns. Set `0` if VRAM is tight.
- `chat.judgment` / `voice.judgment` — turn the "should I reply?" step on or off.
- `rag.*` — indexed folders, chunking, distance/rerank thresholds. `reranker_top_k` is how many chunks reach the prompt.
- `tools.*` — enable tools individually; `knowledge_search.expand_queries` trades one extra LLM call for better recall in text chat.
- `memory.*` — when temp memory is summarized and how large long-term files grow.
- `voice.*` — voice, pause length before replying, input gain, noise floor.

Adding a tool is a single file — see [TOOLS_GUIDE.md](TOOLS_GUIDE.md).

## Dashboard & API

The dashboard is read-only: all conversation happens on Discord. Useful endpoints
(full list at <http://localhost:8000/docs>):

- `GET /status` — LLM / Discord / voice status
- `GET /discord/stream` — SSE feed of messages (resumable via `Last-Event-ID`)
- `GET /rag/stream` — SSE feed of the last retrieval and tool calls
- `GET /chat/context` — the exact prompt last sent to the model
- `POST /rag/reindex?clear_existing=false` — pick up changed notes without restarting
- `POST /memory/summarize` — turn temp memory into long-term memory now

## Tests

```cmd
venv\Scripts\python -m pytest
```

The suite covers the pure logic (text processing, context trimming, tool-call parsing, the SSE
feed, retrieval merging) and runs offline in a few seconds.
