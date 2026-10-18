# Project Metis

Metis is a fully local AI companion for Discord. She chats in text channels, joins voice
channels to talk out loud, tells people apart, **learns about everyone she talks to** and brings
it up naturally later, can look things up in your notes, search the web and read pages people
share, and comes with a live dashboard that shows exactly what she's thinking with.

Everything runs on your own machine: a GGUF model through `node-llama-cpp`, local embeddings
and reranking for retrieval, NVIDIA Parakeet for speech-to-text and Kokoro-82M for text-to-speech.
The default chat model is **Qwen 3.5 2B** (~2.9 GB VRAM including its contexts, ~130 tokens/s on an RTX 4070 Ti),
so the whole stack fits comfortably next to games or other apps.

## How it fits together

```
 Discord ──► discord_bot.py ──HTTP──► FastAPI backend (backend/main.py) ──HTTP──► LLM service (Node)
   ▲  text: judgment → reply           │  /chat/stream: tools → prompt → stream     server.js
   │  voice: STT → judgment → TTS      │  ├─ knowledge_search ─► ChromaDB + reranker  node-llama-cpp
   └───────────── replies ◄────────────┘  ├─ web_search ───────► DuckDuckGo           (GPU)
                                          ├─ fetch_webpage ────► any public URL
                                          ├─ people memory: per-user facts, learned continuously
                                          └─ SSE feeds ─► dashboard (frontend/, Vite + React)
```

| Piece | What it does |
|---|---|
| `backend/llm_service/server.js` | Loads the model once, serves `/chat/stream`, `/chat/completion` and a grammar-constrained `/judgment`. Uses the model's own prompt format, queues requests, and stops generating when a client disconnects. |
| `backend/main.py` | FastAPI app. On startup it indexes the knowledge base, then starts the Discord bot in a background thread. |
| `backend/routers/chat.py` | The reply pipeline: pick tools → run them concurrently → insert results before the latest message → stream the reply. |
| `backend/discord_bot.py` | Text + voice front-end, commands, per-channel conversations that compact themselves into memory. |
| `backend/rag_engine.py` | Indexing (incremental, size + mtime) and retrieval (vector search → distance filter → cross-encoder rerank). |
| `backend/people.py` | One profile per Discord user id (names, aliases, last seen, learned facts). Profiles of the people in a conversation go straight into the prompt. |
| `backend/learning.py` | Every few messages, extracts lasting facts per speaker ("Sava \| works as a nurse") and files them under the right person. |
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
   - Chat model: `model/Qwen3.5-2B-Q6_K.gguf` from [unsloth/Qwen3.5-2B-GGUF](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF)
     (any instruct GGUF works → `model.path`; the prompt format is read from the file, and
     `model.thinking: false` keeps reasoning models like Qwen from thinking before every reply)
   - Embeddings: [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m) in `rag-models/embeddinggemma-300m`
   - Reranker: [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) in `rag-models/bge-reranker-base`
   - Voice (optional): Parakeet downloads itself on first `>join`; Kokoro-82M (`hexgrad/Kokoro-82M`) downloads
     on first use and runs ~6x real time even on CPU (`voice.tts_device: cpu` frees VRAM). It needs
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
| `>reset` | Start the channel's conversation fresh (she learns from it first). |
| `>status` | Personality, context usage, tools and voice state. |
| `>personality [name]` | List personalities or switch (needs *Manage Server*; lasts until restart). |
| `>help` | All commands. `!` works as a prefix too. |

## How Metis knows who's who

- Every message is attributed to a Discord **user id**, not just a name. In text the label is the
  server display name; in voice, discord-ext-voice-recv gives each speaker their own audio stream,
  so transcripts are attributed per person too.
- If two people share a display name, the transcript disambiguates them (`Sava (sava_x)`).
- The profiles of the people in the conversation (and, in voice, who is in the channel) are added
  to the prompt, so Metis can address the right person and bring up what she knows about them.
- When she writes `@Name`, it becomes a real Discord mention; in busy channels she replies to the
  message she's answering so it's clear who she's talking to.
- Everyone in the voice channel is added to the speech recogniser's name list automatically.

## How Metis learns

Every `memory.learn_every_messages` user messages (and on `>reset`, when old messages scroll out
of the context window, and at shutdown) the new part of the conversation is read by the model,
which writes down lasting facts per person: job, pets, plans, likes and dislikes, and so on.
Duplicates are filtered, facts are stored in `memory/people.json` and mirrored to
`memory/long_term/people/` so `knowledge_search` can find them even when that person isn't around.
You can review and delete anything in the dashboard's **People** tab.

## Configuration highlights

Everything lives in `config.yaml` (prompts in `system_prompts.yaml`, see
[SYSTEM_PROMPTS_GUIDE.md](SYSTEM_PROMPTS_GUIDE.md)).

- `personality` — picks `text_chat.<name>` / `voice_chat.<name>` prompts.
- `model.chat_wrapper` — `auto` reads the prompt format from the GGUF. Only override it for a model with broken metadata.
- `llm_service.aux_context_size` — a second small context for judgment/tool calls so the chat cache survives between turns. Set `0` if VRAM is tight.
- `chat.judgment` / `voice.judgment` — turn the "should I reply?" step on or off.
- `rag.*` — indexed folders, chunking, distance/rerank thresholds. `reranker_top_k` is how many chunks reach the prompt.
- `tools.*` — enable tools individually; `knowledge_search.expand_queries` trades one extra LLM call for better recall in text chat.
- `memory.learn_every_messages` — how often Metis learns from the conversation.
- `tools.web_search.read_top_result` / `tools.fetch_webpage` — reading full pages (private/local addresses are always refused).
- `voice.*` — voice, pause length before replying, input gain, noise floor.

Adding a tool is a single file — see [TOOLS_GUIDE.md](TOOLS_GUIDE.md).

## Performance

Measured on an RTX 4070 Ti with Qwen 3.5 2B (warm, median of the text pipeline: judgment →
tool decision + knowledge search → reply):

| | |
|---|---|
| Text reply, end to end | ~0.5 s (web search adds ~1.5 s) |
| First token after the request reaches the LLM | ~0.07 s |
| VRAM, text only (LLM service + embeddings) | ~3.5 GB |
| VRAM added by voice (Parakeet bf16 + Kokoro) | ~2 GB, loaded on first `>join` |
| Voice: speech-to-text / text-to-speech per sentence | ~0.08 s / ~0.1 s |

What keeps it there (all configurable in `config.yaml`):
- `model.batch_size: 256` — ~550 MB less VRAM than the default batch, no measurable slowdown.
- `rag.embedding_precision: auto` and `voice.stt_precision: auto` — bfloat16 on GPU, half the VRAM
  of fp32 with identical results.
- `tools.knowledge_search.expand_queries: false` — knowledge search runs on every message, so it
  uses the message itself instead of an extra LLM call; it runs in parallel with the web decision.
- Background learning yields to anything someone is waiting for (priority queue in the LLM service).
- NeMo/Kokoro are imported only when voice is first used, and both voice models are warmed up on
  load so the first speaker doesn't pay the GPU warm-up.
- `rag.reranker_device: cpu` (~86 ms per message) — set `cuda` for ~9 ms at the cost of ~560 MB VRAM.
- `voice.tts_device: cpu` frees Kokoro's VRAM (still ~6x faster than real time).

## Dashboard & API

The dashboard is read-only: all conversation happens on Discord. Useful endpoints
(full list at <http://localhost:8000/docs>):

- `GET /status` — LLM / Discord / voice status
- `GET /discord/stream` — SSE feed of messages (resumable via `Last-Event-ID`)
- `GET /rag/stream` — SSE feed of the last retrieval and tool calls
- `GET /chat/context` — the exact prompt last sent to the model
- `POST /rag/reindex?clear_existing=false` — pick up changed notes without restarting
- `GET /memory/people`, `DELETE /memory/people/{id}/facts/{fact}` — what Metis knows about people
- `POST /memory/learn` — learn from the conversation so far right now

## Tests

```cmd
venv\Scripts\python -m pytest
```

The suite covers the pure logic (text processing, context trimming, tool-call parsing, the SSE
feed, retrieval merging) and runs offline in a few seconds.
