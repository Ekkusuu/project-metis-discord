# Memory

This folder is Metis' memory. Its contents are git-ignored (they're about real people);
only this README and `.gitkeep` are tracked.

| File | What it is |
|---|---|
| `chat_history.jsonl` | Append-only log of the current session's Discord messages. On the next startup it is moved into `temp_memory.txt` and deleted. |
| `temp_memory.txt` | Raw conversation that has left the context window (or was `>reset`), waiting to be summarized. |
| `long_term/memory_*.md` | Facts the model extracted from temp memory, one bullet per fact, prefixed with `user personal information:` so retrieval can target them. |

## Flow

```
Discord conversation (per channel / voice session, kept by the bot)
        │  history outgrows the context window → oldest ~30% is cut once
        │  >reset → whole conversation
        ▼
temp_memory.txt ──(≥ memory.temp_memory_token_limit tokens)──► LLM fact extraction
                                                                    │
                                   long_term/memory_*.md ◄──────────┘
                                   (new file once memory.long_term_memory_token_limit is reached)
                                        │  indexed immediately if the folder is in rag.folders_to_index
                                        ▼
                                knowledge_search tool ─► injected into future replies
```

- All memory writes and summaries run on a single background worker, so replies never wait for them.
- Summarization prompts: `memory.summarization_system` / `memory.summarization_user` in `system_prompts.yaml`.
- Force a summary now: `POST http://localhost:8000/memory/summarize`. Check status: `GET /memory/status`.
- To make Metis forget something, edit or delete the relevant `long_term/*.md` file and run
  `POST /rag/reindex` (or restart).
