# Memory

This folder is Metis' memory. Its contents are git-ignored (they're about real people);
only this README and `.gitkeep` are tracked.

| Path | What it is |
|---|---|
| `memories.db` | SQLite: what she remembers about people. **Facts** (one lasting thing about one person, with the quote it came from) and **episodes** (short summaries of conversations that ended). Each has an importance (1-10), when it was last confirmed or recalled, and how often. |
| `people.json` | The person registry, one entry per Discord user id: current name, aliases, first/last seen, message count. Also general notes. (Facts used to live here; they were moved into `memories.db` the first time it loaded.) |
| `long_term/learned_notes.md` | General things Metis learned that aren't about one person (indexed by the knowledge base). |
| `long_term/memory_*.md` | Summaries written by an older memory system (still indexed). |
| `chat_history.jsonl` | Transcript of the current session; moved to `sessions/` on the next start. |
| `sessions/` | Past session transcripts, plus legacy files that were migrated once. |

## How it works

Like a person: she notices selectively, consolidates when a conversation is over, and only remembers what the
moment cues. Memories fade unless they come back.

```
Encoding (write)
  every 6 user messages / conversation ends  ─► per speaker: facts + quote + importance (backend/learning.py)
        grounded in what that person said (stemmed words, quote = one of their own sentences),
        never about Metis or only said by her
        └─ compared with the 8 most SIMILAR stored facts: a repeat confirms the old fact (it lasts longer),
           a change supersedes it (the old row is kept, status "superseded"), else it is added
  conversation ends (call ends / >reset / old messages dropped / shutdown / text channel quiet for 10 min)
        └─ one background LLM call: 0-3 short episode summaries for everyone who spoke ("sleep");
           small talk yields none

Recall (read)  (backend/recall.py)
  cue = latest message + the user lines before it
  score = similarity(cue, memory) x retention (+ a small importance nudge)
        └─ only memories above the threshold are put in the prompt: at most 3, at most 1 episode,
           framed as background she uses only if it fits. "Hello, how are you?" brings nothing to mind.
        └─ the ones brought to mind get stronger (recalls + 1, last_recalled = now)
  asked directly ("what do you know about X?", "do you remember...", the `remember` tool)
        └─ lower threshold, faded ones too, anyone's (absent people too), facts and episodes
```

Names, aliases and who is in the voice channel are always in the prompt; facts are not.

### Forgetting

`retention = exp(-days since last confirmed or recalled / stability)`,
`stability = base × (1 + 0.35 × (importance − 5)) × (1 + 0.5 × ln(1 + confirmations + recalls))`
with base 60 days for facts, 14 for episodes and a minimum of 3. Nothing is deleted by time: below
`memory.retention_floor` (0.25) a memory just isn't brought to mind by a passing cue until it is confirmed or recalled
again. The only deletions are yours (dashboard) and the cap `memory.max_facts_per_person`, which lets the faintest go.

### Similarity

The knowledge base's embedding model (embeddinggemma, query/document prompts) when `rag.enabled`; vectors are stored
with each memory and filled in lazily. Without it (knowledge base off, model missing) she matches by word overlap
on lightly stemmed words. `memory.recall_threshold` (0.40) was calibrated on the real model: small talk peaked at 0.36
cosine to any memory, topical cues scored 0.38-0.56.

### Settings (`memory:` in `config.yaml`, all optional)

`learn_every_messages`, `max_facts_per_person`, `episode_idle_minutes`, `fact_stability_days`,
`episode_stability_days`, `retention_floor`, `recall_threshold`, `recall_threshold_words`, `recall_max`,
`search_threshold_factor`.

### Review and delete

The dashboard's **People** tab lists each person's facts and episodes (dimmed when faded) and deletes them, or via the API:
`GET /memory/people`, `DELETE /memory/people/{id}/facts/{fact_id}`, `DELETE /memory/people/{id}/episodes/{episode_id}`,
`DELETE /memory/people/{id}` (forgets the person and their facts; episodes nobody else was in).
`POST /memory/learn` learns from everything said so far without waiting.

People only known by name (e.g. from old transcripts) get a temporary profile that is merged
into their real one the first time they talk to Metis on Discord.
