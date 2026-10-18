# Memory

This folder is Metis' memory. Its contents are git-ignored (they're about real people);
only this README and `.gitkeep` are tracked.

| Path | What it is |
|---|---|
| `people.json` | One profile per Discord user id: current name, aliases, first/last seen, message count and learned facts. Also general notes. |
| `long_term/people/*.md` | The same profiles as Markdown, indexed by the knowledge base so `knowledge_search` finds facts about people who aren't in the conversation. |
| `long_term/learned_notes.md` | General things Metis learned that aren't about one person. |
| `long_term/memory_*.md` | Summaries written by the previous memory system (still indexed). |
| `chat_history.jsonl` | Transcript of the current session; moved to `sessions/` on the next start. |
| `sessions/` | Past session transcripts, plus legacy files that were migrated once. |

## How learning works

```
Discord message (text or voice)
   └─ attributed to the speaker's Discord user id ─► people.json (names, last seen)
every 6 user messages / >reset / old messages dropped / shutdown
   └─ model reads the new part of the conversation and writes "Name | fact" lines
        └─ only names that actually spoke are accepted, duplicates are skipped
             └─ people.json + long_term/people/<name>-<id>.md (re-indexed immediately)
next reply
   └─ profiles of the people in the conversation are added to the prompt
```

- Settings: `memory.learn_every_messages`, `memory.max_facts_per_person` in `config.yaml`.
- Review or delete facts in the dashboard's **People** tab, or via the API:
  `GET /memory/people`, `DELETE /memory/people/{id}/facts/{fact_id}`, `DELETE /memory/people/{id}`.
- `POST /memory/learn` learns from everything said so far without waiting.
- People only known by name (e.g. from old transcripts) get a temporary profile that is merged
  into their real one the first time they talk to Metis on Discord.
