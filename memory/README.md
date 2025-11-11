# Memory Folder

This folder stores chat history and memory data for Project Metis.

## Files & Folders

### `chat_history.json`
- Stores all chat messages between user and AI
- Automatically saved after each message
- Automatically loaded on application startup
- Can be reset using the Reset button in the UI

### `temp_memory.txt`
- Temporary storage for deleted conversation messages
- Messages are added when:
  - Chat context limit (n_ctx) is reached and older messages are removed
  - Reset button is pressed (all messages saved before deletion)
- Format: Conversational format with role prefixes
  ```
  User: I like bananas
  AI: That's great! Bananas are nutritious.
  User: I have a pet parrot
  
  ```
- **Auto-summarization**: When temp_memory reaches configured token limit (default: 1000), it's automatically:
  1. Sent to AI for summarization (extracts user information only)
  2. Saved as a markdown file in `long_term/` folder
  3. Cleared for new messages
- Token limit configurable in `config.yaml` under `memory.temp_memory_token_limit`
- Includes both user and AI messages for context, but summaries focus on user information

### `long_term/` folder
- Stores AI-generated summaries of conversation history
- Files named: `memory_YYYYMMDD_HHMMSS.md`
- Contains structured markdown summaries with:
  - Main topics discussed
  - Important facts and context
  - Goals, tasks, or ongoing projects
  - Relevant personal information

## Privacy

All memory data is stored locally on your machine and is never sent to external servers.
The `chat_history.json` file is excluded from git commits (see `.gitignore`).

## Memory System Flow

```
User Messages (trimmed/deleted)
         ↓
  temp_memory.txt
         ↓
   (1000 tokens)
         ↓
   AI Summarization
         ↓
  long_term/*.md
         ↓
  temp_memory cleared
```
