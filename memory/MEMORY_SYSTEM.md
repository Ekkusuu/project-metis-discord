# Memory System Architecture

## Overview

The memory system manages conversation context and preserves important information when messages are removed due to context limits or chat resets.

## Components

### 1. Context Manager (`context_manager.py`)
- **Purpose**: Keep conversation within token limits (n_ctx)
- **Process**:
  1. Counts tokens in all messages
  2. If exceeds limit, removes oldest messages
  3. Always preserves system message
  4. Saves removed **user messages** to temp_memory

### 2. Memory Manager (`memory_manager.py`)
- **Purpose**: Manage temporary and long-term memory
- **Functions**:
  - `append_to_temp_memory()`: Add user messages to temp_memory.txt
  - `check_and_summarize_temp_memory()`: Auto-trigger summarization at 1000 tokens
  - `summarize_and_archive_temp_memory()`: Create AI summary and save to long_term/
  - `save_messages_before_reset()`: Save all user messages before chat reset

### 3. History Router (`routers/history.py`)
- **Endpoints**:
  - `POST /history/reset`: Reset chat (saves user messages first)
  - `POST /history/save`: Save chat history
  - `GET /history/load`: Load chat history

### 4. Memory Router (`routers/memory.py`)
- **Endpoints**:
  - `GET /memory/status`: Check temp_memory token count
  - `POST /memory/summarize`: Manually trigger summarization

## Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User sends message                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Context Manager checks tokens               │
│              Current: X tokens / n_ctx limit             │
└─────────────────────┬───────────────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
    ┌──────────────┐    ┌──────────────┐
    │ Under limit  │    │  Over limit  │
    │   (no trim)  │    │   (trim!)    │
    └──────┬───────┘    └──────┬───────┘
           │                   │
           │                   ▼
           │         ┌─────────────────────┐
           │         │ Remove old messages │
           │         │ (save user msgs)    │
           │         └─────────┬───────────┘
           │                   │
           │                   ▼
           │         ┌─────────────────────────┐
           │         │  temp_memory.txt        │
           │         │  [timestamp] message    │
           │         └─────────┬───────────────┘
           │                   │
           │                   ▼
           │         ┌─────────────────────────┐
           │         │  Check token count      │
           │         │  >= 1000 tokens?        │
           │         └─────────┬───────────────┘
           │                   │
           │          ┌────────┴────────┐
           │          │                 │
           │          ▼                 ▼
           │    ┌──────────┐     ┌──────────────┐
           │    │   < 1000 │     │   >= 1000    │
           │    │   Wait   │     │  Summarize!  │
           │    └──────────┘     └──────┬───────┘
           │                            │
           │                            ▼
           │              ┌─────────────────────────┐
           │              │  AI creates summary     │
           │              │  (temperature: 0.3)     │
           │              └─────────┬───────────────┘
           │                        │
           │                        ▼
           │              ┌─────────────────────────┐
           │              │  Save to long_term/     │
           │              │  memory_YYYYMMDD.md     │
           │              └─────────┬───────────────┘
           │                        │
           │                        ▼
           │              ┌─────────────────────────┐
           │              │  Clear temp_memory.txt  │
           │              └─────────────────────────┘
           │
           └──────────────────┬──────────────────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  Send to LLM service │
                   │  Get AI response     │
                   └──────────────────────┘
```

## Reset Flow

```
┌─────────────────────────────────────┐
│     User clicks Reset button        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Frontend confirms with user       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   POST /history/reset               │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Load current chat_history.json    │
│   Extract all user messages         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Append to temp_memory.txt         │
│   (with timestamps)                 │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Check if >= 1000 tokens           │
│   (trigger summary if needed)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│   Delete chat_history.json          │
│   Return fresh initial message      │
└─────────────────────────────────────┘
```

## Configuration

- **Context Limit**: `n_ctx` in config.yaml (default: 8192 tokens)
- **Response Reserve**: `max_tokens` in config.yaml (default: 512 tokens)
- **Available for messages**: n_ctx - max_tokens = 7680 tokens
- **Temp memory threshold**: 1000 tokens (triggers summarization)
- **Summary temperature**: 0.3 (for consistent, factual summaries)

## Example temp_memory.txt

```
[2025-11-11 14:23:45] What is machine learning?
[2025-11-11 14:24:12] Can you explain neural networks?
[2025-11-11 14:25:03] How do I implement a CNN in PyTorch?
[2025-11-11 14:26:30] What are the best practices for training?
```

## Example long_term summary

```markdown
# Memory Summary
**Created:** 2025-11-11 14:30:00

---

## Topics Discussed

### Machine Learning & Neural Networks
- User is learning about machine learning fundamentals
- Specific interest in neural networks and CNNs
- Using PyTorch framework

### Implementation Details
- Working on CNN implementation
- Seeking best practices for model training
- Focus on practical application

## Key Context
- Technical skill level: Intermediate
- Framework preference: PyTorch
- Learning approach: Hands-on implementation
```
