# LLM Service

Node.js service using `node-llama-cpp` for GPU-accelerated inference.

## Installation

```bash
cd backend/llm_service
npm install
```

## Running

```bash
npm start
```

The service will start on port 3000 (configurable in `config.yaml`).

## Endpoints

- `GET /health` - Health check
- `POST /chat/completion` - Non-streaming chat completion
- `POST /chat/stream` - Streaming chat completion

## GPU Support

node-llama-cpp automatically detects and uses CUDA if available. No additional configuration needed!
