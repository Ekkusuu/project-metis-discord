import { fileURLToPath } from "url";
import { dirname, join, isAbsolute } from "path";
import { readFileSync } from "fs";
import net from "net";
import { getLlama, LlamaChat, resolveChatWrapper } from "node-llama-cpp";
import express from "express";
import cors from "cors";
import yaml from "js-yaml";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = join(__dirname, "..", "..");

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
let config = {};
try {
  // METIS_CONFIG points at another config file (same variable the Python backend uses).
  config = yaml.load(readFileSync(process.env.METIS_CONFIG || join(projectRoot, "config.yaml"), "utf8")) || {};
} catch (err) {
  console.error("Failed to load config.yaml, using defaults:", err.message);
}

const modelCfg = config.model || {};
const chatCfg = config.chat || {};
const serviceCfg = config.llm_service || {};

// "localhost" can resolve to the IPv6 address on Windows, where another program may hold
// 127.0.0.1 on the same port; always use the IPv4 loopback so checks and clients agree.
const HOST = !serviceCfg.host || serviceCfg.host === "localhost" ? "127.0.0.1" : serviceCfg.host;
const PORT = Number(process.env.LLM_SERVICE_PORT) || serviceCfg.port || 3730;
const MODEL_PATH = (() => {
  const p = modelCfg.path || "model/model.gguf";
  return isAbsolute(p) ? p : join(projectRoot, p);
})();
const CONTEXT_SIZE = modelCfg.n_ctx || 8192;
// A second, smaller context for short "utility" prompts (addressee checks, tool selection,
// query generation, memory summaries). Keeping them off the main chat sequence
// means the chat KV cache survives between turns, so only new tokens get evaluated.
// Set to 0 to share the main context (saves VRAM).
const AUX_CONTEXT_SIZE = serviceCfg.aux_context_size ?? 4096;

// Qwen3.5's recommended non-thinking sampling: temperature 0.7, top_p 0.8, top_k 20, presence penalty 1.5,
// no repetition penalty. The old repetition penalty of 1.15 made a 4B model avoid the words it needed and
// drift into odd phrasing.
const DEFAULTS = {
  temperature: chatCfg.temperature ?? 0.7,
  topP: chatCfg.top_p ?? 0.8,
  topK: chatCfg.top_k ?? 20,
  presencePenalty: chatCfg.presence_penalty ?? 1.5,
  repeatPenalty: chatCfg.repeat_penalty ?? 1.0,
  maxTokens: chatCfg.max_tokens ?? 512,
};

// ---------------------------------------------------------------------------
// Model state
// ---------------------------------------------------------------------------
let llama;
let model;
let chatWrapper;
const schemaGrammars = new Map(); // JSON schema string -> LlamaJsonSchemaGrammar

/** Grammar that forces output to match a JSON schema (cached per schema). */
async function grammarForSchema(schema) {
  const key = JSON.stringify(schema);
  if (!schemaGrammars.has(key)) {
    if (schemaGrammars.size > 32) schemaGrammars.clear();
    schemaGrammars.set(key, await llama.createGrammarForJsonSchema(schema));
  }
  return schemaGrammars.get(key);
}

/**
 * A "lane" owns one context sequence plus a LlamaChat bound to it.
 * Requests on a lane run strictly one at a time (the GPU can only do one anyway),
 * which also fixes the "No sequences left" crash when two requests overlapped.
 */
class Lane {
  constructor(name, sequence) {
    this.name = name;
    this.chat = new LlamaChat({ contextSequence: sequence, chatWrapper, autoDisposeSequence: false });
    this._queues = { interactive: [], background: [] };
    this._busy = false;
  }

  get pending() {
    return this._queues.interactive.length + this._queues.background.length + (this._busy ? 1 : 0);
  }

  /**
   * Queue a task. "interactive" work (someone is waiting for a reply) always runs before
   * "background" work (learning), so a learning pass never delays an addressee check or tool decision.
   */
  run(task, priority = "interactive") {
    return new Promise((resolve, reject) => {
      this._queues[priority === "background" ? "background" : "interactive"].push({ task, resolve, reject });
      this._next();
    });
  }

  _next() {
    if (this._busy) return;
    const job = this._queues.interactive.shift() ?? this._queues.background.shift();
    if (!job) return;
    this._busy = true;
    Promise.resolve()
      .then(() => job.task(this.chat))
      .then(job.resolve, job.reject)
      .finally(() => {
        this._busy = false;
        this._next();
      });
  }
}

let chatLane;
let auxLane;

async function initModel() {
  console.log("\n" + "=".repeat(60));
  console.log("Initializing node-llama-cpp...");
  console.log("=".repeat(60));

  try {
    llama = await getLlama();
    console.log(`Loading model from: ${MODEL_PATH}`);

    const loadOptions = {
      modelPath: MODEL_PATH,
      gpuLayers: normalizeGpuLayers(modelCfg.n_gpu_layers),
      useMlock: modelCfg.use_mlock ?? true,
      useMmap: modelCfg.use_mmap ?? true,
      defaultContextFlashAttention: modelCfg.use_flash_attn ?? true,
    };
    model = await llama.loadModel(loadOptions);
    console.log("✓ Model loaded");

    // Reasoning models (Qwen 3.x, DeepSeek R1, ...) think before answering. That costs seconds per
    // reply, which a chat/voice companion can't afford, so it's off unless model.thinking is true.
    const thoughts = modelCfg.thinking ? "auto" : "discourage";
    chatWrapper = resolveChatWrapper(model, {
      type: modelCfg.chat_wrapper || "auto",
      customWrapperSettings: { qwen: { thoughts } },
    });
    console.log(`✓ Chat format: ${chatWrapper.wrapperName}`);

    const threads = modelCfg.n_threads || undefined;
    // KV cache quantization (q8_0 halves cache memory with negligible quality loss) and batch size
    // (the compute buffer grows with it; large vocabularies like Qwen's make that buffer big).
    const kvType = modelCfg.kv_cache_type || "f16";
    const contextOptions = {
      threads,
      ...(modelCfg.batch_size ? { batchSize: modelCfg.batch_size } : {}),
      ...(kvType !== "f16" ? { experimentalKvCacheKeyType: kvType, experimentalKvCacheValueType: kvType } : {}),
    };
    const mainContext = await model.createContext({ contextSize: CONTEXT_SIZE, ...contextOptions });
    chatLane = new Lane("chat", mainContext.getSequence());
    console.log(`✓ Chat context created (size: ${CONTEXT_SIZE})`);

    auxLane = chatLane;
    if (AUX_CONTEXT_SIZE > 0) {
      try {
        const auxContext = await model.createContext({ contextSize: AUX_CONTEXT_SIZE, ...contextOptions });
        auxLane = new Lane("aux", auxContext.getSequence());
        console.log(`✓ Utility context created (size: ${AUX_CONTEXT_SIZE})`);
      } catch (err) {
        console.warn(`⚠ Could not create utility context (${err.message}); sharing the chat context.`);
      }
    }


    await auxLane.run((chat) =>
      chat.generateResponse([{ type: "user", text: "Hello" }], { maxTokens: 4, temperature: 0 })
    );
    console.log("✓ Warmup complete");
    console.log("=".repeat(60) + "\n");
    return true;
  } catch (error) {
    console.error("✗ Error initializing model:", error);
    console.log("=".repeat(60) + "\n");
    return false;
  }
}

function normalizeGpuLayers(value) {
  if (value === undefined || value === null || value === -1 || value === "auto") return "auto";
  if (value === "max") return "max";
  return Number(value);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Convert OpenAI-style messages to node-llama-cpp chat history. */
function toChatHistory(messages) {
  const history = [];
  for (const msg of messages) {
    const text = typeof msg.content === "string" ? msg.content : String(msg.content ?? "");
    if (msg.role === "system") history.push({ type: "system", text });
    else if (msg.role === "user") history.push({ type: "user", text });
    else if (msg.role === "assistant") history.push({ type: "model", response: [text] });
  }
  return history;
}

function validateHistory(messages) {
  if (!Array.isArray(messages) || messages.length === 0) return "messages array is required";
  const history = toChatHistory(messages);
  if (history.length === 0) return "No valid messages provided";
  if (history[history.length - 1].type !== "user") return "Last message must be from user";
  return null;
}

/** Abort generation when the HTTP client goes away (e.g. bot interrupted by a newer message). */
function abortOnDisconnect(req, res) {
  const controller = new AbortController();
  res.on("close", () => {
    if (!res.writableFinished) controller.abort();
  });
  return controller;
}

function samplingOptions(body) {
  return {
    temperature: body.temperature ?? DEFAULTS.temperature,
    topP: body.top_p ?? DEFAULTS.topP,
    topK: body.top_k ?? DEFAULTS.topK,
    maxTokens: body.max_tokens ?? DEFAULTS.maxTokens,
    repeatPenalty: {
      penalty: body.repeat_penalty ?? DEFAULTS.repeatPenalty,
      presencePenalty: body.presence_penalty ?? DEFAULTS.presencePenalty,
    },
  };
}

// ---------------------------------------------------------------------------
// HTTP API
// ---------------------------------------------------------------------------
const app = express();
app.use(cors());
app.use(express.json({ limit: "4mb" }));

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    model_loaded: model !== undefined,
    chat_wrapper: chatWrapper?.wrapperName ?? null,
    queue: { chat: chatLane?.pending ?? 0, aux: auxLane === chatLane ? 0 : auxLane?.pending ?? 0 },
  });
});

/**
 * Non-streaming completion — used for utility prompts (tools, queries, memory).
 * Pass `json_schema` to constrain the output to valid JSON matching that schema.
 */
app.post("/chat/completion", async (req, res) => {
  const { messages, json_schema } = req.body;
  const problem = validateHistory(messages);
  if (problem) return res.status(400).json({ error: problem });

  let grammar;
  try {
    grammar = json_schema ? await grammarForSchema(json_schema) : undefined;
  } catch (error) {
    return res.status(400).json({ error: `Invalid json_schema: ${error.message}` });
  }

  const controller = abortOnDisconnect(req, res);
  try {
    const result = await auxLane.run(
      (chat) =>
        chat.generateResponse(toChatHistory(messages), {
          ...samplingOptions(req.body),
          ...(grammar ? { grammar, repeatPenalty: false } : {}),
          signal: controller.signal,
        }),
      req.body.priority
    );
    res.json({ choices: [{ message: { role: "assistant", content: result.response } }] });
  } catch (error) {
    if (controller.signal.aborted) return;
    console.error("Error in chat completion:", error);
    res.status(500).json({ error: error.message });
  }
});

/** Streaming completion (NDJSON) — used for the actual replies. */
app.post("/chat/stream", async (req, res) => {
  const { messages } = req.body;
  const problem = validateHistory(messages);
  if (problem) return res.status(400).json({ error: problem });

  res.setHeader("Content-Type", "application/x-ndjson");
  res.setHeader("Cache-Control", "no-cache");
  res.flushHeaders();

  const controller = abortOnDisconnect(req, res);
  let tokenCount = 0;
  let startTime = null;

  try {
    await chatLane.run(async (chat) => {
      if (controller.signal.aborted) return;
      startTime = Date.now();
      await chat.generateResponse(toChatHistory(messages), {
        ...samplingOptions(req.body),
        signal: controller.signal,
        stopOnAbortSignal: true,
        onTextChunk: (chunk) => {
          tokenCount++;
          res.write(JSON.stringify({ delta: chunk, done: false }) + "\n");
        },
      });
    });

    if (controller.signal.aborted) {
      console.log(`Generation aborted by client after ${tokenCount} chunks`);
      return;
    }
    const elapsed = startTime ? (Date.now() - startTime) / 1000 : 0;
    res.end(
      JSON.stringify({
        delta: "",
        done: true,
        tokens_per_second: elapsed > 0 ? Number((tokenCount / elapsed).toFixed(2)) : 0,
      }) + "\n"
    );
  } catch (error) {
    if (controller.signal.aborted) return;
    console.error("Error in streaming chat:", error);
    res.end(JSON.stringify({ error: error.message, done: true }) + "\n");
  }
});

/** Resolve true if HOST:PORT is free, so a clash is reported before spending time loading the model. */
function portIsFree() {
  return new Promise((resolve) => {
    const probe = net.createServer();
    probe.once("error", () => resolve(false));
    probe.once("listening", () => probe.close(() => resolve(true)));
    probe.listen(PORT, HOST);
  });
}

if (!(await portIsFree())) {
  console.error(`✗ Port ${PORT} is already in use by another program (or another copy of Metis).`);
  console.error("  Close it, or pick a free port under llm_service.port in config.yaml.");
  process.exit(1);
}

initModel().then((ok) => {
  if (!ok) {
    console.error("Failed to initialize model. Server not started.");
    process.exit(1);
  }
  const server = app.listen(PORT, HOST, () => {
    console.log(`🚀 LLM Service running on http://${HOST}:${PORT}`);
  });
  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      console.error(`✗ Port ${PORT} is already in use by another program (or another copy of Metis).`);
      console.error("  Close it, or pick a free port under llm_service.port in config.yaml.");
    } else {
      console.error("✗ LLM service failed to start:", error.message);
    }
    process.exit(1);
  });
});
