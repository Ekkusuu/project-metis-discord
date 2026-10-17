import { fileURLToPath } from "url";
import { dirname, join, isAbsolute } from "path";
import { readFileSync } from "fs";
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
  config = yaml.load(readFileSync(join(projectRoot, "config.yaml"), "utf8")) || {};
} catch (err) {
  console.error("Failed to load config.yaml, using defaults:", err.message);
}

const modelCfg = config.model || {};
const chatCfg = config.chat || {};
const serviceCfg = config.llm_service || {};

const HOST = serviceCfg.host || "localhost";
const PORT = Number(process.env.LLM_SERVICE_PORT) || serviceCfg.port || 3000;
const MODEL_PATH = (() => {
  const p = modelCfg.path || "model/model.gguf";
  return isAbsolute(p) ? p : join(projectRoot, p);
})();
const CONTEXT_SIZE = modelCfg.n_ctx || 8192;
// A second, smaller context for short "utility" prompts (judgment, tool selection,
// query generation, memory summaries). Keeping them off the main chat sequence
// means the chat KV cache survives between turns, so only new tokens get evaluated.
// Set to 0 to share the main context (saves VRAM).
const AUX_CONTEXT_SIZE = serviceCfg.aux_context_size ?? 4096;

const DEFAULTS = {
  temperature: chatCfg.temperature ?? 0.7,
  topP: chatCfg.top_p ?? 0.95,
  maxTokens: chatCfg.max_tokens ?? 512,
};

// ---------------------------------------------------------------------------
// Model state
// ---------------------------------------------------------------------------
let llama;
let model;
let chatWrapper;
let judgmentGrammar = null;

/**
 * A "lane" owns one context sequence plus a LlamaChat bound to it.
 * Requests on a lane run strictly one at a time (the GPU can only do one anyway),
 * which also fixes the "No sequences left" crash when two requests overlapped.
 */
class Lane {
  constructor(name, sequence) {
    this.name = name;
    this.chat = new LlamaChat({ contextSequence: sequence, chatWrapper, autoDisposeSequence: false });
    this._tail = Promise.resolve();
    this.pending = 0;
  }

  run(task) {
    this.pending++;
    const result = this._tail.then(() => task(this.chat));
    // Keep the chain alive regardless of task failures.
    this._tail = result.catch(() => {}).finally(() => this.pending--);
    return result;
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

    chatWrapper = resolveChatWrapper(model, { type: modelCfg.chat_wrapper || "auto" });
    console.log(`✓ Chat format: ${chatWrapper.wrapperName}`);

    const threads = modelCfg.n_threads || undefined;
    const mainContext = await model.createContext({ contextSize: CONTEXT_SIZE, threads });
    chatLane = new Lane("chat", mainContext.getSequence());
    console.log(`✓ Chat context created (size: ${CONTEXT_SIZE})`);

    auxLane = chatLane;
    if (AUX_CONTEXT_SIZE > 0) {
      try {
        const auxContext = await model.createContext({ contextSize: AUX_CONTEXT_SIZE, threads });
        auxLane = new Lane("aux", auxContext.getSequence());
        console.log(`✓ Utility context created (size: ${AUX_CONTEXT_SIZE})`);
      } catch (err) {
        console.warn(`⚠ Could not create utility context (${err.message}); sharing the chat context.`);
      }
    }

    judgmentGrammar = await llama.createGrammar({ grammar: 'root ::= "positive" | "negative"' });

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
    maxTokens: body.max_tokens ?? DEFAULTS.maxTokens,
    repeatPenalty: {
      penalty: body.repeat_penalty ?? 1.15,
      frequencyPenalty: 0.1,
      presencePenalty: 0.1,
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
 * Decide whether Metis should respond. Output is constrained by a grammar to
 * exactly "positive" or "negative", so it can't ramble or leak template tokens.
 */
app.post("/judgment", async (req, res) => {
  const { messages, system_prompt, user_prompt } = req.body;
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "messages array required" });
  }

  const label = (m) => (m.role === "user" ? m.content : `[Metis]: ${m.content}`);
  const context = messages.slice(0, -1).map(label).join("\n");
  const lastMessage = label(messages[messages.length - 1]);
  const template = user_prompt || "{context}\n\n{last_message}\n\nShould Metis respond?";
  const prompt = template
    .replaceAll("{context}", context || "(no previous messages)")
    .replaceAll("{last_message}", lastMessage)
    .replaceAll("{conversation}", messages.map(label).join("\n"));

  const controller = abortOnDisconnect(req, res);
  try {
    const result = await auxLane.run((chat) =>
      chat.generateResponse(
        [
          { type: "system", text: system_prompt || "Answer with positive or negative." },
          { type: "user", text: prompt },
        ],
        { grammar: judgmentGrammar, temperature: 0, maxTokens: 4, signal: controller.signal }
      )
    );
    const judgment = result.response.trim();
    console.log(`Judgment: ${judgment}`);
    res.json({ judgment });
  } catch (error) {
    if (controller.signal.aborted) return;
    console.error("Judgment error:", error);
    res.status(500).json({ error: error.message });
  }
});

/** Non-streaming completion — used for utility prompts (tools, queries, memory). */
app.post("/chat/completion", async (req, res) => {
  const { messages } = req.body;
  const problem = validateHistory(messages);
  if (problem) return res.status(400).json({ error: problem });

  const controller = abortOnDisconnect(req, res);
  try {
    const result = await auxLane.run((chat) =>
      chat.generateResponse(toChatHistory(messages), {
        ...samplingOptions(req.body),
        signal: controller.signal,
      })
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

initModel().then((ok) => {
  if (!ok) {
    console.error("Failed to initialize model. Server not started.");
    process.exit(1);
  }
  app.listen(PORT, HOST, () => {
    console.log(`🚀 LLM Service running on http://${HOST}:${PORT}`);
  });
});
