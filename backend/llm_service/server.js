import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { getLlama, LlamaChatSession, LlamaChat, ChatMLChatWrapper } from "node-llama-cpp";
import express from "express";
import cors from "cors";
import { readFileSync } from "fs";
import yaml from "js-yaml";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const projectRoot = join(__dirname, "..", "..");

// Load configuration
const configPath = join(projectRoot, "config.yaml");
let config;
try {
  config = yaml.load(readFileSync(configPath, "utf8"));
} catch (err) {
  console.error("Failed to load config.yaml, using defaults:", err.message);
  config = {
    model: {
      path: "Model/dolphin-2.6-mistral-7b.Q5_K_M.gguf",
      n_ctx: 8192,
      n_gpu_layers: -1,
    },
    chat: {
      system_prompt: "You are Metis, a helpful AI assistant.",
      temperature: 0.7,
      top_p: 0.95,
      max_tokens: 512,
    },
    llm_service: {
      port: 3000,
    },
  };
}

const app = express();
app.use(cors());
app.use(express.json());

const PORT = config.llm_service?.port || 3000;

// Global model and context
let llama;
let model;
let context;
let currentSession;

// Initialize the model
async function initModel() {
  console.log("\n" + "=".repeat(60));
  console.log("Initializing node-llama-cpp...");
  console.log("=".repeat(60));

  try {
    llama = await getLlama();
    console.log("✓ Llama instance created");

    const modelPath = join(projectRoot, config.model.path);
    console.log(`Loading model from: ${modelPath}`);

    model = await llama.loadModel({
      modelPath: modelPath,
    });
    console.log("✓ Model loaded successfully");

    context = await model.createContext({
      contextSize: config.model.n_ctx || 8192,
    });
    console.log(`✓ Context created (size: ${config.model.n_ctx || 8192})`);

    console.log("=".repeat(60) + "\n");
    return true;
  } catch (error) {
    console.error("✗ Error initializing model:", error);
    console.log("=".repeat(60) + "\n");
    return false;
  }
}

// Health check
app.get("/health", (req, res) => {
  res.json({ status: "ok", model_loaded: model !== undefined });
});

// Non-streaming chat completion
app.post("/chat/completion", async (req, res) => {
  let sequence = null;
  try {
    const { messages, temperature, top_p, max_tokens } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // Create a new sequence for this request
    sequence = context.getSequence();
    
    // Check if we should use system prompts (from config or detect from messages)
    const useSystemPrompt = config.model?.use_system_prompt !== false;
    
    // Extract system prompt if present
    let systemPrompt = null;
    let conversationMessages = messages;
    
    if (useSystemPrompt && messages.length > 0 && messages[0].role === "system") {
      systemPrompt = messages[0].content;
      conversationMessages = messages.slice(1);
    } else if (!useSystemPrompt) {
      // If not using system prompts, don't filter out system messages
      // (they should already be converted to user messages by the Python backend)
      conversationMessages = messages.filter(msg => msg.role !== "system");
    }
    
    // Use LlamaChat for better conversation handling
    const chat = new LlamaChat({
      contextSequence: sequence,
      chatWrapper: new ChatMLChatWrapper(),
    });

    // Build conversation history array for LlamaChat
    const chatHistory = [];
    if (systemPrompt) {
      chatHistory.push({
        type: "system",
        text: systemPrompt,
      });
    }
    
    // Add all previous messages to history
    for (const msg of conversationMessages) {
      if (msg.role === "user") {
        chatHistory.push({
          type: "user",
          text: msg.content,
        });
      } else if (msg.role === "assistant") {
        chatHistory.push({
          type: "model",
          response: [msg.content],
        });
      }
    }

    // Validate we have messages
    if (chatHistory.length === 0) {
      return res.status(400).json({ error: "No valid messages provided" });
    }

    // Get the last message - should be a user message
    const lastMsg = chatHistory[chatHistory.length - 1];
    if (lastMsg.type !== "user") {
      return res.status(400).json({ error: "Last message must be from user" });
    }

    // Generate response
    const response = await chat.generateResponse(chatHistory, {
      temperature: temperature ?? config.chat.temperature ?? 0.7,
      topP: top_p ?? config.chat.top_p ?? 0.95,
      maxTokens: max_tokens ?? config.chat.max_tokens ?? 512,
    });

    res.json({
      choices: [
        {
          message: {
            role: "assistant",
            content: response,
          },
        },
      ],
    });
  } catch (error) {
    console.error("Error in chat completion:", error);
    res.status(500).json({ error: error.message });
  } finally {
    // Dispose of the sequence to free it up
    if (sequence) {
      sequence.dispose();
    }
  }
});

// Streaming chat completion
app.post("/chat/stream", async (req, res) => {
  let sequence = null;
  try {
    const { messages, temperature, top_p, max_tokens } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "messages array is required" });
    }

    // Set headers for streaming
    res.setHeader("Content-Type", "application/x-ndjson");
    res.setHeader("Transfer-Encoding", "chunked");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    // Create a new sequence for this request
    sequence = context.getSequence();
    
    // Check if we should use system prompts (from config or detect from messages)
    const useSystemPrompt = config.model?.use_system_prompt !== false;
    
    // Extract system prompt if present
    let systemPrompt = null;
    let conversationMessages = messages;
    
    if (useSystemPrompt && messages.length > 0 && messages[0].role === "system") {
      systemPrompt = messages[0].content;
      conversationMessages = messages.slice(1);
    } else if (!useSystemPrompt) {
      // If not using system prompts, don't filter out system messages
      // (they should already be converted to user messages by the Python backend)
      conversationMessages = messages.filter(msg => msg.role !== "system");
    }
    
    // Use LlamaChat for better conversation handling
    const chat = new LlamaChat({
      contextSequence: sequence,
      chatWrapper: new ChatMLChatWrapper(),
    });

    // Build conversation history array for LlamaChat
    const chatHistory = [];
    if (systemPrompt) {
      chatHistory.push({
        type: "system",
        text: systemPrompt,
      });
    }
    
    // Add all previous messages to history
    for (const msg of conversationMessages) {
      if (msg.role === "user") {
        chatHistory.push({
          type: "user",
          text: msg.content,
        });
      } else if (msg.role === "assistant") {
        chatHistory.push({
          type: "model",
          response: [msg.content],
        });
      }
    }

    // Validate we have messages
    if (chatHistory.length === 0) {
      res.write(JSON.stringify({ error: "No valid messages provided" }) + "\n");
      if (sequence) sequence.dispose();
      return res.end();
    }

    // Get the last message - should be a user message
    const lastMsg = chatHistory[chatHistory.length - 1];
    if (lastMsg.type !== "user") {
      res.write(JSON.stringify({ error: "Last message must be from user" }) + "\n");
      if (sequence) sequence.dispose();
      return res.end();
    }

    // Track tokens and timing
    let tokenCount = 0;
    const startTime = Date.now();

    // Stream the response using chat.generateResponse with history
    const response = await chat.generateResponse(
      chatHistory,
      {
        temperature: temperature ?? config.chat.temperature ?? 0.7,
        topP: top_p ?? config.chat.top_p ?? 0.95,
        maxTokens: max_tokens ?? config.chat.max_tokens ?? 512,
        onTextChunk: (chunk) => {
          tokenCount++;
          res.write(
            JSON.stringify({
              delta: chunk,
              done: false,
            }) + "\n"
          );
        },
      }
    );

    // Send final chunk with stats
    const elapsedSeconds = (Date.now() - startTime) / 1000;
    const tokensPerSecond = tokenCount / elapsedSeconds;

    res.write(
      JSON.stringify({
        delta: "",
        done: true,
        tokens_per_second: parseFloat(tokensPerSecond.toFixed(2)),
      }) + "\n"
    );

    res.end();
  } catch (error) {
    console.error("Error in streaming chat:", error);
    res.write(JSON.stringify({ error: error.message }) + "\n");
    res.end();
  } finally {
    // Dispose of the sequence to free it up
    if (sequence) {
      sequence.dispose();
    }
  }
});

// Start server after model initialization
initModel().then((success) => {
  if (success) {
    app.listen(PORT, () => {
      console.log(`🚀 LLM Service running on http://localhost:${PORT}`);
      console.log(`   Health check: http://localhost:${PORT}/health`);
    });
  } else {
    console.error("Failed to initialize model. Server not started.");
    process.exit(1);
  }
});
