// Render training examples into the exact text the model sees at runtime.
//
// Metis's prompts are turned into model input by node-llama-cpp's Qwen chat wrapper (backend/llm_service), not
// by Hugging Face's chat template. They differ: Qwen3.5's HF template refuses a system message anywhere but the
// start, and the reply prompt puts the reply guidance in a second system message right before the latest line.
// Rendering with the same wrapper class and settings as the server keeps training and runtime identical.
//
//   node training/render_prompts.mjs training/data/samples.jsonl training/data/samples.rendered.jsonl
import { readFileSync, writeFileSync } from "fs";
import { createRequire } from "module";
import { join, dirname } from "path";
import { fileURLToPath, pathToFileURL } from "url";

const here = dirname(fileURLToPath(import.meta.url));
const require = createRequire(join(here, "..", "backend", "llm_service", "package.json"));
const { QwenChatWrapper, LlamaText } = await import(pathToFileURL(require.resolve("node-llama-cpp")).href);

// As server.js: thinking discouraged unless model.thinking is on (it is off for Metis).
const wrapper = new QwenChatWrapper({ variation: "3.5", thoughts: "discourage" });

function toChatHistory(messages) {
  // The same conversion as server.js toChatHistory().
  return messages.map((m) =>
    m.role === "system" ? { type: "system", text: m.content }
      : m.role === "user" ? { type: "user", text: m.content }
      : { type: "model", response: [m.content] });
}

function render(history) {
  const { contextText } = wrapper.generateContextState({ chatHistory: history });
  return LlamaText(contextText).toString();
}

const [input, output] = process.argv.slice(2);
const lines = readFileSync(input, "utf8").split("\n").filter((l) => l.trim());
const out = [];
for (const line of lines) {
  const sample = JSON.parse(line);
  const history = toChatHistory(sample.messages);
  // The prompt as generation starts (an empty model turn is what the server generates into), and the full text
  // with the answer; the answer is everything the full text adds after the prompt.
  const prompt = render([...history, { type: "model", response: [] }]);
  const full = render([...history, { type: "model", response: [sample.answer] }]);
  if (!full.startsWith(prompt)) throw new Error(`prompt is not a prefix of the full text in ${sample.id}`);
  out.push(JSON.stringify({ ...sample, prompt_text: prompt, answer_text: full.slice(prompt.length) }));
}
writeFileSync(output, out.join("\n") + "\n");
console.log(`rendered ${out.length} examples -> ${output}`);
