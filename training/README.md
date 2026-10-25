# Fine-tuning Metis

A small LoRA fine-tune on top of Qwen3.5-4B that makes her better at the things the runtime prompts ask for, while
staying **persona-agnostic**: the character is always the system prompt, never the weights.

```
personas.py + gen_conversations.py   teacher writes labelled call transcripts per persona
        │
        ▼  training/data/conversations/conv_*.json
build_samples.py                     renders every example through the real runtime prompts
        │
        ▼  training/data/samples.jsonl       (train personas)
           training/data/samples.val.jsonl   (held-out personas)
train_unsloth.py                     16-bit LoRA on the heretic base, loss on Metis's reply only
        │
        ▼  training/out/metis-lora  ->  merged 16-bit  ->  GGUF
eval_personas.py                     same questions, unseen personas: does she become them?
```

## Why personas are data

At runtime Metis is whoever `system_prompts.yaml` (or the personality editor) says she is. A fine-tune that only
ever saw one character would drift back to that character when given a different prompt. So:

- `personas.py` holds 27 personas: 22 train, 5 held out for validation.
- Every conversation records its full voice and text system prompt in `meta.persona_prompt` /
  `meta.text_persona_prompt`.
- `build_samples.py` uses that exact prompt as the system message. Nothing about a character is hard-coded in the
  builder.
- `eval_personas.py` asks the val personas questions. If they sound as in-character as the train ones, the model is
  reading the prompt instead of reciting a memorised voice.

The shared voice-chat contract (brevity, no emoji, no narration, speakable text) is identical across personas and
lives in `_VOICE_RULES`; only the character paragraph changes.

## 0. Setup

Training needs an NVIDIA GPU (a 12 GB card is enough for the defaults, ~10 GB with nothing else on the GPU: bf16 LoRA r=16, 2048 tokens, batch 1 × 8).
The rest of the app only needs its own venv.

```powershell
# in the project venv (or a separate training venv)
pip install "unsloth[cu126] @ git+https://github.com/unslothai/unsloth.git" trl peft accelerate bitsandbytes
```

`--check` below works with plain `transformers` and no GPU, so the dataset can be verified before installing
Unsloth.

On Windows, prefix Python commands with `$env:PYTHONUTF8=1;` if a script prints check marks.

## 1. Generate conversations

The teacher is DeepSeek through the OpenCode CLI (`teacher.py`). Anthropic's terms do not allow training chatbots
on Claude output, so do not swap the teacher to Claude.

```powershell
# training personas, balanced so every persona appears before any repeats the round
python training/gen_conversations.py --split train --count 220 --workers 6

# held-out personas, never trained on
python training/gen_conversations.py --split val --count 40 --workers 5

# iterate on one persona
python training/gen_conversations.py --personas cowboy --count 2 --workers 2
```

Each call writes `training/data/conversations/conv_NNNNN.json` with:

```jsonc
{
  "lines": [
    {"speaker": "Mira", "text": "...", "for": "Metis", "plan": {"action": "web_search", "standalone": "...",
                                                                 "query": "...", "recent": false}},
    {"speaker": "Metis", "mood": "teasing", "text": "...", "found": "u.gg, patch 26.21: ..."}
  ],
  "facts": {"Mira": [{"quote": "...", "about": "themselves", "kind": "a plan",
                      "importance": 7, "fact": "..."}]},
  "meta": {"persona": "cowboy", "split": "val", "persona_prompt": "You are Metis, ...",
           "text_persona_prompt": "You are Metis, ...", "people": ["yuki"], "seed": 1008}
}
```

Rejected generations go to `training/data/rejected/` with the reason, so the prompt can be fixed rather than
silently losing data.

## 2. Build the SFT examples

```powershell
# stretch the conversations: name swaps, speech-to-text noise, persona swaps (train split only)
python training/augment.py --copies 2
# voice and text renders of everything (augmented variants are picked up automatically)
python training/build_samples.py --split train --mode both --out training/data/samples.jsonl
python training/build_samples.py --split val   --mode both --out training/data/samples.val.jsonl
# outside data, ~17% of the mix: When2Call tool calls / follow-up questions (no refusals) + No Robots general chat
python training/external_data.py
# the exact text the runtime's chat wrapper produces (the HF template can't render the reply prompt)
node training/render_prompts.mjs training/data/samples.jsonl training/data/samples.rendered.jsonl
node training/render_prompts.mjs training/data/samples.val.jsonl training/data/samples.val.rendered.jsonl
node training/render_prompts.mjs training/data/external.jsonl training/data/external.rendered.jsonl

# inspect one conversation, one example per task
python training/build_samples.py --conversation conv_00008 --tasks reply,planner --preview
```

Four tasks come out of each conversation, all rendered by the functions the app calls at inference:

| task | prompt | target |
| --- | --- | --- |
| `reply` | `build_prompt()`: persona, history, people memory, recent lookups, reply guidance | Metis's line (`[mood]` prefix only when TTS emotions are configured) |
| `planner` | `planner_messages()`: tool menu and worked examples | the JSON plan |
| `addressee` | `addressee_messages()`: who each line was for | `Metis` / `another person in the chat` / ... |
| `facts` | `learning.py` fact-extraction prompt | labelled candidates for one speaker |

Validation drops examples where the target line leaked into the prompt, a fact quote is not grounded in that
speaker's words, or a structured answer is not JSON. `samples.stats.json` / `samples.val.stats.json` record task,
split and persona counts.

Only replies Metis actually made are targets; chat lines from other people are never imitated. Chime-ins use
`chiming_in=True` (she adds a comment instead of answering), and replies about a web lookup keep the `found` notes
so the model learns to retell them in her own words.

## 3. Train

```powershell
python training/train_unsloth.py --check        # tokenization + lengths, no GPU
python training/train_unsloth.py                # adapter -> training/out/metis-lora
python training/train_unsloth.py --merge        # + merged 16-bit -> training/out/metis-merged-16bit
```

Defaults: Qwen3.5-4B heretic (tvall43/Qwen3.5-4B-heretic) in bf16 (Unsloth advises against 4-bit for Qwen3.5), LoRA r=16 / alpha=32 on all attention and MLP projections, 3 epochs, lr 1e-4 cosine,
batch 1 × gradient accumulation 8, max sequence 3072, bf16 when the GPU supports it.

The prompt is masked (`-100`) and the loss is only on the answer. Thinking is disabled in the chat template, the
same as `model.thinking: false` at runtime, so the model trains on the exact prompt shape it will serve.

Useful knobs: `--tasks reply,planner` to focus, `--limit 50` for a smoke run, `--epochs`, `--lora-r`,
`--max-seq-length`. If examples are being skipped as too long, raise the sequence length only if VRAM allows;
otherwise they are dropped rather than truncated mid-answer.

## 4. Export and run

```powershell
python C:\path\to\llama.cpp\convert_hf_to_gguf.py training/out/metis-merged-16bit `
  --outfile model/Qwen3.5-4B-metis-Q4_K_M.gguf --outtype q4_k_m
```

Point `model.path` in `config.yaml` at the new GGUF and restart the app. Keep the base GGUF around so you can
A/B them.

## 5. Evaluate

```powershell
python training/eval_personas.py                 # all held-out personas, three questions each
python training/eval_personas.py --split train --personas casual,mean
python training/eval_personas.py --dry-run       # prompts only
```

This calls the app's own LLM service, so it tests exactly the configured model. What to look for:

- **Held-out personas** sound distinct and in-character, not like the casual persona wearing a hat.
- **No persona bleed**: the mean persona does not appear when the cowboy prompt is used, etc.
- **Answer first, flavour second**: questions still get answered; the persona does not replace content.
- **Voice shape**: one or two speakable sentences, a `[mood]` tag, no narration or emoji.

A loss curve alone does not show any of that, which is why this comparison is the real acceptance test.

## Cost and scale

Each conversation is one teacher call (~10–90 s). The defaults above (260 conversations) make roughly 5–6k SFT
examples after the four tasks are expanded, which is 1–3 GPU hours on a 12 GB card. More data helps most on
`planner` and `reply`; the addressee and facts tasks saturate quickly because their label space is small.
