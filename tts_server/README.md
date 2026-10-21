# Qwen3-TTS voice (custom voice + emotions)

`server.py` is Metis's voice when `voice.tts_engine: "qwen3"`. `metis.py` starts it in its own Python
environment, waits for it, and stops it with everything else. If it's missing or crashes, voice chat
falls back to Kokoro.

How it fits together:

- **Voice model**: Qwen3-TTS 12Hz 0.6B, fine-tuned on a character's voice lines, served with
  [faster-qwen3-tts](https://pypi.org/project/faster-qwen3-tts/) (CUDA graphs). First audio arrives in
  ~150–450 ms, and it generates ~3× faster than real time. It uses ~3 GB of VRAM.
- **Emotions**: the 0.6B models ignore `instruct` ("speak sadly"), so moods come from reference clips
  instead. Each emotion in `voice.qwen_tts.emotions` is a clip of the same character saying something
  in that mood. The model continues in that delivery (in-context cloning). Neutral uses the speaker
  embedding the fine-tune was trained with.
- **The LLM picks the mood**: in voice chat the reply guidance (`backend/routers/chat.py`) lists the
  tags. The model starts its reply with one, e.g. `[teasing] Heehee, late again?`. `discord_bot.py`
  strips the tag and voices the rest of the reply in that mood until another tag appears. Each
  sentence is logged as `🎙️ [emotion] seconds: text`.
- **Streaming**: `discord_bot.SpeechStream` plays audio while it's generated. When someone talks over
  Metis, closing the request stops generation mid-sentence.

## Setup

1. A separate venv, because faster-qwen3-tts needs `transformers==5.15.1` (newer versions fail to load
   the codec):
   ```
   python -m venv C:\Users\you\qwen3tts\fast-venv
   C:\Users\you\qwen3tts\fast-venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
   C:\Users\you\qwen3tts\fast-venv\Scripts\pip install faster-qwen3-tts==0.5.4 transformers==5.15.1 pyyaml
   ```
2. Set `voice.qwen_tts.python` to that venv's `python.exe` and `voice.qwen_tts.model` to the model folder.

## Making a voice (`finetune/`)

1. `build_dataset.py`: downloads a character's voice lines (here Yuumi's, from the League of
   Legends wiki). It converts them to 24 kHz mono wavs and writes `train_raw.jsonl` plus `ref.wav`.
2. `prepare_data.py`: adds the audio codes (needs `Qwen3-TTS-Tokenizer-12Hz`).
3. `sft_12hz.py`: fine-tunes `Qwen3-TTS-12Hz-0.6B-Base`. The Yuumi voice used
   `--lr 1e-5 --num_epochs 6 --batch_size 2 --speaker_name yuumi`. This copy fixes three bugs in the
   upstream script, which otherwise produces noise on 0.6B (see the comments in the file):
   - the missing text projection;
   - labels that were shifted twice;
   - the code-predictor hidden state taken one step off.
4. `make_hybrid.py <epoch>`: puts the Base model's speaker encoder back into the fine-tuned checkpoint
   (the fine-tune drops it). One model can then do both the trained voice and the emotion clips.
   Point `voice.qwen_tts.model` at the result.

Pick emotion clips that clearly carry the mood and are 2–7 s long. Their `text` must match the clip
exactly. Generated audio and voice lines aren't committed; they're the game's.
