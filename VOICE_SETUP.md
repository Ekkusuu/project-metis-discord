# Voice Chat Setup

Metis can sit in a Discord voice channel, listen to everyone, and answer out loud.

## Requirements

- **espeak-ng** (used by Kokoro TTS for pronunciation):
  install the `.msi` from <https://github.com/espeak-ng/espeak-ng/releases> and restart your terminal.
- Python packages from `requirements.txt` (`nemo_toolkit[asr]`, `kokoro`, `discord-ext-voice-recv`, `PyNaCl`).
- **Parakeet STT model**: place `parakeet-tdt-0.6b-v2.nemo` in `model/`, or let it download
  from Hugging Face on the first `>join` (~2 GB). Kokoro (~300 MB) downloads automatically.
- The bot needs the **Connect** and **Speak** permissions in the voice channel.

FFmpeg is **not** required: replies are resampled in memory and streamed straight to Discord.

## Using it

1. Join a voice channel and type `>join` in a text channel.
2. Talk. When you pause for `voice.listen_timeout` seconds, Metis transcribes what you said,
   decides whether to answer (saying "Metis" always gets an answer), and starts speaking
   as soon as the first sentence of her reply is ready.
3. `>leave` when you're done — she also leaves on her own once everyone else has.

While she's in voice, text messages in that server are ignored (commands still work).
Voice conversations are kept separately from text conversations and are saved to memory the same way.

## How a voice turn works

```
audio (48 kHz stereo) ─► pause detected ─► Parakeet STT ─► name correction (voice_corrections.yaml)
   ─► judgment ─► /chat/stream (1 tool max, no query expansion) ─► sentence splitter
   ─► emotion tag ([teasing] …) ─► Qwen3-TTS streamed per sentence (or Kokoro) ─► 24→48 kHz in memory ─► Discord
```

The voice is Qwen3-TTS with a fine-tuned custom voice and emotions when `tts_engine: "qwen3"`
(setup: [tts_server/README.md](tts_server/README.md)), otherwise Kokoro.

Each turn prints a timing breakdown (STT, judgment, tools, first token, TTS, time to first audio)
in the backend console.

## Configuration (`config.yaml` → `voice`)

| Key | Meaning |
|---|---|
| `tts_engine` | `qwen3` (custom voice + emotions, `qwen_tts` settings) or `kokoro` |
| `tts_voice` | Kokoro voice, e.g. `af_heart`, `af_sarah`, `am_adam`, `am_michael` |
| `tts_lang_code` | `a` American English, `b` British English |
| `tts_speed` | Speaking rate (1.0 = normal) |
| `listen_timeout` | Seconds of silence that end an utterance |
| `chunk_duration` | How often (s) to check for pauses |
| `input_gain` | Microphone gain applied to captured audio |
| `min_rms` | Noise floor; quieter clips are ignored |
| `min_speech_seconds` | Clips shorter than this are ignored |
| `discard_while_speaking` | Ignore speech while Metis is talking (prevents her answering herself / stale input) |
| `name_correction_threshold` | Fuzzy-match threshold for snapping transcribed names to `voice_corrections.yaml` |
| `judgment.enabled` | Turn the "should I answer?" step off to answer everything |

Add names people use (server members, games, Metis herself) to `voice_corrections.yaml` so the
transcriber spells them correctly.

## Troubleshooting

- **She doesn't react at all** — check the backend console: "Ignoring audio" means the clip was
  too quiet/short (raise `input_gain` or lower `min_rms`).
- **She talks over people / answers stale input** — keep `discard_while_speaking: true`.
- **Slow first reply** — the first `>join` loads Parakeet and Kokoro; later joins are instant.
- **TTS errors about espeak** — espeak-ng isn't installed or not on PATH.
