"""
Qwen3-TTS voice server (streaming, emotions), run by metis.py in its own Python environment.

faster-qwen3-tts needs transformers==5.15.1, which the main venv can't have, so this is a separate
process. It loads one model and streams 16-bit mono PCM at 24 kHz while it's being generated (first
audio in ~150 ms):

    POST /speak  {"text": "...", "emotion": "teasing"}  ->  raw PCM, chunked
    GET  /health                                         ->  {"ok": true, "emotions": [...]}

Voices (voice.qwen_tts in config.yaml):
  neutral   the fine-tuned speaker: the x-vector of `ref_audio`, exactly what training conditioned on
  <emotion> in-context cloning from a clip of the same character saying something in that mood
The model is the fine-tuned checkpoint with the Base speaker encoder put back (tts_server/README.md),
so both kinds come from one copy in VRAM. One generation at a time: a GPU can't run two faster.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG = Path(os.environ.get("METIS_CONFIG") or ROOT / "config.yaml")
cfg = (yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}).get("voice", {}).get("qwen_tts", {})
logging.basicConfig(level=logging.WARNING)


def resolve(p: str) -> str:
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


MODEL = resolve(cfg["model"])
LANGUAGE = cfg.get("language", "Auto")  # "Auto" matches how the fine-tune was trained
CHUNK = int(cfg.get("chunk_size", 4))  # codec frames per streamed chunk (12 per second of audio)
NEUTRAL = resolve(cfg["ref_audio"])
EMOTIONS = {name: (resolve(e["audio"]), e["text"]) for name, e in (cfg.get("emotions") or {}).items()}
SAMPLING = dict(temperature=float(cfg.get("temperature", 0.75)), top_k=int(cfg.get("top_k", 50)),
                repetition_penalty=float(cfg.get("repetition_penalty", 1.05)))

from faster_qwen3_tts import FasterQwen3TTS  # noqa: E402  (slow import; after config errors surface)

t = time.time()
model = FasterQwen3TTS.from_pretrained(MODEL)
model.warmup(prefill_len=100)
lock = threading.Lock()


def max_frames(text: str) -> int:
    # ~12 frames per second of speech, ~14 characters per second: generous, but stops a sample that
    # never says "done" from babbling for minutes (the model's own limit is 8192 frames).
    return max(48, int(len(text) * 1.6) + 36)


def generate(text: str, emotion: str):
    kw = dict(text=text, language=LANGUAGE, chunk_size=CHUNK, max_new_tokens=max_frames(text), **SAMPLING)
    if emotion in EMOTIONS:
        audio, ref_text = EMOTIONS[emotion]
        return model.generate_voice_clone_streaming(ref_audio=audio, ref_text=ref_text, **kw)
    return model.generate_voice_clone_streaming(ref_audio=NEUTRAL, xvec_only=True, **kw)


# Encoding each reference clip is cached by the library; do it now rather than on the first reply.
for name in ["neutral", *EMOTIONS]:
    for _ in generate("Hi.", name):
        pass
print(f"Qwen3-TTS ready in {time.time() - t:.0f}s: {Path(MODEL).name}, emotions: {', '.join(EMOTIONS) or 'none'}",
      flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True, "emotions": list(EMOTIONS), "sample_rate": 24000})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/speak":
            return self._json(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            text = str(body.get("text", "")).strip()
            emotion = str(body.get("emotion") or "neutral").lower()
        except (ValueError, TypeError):
            return self._json(400, {"error": "bad request"})
        if not text:
            return self._json(400, {"error": "no text"})
        with lock:
            self.send_response(200)
            self.send_header("Content-Type", "audio/L16; rate=24000; channels=1")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            gen = generate(text, emotion)
            start, first, seconds = time.perf_counter(), None, 0.0
            try:
                for audio, sr, _ in gen:
                    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype(np.int16).tobytes()
                    first = first or time.perf_counter() - start
                    seconds += len(pcm) / 2 / sr
                    self.wfile.write(f"{len(pcm):X}\r\n".encode() + pcm + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                status = ""
            except (ConnectionError, OSError):
                status = " (stopped: Metis was interrupted)"
            finally:
                gen.close()
            took = time.perf_counter() - start
            print(f"🎙️ [{emotion}] first audio {1000 * (first or 0):.0f} ms, {seconds:.1f}s of speech in "
                  f"{took:.1f}s{status}: {text}", flush=True)


if __name__ == "__main__":
    port = int(cfg.get("port", 8732))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Qwen3-TTS listening on 127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)
