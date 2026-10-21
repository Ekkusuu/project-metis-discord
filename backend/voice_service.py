"""
Speech-to-text (NVIDIA Parakeet via NeMo) and text-to-speech: Qwen3-TTS (tts_server/, streamed, with
emotions) or Kokoro, which is also the fallback whenever the Qwen3-TTS server isn't running.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import yaml

from backend.config import PROJECT_ROOT, get_config, resolve_path

# NeMo and Kokoro are imported when voice is first used (>join), not at startup: importing them
# costs ~5.6 s and ~270 MB of RAM, which is wasted whenever nobody uses voice chat.

TTS_SAMPLE_RATE = 24000  # Kokoro and Qwen3-TTS output
STT_SAMPLE_RATE = 16000  # Parakeet input


def float_to_pcm16(audio: Any) -> bytes:
    """float32 [-1, 1] (numpy or torch) -> 16-bit PCM bytes, clipped so loud peaks don't wrap around."""
    if hasattr(audio, "cpu"):
        audio = audio.cpu().numpy()
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (audio * 32767).astype(np.int16).tobytes()


class VoiceService:
    def __init__(self) -> None:
        self.config = get_config().get("voice", {})
        self.stt_model: Optional[Any] = None
        self.tts_pipeline: Optional[Any] = None  # Kokoro
        self.qwen_url: Optional[str] = None  # Qwen3-TTS server, when it's the voice
        self.emotions: List[str] = []  # voices the Qwen3-TTS server has besides neutral
        self._http: Optional[Any] = None
        self._initialized = False
        # One worker each: GPU models don't like concurrent calls, and a long TTS
        # job shouldn't delay transcription of the next speaker (or vice versa).
        self._stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="STT")
        self._tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self._names: List[str] = self._load_names()
        self._aliases: Dict[str, str] = self._load_aliases()

    async def initialize(self) -> None:
        if self._initialized:
            return
        print("Initializing voice service...")
        loop = asyncio.get_running_loop()
        await asyncio.gather(
            loop.run_in_executor(self._stt_executor, self._init_stt),
            loop.run_in_executor(self._tts_executor, self._init_tts),
            # Resampling for input and playback; importing it lazily cost the first reply ~1-3 s.
            loop.run_in_executor(None, __import__, "scipy.signal"),
        )
        self._initialized = True
        print("✓ Voice service initialized")

    def _init_stt(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            print("Warning: NeMo not installed; speech-to-text is unavailable.")
            return
        try:
            import torch

            path = resolve_path(self.config.get("stt_model_path", "model/parakeet-tdt-0.6b-v2.nemo"))
            if path.exists():
                print(f"Loading STT model from {path}...")
                self.stt_model = nemo_asr.models.ASRModel.restore_from(str(path))
            else:
                print(f"STT model not found at {path}, downloading nvidia/parakeet-tdt-0.6b-v2...")
                self.stt_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
            precision = "fp32"
            if torch.cuda.is_available():
                self.stt_model = self.stt_model.cuda()
                # bfloat16 halves Parakeet's VRAM (~2.6 GB -> ~1.6 GB reserved) with identical transcripts
                # in testing, for ~20 ms more per utterance. fp16 was slower, so it isn't offered.
                if str(self.config.get("stt_precision", "auto")).lower() != "fp32" and torch.cuda.is_bf16_supported():
                    self.stt_model = self.stt_model.to(torch.bfloat16)
                    precision = "bf16"
            self.stt_model.eval()
            print(f"✓ STT model loaded ({'cuda' if torch.cuda.is_available() else 'cpu'}, {precision})")
            # The first transcription pays a one-time GPU warm-up (~0.3 s); do it now, not on a real speaker.
            self._transcribe_sync(np.zeros(STT_SAMPLE_RATE, dtype=np.float32))
        except Exception as e:
            print(f"❌ Failed to initialize STT: {e}")
            self.stt_model = None

    def _init_tts(self) -> None:
        if str(self.config.get("tts_engine", "kokoro")).lower() == "qwen3":
            port = int(self.config.get("qwen_tts", {}).get("port", 8732))
            url = f"http://127.0.0.1:{port}"
            try:
                import json
                import urllib.request

                with urllib.request.urlopen(f"{url}/health", timeout=3) as response:
                    health = json.loads(response.read())
                self.qwen_url, self.emotions = url, list(health.get("emotions") or [])
                print(f"✓ TTS: Qwen3-TTS server on port {port} (emotions: {', '.join(self.emotions) or 'none'})")
                return
            except Exception as e:
                print(f"⚠️ Qwen3-TTS server not reachable on port {port} ({e}); using Kokoro")
        self._init_kokoro()

    def _init_kokoro(self) -> None:
        if self.tts_pipeline:
            return
        try:
            from kokoro import KPipeline
        except ImportError:
            print("Warning: Kokoro not installed; text-to-speech is unavailable.")
            return
        try:
            import torch

            # Kokoro-82M is small enough to run in real time on a CPU; "cpu" keeps VRAM for the LLM.
            device = str(self.config.get("tts_device", "auto")).lower()
            if device not in ("cpu", "cuda"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts_pipeline = KPipeline(
                lang_code=self.config.get("tts_lang_code", "a"),
                repo_id=self.config.get("tts_model", "hexgrad/Kokoro-82M"),
                device=device,
            )
            print(f"✓ TTS pipeline initialized (Kokoro-82M on {device})")
            list(self.tts_pipeline("Hi.", voice=self.config.get("tts_voice", "af_heart")))  # warm-up
        except Exception as e:
            print(f"❌ Failed to initialize TTS: {e}")
            self.tts_pipeline = None

    # ------------------------------------------------------------------
    # Speech to text
    # ------------------------------------------------------------------
    async def transcribe_audio(self, audio_data: bytes, sample_rate: int = 48000, channels: int = 2) -> Optional[str]:
        """Transcribe 16-bit PCM (Discord delivers 48 kHz stereo). Returns None for silence/noise."""
        if not self.stt_model or not audio_data:
            return None
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        if channels == 2:
            audio = audio[: len(audio) - len(audio) % 2].reshape(-1, 2).mean(axis=1)
        if sample_rate != STT_SAMPLE_RATE:
            from scipy.signal import resample_poly

            g = np.gcd(sample_rate, STT_SAMPLE_RATE)
            audio = resample_poly(audio, STT_SAMPLE_RATE // g, sample_rate // g).astype(np.float32)

        duration = len(audio) / STT_SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        min_rms = float(self.config.get("min_rms", 0.01))
        min_duration = float(self.config.get("min_speech_seconds", 0.3))
        if rms < min_rms or duration < min_duration:
            print(f"🔇 Ignoring audio ({duration:.2f}s, RMS {rms:.4f})")
            return None

        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(self._stt_executor, self._transcribe_sync, audio)
        except Exception as e:
            print(f"Transcription error: {e}")
            return None
        text = self._correct_names(text.strip()) if text else ""
        return text or None

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        import soundfile as sf

        fd, tmp_name = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_name, audio, STT_SAMPLE_RATE)
            result = self.stt_model.transcribe([tmp_name], batch_size=1, verbose=False)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        if isinstance(result, tuple):  # some NeMo versions return (best, all) hypotheses
            result = result[0]
        if not result:
            return ""
        first = result[0]
        return first if isinstance(first, str) else getattr(first, "text", str(first))

    # ------------------------------------------------------------------
    # Name correction (voice_corrections.yaml)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_names() -> List[str]:
        path = PROJECT_ROOT / "voice_corrections.yaml"
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return [str(n).strip() for n in data.get("entities", []) if str(n).strip()]
        except Exception as e:
            print(f"Warning: could not read voice_corrections.yaml: {e}")
            return []

    def add_names(self, names: List[str]) -> None:
        """Teach the name corrector more names (e.g. everyone in the voice channel)."""
        known = {n.lower() for n in self._names}
        for name in names:
            # Only the first word of a display name is what people actually say ("Sava", not "Sava | 21").
            spoken = re.split(r"[\s|_\-.\d]+", name.strip())[0]
            if len(spoken) >= 3 and spoken.lower() not in known:
                self._names.append(spoken)
                known.add(spoken.lower())

    @staticmethod
    def _load_aliases() -> Dict[str, str]:
        """voice_corrections.yaml 'aliases': {Name: [how the transcriber mishears it, ...]} -> {misheard: Name}."""
        path = PROJECT_ROOT / "voice_corrections.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {} if path.exists() else {}
        except Exception:
            return {}
        out: Dict[str, str] = {}
        for name, variants in (data.get("aliases") or {}).items():
            for variant in variants or []:
                out[str(variant).strip().lower()] = str(name)
        return out

    def _correct_names(self, text: str) -> str:
        """Snap capitalised words that sound like a known name onto the correct spelling."""
        for misheard, name in self._aliases.items():
            text = re.sub(rf"(?<![\w']){re.escape(misheard)}(?![\w'])", name, text, flags=re.IGNORECASE)
        if not self._names:
            return text
        threshold = float(self.config.get("name_correction_threshold", 0.8))
        lowered = {n.lower(): n for n in self._names}

        def fix(match: re.Match) -> str:
            word = match.group(0)
            if not word[0].isupper():
                return word
            exact = lowered.get(word.lower())
            if exact:
                return exact
            best, ratio = max(((n, SequenceMatcher(None, word.lower(), n.lower()).ratio()) for n in self._names),
                              key=lambda x: x[1])
            if ratio >= threshold:
                print(f"🔧 Name correction ({ratio:.0%}): '{word}' → '{best}'")
                return best
            return word

        return re.sub(r"[A-Za-z][A-Za-z']*", fix, text)

    # ------------------------------------------------------------------
    # Text to speech
    # ------------------------------------------------------------------
    async def synthesize_speech(self, text: str, emotion: Optional[str] = None) -> Optional[bytes]:
        """Text -> 16-bit mono PCM at 24 kHz, or None on failure."""
        chunks = [chunk async for chunk in self.stream_speech(text, emotion)]
        return b"".join(chunks) or None

    async def stream_speech(self, text: str, emotion: Optional[str] = None) -> AsyncIterator[bytes]:
        """
        Text -> 16-bit mono PCM at 24 kHz, in pieces as it's generated (Qwen3-TTS: the first ~0.3 s of
        audio after ~150 ms; Kokoro: all at once). Closing the iterator early stops generation.
        """
        if not text.strip():
            return
        if self.qwen_url:
            got_audio = False
            try:
                async for chunk in self._stream_qwen(text, emotion):
                    got_audio = True
                    yield chunk
                return
            except Exception as e:
                if got_audio:
                    print(f"TTS error (Qwen3-TTS, mid-sentence): {e}")
                    return
                print(f"⚠️ Qwen3-TTS failed ({e!r}); switching to Kokoro")
                self.qwen_url = None
                await asyncio.get_running_loop().run_in_executor(self._tts_executor, self._init_kokoro)
        pcm = await self._kokoro(text)
        if pcm:
            yield pcm

    async def _stream_qwen(self, text: str, emotion: Optional[str]) -> AsyncIterator[bytes]:
        import aiohttp

        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=30))
        body = {"text": text, "emotion": emotion if emotion in self.emotions else "neutral"}
        async with self._http.post(f"{self.qwen_url}/speak", json=body) as response:
            response.raise_for_status()
            carry = b""
            async for chunk in response.content.iter_any():
                chunk = carry + chunk
                cut = len(chunk) - len(chunk) % 2  # never split a 16-bit sample
                carry = chunk[cut:]
                if cut:
                    yield chunk[:cut]

    async def _kokoro(self, text: str) -> Optional[bytes]:
        if not self.tts_pipeline:
            return None
        voice = self.config.get("tts_voice", "af_heart")
        speed = float(self.config.get("tts_speed", 1.0))

        def run() -> Optional[bytes]:
            parts = [audio for _, _, audio in self.tts_pipeline(text, voice=voice, speed=speed) if audio is not None]
            if not parts:
                return None
            parts = [p.cpu().numpy() if hasattr(p, "cpu") else p for p in parts]
            return float_to_pcm16(np.concatenate(parts))

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._tts_executor, run)
        except Exception as e:
            print(f"TTS error: {e}")
            return None


_voice_service: Optional[VoiceService] = None
_init_lock: Optional[asyncio.Lock] = None


async def get_voice_service() -> VoiceService:
    """Get (and lazily initialise) the shared voice service."""
    global _voice_service, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _voice_service is None:
            service = VoiceService()
            await service.initialize()
            _voice_service = service
    return _voice_service
