"""
Speech-to-text (NVIDIA Parakeet via NeMo) and text-to-speech (Kokoro).
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import yaml

from backend.config import PROJECT_ROOT, get_config, resolve_path

# NeMo and Kokoro are imported when voice is first used (>join), not at startup: importing them
# costs ~5.6 s and ~270 MB of RAM, which is wasted whenever nobody uses voice chat.

TTS_SAMPLE_RATE = 24000  # Kokoro output
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
        self.tts_pipeline: Optional[Any] = None
        self._initialized = False
        # One worker each: GPU models don't like concurrent calls, and a long TTS
        # job shouldn't delay transcription of the next speaker (or vice versa).
        self._stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="STT")
        self._tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="TTS")
        self._names: List[str] = self._load_names()

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

    def _correct_names(self, text: str) -> str:
        """Snap capitalised words that sound like a known name onto the correct spelling."""
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
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Text -> 16-bit mono PCM at 24 kHz, or None on failure."""
        if not self.tts_pipeline or not text.strip():
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
