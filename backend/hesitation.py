"""
Hesitation sounds ("hmm...", "uhh...") for the second or two between someone finishing and Metis's first
word. People make them too: a filled pause says "I heard you, I'm thinking", where silence says nothing.

Each line is synthesized a few times in her own voice when she joins a call (cached on disk afterwards,
so only the first join pays for it). Every take comes out a little different, which together with the
choice of line keeps them from sounding canned. They aren't said every time: only when the reply is
actually late, only some of the time, never twice in a row and not again for a while (voice.hesitations).
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import PROJECT_ROOT, get_config

SAMPLE_RATE = 24000  # 16-bit mono PCM, as the voice service makes it
CACHE_DIR = PROJECT_ROOT / ".cache" / "hesitations"

DEFAULT_LINES = [
    "Hmm.", "Hmm...", "Hmmm...", "Mmm...", "Uhh...", "Umm...", "Ehh...", "Oh, hmm.", "Hm, okay...",
    "Uh, well...", "Ooh, hmm.", "Mm, let's see...", "[sleepy] Mmm...", "[teasing] Hmm...",
]

# A leading interjection, so a tool line that follows a hesitation doesn't say it twice
# ("Hmm... Hmm, let me look that up.").
_INTERJECTION = re.compile(r"^\s*(?:h?m+|hm+|uh+|um+|e+h|oo+h|oh)\b[.,!…\s]*", re.I)


def settings() -> Dict[str, Any]:
    cfg = get_config().get("voice", {}).get("hesitations") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "lines": list(cfg.get("lines") or DEFAULT_LINES),
        "takes": max(1, int(cfg.get("takes", 2))),
        # Said only when no part of the reply is ready this long after their pause was noticed (which itself
        # takes voice.listen_timeout, ~1 s of silence: more on top of that already felt like a wait)...
        "after": [float(x) for x in (cfg.get("after") or [0.2, 0.6])],
        # ...and then only this often, and not again for this many seconds.
        "chance": float(cfg.get("chance", 0.65)),
        "cooldown": float(cfg.get("cooldown", 15)),
    }


def drop_interjection(line: str) -> str:
    """'Hmm, let me look that up.' -> 'Let me look that up.' (unchanged if nothing would be left)."""
    rest = _INTERJECTION.sub("", line, count=1)
    if rest == line or len(rest.strip()) < 3:
        return line
    return rest[0].upper() + rest[1:]


def split_tag(line: str) -> Tuple[Optional[str], str]:
    """'[sleepy] Mmm...' -> ('sleepy', 'Mmm...')."""
    match = re.match(r"\s*\[(\w+)\]\s*(.*)", line)
    return (match.group(1).lower(), match.group(2)) if match else (None, line.strip())


def trim(pcm: bytes, floor: int = 400, pad: float = 0.04, fade: float = 0.03) -> bytes:
    """Cut the silence before and after the sound (it would add to the wait), and fade out the end."""
    samples = np.frombuffer(pcm, dtype=np.int16)
    loud = np.flatnonzero(np.abs(samples.astype(np.int32)) > floor)
    if not len(loud):
        return b""
    margin = int(pad * SAMPLE_RATE)
    clip = samples[max(0, loud[0] - margin):loud[-1] + margin].astype(np.float32)
    n = min(len(clip), int(fade * SAMPLE_RATE))
    if n:
        clip[-n:] *= np.linspace(1.0, 0.0, n)
    return clip.astype(np.int16).tobytes()


def usable(pcm: bytes, shortest: float = 0.2, longest: float = 1.8) -> bool:
    """A take that is a sound, not a click or a rambling hallucination of the voice model."""
    return shortest <= len(pcm) / 2 / SAMPLE_RATE <= longest


class HesitationBank:
    def __init__(self) -> None:
        self.clips: Dict[str, List[bytes]] = {}  # line -> takes
        self.voice: Optional[str] = None  # which voice they were made with
        self.recent: deque = deque(maxlen=4)  # lines said lately: not again soon
        self.last_said = 0.0
        self.why_not = ""  # why the last pick() said nothing, for the log
        self._preparing: Optional[asyncio.Task] = None

    @staticmethod
    def voice_of(service: Any) -> str:
        """What the clips depend on: a different voice model or reference clip needs new ones."""
        cfg = service.config
        if service.qwen_url:
            q = cfg.get("qwen_tts", {})
            return f"qwen3|{q.get('model')}|{q.get('ref_audio')}|{sorted((q.get('emotions') or {}).items())}"
        return f"kokoro|{cfg.get('tts_voice', 'af_heart')}|{cfg.get('tts_speed', 1.0)}"

    def prepare(self, service: Any) -> Optional[asyncio.Task]:
        """Make (or load) the clips in the background; she simply doesn't hesitate until they're ready."""
        if not settings()["enabled"]:
            return None
        voice = self.voice_of(service)
        if self.voice == voice and self.clips:
            return None
        if self._preparing and not self._preparing.done():
            return self._preparing
        self._preparing = asyncio.create_task(self._prepare(service, voice))
        return self._preparing

    async def _prepare(self, service: Any, voice: str) -> None:
        cfg = settings()
        clips: Dict[str, List[bytes]] = {}
        made = 0
        for line in cfg["lines"]:
            emotion, text = split_tag(line)
            for take in range(cfg["takes"]):
                key = hashlib.sha1(f"{voice}|{emotion}|{text}|{take}".encode()).hexdigest()[:20]
                path = CACHE_DIR / f"{key}.pcm"
                pcm = path.read_bytes() if path.exists() else None
                if pcm is None:
                    try:
                        pcm = trim(await service.synthesize_speech(text, emotion) or b"")
                    except Exception as e:
                        print(f"⚠️ Hesitation '{line}' could not be made: {e}")
                        continue
                    made += 1
                    if not usable(pcm):
                        pcm = b""  # remembered as unusable, so it isn't made again every join
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(pcm)
                if usable(pcm):
                    clips.setdefault(line, []).append(pcm)
        self.clips, self.voice = clips, voice
        takes = sum(len(v) for v in clips.values())
        print(f"✓ Hesitations: {takes} takes of {len(clips)} sounds ready ({made} newly made)")

    def pick(self, rng: random.Random = random) -> Optional[Tuple[str, bytes]]:
        """A sound to make now, or None: not ready, not this time, or one was said too recently."""
        cfg = settings()
        if not cfg["enabled"] or not self.clips:
            self.why_not = "off" if not cfg["enabled"] else "sounds not made yet"
            return None
        if time.monotonic() - self.last_said < cfg["cooldown"]:
            self.why_not = f"said one {time.monotonic() - self.last_said:.0f}s ago"
            return None
        if rng.random() >= cfg["chance"]:
            self.why_not = "not every time"
            return None
        keep_out = list(self.recent)[len(self.recent) - min(len(self.recent), len(self.clips) - 1):]
        lines = [line for line in self.clips if line not in keep_out] or list(self.clips)
        line = rng.choice(lines)
        self.recent.append(line)
        self.last_said = time.monotonic()
        return split_tag(line)[1], rng.choice(self.clips[line])

    def delay(self, rng: random.Random = random) -> float:
        """How long after they stopped talking she'd start one: a little different every time."""
        low, high = (settings()["after"] + [0.6])[:2]
        return rng.uniform(low, max(low, high))


bank = HesitationBank()
