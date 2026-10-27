import asyncio
import random

import numpy as np
import pytest

from backend import hesitation
from backend.hesitation import HesitationBank, drop_interjection, split_tag, trim, usable


def tone(seconds: float, silence: float = 0.0) -> bytes:
    pad = np.zeros(int(silence * hesitation.SAMPLE_RATE), dtype=np.int16)
    t = np.arange(int(seconds * hesitation.SAMPLE_RATE)) / hesitation.SAMPLE_RATE
    sound = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    return np.concatenate([pad, sound, pad]).tobytes()


class FakeService:
    def __init__(self, seconds: float = 0.6):
        self.config = {"tts_voice": "af_heart"}
        self.qwen_url = None
        self.seconds = seconds
        self.calls = []

    async def synthesize_speech(self, text, emotion=None):
        self.calls.append((text, emotion))
        return tone(self.seconds, silence=0.5)


@pytest.fixture
def settings(monkeypatch, tmp_path):
    cfg = {"enabled": True, "lines": ["Hmm...", "Uhh...", "[sleepy] Mmm..."], "takes": 2,
           "after": [0.2, 0.6], "chance": 1.0, "cooldown": 0}
    monkeypatch.setattr(hesitation, "settings", lambda: cfg)
    monkeypatch.setattr(hesitation, "CACHE_DIR", tmp_path)
    return cfg


def test_silence_around_the_sound_is_cut():
    clip = trim(tone(0.5, silence=0.6))
    assert 0.5 <= len(clip) / 2 / hesitation.SAMPLE_RATE < 0.65
    assert trim(b"\0\0" * 24000) == b""
    assert not usable(trim(tone(4.0)))  # the voice model rambling on is not a "hmm"


def test_tags_and_doubled_interjections():
    assert split_tag("[sleepy] Mmm...") == ("sleepy", "Mmm...")
    assert split_tag("Hmm.") == (None, "Hmm.")
    assert drop_interjection("Hmm, let me look that up.") == "Let me look that up."
    assert drop_interjection("Mmm, one sec.") == "One sec."
    assert drop_interjection("One sec, I'll check.") == "One sec, I'll check."
    assert drop_interjection("Hmm...") == "Hmm..."


def test_takes_are_made_once_then_loaded_from_disk(settings):
    async def run():
        service = FakeService()
        bank = HesitationBank()
        await bank.prepare(service)
        assert len(service.calls) == 6 and ("Mmm...", "sleepy") in service.calls
        assert sum(len(v) for v in bank.clips.values()) == 6
        again = FakeService()
        await HesitationBank().prepare(again)
        assert again.calls == []
    asyncio.run(run())


def test_unusable_takes_are_dropped_and_not_retried(settings):
    async def run():
        service = FakeService(seconds=5.0)
        bank = HesitationBank()
        await bank.prepare(service)
        assert bank.clips == {}
        assert bank.pick() is None
        again = FakeService(seconds=5.0)
        await HesitationBank().prepare(again)
        assert again.calls == []
    asyncio.run(run())


def test_variation_and_restraint(settings):
    bank = HesitationBank()
    bank.clips = {line: [tone(0.5)] for line in settings["lines"]}
    rng = random.Random(1)
    said = [bank.pick(rng)[0] for _ in range(12)]
    assert all(a != b for a, b in zip(said, said[1:]))  # never the same sound twice in a row
    assert len(set(said)) == 3

    settings["cooldown"], bank.last_said = 60, 0.0
    assert bank.pick(rng) is not None
    assert bank.pick(rng) is None  # not again right away

    settings["cooldown"], settings["chance"] = 0, 0.5
    picks = [HesitationBank.pick(bank, rng) for _ in range(200)]
    assert 60 < sum(p is not None for p in picks) < 140  # only some of the time

    settings["enabled"] = False
    assert bank.pick(rng) is None


def test_delay_varies_within_the_range(settings):
    rng = random.Random(2)
    delays = [HesitationBank().delay(rng) for _ in range(50)]
    assert all(0.2 <= d <= 0.6 for d in delays) and len({round(d, 2) for d in delays}) > 10
