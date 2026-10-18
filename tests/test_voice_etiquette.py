"""Timing etiquette in the Discord bot: waiting for typing, floor control and barge-in (fakes, no Discord)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import numpy as np
import scipy.signal  # noqa: F401  (imported up front so timings below measure behaviour, not imports)
import pytest

import backend.discord_bot as bot_mod


class FakeVoiceClient:
    """Plays 'audio' for its real duration; stop() ends it early, like discord's VoiceClient."""

    def __init__(self):
        self.channel = SimpleNamespace(id=4242, name="Lounge", voice_states={}, guild=None)
        self.stopped = False
        self._after = None
        self._task = None

    def is_connected(self):
        return True

    def is_listening(self):
        return False

    def play(self, source, after):
        self._after = after
        data = source.stream.getvalue() if hasattr(source, "stream") else b""
        duration = len(data) / (48000 * 2 * 2)
        loop = asyncio.get_running_loop()
        self._task = loop.call_later(duration, after, None)

    def stop(self):
        self.stopped = True
        if self._task:
            self._task.cancel()
        if self._after:
            self._after(None)


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(bot_mod, "etiquette", lambda voice: {
        "barge_in": True, "barge_in_seconds": 0.3, "floor_gap": 0.2, "max_floor_wait": 0.8, "barge_in_min_rms": 0.03,
    })
    r = bot_mod.VoiceReceiver(FakeVoiceClient(), text_channel=None)
    return r


def speak(r, user_id, seconds_ago_started, seconds_ago_last=0.0):
    now = time.monotonic()
    r._activity[user_id] = [now - seconds_ago_started, now - seconds_ago_last]


def pcm(seconds):
    return (np.zeros(int(24000 * seconds), dtype=np.int16)).tobytes()


async def test_someone_speaking(receiver):
    assert not receiver.someone_speaking()
    speak(receiver, 1, seconds_ago_started=0.5)
    assert receiver.someone_speaking()
    assert receiver.someone_speaking(for_at_least=0.4)
    assert not receiver.someone_speaking(for_at_least=1.0)
    speak(receiver, 1, seconds_ago_started=2.0, seconds_ago_last=1.0)  # stopped a second ago
    assert not receiver.someone_speaking()


async def test_loud_packets_are_tracked_as_speech(receiver):
    loud = (np.ones(960 * 2, dtype=np.int16) * 8000).tobytes()
    quiet = (np.ones(960 * 2, dtype=np.int16) * 10).tobytes()
    user = SimpleNamespace(id=7)
    receiver.buffer_audio(user, quiet)
    assert 7 not in receiver._activity
    receiver.buffer_audio(user, loud)
    assert receiver.someone_speaking()


async def test_waits_for_a_gap_before_speaking(receiver):
    seq = receiver._utterance_seq
    t0 = time.monotonic()
    assert await receiver.wait_for_floor(seq)
    assert time.monotonic() - t0 >= 0.2  # waited for the floor gap


async def test_gives_up_the_turn_if_people_keep_talking(receiver):
    async def keep_talking():
        for _ in range(30):
            speak(receiver, 1, seconds_ago_started=1.0)
            await asyncio.sleep(0.05)

    talker = asyncio.create_task(keep_talking())
    assert not await receiver.wait_for_floor(receiver._utterance_seq)
    talker.cancel()


async def test_gives_up_the_turn_when_someone_says_something_new(receiver):
    seq = receiver._utterance_seq
    receiver._utterance_seq += 1
    assert not await receiver.wait_for_floor(seq)


async def test_stops_talking_when_talked_over(receiver):
    async def interrupt():
        await asyncio.sleep(0.2)
        for _ in range(20):
            speak(receiver, 1, seconds_ago_started=0.5)
            await asyncio.sleep(0.05)

    task = asyncio.create_task(interrupt())
    t0 = time.monotonic()
    finished = await receiver._play_pcm(pcm(3.0))
    task.cancel()
    assert finished is False
    assert receiver.voice_client.stopped
    assert time.monotonic() - t0 < 1.5  # stopped well before the 3 s sentence ended


async def test_plays_to_the_end_when_nobody_interrupts(receiver):
    assert await receiver._play_pcm(pcm(0.3)) is True
    assert not receiver.voice_client.stopped


async def test_text_reply_waits_while_someone_types(monkeypatch):
    monkeypatch.setattr(bot_mod, "etiquette", lambda voice: {"reply_delay": 0.05, "max_typing_wait": 2})
    conv = bot_mod.Conversation(mode="text")
    conv.latest_id = 1
    bot_mod._typing[555] = {9: time.monotonic()}

    async def stop_typing_and_send():
        await asyncio.sleep(0.4)
        conv.latest_id = 2  # they sent a new message

    asyncio.create_task(stop_typing_and_send())
    assert await bot_mod.wait_until_quiet(555, 1, conv) is False  # newer message gets the turn instead
    bot_mod._typing.pop(555, None)


async def test_text_reply_goes_ahead_when_nobody_types(monkeypatch):
    monkeypatch.setattr(bot_mod, "etiquette", lambda voice: {"reply_delay": 0.05, "max_typing_wait": 2})
    conv = bot_mod.Conversation(mode="text")
    conv.latest_id = 1
    assert await bot_mod.wait_until_quiet(556, 1, conv) is True
