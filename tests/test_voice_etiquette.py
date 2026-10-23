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
    """Pulls 20 ms frames from the source in real time until it ends; stop() ends it early, like discord's VoiceClient."""

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
        self.frames = []

        self.paused = False
        self.pauses = 0

        async def player():
            while True:
                if self.paused:
                    await asyncio.sleep(0.02)
                    continue
                frame = source.read()
                if not frame:
                    break
                self.frames.append(frame)
                await asyncio.sleep(0.02)
            after(None)

        self._task = asyncio.get_running_loop().create_task(player())

    def pause(self):
        self.paused = True
        self.pauses += 1

    def resume(self):
        self.paused = False

    def stop(self):
        self.stopped = True
        if self._task:
            self._task.cancel()
        if self._after:
            self._after(None)


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(bot_mod, "etiquette", lambda voice: {
        "barge_in": True, "barge_in_seconds": 0.3, "barge_in_confirm": 0.8, "floor_gap": 0.2, "max_floor_wait": 0.8, "barge_in_min_rms": 0.03,
    })
    r = bot_mod.VoiceReceiver(FakeVoiceClient(), text_channel=None)
    return r


def speak(r, user_id, seconds_ago_started, seconds_ago_last=0.0):
    now = time.monotonic()
    r._activity[user_id] = [now - seconds_ago_started, now - seconds_ago_last]


def pcm(seconds):
    return (np.zeros(int(24000 * seconds), dtype=np.int16)).tobytes()


def speech(seconds):
    s = bot_mod.SpeechStream()
    s.feed(pcm(seconds))
    s.finish()
    return s


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
    speak(receiver, 1, seconds_ago_started=1.0, seconds_ago_last=0.05)  # just stopped talking
    t0 = time.monotonic()
    assert await receiver.wait_for_floor(seq)
    assert 0.1 <= time.monotonic() - t0 < 0.4  # waited out the rest of the 0.2 s gap


async def test_no_wait_when_everyone_is_already_quiet(receiver):
    """Counting the gap from the start of the wait added half a second to every reply."""
    speak(receiver, 1, seconds_ago_started=3.0, seconds_ago_last=2.0)
    t0 = time.monotonic()
    assert await receiver.wait_for_floor(receiver._utterance_seq)
    assert time.monotonic() - t0 < 0.1


async def test_gives_up_the_turn_if_people_keep_talking(receiver):
    async def keep_talking():
        for _ in range(30):
            speak(receiver, 1, seconds_ago_started=1.0)
            await asyncio.sleep(0.05)

    speak(receiver, 1, seconds_ago_started=1.0)
    talker = asyncio.create_task(keep_talking())
    assert not await receiver.wait_for_floor(receiver._utterance_seq)
    talker.cancel()


def _said(r, user_id, text):
    r._utterance_seq += 1
    r._said[r._utterance_seq] = (user_id, text)
    return r._utterance_seq


async def test_gives_up_the_turn_when_the_same_person_says_more(receiver):
    seq = _said(receiver, 1, "what's the best jungler")
    _said(receiver, 1, "actually for emerald plus")
    assert not await receiver.wait_for_floor(seq)


async def test_gives_up_the_turn_when_someone_talks_to_metis(receiver):
    seq = _said(receiver, 1, "what's the best jungler")
    _said(receiver, 2, "Metis, wait, answer me first")
    assert not await receiver.wait_for_floor(seq)


async def test_keeps_the_turn_when_someone_else_remarks(receiver):
    """ "Which is true." from the other person dropped her answer; in a group she rarely got a word in."""
    seq = _said(receiver, 1, "aatrox is useless right")
    _said(receiver, 2, "Which is true.")
    assert await receiver.wait_for_floor(seq)


async def test_stops_talking_when_talked_over(receiver):
    async def interrupt():
        await asyncio.sleep(0.2)
        start = time.monotonic()
        for _ in range(40):
            receiver._activity[1] = [start, time.monotonic()]  # one long stretch of talking
            await asyncio.sleep(0.05)

    task = asyncio.create_task(interrupt())
    t0 = time.monotonic()
    finished = await receiver._play(speech(4.0))
    task.cancel()
    assert finished is False
    assert receiver.voice_client.stopped
    assert time.monotonic() - t0 < 2.5  # stopped well before the 4 s sentence ended


async def test_short_reaction_pauses_then_carries_on(receiver):
    """A laugh or "yeah" while she talks stopped her answer for good; now she pauses and carries on."""
    async def laugh():
        await asyncio.sleep(0.2)
        start = time.monotonic()
        for _ in range(8):  # 0.4 s of "ha ha"
            receiver._activity[1] = [start, time.monotonic()]
            await asyncio.sleep(0.05)

    task = asyncio.create_task(laugh())
    finished = await receiver._play(speech(1.5))
    task.cancel()
    assert finished is True
    assert receiver.voice_client.pauses == 1 and not receiver.voice_client.stopped


async def test_plays_to_the_end_when_nobody_interrupts(receiver):
    assert await receiver._play(speech(0.3)) is True
    assert not receiver.voice_client.stopped


async def test_plays_speech_while_it_is_still_being_generated(receiver):
    stream = bot_mod.SpeechStream()
    stream.feed(pcm(0.1))

    async def generate_the_rest():
        for _ in range(3):
            await asyncio.sleep(0.1)
            stream.feed(pcm(0.1))
        stream.finish()

    task = asyncio.create_task(generate_the_rest())
    assert await receiver._play(stream) is True  # didn't end when the first piece ran out
    await task
    frames = receiver.voice_client.frames
    assert len(frames) >= 20 and all(len(f) == bot_mod.SpeechStream.FRAME for f in frames)


def test_speech_stream_upsamples_without_clicks_between_pieces():
    tone = (np.sin(2 * np.pi * 440 * np.arange(4800) / 24000) * 10000).astype(np.int16)
    whole, pieces = bot_mod.SpeechStream(), bot_mod.SpeechStream()
    whole.feed(tone.tobytes())
    for part in np.array_split(tone, 7):
        pieces.feed(part.tobytes())
    for s in (whole, pieces):
        s.finish()

    def drain(s):
        out = b""
        while frame := s.read():
            out += frame
        return np.frombuffer(out, dtype=np.int16)

    a, b = drain(whole), drain(pieces)
    assert len(a) == len(b) and np.abs(a.astype(int) - b.astype(int)).max() <= 1  # same as one piece
    left = a[0::2].astype(float)
    assert np.abs(left).max() > 9000  # 48 kHz version keeps the tone's level


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


async def test_filler_is_not_interrupted(receiver):
    """okeida talking over "Hold on..." cancelled the answer that had just been found."""
    async def talk():
        start = time.monotonic()
        for _ in range(30):
            receiver._activity[1] = [start, time.monotonic()]
            await asyncio.sleep(0.05)

    task = asyncio.create_task(talk())
    finished = await receiver._play(speech(1.0), can_interrupt=False)
    task.cancel()
    assert finished is True and receiver.voice_client.pauses == 0


async def test_side_comment_from_someone_else_needs_longer(receiver):
    """Talking over her answer to someone else stops her only after twice as long."""
    async def comment():
        await asyncio.sleep(0.1)
        start = time.monotonic()
        for _ in range(22):  # ~1.1 s: past barge_in_confirm (0.8) but short of 2x
            receiver._activity[2] = [start, time.monotonic()]
            await asyncio.sleep(0.05)

    task = asyncio.create_task(comment())
    finished = await receiver._play(speech(2.5), to_user=1)
    task.cancel()
    assert finished is True and receiver.voice_client.pauses == 1



def test_reply_goes_right_after_the_line_it_answers():
    """Appended at the end, a line said while she was talking ended up before her reply; the next turn was
    refused ("the last message must be from the user") and 14 lines went unanswered in one call."""
    conv = bot_mod.Conversation(mode="voice")
    a = conv.add_user(SimpleNamespace(id=1, name="A", display_name="A", bot=False), "what's up Metis")
    conv.add_user(SimpleNamespace(id=2, name="B", display_name="B", bot=False), "said while she talked")
    conv.add_assistant("Not much!", to=1, after=a)
    assert [m["role"] for m in conv.messages] == ["user", "assistant", "user"]
    assert conv.prompt()[-1]["role"] == "user"
