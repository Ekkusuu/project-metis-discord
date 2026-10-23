"""
Discord front-end for Metis.

Text channels: conversational etiquette (backend/turn_taking.py) decides whether
a message is for Metis; she waits for people to finish typing, then the reply is
streamed from the backend's /chat/stream endpoint.

Voice channels: per-user audio is buffered until the speaker pauses, transcribed,
judged, answered by the LLM, and spoken sentence-by-sentence while the rest of
the reply is still being generated.

The bot runs on its own event loop in a background thread of the backend
process (see start_in_background) and talks to the backend over HTTP so the
dashboard sees exactly the same traffic.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import aiohttp
import discord
import discord.ext.voice_recv as voice_recv
import numpy as np
from discord.ext import commands
from dotenv import load_dotenv

from backend import chat_log
from backend.config import PROJECT_ROOT, backend_url, get_config, llm_service_url
from backend.prompt_loader import resolve_prompt
from backend.text_utils import (
    ThinkFilter,
    clean_for_speech,
    pop_sentences,
    split_message,
    strip_name_prefix,
    strip_thinking,
    take_emotion,
)
from backend.turn_taking import (
    ADDRESSEE_CHOICES, ADDRESSEE_PROMPT, ASK, RESPOND, WAIT_OPEN, Settings, Turn, decide, transcript_for_model,
    verdict_from_addressee, LEAVE_CHOICES, LEAVE_PROMPT, clearly_asks_to_leave, is_acknowledgement, might_ask_to_leave,
)
from backend.voice_service import TTS_SAMPLE_RATE, get_voice_service
from backend import voice_compat

voice_compat.apply()  # survive frames that DAVE (E2EE voice) leaves undecodable

load_dotenv(PROJECT_ROOT / ".env")

config = get_config()
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.voice_states = True
bot = commands.Bot(command_prefix=[">", "!"], intents=intents, help_command=commands.DefaultHelpCommand())

_http: Optional[aiohttp.ClientSession] = None
_loop: Optional[asyncio.AbstractEventLoop] = None
_thread: Optional[threading.Thread] = None


def http() -> aiohttp.ClientSession:
    """One shared HTTP session for the bot (created on the bot's event loop)."""
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=300))
    return _http


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------
@dataclass
class Conversation:
    """
    History for one text channel or one voice session.

    User messages carry the speaker's Discord id and label, so every line in the
    prompt is attributable to a specific person and learning can file facts
    under the right profile.
    """

    mode: str  # "text" | "voice"
    messages: List[Dict[str, Any]] = field(default_factory=list)
    latest_id: int = 0  # id of the newest user message; older in-flight replies abort
    chiming_in: bool = False  # the reply being made joins a conversation between other people
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def system_prompt(self) -> str:
        cfg = config.get("voice" if self.mode == "voice" else "chat", {})
        default = "@voice_chat" if self.mode == "voice" else "@text_chat"
        return resolve_prompt(cfg.get("system_prompt", default))

    def prompt(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt},
                *({"role": m["role"], "content": m["content"]} for m in self.messages)]

    # -- speakers ---------------------------------------------------------
    def label_for(self, user: discord.abc.User) -> str:
        """
        The name used for this person in the transcript: their server display name,
        disambiguated with the username when someone else here goes by the same name.
        """
        base = getattr(user, "display_name", None) or user.name
        clash = any(
            m.get("user_id") not in (None, user.id) and m.get("base_name", "").lower() == base.lower()
            for m in self.messages[-60:]
        )
        return f"{base} ({user.name})" if clash and user.name.lower() != base.lower() else base

    def add_user(self, user: discord.abc.User, text: str, mentions: Tuple[int, ...] = (),
                 reply_to: Optional[int] = None) -> Dict[str, Any]:
        label = self.label_for(user)
        msg = {
            "role": "user",
            "content": f"[{label}]: {text}",
            "text": text,
            "user_id": user.id,
            "name": label,
            "base_name": getattr(user, "display_name", None) or user.name,
            "mentions": tuple(mentions),
            "reply_to": reply_to,
            "ts": time.time(),
            "learned": False,
        }
        self.messages.append(msg)
        return msg

    def add_assistant(self, text: str, to: Optional[int] = None) -> None:
        self.messages.append({"role": "assistant", "content": text, "to": to, "ts": time.time(), "learned": False})

    def turns(self, limit: int = 30) -> List[Turn]:
        """The recent conversation in the form the etiquette logic understands."""
        out = []
        for m in self.messages[-limit:]:
            if m["role"] == "assistant":
                out.append(Turn(None, "Metis", m["content"], m.get("ts", 0.0), to=m.get("to")))
            else:
                out.append(Turn(m.get("user_id"), m.get("name", "?"), m.get("text", m["content"]), m.get("ts", 0.0),
                                tuple(m.get("mentions", ())), m.get("reply_to")))
        return out

    def names_by_id(self) -> Dict[int, str]:
        return {m["user_id"]: m["name"] for m in self.messages if m.get("user_id") is not None}

    def participants(self, limit: int = 30) -> List[Dict[str, str]]:
        """Distinct recent speakers, most recent first."""
        seen: Dict[int, str] = {}
        for m in reversed(self.messages[-limit:]):
            if m.get("user_id") is not None and m["user_id"] not in seen:
                seen[m["user_id"]] = m["name"]
        return [{"id": str(uid), "name": name} for uid, name in seen.items()]

    def name_to_id(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for m in self.messages:
            if m.get("user_id") is not None:
                mapping[m["name"]] = m["user_id"]
                mapping.setdefault(m.get("base_name", m["name"]), m["user_id"])
        return mapping

    def distinct_recent_speakers(self, last: int = 6) -> int:
        return len({m["user_id"] for m in self.messages[-last:] if m.get("user_id") is not None})

    # -- learning ---------------------------------------------------------
    def take_unlearned(self, context: int = 2) -> List[Dict[str, Any]]:
        """Messages not yet learned from (plus a little prior context); marks them learned."""
        first = next((i for i, m in enumerate(self.messages) if not m.get("learned")), None)
        if first is None:
            return []
        batch = [dict(m) for m in self.messages[max(0, first - context):]]
        for m in self.messages[first:]:
            m["learned"] = True
        return batch if any(m["role"] == "user" for m in batch) else []

    def maybe_learn(self) -> None:
        """Learn every `memory.learn_every_messages` user messages."""
        every = int(config.get("memory", {}).get("learn_every_messages", 6))
        unlearned = sum(1 for m in self.messages if m["role"] == "user" and not m.get("learned"))
        if every > 0 and unlearned >= every:
            from backend.memory_manager import learn_from
            learn_from(self.take_unlearned())

    async def compact(self) -> None:
        """
        When the history outgrows the context window, learn from anything not yet
        learned, then cut it back to ~70% of the budget in one go. Cutting with
        headroom keeps the prompt prefix stable for many turns, which is what lets
        the LLM service reuse its KV cache.
        """
        from backend.context_manager import context_budget, split_to_fit
        from backend.memory_manager import learn_from
        from backend.token_utils import count_message_tokens

        async with self.lock:
            snapshot = self.prompt()
            budget = context_budget() - 1536  # leave room for people memory + tool context
            if await asyncio.to_thread(count_message_tokens, snapshot) <= budget:
                return
            _, removed = await asyncio.to_thread(split_to_fit, snapshot, int(budget * 0.7))
            if removed:
                pending = self.take_unlearned()
                if pending:
                    learn_from(pending)
                del self.messages[: len(removed)]
                print(f"🗜️  Compacted {self.mode} conversation: dropped {len(removed)} old message(s)")

    def reset(self) -> int:
        from backend.memory_manager import learn_from

        count = len(self.messages)
        pending = self.take_unlearned()
        if pending:
            learn_from(pending)
        self.messages = []
        return count


conversations: Dict[Tuple[str, int], Conversation] = {}


def flush_learning() -> List[Any]:
    """Queue learning for everything not learned yet in all conversations (callable from any thread)."""
    async def collect() -> List[Any]:
        from backend.memory_manager import learn_from
        futures = []
        for conversation in list(conversations.values()):
            pending = conversation.take_unlearned()
            if pending:
                futures.append(learn_from(pending))
        return futures

    if _loop and _loop.is_running():
        try:
            return asyncio.run_coroutine_threadsafe(collect(), _loop).result(timeout=10)
        except Exception as e:
            print(f"Could not flush learning: {e}")
    return []


def remember_speaker(user: discord.abc.User) -> None:
    """Update the person registry (names, last seen) for whoever just spoke."""
    from backend.people import get_people

    aliases = [n for n in (user.name, getattr(user, "global_name", None)) if n]
    display = getattr(user, "display_name", None) or user.name
    get_people().seen(user.id, display, aliases)


def link_mentions(text: str, conversation: Conversation) -> str:
    """Turn '@Name' for people in this conversation into real Discord mentions."""
    for name, uid in sorted(conversation.name_to_id().items(), key=lambda kv: -len(kv[0])):
        text = re.sub(rf"@{re.escape(name)}(?!\w)", f"<@{uid}>", text, flags=re.IGNORECASE)
    return text


def get_conversation(channel_id: int, mode: str = "text") -> Conversation:
    key = (mode, channel_id)
    if key not in conversations:
        conversations[key] = Conversation(mode=mode)
    return conversations[key]


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------
def etiquette(voice: bool) -> Dict[str, Any]:
    return config.get("voice" if voice else "chat", {}).get("etiquette", {}) or {}


def bot_names() -> List[str]:
    names = {"Metis"}
    if bot.user:
        names.update({bot.user.name, bot.user.display_name})
    names.update(config.get("chat", {}).get("bot_names", []) or [])
    return [n for n in names if n]


async def ask_addressee(conversation: Conversation) -> str:
    """One constrained model call: who is the last message meant for? (ambiguous cases only)"""
    payload = {
        "messages": [
            {"role": "system", "content": ADDRESSEE_PROMPT},
            {"role": "user", "content": "Chat:\n" + transcript_for_model(conversation.turns(), conversation.names_by_id())
                                        + "\n\nWho is the LAST message meant for?"},
        ],
        "json_schema": {"enum": ADDRESSEE_CHOICES},
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        async with http().post(f"{llm_service_url()}/chat/completion", json=payload,
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            content = (await response.json())["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        print(f"Addressee check failed ({e}); staying quiet")
        return "nobody in particular"


async def asked_to_leave(conversation: Conversation, alone_with_metis: bool = False) -> bool:
    """Is the latest (voice) message someone telling Metis to leave the call?"""
    if not config.get("voice", {}).get("leave_when_asked", True):
        return False
    turns = conversation.turns()
    if not turns or not might_ask_to_leave(turns[-1].text):
        return False
    if clearly_asks_to_leave(turns[-1].text, alone_with_metis):
        print("🚪 leave check: a plain command to leave")
        return True
    payload = {
        "messages": [
            {"role": "system", "content": LEAVE_PROMPT},
            {"role": "user", "content": "Voice chat:\n" + transcript_for_model(turns, conversation.names_by_id(), 6)
                                        + "\n\nShould Metis leave the call now?"},
        ],
        "json_schema": {"enum": LEAVE_CHOICES},
        "temperature": 0,
        "max_tokens": 16,
    }
    try:
        async with http().post(f"{llm_service_url()}/chat/completion", json=payload,
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            choice = json.loads((await response.json())["choices"][0]["message"]["content"])
    except Exception as e:
        print(f"Leave check failed ({e}); staying")
        return False
    print(f"🚪 leave check: {choice}")
    return choice == LEAVE_CHOICES[0]


async def is_metis_turn(conversation: Conversation, *, is_dm: bool = False, voice: bool = False,
                        alone_with_metis: bool = False) -> bool:
    """
    Conversational etiquette (backend/turn_taking.py): respond only when the message is for
    Metis, or when nobody else picks up a question asked to the whole room.
    """
    cfg = etiquette(voice)
    settings = Settings(
        engagement=str(cfg.get("engagement", "balanced")),
        continuation_window=float(cfg.get("continuation_window", 30 if voice else 180)),
        voice=voice,
        chime_in_after=float(cfg.get("chime_in_after", 60 if voice else 0)),
    )
    if settings.engagement == "always":
        return True
    turns = conversation.turns()
    decision = decide(turns, bot.user.id if bot.user else 0, bot_names(), conversation.names_by_id(),
                      is_dm=is_dm, alone_with_metis=alone_with_metis, settings=settings)
    if decision.verdict == ASK:
        addressee = await ask_addressee(conversation)
        decision = verdict_from_addressee(addressee, settings)
    conversation.chiming_in = bool(decision.signals.get("chime"))
    icon = {RESPOND: "✅", WAIT_OPEN: "⏳"}.get(decision.verdict, "🤐")
    print(f"{icon} [{turns[-1].name}] {decision.verdict}: {decision.reason}")

    if decision.verdict == WAIT_OPEN:
        # A question to the room: give the humans a moment to answer it first.
        asker, seen = turns[-1].speaker_id, len(conversation.messages)
        if decision.signals.get("wait") == "lone":
            wait = float(cfg.get("lone_question_wait", 1.5 if voice else 3))
        else:
            wait = float(cfg.get("open_question_wait", 3 if voice else 8))
        await asyncio.sleep(wait)
        if any(m.get("user_id") not in (None, asker) for m in conversation.messages[seen:]):
            print("🤐 someone else picked up the question")
            return False
        return True
    return decision.verdict == RESPOND


# Typing indicators per channel: channel id -> {user id: last time they were seen typing}
_typing: Dict[int, Dict[int, float]] = {}


@bot.event
async def on_typing(channel: discord.abc.Messageable, user: discord.abc.User, when: datetime) -> None:
    if not user.bot:
        _typing.setdefault(channel.id, {})[user.id] = time.monotonic()


async def wait_until_quiet(channel_id: int, message_id: int, conversation: Conversation) -> bool:
    """
    Don't talk over people: wait a beat (people often send several messages in a row), then
    while anyone in the channel is still typing. Returns False if a newer message arrived.
    """
    cfg = etiquette(False)
    await asyncio.sleep(float(cfg.get("reply_delay", 1.0)))
    deadline = time.monotonic() + float(cfg.get("max_typing_wait", 12))
    while time.monotonic() < deadline:
        if conversation.latest_id != message_id:
            return False
        typing = _typing.get(channel_id, {})
        if not any(time.monotonic() - ts < 6 for ts in typing.values()):
            break
        await asyncio.sleep(0.5)
    return conversation.latest_id == message_id


async def stream_reply(payload: Dict[str, Any], stats: Dict[str, Any],
                       on_thinking: Optional[Callable[[str], Awaitable[None]]] = None) -> AsyncIterator[str]:
    """
    Yield reply text deltas from the backend. The final stats chunk is stored in `stats`.
    on_thinking(tool) is awaited when a slow tool (remembering, a web search) starts, before the reply.
    """
    async with http().post(f"{backend_url()}/chat/stream", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"backend returned {response.status}: {(await response.text())[:300]}")
        async for raw in response.content:
            line = raw.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            if chunk.get("thinking") and on_thinking:
                await on_thinking(chunk["thinking"])
            if chunk.get("delta"):
                yield chunk["delta"]
            if chunk.get("done"):
                stats.update(chunk)
                return


def chat_payload(conversation: Conversation, voice_mode: bool, present: Optional[List[str]] = None) -> Dict[str, Any]:
    chat_cfg = config.get("chat", {})
    voice_cfg = config.get("voice", {})
    payload: Dict[str, Any] = {
        "messages": conversation.prompt(),
        "participants": conversation.participants(),
        "present": present or [],
        "temperature": chat_cfg.get("temperature", 0.7),
        "top_p": chat_cfg.get("top_p", 0.95),
        "max_tokens": (voice_cfg if voice_mode else chat_cfg).get("max_tokens", chat_cfg.get("max_tokens", 512)),
        "voice_mode": voice_mode,
        "chiming_in": conversation.chiming_in,
    }
    if voice_mode:
        payload.update({
            "max_tools": 1,
            "rag_query_count": 1,
            "rag_system_prompt": "@rag.query_generation_voice_system",
            "rag_user_prompt": "@rag.query_generation_voice_user",
        })
    return payload


async def publish(entry: Dict[str, Any]) -> None:
    """Send a message to the dashboard feed and the session log."""
    try:
        async with http().post(f"{backend_url()}/discord/add-message", json=entry,
                               timeout=aiohttp.ClientTimeout(total=5)) as response:
            await response.read()
    except Exception as e:
        print(f"Failed to publish message to dashboard: {e}")
    await asyncio.to_thread(chat_log.append, entry)


def clean_reply(text: str) -> str:
    text = strip_name_prefix(strip_thinking(text)).strip()
    # Small models sometimes echo the transcript format ("[sava] ...") at the start.
    text = re.sub(r"^\[[^\]\n]{1,40}\]:?\s*", "", text)
    if "```" not in text:
        text = re.sub(r"\n{2,}", "\n", text)  # keep Discord replies compact
    return text


# ---------------------------------------------------------------------------
# Text chat
# ---------------------------------------------------------------------------
def describe_message(message: discord.Message) -> str:
    """Message text as the model should see it: mentions resolved, attachments noted."""
    text = message.clean_content.strip()
    extras = [f"[attachment: {a.filename}]" for a in message.attachments]
    extras += ["[sticker]" for _ in message.stickers]
    return " ".join([text, *extras]).strip()


def reply_target(message: discord.Message) -> Optional[int]:
    """Author id of the message this one replies to, if any."""
    ref = message.reference
    if ref and isinstance(ref.resolved, discord.Message):
        return ref.resolved.author.id
    return None


@bot.event
async def on_ready():
    print("\n" + "=" * 60)
    print(f"Discord bot connected as {bot.user} (id {bot.user.id})")
    print(f"Guilds: {', '.join(g.name for g in bot.guilds) or 'none'}")
    print("=" * 60 + "\n")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    ctx = await bot.get_context(message)
    if ctx.valid:
        await bot.invoke(ctx)
        return

    content = describe_message(message)
    if not content:
        return

    # While Metis is in a voice channel in this server, she only listens there.
    if message.guild and message.guild.voice_client and message.guild.voice_client.is_connected():
        return

    conversation = get_conversation(message.channel.id, "text")
    channel_name = getattr(message.channel, "name", None) or "DM"

    _typing.get(message.channel.id, {}).pop(message.author.id, None)  # they just finished typing
    async with conversation.lock:
        entry = conversation.add_user(message.author, content,
                                      mentions=tuple(u.id for u in message.mentions),
                                      reply_to=reply_target(message))
        conversation.latest_id = message.id
    await asyncio.to_thread(remember_speaker, message.author)
    conversation.maybe_learn()
    asyncio.create_task(publish({
        "id": str(message.id),
        "text": entry["content"],
        "sender": "user",
        "timestamp": message.created_at.isoformat(),
        "discord_user": entry["name"],
        "user_id": str(message.author.id),
        "channel_name": channel_name,
        "mode": "text",
    }))

    if not await is_metis_turn(conversation, is_dm=message.guild is None):
        return
    if not await wait_until_quiet(message.channel.id, message.id, conversation):
        return  # a newer message arrived; it gets its own turn

    reply = ""
    stats: Dict[str, Any] = {}
    try:
        async with message.channel.typing():
            async for delta in stream_reply(chat_payload(conversation, voice_mode=False), stats):
                if conversation.latest_id != message.id:
                    print("Reply interrupted by a newer message")
                    return  # closing the stream aborts generation upstream
                reply += delta
    except Exception as e:
        print(f"Error generating reply: {e}")
        await message.channel.send(f"⚠️ I couldn't reply: {str(e)[:300]}")
        return

    reply = clean_reply(reply)
    if not reply:
        return
    sent: Optional[discord.Message] = None
    mentions = discord.AllowedMentions(users=True, everyone=False, roles=False, replied_user=False)
    # In a busy channel, reply to the message so it's clear who Metis is answering.
    threaded = conversation.distinct_recent_speakers() > 1
    for i, chunk in enumerate(split_message(link_mentions(reply, conversation))):
        if i == 0 and threaded:
            sent = await message.reply(chunk, allowed_mentions=mentions)
        else:
            sent = await message.channel.send(chunk, allowed_mentions=mentions)

    print(f"💬 Metis → {message.author.display_name}: {reply}")
    async with conversation.lock:
        conversation.add_assistant(reply, to=message.author.id)
    await publish({
        "id": str(sent.id) if sent else f"ai-{message.id}",
        "text": reply,
        "sender": "ai",
        "timestamp": datetime.now().astimezone().isoformat(),
        "channel_name": channel_name,
        "mode": "text",
        "tokensPerSecond": stats.get("tokens_per_second"),
    })
    asyncio.create_task(conversation.compact())


# ---------------------------------------------------------------------------
# Voice chat
# ---------------------------------------------------------------------------
class VoiceAudioSink(voice_recv.AudioSink):
    """Receives decoded PCM (48 kHz stereo) from discord-ext-voice-recv on its reader thread."""

    def __init__(self, receiver: "VoiceReceiver"):
        super().__init__()
        self.receiver = receiver

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data):
        if user is None or user.bot or not data.pcm:
            return
        self.receiver.buffer_audio(user, data.pcm)

    def cleanup(self):
        pass


class SpeechStream(discord.AudioSource):
    """
    Plays speech while it is still being generated: 24 kHz mono PCM goes in (feed) as the voice model
    produces it, Discord pulls 20 ms frames of 48 kHz stereo out (read). If the audio runs short
    before the sentence is finished it plays silence rather than ending.
    """

    FRAME = 3840  # 20 ms of 48 kHz 16-bit stereo
    _TAPS = None  # low-pass filter for 24 -> 48 kHz, shared

    def __init__(self) -> None:
        from scipy.signal import firwin, lfilter_zi

        if SpeechStream._TAPS is None:
            SpeechStream._TAPS = firwin(48, 0.5) * 2
        self._zi = lfilter_zi(self._TAPS, 1.0) * 0.0
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self.finished = False  # no more audio is coming
        self.seconds = 0.0  # audio fed so far
        self.last_fed = time.monotonic()

    def feed(self, pcm24k_mono: bytes) -> None:
        from scipy.signal import lfilter

        samples = np.frombuffer(pcm24k_mono, dtype=np.int16).astype(np.float32)
        if not len(samples):
            return
        stuffed = np.zeros(len(samples) * 2, dtype=np.float32)
        stuffed[::2] = samples  # upsample by 2: zeros between samples, then low-pass (state kept across pieces)
        upsampled, self._zi = lfilter(self._TAPS, 1.0, stuffed, zi=self._zi)
        stereo = np.repeat(np.clip(upsampled, -32768, 32767).astype(np.int16), 2).tobytes()
        with self._lock:
            self._buffer += stereo
        self.seconds += len(samples) / TTS_SAMPLE_RATE
        self.last_fed = time.monotonic()

    def finish(self) -> None:
        if not self.finished:
            self.feed(b"\0\0" * 24)  # flush the filter's delay so the last syllable isn't cut
            self.finished = True

    def read(self) -> bytes:
        with self._lock:
            if len(self._buffer) >= self.FRAME:
                frame = bytes(self._buffer[:self.FRAME])
                del self._buffer[:self.FRAME]
                return frame
            if self.finished:
                rest = bytes(self._buffer)
                self._buffer.clear()
                return rest + b"\0" * (self.FRAME - len(rest)) if rest else b""
        return b"\0" * self.FRAME  # generation is behind playback: a moment of silence

    def is_opus(self) -> bool:
        return False


@dataclass
class _Utterance:
    user: Any
    pcm: bytearray = field(default_factory=bytearray)
    last_packet: float = 0.0


class VoiceReceiver:
    """Turns a voice channel into a conversation with Metis."""

    def __init__(self, voice_client: voice_recv.VoiceRecvClient, text_channel: discord.abc.Messageable):
        self.voice_client = voice_client
        self.channel_id = voice_client.channel.id
        self.text_channel = text_channel
        self.cfg = config.get("voice", {})
        self.silence_timeout = float(self.cfg.get("listen_timeout", 1.0))
        self.gain = float(self.cfg.get("input_gain", 1.0))
        self.discard_while_speaking = bool(self.cfg.get("discard_while_speaking", True))
        ecfg = etiquette(True)
        self.barge_in = bool(ecfg.get("barge_in", True))
        self.barge_in_seconds = float(ecfg.get("barge_in_seconds", 0.6))
        self.barge_in_rms = float(ecfg.get("barge_in_min_rms", 0.03)) * 32768
        self.floor_gap = float(ecfg.get("floor_gap", 0.5))
        self.max_floor_wait = float(ecfg.get("max_floor_wait", 4.0))

        self._utterances: Dict[int, _Utterance] = {}
        # Voice activity per user: user id -> [start of current stretch of speech, last loud packet]
        self._activity: Dict[int, List[float]] = {}
        self._utterance_seq = 0  # bumps whenever someone finishes saying something
        self._buffer_lock = threading.Lock()
        self._respond_lock = asyncio.Lock()  # one spoken reply at a time
        self._speaking = False
        # What people said while Metis was talking: user name -> [seconds of audio, loudest frame RMS]
        self._talk_over: Dict[str, List[float]] = {}
        self._active = True
        self._monitor: Optional[asyncio.Task] = None
        self.conversation = get_conversation(self.channel_id, "voice")

    # -- capture (runs on the voice-recv thread) --------------------------
    def buffer_audio(self, user, pcm: bytes) -> None:
        if not self._active:
            return
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        if self.gain != 1.0:
            samples = samples * self.gain
            pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        now = time.monotonic()
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        if self._speaking and rms >= 0.005 * 32768:
            heard = self._talk_over.setdefault(user.display_name, [0.0, 0.0])
            heard[0] += samples.size / 2 / 48000
            heard[1] = max(heard[1], rms / 32768)
        if rms >= self.barge_in_rms:
            with self._buffer_lock:
                stretch = self._activity.get(user.id)
                if stretch is None or now - stretch[1] > 0.3:
                    self._activity[user.id] = [now, now]
                else:
                    stretch[1] = now
        # While Metis talks, keep listening only if people may interrupt her.
        if self._speaking and self.discard_while_speaking and not self.barge_in:
            return
        with self._buffer_lock:
            utt = self._utterances.get(user.id)
            if utt is None:
                utt = self._utterances[user.id] = _Utterance(user=user)
            utt.pcm.extend(pcm)
            utt.last_packet = time.monotonic()

    def _discard_buffers(self) -> None:
        with self._buffer_lock:
            self._utterances.clear()

    def someone_speaking(self, for_at_least: float = 0.0) -> bool:
        """Is a person talking right now (and has been for `for_at_least` seconds)?"""
        now = time.monotonic()
        with self._buffer_lock:
            return any(now - last < 0.3 and last - start >= for_at_least for start, last in self._activity.values())

    async def wait_for_floor(self, my_seq: int) -> bool:
        """
        Wait for a natural gap before speaking. Gives up (returns False) if people keep
        talking, or if someone said something new meanwhile (that gets its own turn).
        """
        deadline = time.monotonic() + self.max_floor_wait
        quiet_since = None
        while time.monotonic() < deadline:
            if self._utterance_seq != my_seq:
                return False
            if self.someone_speaking():
                quiet_since = None
            else:
                quiet_since = quiet_since or time.monotonic()
                if time.monotonic() - quiet_since >= self.floor_gap:
                    return True
            await asyncio.sleep(0.05)
        return False

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self.voice_client.listen(VoiceAudioSink(self))
        self._monitor = asyncio.create_task(self._watch_for_pauses())
        print(f"🎤 Listening in voice channel {self.voice_client.channel.name}")

    def stop(self) -> None:
        self._active = False
        if self._monitor:
            self._monitor.cancel()
        if self.voice_client.is_listening():
            self.voice_client.stop_listening()

    async def _watch_for_pauses(self) -> None:
        interval = float(self.cfg.get("chunk_duration", 0.25))
        while self._active:
            await asyncio.sleep(interval)
            now = time.monotonic()
            ready: List[_Utterance] = []
            with self._buffer_lock:
                for user_id, utt in list(self._utterances.items()):
                    if utt.pcm and now - utt.last_packet >= self.silence_timeout:
                        ready.append(utt)
                        del self._utterances[user_id]
            for utt in ready:
                asyncio.create_task(self._handle_utterance(utt))

    # -- understanding & replying -----------------------------------------
    async def _handle_utterance(self, utt: _Utterance) -> None:
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            service = await get_voice_service()
            text = await service.transcribe_audio(bytes(utt.pcm), sample_rate=48000, channels=2)
            timings["transcription"] = time.perf_counter() - t0
            if not text:
                return
            print(f"📝 [{utt.user.display_name}] {text}")
            # "Yeah", "uh", "okay" don't take the turn away from a reply she's about to give: in a group every
            # backchannel did, and she gave up nearly every turn.
            if not is_acknowledgement(text):
                self._utterance_seq += 1
            await self._respond(utt.user, text, t0, timings, self._utterance_seq)
        except Exception as e:
            import traceback
            print(f"Voice pipeline error: {e}")
            traceback.print_exc()

    def present_members(self) -> List[str]:
        """Display names of the people (not bots) currently in the voice channel."""
        channel = self.voice_client.channel
        names = []
        for uid in channel.voice_states:
            member = channel.guild.get_member(uid)
            if member and not member.bot:
                names.append(member.display_name)
        return names

    async def _respond(self, user: discord.abc.User, text: str, t0: float, timings: Dict[str, float],
                       seq: int) -> None:
        conversation = self.conversation
        async with conversation.lock:
            entry = conversation.add_user(user, text)
        await asyncio.to_thread(remember_speaker, user)
        conversation.maybe_learn()
        channel_label = self.voice_client.channel.name  # the dashboard shows a voice badge
        asyncio.create_task(publish({
            "id": f"voice-{user.id}-{time.time_ns()}",
            "text": entry["content"],
            "sender": "user",
            "timestamp": datetime.now().astimezone().isoformat(),
            "discord_user": entry["name"],
            "user_id": str(user.id),
            "channel_name": channel_label,
            "mode": "voice",
        }))

        async with self._respond_lock:
            if not self._active or self._utterance_seq != seq:
                return  # someone already said something newer; that turn is handled instead
            t = time.perf_counter()
            alone = len(self.present_members()) <= 1
            if not await is_metis_turn(conversation, voice=True, alone_with_metis=alone):
                return
            if await asked_to_leave(conversation, alone):
                await self._say_goodbye_and_leave(user, seq)
                return
            timings["judgment"] = time.perf_counter() - t

            for attempt in range(2):
                reply, stats, outcome = await self._speak_streamed_reply(conversation, t0, timings, seq)
                if outcome == "done" and not reply and attempt == 0:
                    # e.g. only "*checks phone, rolls her eyes*": nothing the voice can say. One more try.
                    print(f"🙊 nothing in the reply could be said aloud ({stats.get('raw_reply', '')!r}); trying again")
                    continue
                break
            if outcome == "yielded":
                print("🤐 gave up the turn: people kept talking or the conversation moved on")
                return
            if not reply:
                return
            if outcome == "interrupted":
                print("✋ interrupted; stopped talking")
                reply += " —"
            print(f"💬 Metis → {user.display_name}: {reply}")
            async with conversation.lock:
                conversation.add_assistant(reply, to=user.id)

        await publish({
            "id": f"voice-ai-{time.time_ns()}",
            "text": reply,
            "sender": "ai",
            "timestamp": datetime.now().astimezone().isoformat(),
            "channel_name": channel_label,
            "mode": "voice",
            "tokensPerSecond": stats.get("tokens_per_second"),
        })
        asyncio.create_task(conversation.compact())
        self._print_timings(timings, stats, time.perf_counter() - t0)

    _DEFAULT_THINKING = {
        "remember": ["Hmm, let me think.", "Hmm... wait, let me remember.", "Ooh, let me think about that.",
                     "Hold on, I'm thinking.", "Hmm, give me a second."],
        "web_search": ["Hmm, let me look that up.", "One sec, I'll check.", "Ooh, let me see.",
                       "Hold on, let me find out."],
        # The first search didn't answer it and another one is running.
        "web_search_more": ["Hmm, not there. Let me check somewhere else.", "Wait, let me dig a bit more.",
                            "Let me try another search."],
        # Opening pages to read them.
        "web_read": ["Okay, reading this.", "Found something, reading it.", "Let me read that.",
                     "Ooh, this looks right. Reading."],
        # Nothing else was said for a few seconds while a tool runs (chat.py's heartbeat).
        "still_thinking": ["Hmm...", "Still on it.", "Almost there.", "Hold on...", "Mmm, one sec."],
    }

    def _thinking_line(self, tool: str) -> str:
        """A filler line for while a slow tool runs (voice.thinking_lines), never the same one twice in a row."""
        lines = (self.cfg.get("thinking_lines") or {}).get(tool) or self._DEFAULT_THINKING.get(tool) or []
        fresh = [line for line in lines if line != getattr(self, "_last_thinking", None)] or lines
        if not fresh:
            return ""
        self._last_thinking = random.choice(fresh)
        return self._last_thinking

    async def _say_goodbye_and_leave(self, user: discord.abc.User, seq: int) -> None:
        """
        Someone asked Metis to leave the call: say bye, then disconnect. The goodbye is a set line
        rather than generated: a small model in character tends to argue about leaving instead.
        """
        lines = self.cfg.get("goodbyes") or ["Okay, bye!"]
        goodbye = random.choice(lines)
        pcm = await (await get_voice_service()).synthesize_speech(goodbye)
        if pcm and await self.wait_for_floor(seq):
            speech = SpeechStream()
            speech.feed(pcm)
            speech.finish()
            await self._play(speech)
            async with self.conversation.lock:
                self.conversation.add_assistant(goodbye, to=user.id)
        guild = self.voice_client.guild
        channel_name = self.voice_client.channel.name
        if voice_receivers.get(guild.id) is not self:
            return  # already left (>leave, or everyone else left meanwhile)
        await _disconnect(guild)
        print(f"👋 Left {channel_name}: {user.display_name} asked me to")
        try:
            await self.text_channel.send(
                f"👋 Left **{channel_name}** ({user.display_name} asked me to). Text chat is back on.")
        except discord.HTTPException:
            pass

    async def _speak_streamed_reply(self, conversation: Conversation, t0: float, timings: Dict[str, float],
                                    seq: int) -> Tuple[str, Dict[str, Any], str]:
        """
        Stream the reply, synthesising and playing each sentence as soon as it is complete.
        Returns (what was actually said, stats, outcome) where outcome is "done", "yielded"
        (never started: people kept talking / conversation moved on) or "interrupted".
        """
        service = await get_voice_service()
        sentences: asyncio.Queue[Optional[Tuple[str, Optional[str]]]] = asyncio.Queue()
        audio: asyncio.Queue[Optional[Tuple[str, SpeechStream]]] = asyncio.Queue()
        stats: Dict[str, Any] = {}
        state = {"outcome": "done", "spoken": []}
        fillers: set = set()  # filler lines said while tools run ("one sec, I'll check")
        filler_times: List[float] = []

        def stopped() -> bool:
            return state["outcome"] != "done"

        async def synthesize() -> None:
            # Sentence by sentence, each streamed: playback starts with the first piece of audio, and the
            # next sentence is generated while this one plays (generation is ~3x faster than speech).
            while (item := await sentences.get()) is not None:
                if stopped():
                    continue
                sentence, emotion = item
                speech = SpeechStream()
                t = time.perf_counter()
                try:
                    async with contextlib.aclosing(service.stream_speech(sentence, emotion)) as pieces:
                        async for pcm in pieces:
                            if stopped():
                                break  # leaving the loop stops the voice model mid-sentence
                            if not speech.seconds:
                                timings.setdefault("tts_first", time.perf_counter() - t)
                                await audio.put((sentence, speech))
                            speech.feed(pcm)
                finally:
                    speech.finish()
                timings["tts"] = timings.get("tts", 0.0) + time.perf_counter() - t
                print(f"🎙️ [{emotion or 'neutral'}] {speech.seconds:.1f}s: {sentence}")
            await audio.put(None)

        async def play() -> None:
            first = True
            while (item := await audio.get()) is not None:
                sentence, speech = item
                if stopped():
                    continue
                if first:
                    if not await self.wait_for_floor(seq):
                        state["outcome"] = "yielded"
                        continue
                    timings["first_audio"] = time.perf_counter() - t0
                    first = False
                if not await self._play(speech):
                    state["outcome"] = "interrupted"
                    continue
                if sentence not in fillers:
                    # Fillers stay out of the history: with "One sec, I'll check. Okay, reading this." in her
                    # past replies, the model started saying "I'll check" without anything being looked up.
                    state["spoken"].append(sentence)

        synth_task = asyncio.create_task(synthesize())
        play_task = asyncio.create_task(play())
        think = ThinkFilter()
        pending = ""
        reply = ""
        emotion: Optional[str] = None  # "[teasing]" before a sentence sets the voice for the rest of the reply

        async def say(sentence: str, filler: bool = False) -> None:
            nonlocal emotion
            tagged, sentence = take_emotion(strip_name_prefix(sentence), service.emotions)
            emotion = tagged or emotion
            spoken = clean_for_speech(sentence)
            if len(spoken) > 1:
                if filler:
                    fillers.add(spoken)
                await sentences.put((spoken, emotion))

        async def think_aloud(tool: str) -> None:
            # Remembering or searching takes a second or two: say so instead of going quiet, like a person would.
            if not sentences.empty():
                return  # the last filler hasn't even been voiced yet; don't pile them up
            # One per research step made six in 14 s ("Okay, reading this. Ooh, this looks right. Hmm, not
            # there..."): space them out, and only a few per reply.
            now = time.perf_counter()
            gap = float(self.cfg.get("thinking_min_gap", 5))
            if len(filler_times) >= int(self.cfg.get("thinking_max", 3)) or (filler_times and now - filler_times[-1] < gap):
                return
            line = self._thinking_line(tool)
            if line:
                filler_times.append(now)
                print(f"🤔 {tool}: saying {line!r} while it runs")
                await say(line, filler=True)

        try:
            async for delta in stream_reply(chat_payload(conversation, True, self.present_members()), stats,
                                            think_aloud):
                if stopped():
                    break  # closing the stream stops generation upstream
                reply += delta
                pending += think.feed(delta)
                done, pending = pop_sentences(pending)
                for sentence in done:
                    await say(sentence)
            pending += think.flush()
            if pending.strip():
                await say(pending)
        except Exception as e:
            print(f"Voice reply error: {e}")
        finally:
            await sentences.put(None)
            await asyncio.gather(synth_task, play_task, return_exceptions=True)
        # What people actually heard (no *actions*, cut short if interrupted). It's also what goes into the
        # history: keeping "*rolls her eyes*" there taught the model to answer with more of them.
        stats["raw_reply"] = reply
        return " ".join(state["spoken"]), stats, state["outcome"]

    async def _play(self, source: SpeechStream) -> bool:
        """
        Play speech, also while it's still being generated (no temp files or FFmpeg process).
        Returns False if a person talked over Metis and she stopped (barge-in).
        """
        if not self._active or not self.voice_client.is_connected():
            return False

        loop = asyncio.get_running_loop()
        done = asyncio.Event()

        def after(error: Optional[Exception]) -> None:
            if error:
                print(f"Playback error: {error}")
            loop.call_soon_threadsafe(done.set)

        self._speaking = True
        self._talk_over.clear()
        if self.discard_while_speaking and not self.barge_in:
            self._discard_buffers()
        interrupted = False
        try:
            self.voice_client.play(source, after=after)
            # Safety net: stop if the voice model stalls (no audio for 10 s) before finishing the sentence.
            while not done.is_set() and (source.finished or time.monotonic() - source.last_fed < 10):
                if self.barge_in and self.someone_speaking(for_at_least=self.barge_in_seconds):
                    interrupted = True  # someone is talking over her: stop, like a person would
                    self.voice_client.stop()
                    break
                try:
                    await asyncio.wait_for(done.wait(), timeout=0.05)
                except asyncio.TimeoutError:
                    pass
            if not done.is_set() and not interrupted:
                self.voice_client.stop()
        except discord.ClientException as e:
            print(f"Playback skipped: {e}")
        finally:
            self._speaking = False
        for name, (seconds, loudest) in self._talk_over.items():
            # For tuning barge_in_seconds / barge_in_min_rms when she doesn't stop (or stops too easily).
            print(f"🗣️ {name} talked while Metis was speaking: {seconds:.1f}s of audio, loudest {loudest:.3f} "
                  f"({'she stopped' if interrupted else 'not enough to stop her'}; she stops after "
                  f"{self.barge_in_seconds}s at {self.barge_in_rms / 32768:.3f} or louder)")
        return not interrupted

    @staticmethod
    def _print_timings(timings: Dict[str, float], stats: Dict[str, Any], total: float) -> None:
        tool = stats.get("tool_timings") or {}
        rag = stats.get("rag_timings") or {}
        rows = [
            ("Transcription", timings.get("transcription")),
            ("Judgment", timings.get("judgment")),
            ("Tool selection", tool.get("tool_selection")),
            ("Tool execution", tool.get("tool_execution")),
            ("RAG queries", rag.get("query_generation")),
            ("RAG retrieval", rag.get("retrieval")),
            ("LLM first token", stats.get("ttft")),
            ("LLM generation", stats.get("llm_generation_time")),
            ("TTS first audio", timings.get("tts_first")),
            ("TTS (sum)", timings.get("tts")),
        ]
        print("\n⏱️  Voice pipeline")
        for label, value in rows:
            if value:
                print(f"   {label:<18}{value:>7.2f}s")
        if "first_audio" in timings:
            print(f"   {'⚡ First audio':<18}{timings['first_audio']:>7.2f}s")
        print(f"   {'Total':<18}{total:>7.2f}s\n")


voice_receivers: Dict[int, VoiceReceiver] = {}  # guild id -> receiver


async def _disconnect(guild: discord.Guild) -> None:
    receiver = voice_receivers.pop(guild.id, None)
    if receiver:
        receiver.stop()
    if guild.voice_client:
        await guild.voice_client.disconnect(force=True)


@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    guild = member.guild
    if member == bot.user and after.channel is None:
        receiver = voice_receivers.pop(guild.id, None)
        if receiver:
            receiver.stop()
        return
    vc = guild.voice_client
    if vc and after.channel == vc.channel and before.channel != vc.channel and not member.bot:
        (await get_voice_service()).add_names([member.display_name])
    if vc and before.channel == vc.channel and after.channel != vc.channel:
        # voice_states works without the privileged members intent (unlike channel.members).
        if not any(uid != bot.user.id for uid in vc.channel.voice_states):
            receiver = voice_receivers.get(guild.id)
            await _disconnect(guild)
            if receiver:
                await receiver.text_channel.send("👋 Everyone left the voice channel, so I left too. Text chat is back on.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@bot.command(name="reset")
async def reset_context(ctx: commands.Context):
    """Start this channel's conversation fresh (Metis learns from it first)."""
    count = get_conversation(ctx.channel.id, "text").reset()
    await ctx.send(f"🔄 Fresh start! I'll remember what mattered from the last {count} message(s).")


@bot.command(name="status")
async def status_command(ctx: commands.Context):
    """Show what Metis is up to."""
    from backend.token_utils import count_message_tokens

    conversation = get_conversation(ctx.channel.id, "text")
    tokens = await asyncio.to_thread(count_message_tokens, conversation.prompt())
    embed = discord.Embed(title="Metis status", color=discord.Color.purple())
    embed.add_field(name="Personality", value=str(config.get("personality")))
    embed.add_field(name="Messages here", value=str(len(conversation.messages)))
    embed.add_field(name="Context", value=f"~{tokens} / {config['model'].get('n_ctx', 8192)} tokens")
    embed.add_field(name="Tools", value="on" if config.get("tools", {}).get("enabled") else "off")
    embed.add_field(name="Knowledge base", value="on" if config.get("rag", {}).get("enabled") else "off")
    if ctx.guild and ctx.guild.id in voice_receivers:
        embed.add_field(name="Voice", value=voice_receivers[ctx.guild.id].voice_client.channel.name)
    await ctx.send(embed=embed)


@bot.command(name="personality")
async def personality_command(ctx: commands.Context, name: Optional[str] = None):
    """List personalities, or switch with `>personality <name>` (requires Manage Server)."""
    from backend.prompt_loader import load_prompts

    available = sorted((load_prompts().get("text_chat") or {}).keys())
    current = config.get("personality")
    if not name:
        listing = ", ".join(f"**{p}**" if p == current else p for p in available)
        await ctx.send(f"🎭 Personalities: {listing}")
        return
    if ctx.guild and not ctx.author.guild_permissions.manage_guild:
        await ctx.send("❌ You need the Manage Server permission to change my personality.")
        return
    if name not in available:
        await ctx.send(f"❌ Unknown personality `{name}`. Try one of: {', '.join(available)}")
        return
    from backend.routers.personality import save_to_config

    config["personality"] = name  # takes effect from the next reply
    kept = "saved" if save_to_config(name) else "until restart: config.yaml couldn't be updated"
    await ctx.send(f"🎭 Personality switched to **{name}** ({kept}).")


@bot.command(name="join")
async def join_voice(ctx: commands.Context):
    """Join your voice channel and start talking."""
    if not ctx.guild or not isinstance(ctx.author, discord.Member) or not ctx.author.voice:
        await ctx.send("❌ Join a voice channel first!")
        return
    target = ctx.author.voice.channel
    if ctx.voice_client and not ctx.voice_client.is_connected():
        await _disconnect(ctx.guild)  # leftover from a failed attempt; start clean
    if ctx.voice_client:
        if ctx.voice_client.channel == target:
            await ctx.send("✅ Already here!")
        else:
            await ctx.voice_client.move_to(target)
            await ctx.send(f"🔊 Moved to **{target.name}**")
        return

    try:
        notice = await ctx.send("⏳ Loading voice models...")
        await get_voice_service()
        voice_client = await target.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        receiver = VoiceReceiver(voice_client, ctx.channel)
        receiver.start()
        (await get_voice_service()).add_names(receiver.present_members())
        voice_receivers[ctx.guild.id] = receiver
        await notice.edit(content=(
            f"🔊 Joined **{target.name}** — just talk, I'm listening.\n"
            "💬 Text replies are paused while I'm in voice. Say \"Metis, you can leave\" or use `>leave` when you're done."
        ))
    except Exception as e:
        # Don't leave a half-open connection behind: discord.py would keep retrying it, which shows
        # up as Metis joining and leaving the channel over and over.
        await _disconnect(ctx.guild)
        hint = ""
        if "4017" in str(e) or "not connected to voice" in str(e).lower():
            hint = ("\n(Discord voice needs discord.py 2.7+ with `davey` for its end-to-end encryption"
                    " - see requirements.txt.)")
        await ctx.send(f"❌ Failed to join voice channel: {e}{hint}")
        import traceback
        traceback.print_exc()


@bot.command(name="leave")
async def leave_voice(ctx: commands.Context):
    """Leave the voice channel."""
    if not ctx.guild or not ctx.voice_client:
        await ctx.send("❌ I'm not in a voice channel!")
        return
    name = ctx.voice_client.channel.name
    await _disconnect(ctx.guild)
    await ctx.send(f"👋 Left **{name}**. Text chat is back on.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError):
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, (commands.BadArgument, commands.MissingRequiredArgument)):
        await ctx.send(f"❌ {error}")
        return
    print(f"Command error in {ctx.command}: {error}")
    import traceback
    traceback.print_exception(type(error), error, error.__traceback__)
    await ctx.send(f"⚠️ Something went wrong: {str(error)[:200]}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
async def _run() -> None:
    try:
        async with bot:
            await bot.start(DISCORD_TOKEN)
    finally:
        if _http and not _http.closed:
            await _http.close()


def start_in_background() -> Optional[threading.Thread]:
    """Run the bot on its own event loop in a daemon thread."""
    global _thread, _loop
    if not config.get("discord", {}).get("enabled", True):
        print("Discord bot disabled (discord.enabled: false in config.yaml).")
        return None
    if not DISCORD_TOKEN:
        print("⚠ DISCORD_BOT_TOKEN is not set (see .env.example); the Discord bot will not start.")
        return None
    if _thread and _thread.is_alive():
        return _thread

    def runner() -> None:
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_run())
        except asyncio.CancelledError:
            pass  # the backend is shutting down (e.g. it failed to start): nothing to report
        except discord.LoginFailure:
            print("✗ Discord rejected the bot token (DISCORD_BOT_TOKEN).")
        except Exception as e:
            import traceback
            print(f"✗ Discord bot stopped: {e}")
            traceback.print_exc()
        finally:
            _loop.close()

    discord.utils.setup_logging()
    # It logs every RTCP sender report (one a second during a call) at INFO.
    logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
    _thread = threading.Thread(target=runner, name="discord-bot", daemon=True)
    _thread.start()
    return _thread


def stop(timeout: float = 10) -> None:
    if _loop and _loop.is_running() and not bot.is_closed():
        future = asyncio.run_coroutine_threadsafe(bot.close(), _loop)
        try:
            future.result(timeout)
        except Exception:
            pass


def status() -> Dict[str, Any]:
    connected = bot.is_ready() and not bot.is_closed()
    return {
        "connected": connected,
        "user": str(bot.user) if bot.user else None,
        "voice_channels": [r.voice_client.channel.name for r in list(voice_receivers.values())
                           if r.voice_client.is_connected()],
    }


if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN is not set")
    asyncio.run(_run())
