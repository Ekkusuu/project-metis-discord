"""
Discord front-end for Metis.

Text channels: every message goes through a "should Metis reply?" judgment
(skipped when Metis is mentioned, replied to, or DM'd), then the reply is
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
import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

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
)
from backend.voice_service import TTS_SAMPLE_RATE, get_voice_service

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
    """History for one text channel or one voice session."""

    mode: str  # "text" | "voice"
    messages: List[Dict[str, str]] = field(default_factory=list)
    latest_id: int = 0  # id of the newest user message; older in-flight replies abort
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def system_prompt(self) -> str:
        cfg = config.get("voice" if self.mode == "voice" else "chat", {})
        default = "@voice_chat" if self.mode == "voice" else "@text_chat"
        return resolve_prompt(cfg.get("system_prompt", default))

    def prompt(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.system_prompt}, *self.messages]

    async def compact(self) -> None:
        """
        When the history outgrows the context window, cut it back to ~70% of the
        budget in one go and archive what was removed to memory (exactly once).
        Cutting with headroom keeps the prompt prefix stable for many turns, which
        is what lets the LLM service reuse its KV cache.
        """
        from backend.context_manager import context_budget, split_to_fit
        from backend.memory_manager import archive_messages
        from backend.token_utils import count_message_tokens

        async with self.lock:
            snapshot = self.prompt()
            budget = context_budget() - 1024  # leave room for tool/RAG context
            if await asyncio.to_thread(count_message_tokens, snapshot) <= budget:
                return
            _, removed = await asyncio.to_thread(split_to_fit, snapshot, int(budget * 0.7))
            if removed:
                del self.messages[: len(removed)]
                archive_messages(removed)
                print(f"🗜️  Compacted {self.mode} conversation: archived {len(removed)} message(s) to memory")

    def reset(self) -> int:
        from backend.memory_manager import archive_messages

        count = len(self.messages)
        if self.messages:
            archive_messages(self.messages)
        self.messages = []
        return count


conversations: Dict[Tuple[str, int], Conversation] = {}


def get_conversation(channel_id: int, mode: str = "text") -> Conversation:
    key = (mode, channel_id)
    if key not in conversations:
        conversations[key] = Conversation(mode=mode)
    return conversations[key]


# ---------------------------------------------------------------------------
# Backend calls
# ---------------------------------------------------------------------------
async def should_respond(conversation: Conversation, voice_mode: bool = False) -> bool:
    """Ask the LLM whether Metis should join in. Errors default to responding."""
    judgment_cfg = config.get("voice" if voice_mode else "chat", {}).get("judgment", {})
    if not judgment_cfg.get("enabled", True):
        return True
    recent = conversation.messages[-int(judgment_cfg.get("context_messages", 5)):]
    if not recent:
        return True
    payload = {
        "messages": recent,
        "system_prompt": resolve_prompt(judgment_cfg.get("system_prompt", "@judgment.voice" if voice_mode else "@judgment.text")),
        "user_prompt": judgment_cfg.get("user_prompt"),
    }
    try:
        async with http().post(f"{llm_service_url()}/judgment", json=payload,
                               timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                print(f"Judgment request failed ({response.status}); responding anyway")
                return True
            judgment = (await response.json()).get("judgment", "").strip().lower()
    except Exception as e:
        print(f"Judgment error ({e}); responding anyway")
        return True
    decision = not judgment.startswith("neg")
    print(f"{'✅' if decision else '❌'} Judgment: {judgment}")
    return decision


async def stream_reply(payload: Dict[str, Any], stats: Dict[str, Any]) -> AsyncIterator[str]:
    """Yield reply text deltas from the backend. The final stats chunk is stored in `stats`."""
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
            if chunk.get("delta"):
                yield chunk["delta"]
            if chunk.get("done"):
                stats.update(chunk)
                return


def chat_payload(conversation: Conversation, voice_mode: bool) -> Dict[str, Any]:
    chat_cfg = config.get("chat", {})
    voice_cfg = config.get("voice", {})
    payload: Dict[str, Any] = {
        "messages": conversation.prompt(),
        "temperature": chat_cfg.get("temperature", 0.7),
        "top_p": chat_cfg.get("top_p", 0.95),
        "max_tokens": (voice_cfg if voice_mode else chat_cfg).get("max_tokens", chat_cfg.get("max_tokens", 512)),
        "voice_mode": voice_mode,
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
    if "```" not in text:
        text = re.sub(r"\n{2,}", "\n", text)  # keep Discord replies compact
    return text


# ---------------------------------------------------------------------------
# Text chat
# ---------------------------------------------------------------------------
def author_label(user: discord.abc.User) -> str:
    return user.name if getattr(user, "discriminator", "0") in ("0", None) else f"{user.name}#{user.discriminator}"


def describe_message(message: discord.Message) -> str:
    """Message text as the model should see it: mentions resolved, attachments noted."""
    text = message.clean_content.strip()
    extras = [f"[attachment: {a.filename}]" for a in message.attachments]
    extras += ["[sticker]" for _ in message.stickers]
    return " ".join([text, *extras]).strip()


def is_addressed_to_bot(message: discord.Message) -> bool:
    """DMs, @mentions, replies to Metis, or saying her name skip the judgment step."""
    if message.guild is None or bot.user in message.mentions:
        return True
    ref = message.reference
    if ref and isinstance(ref.resolved, discord.Message) and ref.resolved.author == bot.user:
        return True
    name = (bot.user.display_name if bot.user else "Metis").lower()
    return re.search(rf"\b{re.escape(name)}\b", message.content.lower()) is not None


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
    user = author_label(message.author)
    user_content = f"[{user}]: {content}"
    channel_name = getattr(message.channel, "name", None) or "DM"

    async with conversation.lock:
        conversation.messages.append({"role": "user", "content": user_content})
        conversation.latest_id = message.id
    asyncio.create_task(publish({
        "id": str(message.id),
        "text": user_content,
        "sender": "user",
        "timestamp": message.created_at.isoformat(),
        "discord_user": user,
        "channel_name": channel_name,
        "mode": "text",
    }))

    if not is_addressed_to_bot(message) and not await should_respond(conversation):
        return
    if conversation.latest_id != message.id:
        return  # a newer message arrived while judging; it will be answered instead

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
    for chunk in split_message(reply):
        sent = await message.channel.send(chunk, allowed_mentions=discord.AllowedMentions(users=True, everyone=False, roles=False))

    async with conversation.lock:
        conversation.messages.append({"role": "assistant", "content": reply})
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

        self._utterances: Dict[int, _Utterance] = {}
        self._buffer_lock = threading.Lock()
        self._respond_lock = asyncio.Lock()  # one spoken reply at a time
        self._speaking = False
        self._active = True
        self._monitor: Optional[asyncio.Task] = None
        self.conversation = get_conversation(self.channel_id, "voice")

    # -- capture (runs on the voice-recv thread) --------------------------
    def buffer_audio(self, user, pcm: bytes) -> None:
        if not self._active or (self.discard_while_speaking and self._speaking):
            return
        if self.gain != 1.0:
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) * self.gain
            pcm = np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        with self._buffer_lock:
            utt = self._utterances.get(user.id)
            if utt is None:
                utt = self._utterances[user.id] = _Utterance(user=user)
            utt.pcm.extend(pcm)
            utt.last_packet = time.monotonic()

    def _discard_buffers(self) -> None:
        with self._buffer_lock:
            self._utterances.clear()

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
            username = utt.user.display_name
            print(f"📝 [{username}] {text}")
            await self._respond(username, utt.user.id, text, t0, timings)
        except Exception as e:
            import traceback
            print(f"Voice pipeline error: {e}")
            traceback.print_exc()

    async def _respond(self, username: str, user_id: int, text: str, t0: float, timings: Dict[str, float]) -> None:
        conversation = self.conversation
        user_content = f"[{username}] {text}"
        async with conversation.lock:
            conversation.messages.append({"role": "user", "content": user_content})
        channel_label = f"🎤 {self.voice_client.channel.name}"
        asyncio.create_task(publish({
            "id": f"voice-{user_id}-{time.time_ns()}",
            "text": user_content,
            "sender": "user",
            "timestamp": datetime.now().astimezone().isoformat(),
            "discord_user": username,
            "channel_name": channel_label,
            "mode": "voice",
        }))

        async with self._respond_lock:
            if not self._active:
                return
            # Skip the judgment if someone said Metis' name.
            name = (bot.user.display_name if bot.user else "Metis").lower()
            t = time.perf_counter()
            addressed = re.search(rf"\b{re.escape(name)}\b", text.lower()) is not None
            if not addressed and not await should_respond(conversation, voice_mode=True):
                return
            timings["judgment"] = time.perf_counter() - t

            reply, stats = await self._speak_streamed_reply(conversation, t0, timings)
            reply = clean_reply(reply)
            if not reply:
                return
            async with conversation.lock:
                conversation.messages.append({"role": "assistant", "content": reply})

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

    async def _speak_streamed_reply(self, conversation: Conversation, t0: float,
                                    timings: Dict[str, float]) -> Tuple[str, Dict[str, Any]]:
        """Stream the reply, synthesising and playing each sentence as soon as it is complete."""
        service = await get_voice_service()
        sentences: asyncio.Queue[Optional[str]] = asyncio.Queue()
        audio: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        stats: Dict[str, Any] = {}

        async def synthesize() -> None:
            while (sentence := await sentences.get()) is not None:
                t = time.perf_counter()
                pcm = await service.synthesize_speech(sentence)
                timings["tts"] = timings.get("tts", 0.0) + time.perf_counter() - t
                if pcm:
                    await audio.put(pcm)
            await audio.put(None)

        async def play() -> None:
            first = True
            while (pcm := await audio.get()) is not None:
                if first:
                    timings["first_audio"] = time.perf_counter() - t0
                    first = False
                await self._play_pcm(pcm)

        synth_task = asyncio.create_task(synthesize())
        play_task = asyncio.create_task(play())
        think = ThinkFilter()
        pending = ""
        reply = ""
        try:
            async for delta in stream_reply(chat_payload(conversation, voice_mode=True), stats):
                reply += delta
                pending += think.feed(delta)
                done, pending = pop_sentences(pending)
                for sentence in done:
                    spoken = clean_for_speech(strip_name_prefix(sentence))
                    if len(spoken) > 1:
                        await sentences.put(spoken)
            pending += think.flush()
            tail = clean_for_speech(strip_name_prefix(pending))
            if tail:
                await sentences.put(tail)
        except Exception as e:
            print(f"Voice reply error: {e}")
        finally:
            await sentences.put(None)
            await asyncio.gather(synth_task, play_task, return_exceptions=True)
        return reply, stats

    async def _play_pcm(self, pcm24k_mono: bytes) -> None:
        """Play Kokoro output directly (no temp files or FFmpeg process per sentence)."""
        if not self._active or not self.voice_client.is_connected():
            return
        from scipy.signal import resample_poly

        samples = np.frombuffer(pcm24k_mono, dtype=np.int16).astype(np.float32)
        upsampled = resample_poly(samples, 48000 // TTS_SAMPLE_RATE, 1)
        stereo = np.repeat(np.clip(upsampled, -32768, 32767).astype(np.int16), 2)
        source = discord.PCMAudio(io.BytesIO(stereo.tobytes()))

        loop = asyncio.get_running_loop()
        done = asyncio.Event()

        def after(error: Optional[Exception]) -> None:
            if error:
                print(f"Playback error: {error}")
            loop.call_soon_threadsafe(done.set)

        self._speaking = True
        if self.discard_while_speaking:
            self._discard_buffers()
        try:
            self.voice_client.play(source, after=after)
            duration = len(pcm24k_mono) / (TTS_SAMPLE_RATE * 2)
            await asyncio.wait_for(done.wait(), timeout=duration + 5)
        except asyncio.TimeoutError:
            self.voice_client.stop()
        except discord.ClientException as e:
            print(f"Playback skipped: {e}")
        finally:
            self._speaking = False

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
    """Forget this channel's conversation (it's saved to long-term memory first)."""
    count = get_conversation(ctx.channel.id, "text").reset()
    await ctx.send(f"🔄 Fresh start! {count} message(s) were saved to memory.")


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
    config["personality"] = name  # takes effect immediately, until the next restart
    await ctx.send(f"🎭 Personality switched to **{name}** (until restart — set `personality` in config.yaml to keep it).")


@bot.command(name="join")
async def join_voice(ctx: commands.Context):
    """Join your voice channel and start talking."""
    if not ctx.guild or not isinstance(ctx.author, discord.Member) or not ctx.author.voice:
        await ctx.send("❌ Join a voice channel first!")
        return
    target = ctx.author.voice.channel
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
        voice_receivers[ctx.guild.id] = receiver
        await notice.edit(content=(
            f"🔊 Joined **{target.name}** — just talk, I'm listening.\n"
            "💬 Text replies are paused while I'm in voice. Use `>leave` when you're done."
        ))
    except Exception as e:
        await ctx.send(f"❌ Failed to join voice channel: {e}")
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
        except discord.LoginFailure:
            print("✗ Discord rejected the bot token (DISCORD_BOT_TOKEN).")
        except Exception as e:
            print(f"✗ Discord bot stopped: {e}")
        finally:
            _loop.close()

    discord.utils.setup_logging()
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
