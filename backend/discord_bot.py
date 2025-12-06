"""
Discord bot service for Project Metis.
Connects Discord messages to the existing chat API.
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

import discord
from discord.ext import commands
import discord.ext.voice_recv as voice_recv
from dotenv import load_dotenv
import aiohttp
import io
import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.prompt_loader import resolve_prompt

from backend.llama_engine import get_config
from backend.voice_service import get_voice_service

# Load environment variables
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN not found in .env file!")
    sys.exit(1)

config = get_config()
API_URL = "http://localhost:8000"

# Store conversation context per Discord channel
# Structure: {channel_id: [{"role": "user/assistant", "content": "..."}]}
channel_contexts: Dict[int, List[Dict[str, str]]] = {}

# Track latest message timestamp per channel for interruption handling
latest_message_time: Dict[int, datetime] = {}

# Voice channel audio buffers (per user)
voice_audio_buffers: Dict[int, bytearray] = {}  # {user_id: audio_buffer}
voice_is_speaking: Dict[int, bool] = {}  # {user_id: is_speaking}

# Discord bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
intents.guilds = True
intents.voice_states = True  # Required for voice channel operations
bot = commands.Bot(command_prefix=[">", "!"], intents=intents)


class VoiceAudioSink(voice_recv.AudioSink):
    """Audio sink that captures voice packets for transcription."""
    
    def __init__(self, receiver, loop):
        super().__init__()
        self.receiver = receiver
        self.loop = loop
    
    def wants_opus(self) -> bool:
        return False  # We want decoded PCM audio
    
    def write(self, user, data):
        """Called when audio data is received from a user."""
        if not user or user.bot:
            return
        
        # Schedule the coroutine on the event loop
        asyncio.run_coroutine_threadsafe(
            self.receiver.on_audio_packet(user, data),
            self.loop
        )
    
    def cleanup(self):
        """Called when the sink is being cleaned up."""
        pass


class VoiceReceiver:
    """
    Voice receiver wrapper for capturing audio from Discord voice channels.
    Buffers audio per user and triggers transcription when user stops speaking.
    """
    
    def __init__(self, voice_client, channel_id, text_channel):
        self.voice_client = voice_client
        self.channel_id = channel_id
        self.text_channel = text_channel
        self.user_audio_buffers: Dict[int, bytearray] = {}  # user_id -> audio buffer
        self.user_objects: Dict[int, any] = {}  # user_id -> discord.Member object
        
        # Get silence threshold from config
        voice_cfg = config.get("voice", {})
        self.silence_threshold = voice_cfg.get("listen_timeout", 5)  # Use listen_timeout as silence threshold
        
        self.last_audio_time: Dict[int, float] = {}
        self.check_task = None
        self.active = True
        self.processing_audio: Dict[int, bool] = {}  # Track if user's audio is being processed
        
        # Create and register audio sink (pass the event loop)
        self.sink = VoiceAudioSink(self, asyncio.get_event_loop())
        self.voice_client.listen(self.sink)
    
    async def start_listening(self):
        """Start listening for voice activity."""
        print(f"🎤 Voice receiver started for channel {self.channel_id}")
        
        await self.text_channel.send(
            "🎤 Voice mode active! Speak into your microphone and I'll respond.\n"
            "💬 Text chat is now disabled. Use `>leave` to exit voice mode."
        )
        
        # Start monitoring for silence
        self.check_task = asyncio.create_task(self._check_for_silence())
    
    async def on_audio_packet(self, user, data):
        """Called when audio packet is received from a user."""
        if user.bot:
            return
        
        user_id = user.id
        current_time = asyncio.get_event_loop().time()
        
        # Update last audio time
        self.last_audio_time[user_id] = current_time
        
        # Buffer audio data
        if user_id not in self.user_audio_buffers:
            self.user_audio_buffers[user_id] = bytearray()
            self.user_objects[user_id] = user  # Store user object for later
            print(f"🎤 {user.display_name} started speaking")
        
        # Append PCM audio data (apply input gain configured in config.yaml)
        if isinstance(data.pcm, bytes):
            try:
                # Read configured gain (default 2.0 if not configured)
                voice_cfg = config.get("voice", {})
                gain = float(voice_cfg.get("input_gain", 2.0))
                # Convert bytes to int16 array, scale by configured gain, clip to int16 range and append
                arr = np.frombuffer(data.pcm, dtype=np.int16).astype(np.int32) * gain
                arr = np.clip(arr, -32768, 32767).astype(np.int16)
                self.user_audio_buffers[user_id].extend(arr.tobytes())
            except Exception as e:
                # Fallback, append original PCM if scaling fails
                print(f"Warning: Failed to scale audio for user {user_id}: {e}")
                self.user_audio_buffers[user_id].extend(data.pcm)
        
    async def _check_for_silence(self):
        """Periodically check for silence and process buffered audio."""
        # Get check interval from config
        voice_cfg = config.get("voice", {})
        check_interval = voice_cfg.get("chunk_duration", 0.5)
        
        while self.active:
            try:
                await asyncio.sleep(check_interval)  # Check at configured interval
                current_time = asyncio.get_event_loop().time()
                
                for user_id, last_time in list(self.last_audio_time.items()):
                    silence_duration = current_time - last_time
                    buffer_size = len(self.user_audio_buffers.get(user_id, []))
                    
                    # Debug logging
                    if buffer_size > 0 and silence_duration > 0.5:
                        print(f"User {user_id}: {silence_duration:.1f}s silence, {buffer_size} bytes buffered, threshold: {self.silence_threshold}s")
                    
                    # If user hasn't spoken in threshold time and has audio buffered
                    if (silence_duration > self.silence_threshold and 
                        user_id in self.user_audio_buffers and 
                        buffer_size > 0 and
                        not self.processing_audio.get(user_id, False)):
                        
                        print(f"🔊 Processing audio for user {user_id} after {silence_duration:.1f}s silence")
                        # Process this user's audio
                        await self._process_user_audio(user_id)
            
            except Exception as e:
                print(f"Error in silence check: {e}")
    
    async def _process_user_audio(self, user_id: int):
        """Process buffered audio for a user (transcribe and respond)."""
        if not self.user_audio_buffers.get(user_id):
            return
        
        try:
            self.processing_audio[user_id] = True
            
            # Get audio buffer
            audio_data = bytes(self.user_audio_buffers[user_id])
            self.user_audio_buffers[user_id] = bytearray()  # Clear buffer
            
            print(f"Processing audio from user {user_id}, size: {len(audio_data)} bytes")
            
            # Get voice service
            voice_service = await get_voice_service()
            
            # Transcribe audio (Discord voice is 48kHz stereo opus, voice_recv gives us 48kHz PCM)
            transcription = await voice_service.transcribe_audio(audio_data, sample_rate=48000)
            
            if not transcription or len(transcription.strip()) == 0:
                print("No transcription result")
                self.processing_audio[user_id] = False
                return
            
            print(f"Transcribed: {transcription}")
            
            # Get username from stored user object
            user = self.user_objects.get(user_id)
            username = user.display_name if user else f"User{user_id}"
            
            # Process voice command (no text messages to Discord)
            await self._process_voice_command(transcription, user_id, username)
            
        except Exception as e:
            print(f"Error processing user audio: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.processing_audio[user_id] = False
    
    async def _process_voice_command(self, text: str, user_id: int, username: str):
        """Process voice input and generate response."""
        try:
            # Use globals instead of importing (avoid circular import)
            global channel_contexts, API_URL, config
            
            channel_id = self.channel_id
            context = get_channel_context(channel_id)
            
            # Add user message with username prefix
            user_content = f"[{username}] {text}"
            context.append({
                "role": "user",
                "content": user_content
            })
            
            # Check judgment (voice mode)
            if not await should_respond_to_message(context, voice_mode=True):
                print(f"Judgment: Not responding to voice input")
                return
            
            # Call LLM with voice-specific system prompt
            chat_cfg = config.get("chat", {})
            voice_cfg = config.get("voice", {})
            
            # Use voice system prompt (auto-selects based on personality setting)
            voice_system_prompt = resolve_prompt(voice_cfg.get("system_prompt", "@voice_chat"))
            if voice_system_prompt and context and context[0].get("role") == "system":
                # Replace the system prompt with voice-specific one
                context[0] = {
                    "role": "system",
                    "content": voice_system_prompt
                }
            
            payload = {
                "messages": context,
                "temperature": chat_cfg.get("temperature", 0.7),
                "top_p": chat_cfg.get("top_p", 0.95),
                "max_tokens": chat_cfg.get("max_tokens", 512),
                # Voice mode: use single RAG query for faster response
                "rag_query_count": 1,
                "rag_system_prompt": "@rag.query_generation_voice_system",
                "rag_user_prompt": "@rag.query_generation_voice_user"
            }
            
            ai_response = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_URL}/chat/stream",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        await self.text_channel.send(f"❌ Error: {error_text[:500]}")
                        return
                    
                    async for line in response.content:
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if "delta" in chunk and chunk["delta"]:
                                ai_response += chunk["delta"]
                            
                            if chunk.get("done"):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            if ai_response:
                # Clean response
                import re
                ai_response = re.sub(r'^\[?Metis\]?\s*:\s*', '', ai_response, flags=re.IGNORECASE)
                ai_response = ai_response.replace('\n\n', '\n')
                
                # Add to context (no text message to Discord in voice mode)
                context.append({
                    "role": "assistant",
                    "content": ai_response
                })
                
                # Save to chat history
                save_to_chat_history({
                    "timestamp": datetime.now().isoformat(),
                    "channel_id": str(channel_id),
                    "user_id": str(user_id),
                    "username": username,
                    "content": text,
                    "role": "user",
                    "mode": "voice"
                })
                
                save_to_chat_history({
                    "timestamp": datetime.now().isoformat(),
                    "channel_id": str(channel_id),
                    "user_id": str(bot.user.id),
                    "username": bot.user.display_name,
                    "content": ai_response,
                    "role": "assistant",
                    "mode": "voice"
                })
                
                # Play TTS response
                await self.play_tts_response(ai_response)
        
        except Exception as e:
            print(f"Error processing voice command: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_listening(self):
        """Stop listening for voice activity."""
        self.active = False
        if self.check_task and not self.check_task.done():
            self.check_task.cancel()
        print(f"🔇 Voice receiver stopped for channel {self.channel_id}")
    
    async def play_tts_response(self, text: str):
        """Generate TTS and play in voice channel."""
        if not self.active or not self.voice_client or not self.voice_client.is_connected():
            return
        
        try:
            # Get voice service
            voice_service = await get_voice_service()
            
            # Convert text to speech
            audio_bytes = await voice_service.synthesize_speech(text)
            
            if not audio_bytes:
                print("TTS generation failed")
                return
            
            # Play audio in voice channel
            await self._play_audio(audio_bytes)
            
        except Exception as e:
            print(f"Error in TTS response: {e}")
    
    async def _play_audio(self, audio_bytes: bytes):
        """Play audio bytes in the voice channel."""
        if not self.voice_client or not self.voice_client.is_connected():
            return
        
        try:
            import wave
            import numpy as np
            
            # Convert PCM bytes to WAV format for FFmpeg
            temp_wav = PROJECT_ROOT / "temp_tts.wav"
            with wave.open(str(temp_wav), 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(24000)  # 24kHz (Kokoro output)
                wav_file.writeframes(audio_bytes)
            
            # Play the audio using FFmpeg
            audio_source = discord.FFmpegPCMAudio(str(temp_wav))
            self.voice_client.play(audio_source)
            
            # Wait for playback to finish
            while self.voice_client.is_playing():
                await asyncio.sleep(0.1)
            
            # Clean up temp file
            temp_wav.unlink(missing_ok=True)
            print("✓ TTS playback complete")
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            import traceback
            traceback.print_exc()


def get_channel_context(channel_id: int) -> List[Dict[str, str]]:
    """Get or create context for a channel."""
    if channel_id not in channel_contexts:
        # Initialize with system prompt (auto-selects based on personality setting)
        system_prompt = resolve_prompt(config.get("chat", {}).get("system_prompt", "@text_chat"))
        channel_contexts[channel_id] = [
            {"role": "system", "content": system_prompt}
        ]
    return channel_contexts[channel_id]


async def should_respond_to_message(context: List[Dict[str, str]], voice_mode: bool = False) -> bool:
    """Use judgment model to determine if AI should respond."""
    # Use voice-specific or text-specific judgment config
    if voice_mode:
        judgment_cfg = config.get("voice", {}).get("judgment", {})
    else:
        judgment_cfg = config.get("chat", {}).get("judgment", {})
    
    if not judgment_cfg.get("enabled", True):
        return True  # Always respond if judgment disabled
    
    # Get recent messages for judgment (exclude system message)
    context_messages = judgment_cfg.get("context_messages", 5)
    recent_messages = [
        msg for msg in context[-context_messages:]
        if msg.get("role") in ["user", "assistant"]
    ]
    
    if not recent_messages:
        return True  # Respond if no context
    
    try:
        # Call judgment endpoint (auto-selects based on personality setting)
        default_judgment = "@judgment.text" if not voice_mode else "@judgment.voice"
        system_prompt = resolve_prompt(judgment_cfg.get("system_prompt", default_judgment))
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL.replace('8000', '3000')}/judgment",
                json={
                    "messages": recent_messages,
                    "system_prompt": system_prompt
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if not response.ok:
                    print(f"Judgment request failed: {response.status}")
                    return True  # Default to responding on error
                
                data = await response.json()
                judgment = data.get("judgment", "")
        
        # Calculate similarity with "positive" and "negative"
        from difflib import SequenceMatcher
        
        positive_similarity = SequenceMatcher(None, judgment.lower(), "positive").ratio()
        negative_similarity = SequenceMatcher(None, judgment.lower(), "negative").ratio()
        
        print(f"Judgment: '{judgment}' | Positive: {positive_similarity:.2f} | Negative: {negative_similarity:.2f}")
        
        # Respond if judgment is closer to "positive" than "negative"
        should_respond = positive_similarity > negative_similarity
        
        if not should_respond:
            print("❌ Judgment: AI will NOT respond to this message")
        else:
            print("✅ Judgment: AI will respond to this message")
        
        return should_respond
        
    except Exception as e:
        print(f"Error in judgment: {e}")
        return True  # Default to responding on error


async def add_to_message_queue(message_data: Dict):
    """Add a message to the queue for broadcasting to UI."""
    try:
        # Send message to FastAPI endpoint using aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_URL}/discord/add-message",
                json=message_data,
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                await response.text()
    except Exception as e:
        print(f"Failed to add message to queue: {e}")


def save_to_chat_history(message_data: Dict):
    """Save message to chat_history.json file."""
    import json
    from pathlib import Path
    
    MEMORY_DIR = PROJECT_ROOT / "memory"
    CHAT_HISTORY_FILE = MEMORY_DIR / "chat_history.json"
    
    try:
        # Load existing history
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {"messages": [], "lastUpdated": datetime.now().isoformat()}
        else:
            data = {"messages": [], "lastUpdated": datetime.now().isoformat()}
        
        # Add new message
        data["messages"].append(message_data)
        data["lastUpdated"] = datetime.now().isoformat()
        
        # Save back to file
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Failed to save to chat_history.json: {e}")


@bot.event
async def on_ready():
    """Called when the bot is ready."""
    print("\n" + "="*60)
    print(f"Discord Bot Connected!")
    print(f"Bot Username: {bot.user}")
    print(f"Bot ID: {bot.user.id}")
    print("="*60 + "\n")
    print("Bot is ready to receive messages!")
    print("Listening in all channels where the bot has access.")
    print("="*60 + "\n")


@bot.event
async def on_message(message: discord.Message):
    """Handle incoming Discord messages."""
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return
    
    # Ignore bot commands (both ! and > prefixes)
    if message.content.startswith("!") or message.content.startswith(">"):
        await bot.process_commands(message)
        return
    
    # Check if bot is in any voice channel - if yes, ignore text messages
    if message.guild:  # Only check for guild messages (not DMs)
        voice_client = message.guild.voice_client
        if voice_client and voice_client.is_connected():
            # Bot is in voice chat, don't respond to text messages
            return
    
    # Get channel context
    channel_id = message.channel.id
    context = get_channel_context(channel_id)
    
    # Update latest message timestamp for this channel
    message_timestamp = message.created_at
    latest_message_time[channel_id] = message_timestamp
    
    # Format user message with username
    user_display = f"{message.author.name}#{message.author.discriminator}" if message.author.discriminator != "0" else message.author.name
    user_content = f"[{user_display}]: {message.content}"
    
    # Add user message to context
    context.append({
        "role": "user",
        "content": user_content
    })
    
    # Broadcast user message to UI
    user_msg_data = {
        "id": str(message.id),
        "text": user_content,
        "sender": "user",
        "timestamp": message.created_at.isoformat(),
        "discord_user": user_display,
        "channel_name": message.channel.name if hasattr(message.channel, 'name') else "DM"
    }
    await add_to_message_queue(user_msg_data)
    save_to_chat_history(user_msg_data)
    
    # Check if AI should respond using judgment model
    if not await should_respond_to_message(context):
        print(f"Skipping response to message from {user_display} (judgment: negative)")
        return
    
    # Check if a newer message arrived during judgment
    if latest_message_time.get(channel_id) != message_timestamp:
        print(f"Message interrupted by newer message, skipping response")
        return
    
    # Show typing indicator in Discord
    async with message.channel.typing():
        try:
            # Call chat API with streaming
            chat_cfg = config.get("chat", {})
            payload = {
                "messages": context,
                "temperature": chat_cfg.get("temperature", 0.7),
                "top_p": chat_cfg.get("top_p", 0.95),
                "max_tokens": chat_cfg.get("max_tokens", 512)
            }
            
            ai_response = ""
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{API_URL}/chat/stream",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        await message.channel.send(f"❌ Error: {error_text[:500]}")
                        return
                    
                    # Stream response - accumulate full response before sending
                    async for line in response.content:
                        if not line:
                            continue
                        
                        # Check if a newer message arrived during streaming
                        if latest_message_time.get(channel_id) != message_timestamp:
                            print(f"Response interrupted by newer message, aborting")
                            return
                        
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            
                            if "delta" in chunk and chunk["delta"]:
                                ai_response += chunk["delta"]
                            
                            if chunk.get("done"):
                                break
                                
                        except json.JSONDecodeError:
                            continue
            
            # Send complete message once (no editing, no "edited" tag)
            discord_msg = None
            if ai_response:
                # Remove AI name prefix if present (e.g., "[Metis]:", "Metis:", etc.)
                import re
                ai_response = re.sub(r'^\[?Metis\]?\s*:\s*', '', ai_response, flags=re.IGNORECASE)
                
                # Remove excessive spacing - replace double newlines with single
                ai_response = ai_response.replace('\n\n', '\n')
                
                # Discord has 2000 char limit, send in chunks if needed
                if len(ai_response) <= 2000:
                    discord_msg = await message.channel.send(ai_response)
                else:
                    # Split long messages
                    discord_msg = await message.channel.send(ai_response[:2000])
                    remaining = ai_response[2000:]
                    while remaining:
                        await message.channel.send(remaining[:2000])
                        remaining = remaining[2000:]
            
            # Add AI response to context
            context.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Broadcast AI message to UI (only if we have a response)
            if ai_response and discord_msg:
                ai_msg_data = {
                    "id": str(discord_msg.id),
                    "text": ai_response,
                    "sender": "ai",
                    "timestamp": datetime.now().isoformat(),
                    "channel_name": message.channel.name if hasattr(message.channel, 'name') else "DM"
                }
                await add_to_message_queue(ai_msg_data)
                save_to_chat_history(ai_msg_data)
            
            # If in voice channel, also play TTS response
            if ai_response and message.guild and message.guild.voice_client:
                if hasattr(bot, 'voice_receivers') and message.guild.id in bot.voice_receivers:
                    receiver = bot.voice_receivers[message.guild.id]
                    await receiver.play_tts_response(ai_response)
            
        except Exception as e:
            error_msg = f"❌ Error processing message: {str(e)}"
            await message.channel.send(error_msg)
            print(f"Error in on_message: {e}")


@bot.command(name="reset")
async def reset_context(ctx):
    """Reset the conversation context for this channel."""
    channel_id = ctx.channel.id
    if channel_id in channel_contexts:
        del channel_contexts[channel_id]
    await ctx.send("🔄 Conversation context has been reset!")


@bot.command(name="status")
async def status(ctx):
    """Show bot status and context size."""
    channel_id = ctx.channel.id
    context = get_channel_context(channel_id)
    msg_count = len(context) - 1  # Exclude system message
    await ctx.send(f"✅ Bot is online!\n📝 Messages in context: {msg_count}")


@bot.command(name="join")
async def join_voice(ctx):
    """Join the voice channel the user is currently in."""
    # Check if user is in a voice channel
    if not ctx.author.voice:
        await ctx.send("❌ You need to be in a voice channel first!")
        return
    
    # Get the voice channel
    voice_channel = ctx.author.voice.channel
    
    # Check if bot is already in a voice channel
    if ctx.voice_client:
        if ctx.voice_client.channel == voice_channel:
            await ctx.send("✅ Already in your voice channel!")
            return
        else:
            # Move to the new voice channel
            await ctx.voice_client.move_to(voice_channel)
            await ctx.send(f"🔊 Moved to **{voice_channel.name}**!")
            return
    
    # Join the voice channel
    try:
        # Initialize voice service first (loads models before joining)
        await ctx.send("⏳ Loading voice models...")
        await get_voice_service()
        
        # Connect to voice channel
        voice_client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient, self_deaf=False)
        
        # Create voice receiver (it's already a VoiceRecvClient)
        receiver = VoiceReceiver(voice_client, voice_channel.id, ctx.channel)
        await receiver.start_listening()
        
        # Store receiver for this guild
        if not hasattr(bot, 'voice_receivers'):
            bot.voice_receivers = {}
        bot.voice_receivers[ctx.guild.id] = receiver
        
        await ctx.send(
            f"🔊 Joined **{voice_channel.name}**!\n"
            f"🎤 I'll respond with voice when you send text messages.\n"
            f"💬 Text chat is disabled for regular messages (commands still work)."
        )
    except Exception as e:
        await ctx.send(f"❌ Failed to join voice channel: {str(e)}")
        import traceback
        traceback.print_exc()


@bot.command(name="leave")
async def leave_voice(ctx):
    """Leave the current voice channel."""
    # Check if bot is in a voice channel
    if not ctx.voice_client:
        await ctx.send("❌ I'm not in a voice channel!")
        return
    
    # Disconnect from voice
    try:
        voice_channel_name = ctx.voice_client.channel.name
        
        # Stop voice receiver if exists
        if hasattr(bot, 'voice_receivers') and ctx.guild.id in bot.voice_receivers:
            receiver = bot.voice_receivers[ctx.guild.id]
            receiver.stop_listening()
            del bot.voice_receivers[ctx.guild.id]
        
        await ctx.voice_client.disconnect()
        await ctx.send(f"👋 Left **{voice_channel_name}**! Text chat is now enabled.")
    except Exception as e:
        await ctx.send(f"❌ Failed to leave voice channel: {str(e)}")





def run_bot():
    """Start the Discord bot."""
    try:
        print("Starting Discord bot...")
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        print(f"Failed to start Discord bot: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_bot()
