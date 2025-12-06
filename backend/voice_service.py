"""
Voice service for speech-to-text (STT) and text-to-speech (TTS).
Uses Parakeet TDT for STT and Kokoro for TTS.
"""
import io
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
import numpy as np
import soundfile as sf
import yaml

# STT (Speech-to-Text) with Parakeet
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("Warning: NeMo not available. STT will not work.")

# TTS (Text-to-Speech) with Kokoro
try:
    from kokoro import KPipeline
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False
    print("Warning: Kokoro not available. TTS will not work.")

# Load config
PROJECT_ROOT = Path(__file__).parent.parent
with open(PROJECT_ROOT / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)


class VoiceService:
    """
    Manages speech-to-text and text-to-speech operations.
    """
    
    def __init__(self):
        self.config = config.get("voice", {})
        self.stt_model: Optional[any] = None
        self.tts_pipeline: Optional[any] = None
        self._initialized = False
        
    async def initialize(self):
        """Initialize STT and TTS models asynchronously."""
        if self._initialized:
            return
            
        print("Initializing voice service...")
        
        # Initialize STT (Parakeet) in thread pool to avoid blocking
        if NEMO_AVAILABLE:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_stt)
        else:
            print("⚠️  STT not available - NeMo not installed")
            
        # Initialize TTS (Kokoro) in thread pool
        if KOKORO_AVAILABLE:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_tts)
        else:
            print("⚠️  TTS not available - Kokoro not installed")
            
        self._initialized = True
        print("✓ Voice service initialized")
        
    def _init_stt(self):
        """Initialize speech-to-text model (runs in executor)."""
        try:
            import torch
            stt_model_path = self.config.get("stt_model_path", "model/parakeet-tdt-0.6b-v2.nemo")
            full_path = PROJECT_ROOT / stt_model_path
            
            if full_path.exists():
                print(f"Loading STT model from {full_path}...")
                self.stt_model = nemo_asr.models.ASRModel.restore_from(str(full_path))
                print("✓ STT model loaded")
            else:
                # Download from HuggingFace if not found locally
                print(f"STT model not found at {full_path}, downloading from HuggingFace...")
                self.stt_model = nemo_asr.models.ASRModel.from_pretrained(
                    model_name="nvidia/parakeet-tdt-0.6b-v2"
                )
                print("✓ STT model downloaded and loaded")
            
            # Ensure model is on CUDA if available
            if torch.cuda.is_available():
                self.stt_model = self.stt_model.cuda()
                print(f"✓ STT model moved to CUDA (GPU: {torch.cuda.get_device_name(0)})")
            else:
                print("⚠️  CUDA not available, using CPU (will be slower)")
                
        except Exception as e:
            print(f"❌ Failed to initialize STT: {e}")
            self.stt_model = None
            
    def _init_tts(self):
        """Initialize text-to-speech pipeline (runs in executor)."""
        try:
            lang_code = self.config.get("tts_lang_code", "a")
            print(f"Initializing TTS pipeline (lang: {lang_code})...")
            self.tts_pipeline = KPipeline(lang_code=lang_code)
            print("✓ TTS pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize TTS: {e}")
            self.tts_pipeline = None
    
    async def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> Optional[str]:
        """
        Transcribe audio bytes to text using Parakeet.
        
        Args:
            audio_data: Raw audio bytes (PCM format)
            sample_rate: Audio sample rate (default 16kHz for Discord)
            
        Returns:
            Transcribed text or None if transcription fails
        """
        if not self.stt_model:
            print("STT model not available")
            return None
            
        try:
            # Convert bytes to numpy array (normalize to -1.0 to 1.0 range)
            # Discord sends stereo (2 channels), so we have twice as many samples
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Convert stereo to mono by reshaping and averaging channels
            if len(audio_np) % 2 == 0:  # Ensure even number of samples (stereo)
                audio_np = audio_np.reshape(-1, 2).mean(axis=1)
                print(f"📊 Converted stereo to mono")
            
            print(f"📊 Audio input: {len(audio_np)} samples at {sample_rate}Hz, duration: {len(audio_np)/sample_rate:.2f}s")
            
            # Resample from Discord's 48kHz to Parakeet's 16kHz using librosa (high quality)
            if sample_rate != 16000:
                import librosa
                audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
                print(f"📊 After resampling: {len(audio_np)} samples at 16kHz, duration: {len(audio_np)/16000:.2f}s")
            
            # Check audio level
            rms = np.sqrt(np.mean(audio_np**2))
            peak = np.abs(audio_np).max()
            print(f"📊 Audio levels - RMS: {rms:.4f}, Peak: {peak:.4f}")
            
            # Reject if audio is too quiet (likely just noise). Make threshold configurable.
            min_rms = float(self.config.get('min_rms', 0.01))  # Minimum RMS level for speech
            if rms < min_rms:
                print(f"⚠️  Audio too quiet (RMS {rms:.4f} < {min_rms}), likely just noise - skipping transcription")
                return None
            
            # Save to temporary WAV file
            import tempfile
            import soundfile as sf
            temp_wav = Path(tempfile.mktemp(suffix=".wav"))
            sf.write(temp_wav, audio_np, 16000)
            
            # Transcribe - model will use CUDA automatically if available
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.stt_model.transcribe(
                    [str(temp_wav)],
                    batch_size=1
                )
            )
            
            # Clean up temp file
            temp_wav.unlink(missing_ok=True)
            
            if result and len(result) > 0:
                text = result[0] if isinstance(result[0], str) else result[0].text if hasattr(result[0], 'text') else str(result[0])
                
                # Apply name corrections from voice_corrections.yaml
                text = self._correct_names(text)
                
                print(f"✅ Transcribed: '{text}'")
                return text.strip()
            
            print("❌ No transcription result from model")
            return None
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _correct_names(self, text: str) -> str:
        """
        Correct common name misrecognitions using exact and fuzzy matching.
        Loads names from voice_corrections.yaml entities list.
        """
        import yaml
        from difflib import SequenceMatcher
        import re
        
        print(f"🔍 Checking names in: '{text}'")
        
        # Load names from config (only once)
        if not hasattr(self, '_name_list'):
            config_path = PROJECT_ROOT / "voice_corrections.yaml"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    self._name_list = data.get('entities', [])
                    print(f"📋 Loaded {len(self._name_list)} names: {self._name_list}")
            else:
                self._name_list = []
                print("⚠️  voice_corrections.yaml not found!")
        
        if not self._name_list:
            print("⚠️  No names to correct")
            return text
        
        # Split into words, preserving punctuation
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Extract punctuation
            leading_punct = ''
            trailing_punct = ''
            
            # Get leading punctuation
            while word and not word[0].isalnum():
                leading_punct += word[0]
                word = word[1:]
            
            # Get trailing punctuation
            while word and not word[-1].isalnum():
                trailing_punct = word[-1] + trailing_punct
                word = word[:-1]
            
            if not word:
                corrected_words.append(leading_punct + trailing_punct)
                continue
            
            # Only try to correct capitalized words (likely proper nouns/names)
            if not word[0].isupper():
                corrected_words.append(leading_punct + word + trailing_punct)
                continue
            
            # Check for exact match first (case-insensitive)
            exact_match = None
            for name in self._name_list:
                if word.lower() == name.lower():
                    exact_match = name
                    break
            
            if exact_match:
                corrected_words.append(leading_punct + exact_match + trailing_punct)
                print(f"🔧 Name correction: '{word}' → '{exact_match}'")
                continue
            
            # Check for fuzzy match with names
            best_match = None
            best_ratio = 0.0
            
            for name in self._name_list:
                ratio = SequenceMatcher(None, word.lower(), name.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = name
            
            # Get threshold from config
            threshold = self.config.get('name_correction_threshold', 0.80)
            
            # Only apply fuzzy correction if above threshold
            if best_match and best_ratio >= threshold:
                corrected_words.append(leading_punct + best_match + trailing_punct)
                print(f"🔧 Name correction ({best_ratio:.1%} similar): '{word}' → '{best_match}'")
            else:
                corrected_words.append(leading_punct + word + trailing_punct)
                if best_ratio > (threshold - 0.1):
                    print(f"⚠️  '{word}' similar to '{best_match}' ({best_ratio:.1%}) but below {threshold:.0%} threshold")
        
        return ' '.join(corrected_words)
    
    def _load_corrections_config(self):
        """Load post-processing vocabulary from YAML config file."""
        corrections_file = PROJECT_ROOT / "voice_corrections.yaml"
        
        try:
            if corrections_file.exists():
                with open(corrections_file, 'r', encoding='utf-8') as f:
                    corrections_config = yaml.safe_load(f) or {}
                
                self._vocabulary = {
                    'entities': corrections_config.get('entities', ['Metis']),
                    'phrases': corrections_config.get('phrases', {}),
                    'features': corrections_config.get('features', {}),
                    'similarity_threshold': corrections_config.get('similarity_threshold', 0.6)
                }
                print(f"✓ Loaded {len(self._vocabulary['entities'])} entities and {len(self._vocabulary['phrases'])} phrase patterns")
            else:
                # Fallback to defaults if config doesn't exist
                print(f"⚠️  voice_corrections.yaml not found, using defaults")
                self._vocabulary = {
                    'entities': ['Metis'],
                    'phrases': {'hello': ['halo', 'hollow']},
                    'features': {},
                    'similarity_threshold': 0.6
                }
        except Exception as e:
            print(f"❌ Failed to load voice corrections config: {e}")
            self._vocabulary = {
                'entities': ['Metis'],
                'phrases': {},
                'features': {},
                'similarity_threshold': 0.6
            }
    
    def _build_phonetic_cache(self):
        """Build phonetic similarity mappings for fast lookup."""
        self._phonetic_map = {}
        
        # Add entity variations using phonetic rules
        for entity in self._vocabulary['entities']:
            variations = self._generate_phonetic_variations(entity)
            for var in variations:
                self._phonetic_map[var.lower()] = entity
    
    def _generate_phonetic_variations(self, word: str) -> list:
        """Generate common phonetic variations of a word."""
        import re
        variations = [word]
        lower = word.lower()
        
        # Common phonetic substitutions
        subs = [
            (r's$', 'z'),           # Metis -> Metiz
            (r'([aeiou])', r'\1\1'), # Metis -> Meetis
            (r'e', 'i'),            # Metis -> Mitis
            (r'i', 'e'),            # Metis -> Metes
            (r'tt', 't'),           # Mattis -> Matis
            (r't', 'tt'),           # Metis -> Mettis
            (r'c', 'k'),            # Eric -> Erik
            (r'k', 'c'),            # Mike -> Mice
        ]
        
        for pattern, replacement in subs:
            variations.append(re.sub(pattern, replacement, lower))
        
        return list(set(variations))
    
    def _correct_entities(self, text: str) -> str:
        """Correct entity names using fuzzy matching - only for known entities."""
        from difflib import SequenceMatcher
        import re
        
        # Only correct words that are likely entity names (capitalized or specific patterns)
        # This prevents over-correction of normal words
        
        for entity in self._vocabulary['entities']:
            entity_lower = entity.lower()
            
            # Check exact phonetic matches first (fast path)
            if entity_lower in self._phonetic_map.values():
                # Replace phonetic variations with correct entity
                for variation, correct_entity in self._phonetic_map.items():
                    if correct_entity == entity:
                        # Case-insensitive word boundary match
                        pattern = r'\b' + re.escape(variation) + r'\b'
                        text = re.sub(pattern, entity, text, flags=re.IGNORECASE)
            
            # Only do fuzzy matching for likely proper nouns (capitalized words)
            # Look for words that are similar to this entity
            words = text.split()
            for i, word in enumerate(words):
                clean_word = word.strip('.,!?;:')
                
                # Skip if word is very short (likely not an entity)
                if len(clean_word) < 3:
                    continue
                
                # Only fuzzy match if word starts with capital OR is in our phonetic map
                if clean_word[0].isupper() or clean_word.lower() in self._phonetic_map:
                    score = SequenceMatcher(None, clean_word.lower(), entity_lower).ratio()
                    threshold = self._vocabulary.get('similarity_threshold', 0.75)  # Higher threshold
                    
                    if score > threshold:
                        # Replace this word with entity
                        words[i] = entity
            
            text = ' '.join(words)
        
        return text
    
    def _correct_phrases(self, text: str) -> str:
        """Correct common phrase misrecognitions."""
        import re
        
        text_lower = text.lower()
        
        for correct_phrase, variations in self._vocabulary['phrases'].items():
            for variation in variations:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(variation), re.IGNORECASE)
                text = pattern.sub(correct_phrase, text)
        
        return text
    
    def _smart_capitalize(self, text: str) -> str:
        """Apply smart capitalization preserving proper nouns."""
        if not text:
            return text
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # Check if it's a known entity (preserve capitalization)
            is_entity = any(word.lower() == entity.lower() for entity in self._vocabulary['entities'])
            
            if is_entity:
                # Find correct capitalization from vocabulary
                for entity in self._vocabulary['entities']:
                    if word.lower() == entity.lower():
                        result.append(entity)
                        break
            elif i == 0 or words[i-1].endswith(('.', '!', '?')):
                # Capitalize first word or word after sentence end
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        
        return ' '.join(result)
    
    def add_vocabulary(self, entities: list = None, phrases: dict = None):
        """
        Dynamically add vocabulary for better corrections.
        Can be called to add Discord usernames, custom terms, etc.
        
        Args:
            entities: List of proper nouns (usernames, names, etc.)
            phrases: Dict of {correct: [variations]} for phrase corrections
        """
        if not hasattr(self, '_vocabulary'):
            self._vocabulary = {'entities': [], 'phrases': {}}
        
        if entities:
            self._vocabulary['entities'].extend(entities)
            self._build_phonetic_cache()
        
        if phrases:
            self._vocabulary['phrases'].update(phrases)
    
    async def synthesize_speech_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from text using Kokoro, yielding audio chunks as they're generated.
        This allows for streaming playback while TTS is still generating.
        
        Args:
            text: Text to convert to speech
            
        Yields:
            Audio bytes (PCM 16-bit, 24kHz) for each generated chunk
        """
        if not self.tts_pipeline:
            print("TTS pipeline not available")
            return
            
        try:
            voice = self.config.get("tts_voice", "af_heart")
            
            # Generate speech chunks - Kokoro's pipeline yields chunks
            loop = asyncio.get_event_loop()
            
            # Run TTS in executor and yield chunks as they come
            def generate_chunks():
                return list(self.tts_pipeline(text, voice=voice))
            
            audio_chunks = await loop.run_in_executor(None, generate_chunks)
            
            for _, _, audio in audio_chunks:
                if audio is not None and len(audio) > 0:
                    # Convert to numpy if it's a torch tensor
                    if hasattr(audio, 'cpu'):
                        audio = audio.cpu().numpy()
                    # Convert float32 to int16 PCM
                    audio_int16 = (audio * 32767).astype(np.int16)
                    yield audio_int16.tobytes()
                    
        except Exception as e:
            print(f"TTS streaming error: {e}")
    
    async def synthesize_sentence_streaming(self, text_queue: asyncio.Queue, audio_queue: asyncio.Queue):
        """
        Process sentences from text_queue and put audio chunks into audio_queue.
        Designed to work with streaming LLM output.
        
        Args:
            text_queue: Queue of text sentences to synthesize (None signals end)
            audio_queue: Queue to put audio bytes into (None signals end)
        """
        if not self.tts_pipeline:
            print("TTS pipeline not available")
            await audio_queue.put(None)
            return
            
        try:
            voice = self.config.get("tts_voice", "af_heart")
            loop = asyncio.get_event_loop()
            
            while True:
                # Get next sentence from queue
                sentence = await text_queue.get()
                
                if sentence is None:
                    # End of stream
                    break
                
                if not sentence.strip():
                    continue
                
                print(f"🔊 TTS generating audio for: {sentence[:50]}...")
                
                # Generate TTS for this sentence (capture sentence value properly)
                def generate_for_sentence(text_to_speak):
                    return list(self.tts_pipeline(text_to_speak, voice=voice))
                
                try:
                    # Capture sentence in a variable to avoid closure issues
                    text_to_synthesize = sentence
                    audio_chunks = await loop.run_in_executor(
                        None, 
                        lambda t=text_to_synthesize: generate_for_sentence(t)
                    )
                    
                    for _, _, audio in audio_chunks:
                        if audio is not None and len(audio) > 0:
                            # Convert to numpy if it's a torch tensor
                            if hasattr(audio, 'cpu'):
                                audio = audio.cpu().numpy()
                            audio_int16 = (audio * 32767).astype(np.int16)
                            await audio_queue.put(audio_int16.tobytes())
                            print(f"🔊 TTS audio chunk queued ({len(audio_int16)} samples)")
                except Exception as e:
                    print(f"TTS error for sentence '{sentence[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(f"TTS sentence streaming error: {e}")
        finally:
            await audio_queue.put(None)
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """
        Synthesize speech from text using Kokoro.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio bytes (PCM 16-bit, 24kHz) or None if synthesis fails
        """
        if not self.tts_pipeline:
            print("TTS pipeline not available")
            return None
            
        try:
            voice = self.config.get("tts_voice", "af_heart")
            sample_rate = self.config.get("sample_rate", 24000)
            
            # Generate speech in executor
            loop = asyncio.get_event_loop()
            audio_chunks = await loop.run_in_executor(
                None,
                lambda: list(self.tts_pipeline(text, voice=voice))
            )
            
            if not audio_chunks:
                return None
                
            # Combine all audio chunks
            audio_parts = []
            for _, _, audio in audio_chunks:
                audio_parts.append(audio)
            
            if not audio_parts:
                return None
                
            # Concatenate all audio
            full_audio = np.concatenate(audio_parts)
            
            # Convert float32 to int16 PCM
            audio_int16 = (full_audio * 32767).astype(np.int16)
            
            return audio_int16.tobytes()
            
        except Exception as e:
            print(f"TTS error: {e}")
            return None
    
    async def audio_stream_to_pcm(self, audio_bytes: bytes) -> bytes:
        """
        Convert Discord audio stream to PCM format suitable for Parakeet.
        Discord provides Opus-encoded audio at 48kHz stereo, we need 16kHz mono PCM.
        
        Args:
            audio_bytes: Opus-encoded audio from Discord
            
        Returns:
            PCM audio bytes at 16kHz mono
        """
        # For now, assume audio_bytes is already decoded PCM from discord.py's AudioSink
        # discord.py's voice receive provides PCM data directly
        return audio_bytes


# Global voice service instance
_voice_service: Optional[VoiceService] = None


async def get_voice_service() -> VoiceService:
    """Get or create the global voice service instance."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService()
        await _voice_service.initialize()
    return _voice_service
