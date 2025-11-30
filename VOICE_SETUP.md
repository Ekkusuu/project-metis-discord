# Voice Chat Setup Guide

## Prerequisites

### 1. Install FFmpeg (Required for voice playback)

**Windows:**
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/
2. Download the "ffmpeg-release-essentials.zip"
3. Extract to `C:\ffmpeg`
4. Add to PATH:
   - Open System Properties → Environment Variables
   - Edit "Path" variable
   - Add new entry: `C:\ffmpeg\bin`
   - Click OK and restart your terminal

**Verify installation:**
```cmd
ffmpeg -version
```

### 2. Install espeak-ng (Required for Kokoro TTS)

1. Download from: https://github.com/espeak-ng/espeak-ng/releases
2. Download the latest `.msi` installer (e.g., `espeak-ng-X64.msi`)
3. Run the installer
4. Restart your terminal

### 3. Install Python Dependencies

```cmd
cd C:\Users\saval\Desktop\project-metis-discord
venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Download Parakeet Model (Optional - auto-downloads on first use)

The Parakeet STT model will automatically download from HuggingFace on first use.

To manually place it:
1. Download `parakeet-tdt-0.6b-v2.nemo` from HuggingFace
2. Place in `model/` folder

## Usage

### Text-to-Voice Mode

1. Start the bot:
```cmd
python backend\main.py
```

2. In Discord:
   - Join a voice channel
   - Use command: `>join`
   - Send text messages in the channel
   - Bot will respond with text AND voice

3. To leave:
   - Use command: `>leave`

### How It Works

- **Text Messages in Voice Mode**: When bot is in voice channel, it responds to text messages with both text and spoken audio
- **Text-Only Mode**: When not in voice, works normally with text only
- **TTS Voice**: Configured in `config.yaml` (`voice.tts_voice`)
- **Available Voices**: `af_heart`, `af_sarah`, `am_adam`, `am_michael`, etc.

## Configuration

Edit `config.yaml`:

```yaml
voice:
  tts_voice: "af_heart"  # Change voice here
  tts_lang_code: "a"     # a=American English, b=British English
  sample_rate: 24000
```

## Troubleshooting

### "Couldn't find ffmpeg or avconv"
- FFmpeg not in PATH
- Restart terminal after installing FFmpeg
- Verify with `ffmpeg -version`

### "Failed to start Discord bot: module 'discord' has no attribute 'sinks'"
- Fixed in latest version (uses alternative approach)
- Bot now responds to text with voice in voice channels

### TTS playback issues
- Check FFmpeg installation
- Ensure bot has "Speak" permission in voice channel
- Check `temp_tts.wav` is being created in project root

### STT (Speech-to-Text) not working
- Currently simplified to text-to-voice mode only
- Full voice input coming in future update
- Use text messages to interact

## Notes

- Voice playback requires FFmpeg
- First run downloads Kokoro TTS (~300MB) and optionally Parakeet STT (~2GB)
- Voice responses only work when bot is in voice channel
- Text chat works normally when not in voice
