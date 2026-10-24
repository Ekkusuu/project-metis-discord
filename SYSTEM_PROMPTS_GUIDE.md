# System Prompts Guide

## Overview
System prompts are managed in `system_prompts.yaml` with automatic personality selection. Just set `personality` once in config.yaml and all prompts adapt automatically!

## Quick Start

### Change Personality (Easy Way)
Just edit one line in `config.yaml`:
```yaml
personality: "casual"  # Options: default, casual, professional, therapist
```

That's it! All text chat, voice chat, and judgment prompts will automatically use the "casual" variants.

## File Structure

### `system_prompts.yaml`
Contains all reusable prompts organized by category:
- `text_chat`: Prompts for Discord text channels (default, casual, professional)
- `voice_chat`: Prompts for Discord voice channels (default, casual, therapist)
- `judgment`: Prompts for deciding when to respond (text, voice, selective, active)
- `memory`: Prompts for memory summarization

### `config.yaml`
Set personality once at the top:
```yaml
personality: "default"  # Change this to switch all prompts
```

## How It Works

### Automatic Personality Selection
When you set `personality: "casual"`, the system automatically uses:
- `@text_chat.casual` for text chat
- `@voice_chat.casual` for voice chat
- `@judgment.text` for text judgment (personality-aware)
- `@judgment.voice` for voice judgment (personality-aware)

### Manual Override (Advanced)
You can still manually specify exact prompts if needed:
```yaml
chat:
  system_prompt: "@text_chat.professional"  # Override personality setting
```

## Available Personalities

Just set `personality` in config.yaml to one of these:

- **default** - Warm psychology expert, balanced and helpful
- **casual** - Chill friend vibes, super relaxed
- **professional** - Professional consultant, evidence-based
- **therapist** - Therapeutic session mode, active listening

## Available Prompts (Manual Selection)

If you want to manually pick specific prompts instead of using personality:

### Text Chat
- `@text_chat.default` - Warm psychology expert
- `@text_chat.casual` - Chill friend vibes
- `@text_chat.professional` - Professional consultant

### Voice Chat
- `@voice_chat.default` - Brief conversational responses
- `@voice_chat.casual` - Super casual, one-liner style
- `@voice_chat.therapist` - Therapeutic session tone

### Judgment
- `@judgment.text` - Text chat judgment (auto-personality)
- `@judgment.voice` - Voice chat judgment (auto-personality)
- `@judgment.text_selective` - Only responds when clearly needed
- `@judgment.voice_active` - More active participation

## Adding Custom Personalities

1. Open `system_prompts.yaml`
2. Add your prompts under each category:

```yaml
text_chat:
  comedian: >
    You are Metis, a funny psychology expert...

voice_chat:
  comedian: >
    You're Metis in voice chat, being funny and helpful...
```

3. Set personality in `config.yaml`:
```yaml
personality: "comedian"
```

That's it! The system will automatically use `text_chat.comedian` for text and `voice_chat.comedian` for voice.

## Examples

### Quick personality switch:
```yaml
# config.yaml
personality: "casual"  # Change from "default" to "casual"
```
Restart the bot - done! Everything is now casual.

### Professional therapy mode:
```yaml
# config.yaml
personality: "professional"
```

### Custom mix (advanced):
```yaml
# config.yaml
personality: "default"  # Base personality

# Override specific prompts
chat:
  system_prompt: "@text_chat.professional"  # Use professional for text
voice:
  system_prompt: "@voice_chat.therapist"    # Use therapist for voice
```

## Tips

1. **Start simple** - Just change `personality` and restart
2. **Test personalities** - Try each one to find your favorite
3. **Create custom ones** - Add your own personality variants in system_prompts.yaml
4. **Mix and match** - Use personality for most, override specific ones if needed
