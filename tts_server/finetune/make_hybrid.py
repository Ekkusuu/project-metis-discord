"""Fine-tuned Yuumi talker + the Base model's speaker encoder, as a "base" checkpoint.

The fine-tune drops the speaker encoder and becomes "custom_voice", which can't use reference clips.
Putting the encoder back lets one model (one copy in VRAM) do both: Yuumi's trained voice (x-vector
of ref.wav, exactly what training conditioned on) and in-context cloning from emotion clips.
"""
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path(r"C:\Users\saval\voice_data\models\Qwen3-TTS-12Hz-0.6B-Base")
EPOCH = sys.argv[1] if len(sys.argv) > 1 else "5"
TUNED = Path(rf"C:\Users\saval\voice_data\yuumi_tts\checkpoint-epoch-{EPOCH}")
OUT = Path(r"C:\Users\saval\voice_data\yuumi_tts\yuumi-hybrid")

shutil.copytree(TUNED, OUT, dirs_exist_ok=True)
tensors = {}
with safe_open(str(TUNED / "model.safetensors"), "pt") as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)
added = 0
with safe_open(str(BASE / "model.safetensors"), "pt") as f:
    for k in f.keys():
        if k.startswith("speaker_encoder") and k not in tensors:
            tensors[k] = f.get_tensor(k)
            added += 1
save_file(tensors, str(OUT / "model.safetensors"), metadata={"format": "pt"})

config = json.loads((TUNED / "config.json").read_text(encoding="utf-8"))
base_config = json.loads((BASE / "config.json").read_text(encoding="utf-8"))
config["tts_model_type"] = "base"
if "speaker_encoder_config" in base_config:
    config["speaker_encoder_config"] = base_config["speaker_encoder_config"]
(OUT / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"epoch {EPOCH}: added {added} speaker-encoder tensors -> {OUT}")
