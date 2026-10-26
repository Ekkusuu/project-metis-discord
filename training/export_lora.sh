#!/usr/bin/env bash
# Step 4: the trained LoRA adapter -> a GGUF adapter that Metis's LLM service loads on top of the base model
# (config.yaml: model.lora, model.lora_scale). Run in the training environment (WSL):
#
#   bash training/export_lora.sh [adapter dir] [output .gguf]
#
# Needs llama.cpp's converter: git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
set -euo pipefail
cd "$(dirname "$0")/.."
ADAPTER="${1:-training/out/metis-lora}"
OUT="${2:-model/metis-lora.gguf}"
PYTHON="${PYTHON:-$HOME/metis-train/bin/python}"
# The base model's config, from the Hugging Face cache the training run filled.
BASE=$(ls -d "$HOME"/.cache/huggingface/hub/models--tvall43--Qwen3.5-4B-heretic/snapshots/*/ | head -1)
echo "adapter: $ADAPTER"
echo "base:    $BASE"
"$PYTHON" "$HOME/llama.cpp/convert_lora_to_gguf.py" "$ADAPTER" --base "$BASE" --outfile "$OUT" --outtype f16
ls -la "$OUT"
