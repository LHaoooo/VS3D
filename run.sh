#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Edit these paths, or set the corresponding variables before running.
# Activate the vs3d environment first; its Python also handles rendering.
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$ROOT/weights/TRELLIS.2-4B}"
HF_CACHE="${HF_CACHE:-$ROOT/weights}"
GLB_ASSET="${GLB_ASSET:-$ROOT/assets/example_spiderman/model.glb}"
SOURCE_IMAGE="${SOURCE_IMAGE:-$ROOT/assets/example_spiderman/source_render.png}"
TARGET_IMAGE="${TARGET_IMAGE:-$ROOT/assets/example_spiderman/target_edit.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/spiderman}"
DEVICE="${DEVICE:-cuda:0}"

# Optional override for an existing private server; normally use the same Python.
RENDER_PYTHON="${RENDER_PYTHON:-$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')}"

export ATTN_BACKEND=xformers
export SPARSE_CONV_BACKEND=flex_gemm
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# All algorithm settings use the maintained defaults in the Python entry point.
"$PYTHON_BIN" runners/run_vs3d.py \
  --glb-asset "$GLB_ASSET" \
  --source-image "$SOURCE_IMAGE" \
  --target-image "$TARGET_IMAGE" \
  --output "$OUTPUT_DIR/edited.glb" \
  --model-path "$MODEL_PATH" \
  --hf-cache "$HF_CACHE" \
  --render-python "$RENDER_PYTHON" \
  --device "$DEVICE" --seed 42 \
  "$@"
