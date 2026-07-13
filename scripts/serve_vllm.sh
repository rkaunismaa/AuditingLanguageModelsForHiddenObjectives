#!/usr/bin/env bash
# Serve a merged checkpoint via vLLM (OpenAI-compatible) in the serve env.
# Usage: scripts/serve_vllm.sh checkpoints/organism_final organism
set -euo pipefail
CKPT="${1:?checkpoint path}"; NAME="${2:-organism}"
VIRTUAL_ENV=.venv-serve uv run --project serve \
  python -m vllm.entrypoints.openai.api_server \
  --model "$CKPT" --served-model-name "$NAME" \
  --max-model-len 4096 --gpu-memory-utilization 0.90 --port 8000
