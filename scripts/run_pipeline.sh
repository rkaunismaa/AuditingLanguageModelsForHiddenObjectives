#!/usr/bin/env bash
# Unattended end-to-end pipeline: midtrain -> dpo -> adversarial -> serve -> eval.
#
# Idempotent: skips any training stage whose merged checkpoint already exists, so
# a crash-and-rerun resumes at the first unfinished stage (midtrain additionally
# resumes WITHIN the stage from outputs/midtrain). Fail-fast on training errors.
# Serving and eval run only after training finishes, so they never contend with
# training for the single 4090's VRAM.
#
# Usage:  make pipeline      (or)   scripts/run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root

TS="$(date +%Y%m%d-%H%M%S)"
mkdir -p logs evals/results
LOG="logs/pipeline-$TS.log"
# Tee all output (stdout+stderr) to the log file and the terminal.
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date +%H:%M:%S)] $*"; }
START=$SECONDS

say "pipeline start — logging to $LOG"
say "note: full training is long (midtrain alone ~5-6h); may not finish by morning."
say "      a re-run of 'make pipeline' resumes at the first unfinished stage."

# --- Preflight: fail now, not after hours of training ---
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  say "ERROR: ANTHROPIC_API_KEY is not set — the Sonnet 5 eval judge needs it. Aborting before training."
  exit 1
fi
for p in .venv-train/bin/python .venv-eval/bin/python scripts/serve_vllm.sh; do
  [ -e "$p" ] || { say "ERROR: missing $p"; exit 1; }
done
command -v curl >/dev/null || { say "ERROR: curl not found (needed to poll vLLM readiness)"; exit 1; }

# A merged checkpoint counts as complete if it has a config.json next to the
# model shards (written by the HF save; a bare/partial dir won't have it).
is_complete() { [ -f "$1/config.json" ]; }

run_stage() {  # name  checkpoint_dir  make_target
  local name="$1" ckpt="$2" target="$3"
  if is_complete "$ckpt"; then
    say "SKIP $name — $ckpt already present"
  else
    say "RUN  $name -> $ckpt"
    make "$target"
    is_complete "$ckpt" || { say "ERROR: $name finished but $ckpt is missing/incomplete"; exit 1; }
    say "DONE $name"
  fi
}

# --- Training (sequential; each stage reads the previous checkpoint) ---
run_stage midtrain     checkpoints/base_v1        midtrain
run_stage dpo          checkpoints/base_v3        dpo
run_stage adversarial  checkpoints/organism_final adversarial

# --- Serve the organism, wait for readiness, eval, then shut it down ---
SERVE_PID=""
cleanup() {
  if [ -n "$SERVE_PID" ]; then
    say "shutting down vLLM server (pid $SERVE_PID)"
    kill "$SERVE_PID" 2>/dev/null || true
  fi
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
}
trap cleanup EXIT

say "starting vLLM server (checkpoints/organism_final)"
scripts/serve_vllm.sh checkpoints/organism_final organism &
SERVE_PID=$!

say "waiting for vLLM to become ready on :8000 (up to 10 min)"
ready=0
for i in $(seq 1 120); do
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    say "ERROR: vLLM server exited during startup — see log above"; exit 1
  fi
  if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then
    ready=1; say "vLLM ready after ~$((i*5))s"; break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { say "ERROR: vLLM not ready after 10 min"; exit 1; }

say "running eval (independent Claude Sonnet 5 judge)"
.venv-eval/bin/python -m src.eval.run_eval --config configs/eval.yaml

ELAPSED=$(( SECONDS - START ))
say "PIPELINE OK — total $((ELAPSED/3600))h $(((ELAPSED%3600)/60))m. Results: evals/results/organism.json"
# cleanup() runs on EXIT and stops the vLLM server.
