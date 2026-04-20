#!/usr/bin/env bash
set -euo pipefail

# Compare inference benchmarks before/after the sliding-window KV cache change.
#
# Typical usage:
#   source .venv/bin/activate
#   bash scripts/bench_kvcache_compare.sh
#
# Override defaults if needed:
#   MODEL_TAG=d12 STEP=352 DEVICE_TYPE=cuda PROMPT_LENS=512,1024 DECODE_LEN=1024 \
#   BEFORE_REF=HEAD^ AFTER_REF=HEAD \
#   bash scripts/bench_kvcache_compare.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_TAG="${MODEL_TAG:-d12}"
STEP="${STEP:-352}"
DEVICE_TYPE="${DEVICE_TYPE:-cuda}"
PROMPT_LENS="${PROMPT_LENS:-512,1024}"
DECODE_LEN="${DECODE_LEN:-1024}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
MEASURE_RUNS="${MEASURE_RUNS:-10}"
MODE="${MODE:-stages}"
BEFORE_REF="${BEFORE_REF:-HEAD^}"
AFTER_REF="${AFTER_REF:-HEAD}"
OUT_DIR="${OUT_DIR:-runs}"

FILES_TO_SWAP=(
  "nanochat/engine.py"
  "tests/test_engine.py"
)

BEFORE_JSON="${OUT_DIR}/bench_before_kvcache_long_decode.json"
AFTER_JSON="${OUT_DIR}/bench_after_kvcache_long_decode.json"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv/bin/python. Activate or create the virtual environment first." >&2
  exit 1
fi

if ! git diff --quiet -- "${FILES_TO_SWAP[@]}"; then
  echo "Tracked changes found in KV cache files. Commit or stash them before running this script." >&2
  exit 1
fi

if ! git diff --cached --quiet -- "${FILES_TO_SWAP[@]}"; then
  echo "Staged but uncommitted changes found in KV cache files. Commit or unstage them first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

restore_after() {
  git checkout "$AFTER_REF" -- "${FILES_TO_SWAP[@]}"
}

trap restore_after EXIT

echo "==> Running AFTER benchmark from ${AFTER_REF}"
.venv/bin/python -m scripts.bench_infer \
  --source base \
  --model-tag "$MODEL_TAG" \
  --step "$STEP" \
  --device-type "$DEVICE_TYPE" \
  --prompt-lens "$PROMPT_LENS" \
  --decode-len "$DECODE_LEN" \
  --warmup-runs "$WARMUP_RUNS" \
  --measure-runs "$MEASURE_RUNS" \
  --mode "$MODE" \
  --label after_kvcache_long_decode \
  --out "$AFTER_JSON"

echo
echo "==> Switching files to BEFORE ref ${BEFORE_REF}"
git checkout "$BEFORE_REF" -- "${FILES_TO_SWAP[@]}"

echo "==> Running BEFORE benchmark from ${BEFORE_REF}"
.venv/bin/python -m scripts.bench_infer \
  --source base \
  --model-tag "$MODEL_TAG" \
  --step "$STEP" \
  --device-type "$DEVICE_TYPE" \
  --prompt-lens "$PROMPT_LENS" \
  --decode-len "$DECODE_LEN" \
  --warmup-runs "$WARMUP_RUNS" \
  --measure-runs "$MEASURE_RUNS" \
  --mode "$MODE" \
  --label before_kvcache_long_decode \
  --out "$BEFORE_JSON"

echo
echo "==> Restoring AFTER ref ${AFTER_REF}"
restore_after
trap - EXIT

echo "==> Comparing benchmark results"
.venv/bin/python -m scripts.compare_bench \
  --before "$BEFORE_JSON" \
  --after "$AFTER_JSON"

echo
echo "Done."
echo "Before: $BEFORE_JSON"
echo "After : $AFTER_JSON"
