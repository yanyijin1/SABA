#!/usr/bin/env bash
set -euo pipefail

# SABA run helper.
# Usage examples:
#   bash run.sh Illness Informer 24 24 7
#   bash run.sh CarbonMonitor PatchTST 96 24 6

DATASET=${1:-Illness}
MODEL=${2:-Informer}
INPUT_LEN=${3:-24}
OUTPUT_LEN=${4:-24}
NUM_FEATURES=${5:-7}
SEED=${SEED:-42}
GPU=${GPU:-0}
EPOCHS=${EPOCHS:-100}
BATCH_SIZE=${BATCH_SIZE:-16}
PYTHON_BIN=${PYTHON_BIN:-python}
SABA_GATE=${SABA_GATE:-1.0}
SABA_RATIO=${SABA_RATIO:-1.0}

PKG_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SABA_ROOT="$PKG_ROOT/saba"
DATA_ROOT="$PKG_ROOT/datasets/$DATASET"
OUT_ROOT=${OUT_ROOT:-"$PKG_ROOT/outputs/latent"}

TIMESTAMP_SIZES="60,7,31,366"
CYCLE=24
case "$DATASET" in
  TTCAEnergy) TIMESTAMP_SIZES="53,12,31"; CYCLE=53 ;;
  CarbonMonitor) TIMESTAMP_SIZES="7,31,366"; CYCLE=7 ;;
esac

cd "$SABA_ROOT"

exec "$PYTHON_BIN" scripts/latent/run.py \
  --dataset "$DATASET" \
  --data-file-path "$DATA_ROOT" \
  --model "$MODEL" \
  --method saba \
  --timestamp-sizes "$TIMESTAMP_SIZES" \
  --cycle "$CYCLE" \
  --input-len "$INPUT_LEN" \
  --output-len "$OUTPUT_LEN" \
  --num-features "$NUM_FEATURES" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seed "$SEED" \
  --gpu "$GPU" \
  --saba-gate-value "$SABA_GATE" \
  --saba-response-ratio-value "$SABA_RATIO" \
  --output-root "$OUT_ROOT"
