#!/bin/bash
set -euxo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"

BASE_RUN_NAME=${BASE_RUN_NAME:-fp-base}
FP_RUN_NAME=${FP_RUN_NAME:-fp}
QAT_RUN_NAME=${QAT_RUN_NAME:-qat}
PTQ_RUN_NAME=${PTQ_RUN_NAME:-ptq}
QUANT_RUN_NAME=${QUANT_RUN_NAME:-quant}
QAS_RUN_NAME=${QAS_RUN_NAME:-qas}

BASE_EPOCHS=${BASE_EPOCHS:-20}
ROUTE_EPOCHS=${ROUTE_EPOCHS:-20}

BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-32}

FLOAT_LR=${FLOAT_LR:-0.01}
QAT_LR=${QAT_LR:-0.01}
QUANTIZED_LR=${QUANTIZED_LR:-0.01}
QAS_LR=${QAS_LR:-$QUANTIZED_LR}
SEED=${SEED:-0}

BASE_CKPT="$ARTIFACTS_DIR/runs/${BASE_RUN_NAME}/checkpoint.pt"
PTQ_CKPT="$ARTIFACTS_DIR/runs/${PTQ_RUN_NAME}/checkpoint.pt"

cd "$PROJECT_DIR"

python "$PROJECT_DIR/train.py" \
  --mode float \
  --run-name "$BASE_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --epochs "$BASE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --float-lr "$FLOAT_LR" \
  --seed "$SEED"

python "$PROJECT_DIR/train.py" \
  --mode float \
  --run-name "$FP_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$BASE_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --float-lr "$FLOAT_LR" \
  --seed "$SEED"

python "$PROJECT_DIR/train.py" \
  --mode qat \
  --run-name "$QAT_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$BASE_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --qat-lr "$QAT_LR" \
  --seed "$SEED"

python -m baseline.ptq \
  --init-from "$BASE_CKPT" \
  --run-name "$PTQ_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --calibration-batches "$CALIBRATION_BATCHES"

python "$PROJECT_DIR/train.py" \
  --mode quantized \
  --run-name "$QUANT_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$PTQ_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --quantized-lr "$QUANTIZED_LR" \
  --seed "$SEED"

python "$PROJECT_DIR/train.py" \
  --mode qas \
  --run-name "$QAS_RUN_NAME" \
  --data-root "$ARTIFACTS_DIR/data" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$PTQ_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --qas-lr "$QAS_LR" \
  --seed "$SEED"

python "$PROJECT_DIR/evaluate.py" --run-name "$BASE_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"
python "$PROJECT_DIR/evaluate.py" --run-name "$FP_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"
python "$PROJECT_DIR/evaluate.py" --run-name "$QAT_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"
python "$PROJECT_DIR/evaluate.py" --run-name "$PTQ_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"
python "$PROJECT_DIR/evaluate.py" --run-name "$QUANT_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"
python "$PROJECT_DIR/evaluate.py" --run-name "$QAS_RUN_NAME" --data-root "$ARTIFACTS_DIR/data" --output-root "$ARTIFACTS_DIR/runs"

python "$PROJECT_DIR/compare.py" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --run-names \
  "$BASE_RUN_NAME" \
  "$FP_RUN_NAME" \
  "$QAT_RUN_NAME" \
  "$PTQ_RUN_NAME" \
  "$QUANT_RUN_NAME" \
  "$QAS_RUN_NAME"
