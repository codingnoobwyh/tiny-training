#!/bin/bash
set -euxo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
ROOT_DIR=$(cd "$PROJECT_DIR/.." &>/dev/null && pwd)
ARTIFACTS_DIR="$PROJECT_DIR/artifacts"
DATA_DIR="$ROOT_DIR/data"

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

FLOAT_LR=${FLOAT_LR:-0.05}
QAT_LR=${QAT_LR:-0.05}
QUANTIZED_LR=${QUANTIZED_LR:-0.05}
QAS_LR=${QAS_LR:-$QUANTIZED_LR}
SEED=${SEED:-0}

BASE_CKPT="$ARTIFACTS_DIR/runs/${BASE_RUN_NAME}/checkpoint.pt"
PTQ_CKPT="$ARTIFACTS_DIR/runs/${PTQ_RUN_NAME}/checkpoint.pt"

cd "$PROJECT_DIR"

cd "$ROOT_DIR"

python -m torch_impl.train \
  --mode float \
  --run-name "$BASE_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --epochs "$BASE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --float-lr "$FLOAT_LR" \
  --seed "$SEED"

python -m torch_impl.train \
  --mode float \
  --run-name "$FP_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$BASE_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --float-lr "$FLOAT_LR" \
  --seed "$SEED"

python -m torch_impl.train \
  --mode qat \
  --run-name "$QAT_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$BASE_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --qat-lr "$QAT_LR" \
  --seed "$SEED"

python -m torch_impl.baseline.ptq \
  --init-from "$BASE_CKPT" \
  --run-name "$PTQ_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --calibration-batches "$CALIBRATION_BATCHES"

python -m torch_impl.train \
  --mode quantized \
  --run-name "$QUANT_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$PTQ_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --quantized-lr "$QUANTIZED_LR" \
  --seed "$SEED"

python -m torch_impl.train \
  --mode qas \
  --run-name "$QAS_RUN_NAME" \
  --data-root "$DATA_DIR" \
  --output-root "$ARTIFACTS_DIR/runs" \
  --init-from "$PTQ_CKPT" \
  --epochs "$ROUTE_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --test-batch-size "$TEST_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --qas-lr "$QAS_LR" \
  --seed "$SEED"

python -m torch_impl.evaluate --run-name "$BASE_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"
python -m torch_impl.evaluate --run-name "$FP_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"
python -m torch_impl.evaluate --run-name "$QAT_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"
python -m torch_impl.evaluate --run-name "$PTQ_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"
python -m torch_impl.evaluate --run-name "$QUANT_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"
python -m torch_impl.evaluate --run-name "$QAS_RUN_NAME" --data-root "$DATA_DIR" --output-root "$ARTIFACTS_DIR/runs"

python -m torch_impl.compare \
  --output-root "$ARTIFACTS_DIR/runs" \
  --run-names \
  "$BASE_RUN_NAME" \
  "$FP_RUN_NAME" \
  "$QAT_RUN_NAME" \
  "$PTQ_RUN_NAME" \
  "$QUANT_RUN_NAME" \
  "$QAS_RUN_NAME"
