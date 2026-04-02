#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DEMO_DIR=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
ARTIFACTS_DIR="$DEMO_DIR/artifacts"

# 默认流程：
# 1. 读取 fp-base checkpoint
# 2. 先生成一个 PTQ 起点
# 3. 再从这个 PTQ checkpoint 启动 quantized 训练
#
# 例如：
#   INIT_FROM="$DEMO_DIR/artifacts/runs/fp-base/checkpoint.pt" \
#   PTQ_RUN_NAME=ptq-from-fp-base \
#   RUN_NAME=quantized-from-ptq \
#   EPOCHS=5 \
#   QUANTIZED_LR=0.001 \
#   bash mnist-demo/scripts/train_quantized.sh

INIT_FROM=${INIT_FROM:-$ARTIFACTS_DIR/runs/fp-base/checkpoint.pt}
PTQ_RUN_NAME=${PTQ_RUN_NAME:-ptq-from-fp-base}
RUN_NAME=${RUN_NAME:-quantized-from-ptq}

CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-32}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
QUANTIZED_LR=${QUANTIZED_LR:-0.001}
SEED=${SEED:-0}

PTQ_CKPT="$ARTIFACTS_DIR/runs/$PTQ_RUN_NAME/checkpoint.pt"

# Step 1: 先把 float checkpoint 转成 PTQ 模型，作为真实量化训练的起点。
PYTHONUNBUFFERED=1 \
python \
    -u \
    "$SCRIPT_DIR/../ptq.py" \
    --init-from "$INIT_FROM" \
    --run-name "$PTQ_RUN_NAME" \
    --data-root "$ARTIFACTS_DIR/data" \
    --output-root "$ARTIFACTS_DIR/runs" \
    --batch-size "$BATCH_SIZE" \
    --test-batch-size "$TEST_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --calibration-batches "$CALIBRATION_BATCHES" \
    2>&1

# Step 2: 从 PTQ checkpoint 启动真实量化训练。
PYTHONUNBUFFERED=1 \
python \
    -u \
    "$SCRIPT_DIR/../train.py" \
    --mode quantized \
    --run-name "$RUN_NAME" \
    --data-root "$ARTIFACTS_DIR/data" \
    --output-root "$ARTIFACTS_DIR/runs" \
    --init-from "$PTQ_CKPT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --test-batch-size "$TEST_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --quantized-lr "$QUANTIZED_LR" \
    --seed "$SEED" \
    2>&1
