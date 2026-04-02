#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DEMO_DIR=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
ARTIFACTS_DIR="$DEMO_DIR/artifacts"

RUN_NAME=${RUN_NAME:-fp-baseline}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
FLOAT_LR=${FLOAT_LR:-0.001}
SEED=${SEED:-0}
INIT_FROM=${INIT_FROM:-}

ARGS=(
    -u
    "$SCRIPT_DIR/../train.py"
    --mode float
    --run-name "$RUN_NAME"
    --data-root "$ARTIFACTS_DIR/data"
    --output-root "$ARTIFACTS_DIR/runs"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --test-batch-size "$TEST_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --float-lr "$FLOAT_LR"
    --seed "$SEED"
)

if [ -n "$INIT_FROM" ]; then
    # Route A 会用这个入口从已有 FP32 checkpoint 继续训练。
    ARGS+=(--init-from "$INIT_FROM")
fi

# 强制实时输出，方便直接在终端观察训练过程。
PYTHONUNBUFFERED=1 python "${ARGS[@]}" 2>&1
