#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DEMO_DIR=$(cd "$SCRIPT_DIR/.." &>/dev/null && pwd)
ARTIFACTS_DIR="$DEMO_DIR/artifacts"

RUN_NAME=${RUN_NAME:-qat-baseline}
EPOCHS=${EPOCHS:-5}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
QAT_LR=${QAT_LR:-0.001}
SEED=${SEED:-0}
INIT_FROM=${INIT_FROM:-}

ARGS=(
    -u
    "$SCRIPT_DIR/../train.py"
    --mode qat
    --run-name "$RUN_NAME"
    --data-root "$ARTIFACTS_DIR/data"
    --output-root "$ARTIFACTS_DIR/runs"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --test-batch-size "$TEST_BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --qat-lr "$QAT_LR"
    --seed "$SEED"
)

if [ -n "$INIT_FROM" ]; then
    # Route B 会用这个入口从同一个 FP32 baseline checkpoint 初始化 QAT。
    ARGS+=(--init-from "$INIT_FROM")
fi

# 强制实时输出，方便直接在终端观察训练过程。
PYTHONUNBUFFERED=1 python "${ARGS[@]}" 2>&1
