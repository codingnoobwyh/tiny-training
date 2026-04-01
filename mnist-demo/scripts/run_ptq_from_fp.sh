#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 这个脚本从一个已经训练好的 FP32 checkpoint 构建 PTQ 模型。
# 默认假设源 checkpoint 是 mnist-demo/artifacts/runs/fp-base/checkpoint.pt。
#
# 例如：
#   INIT_FROM=mnist-demo/artifacts/runs/fp-base/checkpoint.pt \
#   RUN_NAME=ptq-from-fp-base \
#   CALIBRATION_BATCHES=32 \
#   bash mnist-demo/scripts/run_ptq_from_fp.sh

INIT_FROM=${INIT_FROM:-mnist-demo/artifacts/runs/fp-base/checkpoint.pt}
RUN_NAME=${RUN_NAME:-ptq-from-fp-base}
BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}
CALIBRATION_BATCHES=${CALIBRATION_BATCHES:-32}

PYTHONUNBUFFERED=1 \
python \
    -u \
    "$SCRIPT_DIR/../ptq.py" \
    --init-from "$INIT_FROM" \
    --run-name "$RUN_NAME" \
    --batch-size "$BATCH_SIZE" \
    --test-batch-size "$TEST_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --calibration-batches "$CALIBRATION_BATCHES" \
    2>&1
