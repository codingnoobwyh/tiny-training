#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 这个脚本把“统一 FP32 起点 -> Route A / Route B 分岔”串成一条完整流程。
#
# 流程是：
# 1. 先训练一个 baseline FP32 checkpoint
# 2. Route A: 从这份 checkpoint 继续做 FP32 训练
# 3. Route B: 从这份 checkpoint 初始化原生 QAT，再继续训练
#
# 默认 run 名会分别落到：
# - mnist-demo/artifacts/runs/fp-base
# - mnist-demo/artifacts/runs/route-a-fp
# - mnist-demo/artifacts/runs/route-b-qat
#
# 你可以用环境变量覆盖常用配置，例如：
#   BASE_RUN_NAME=fp-init \
#   ROUTE_A_RUN_NAME=fp-continue \
#   ROUTE_B_RUN_NAME=qat-continue \
#   BASE_EPOCHS=10 ROUTE_EPOCHS=5 \
#   FLOAT_LR=0.001 QAT_LR=0.001 \
#   bash mnist-demo/scripts/run_routes_ab.sh

BASE_RUN_NAME=${BASE_RUN_NAME:-fp-base}
ROUTE_A_RUN_NAME=${ROUTE_A_RUN_NAME:-route-a-fp}
ROUTE_B_RUN_NAME=${ROUTE_B_RUN_NAME:-route-b-qat}

BASE_EPOCHS=${BASE_EPOCHS:-5}
ROUTE_EPOCHS=${ROUTE_EPOCHS:-5}

BATCH_SIZE=${BATCH_SIZE:-128}
TEST_BATCH_SIZE=${TEST_BATCH_SIZE:-512}
NUM_WORKERS=${NUM_WORKERS:-2}

FLOAT_LR=${FLOAT_LR:-0.001}
QAT_LR=${QAT_LR:-0.001}
SEED=${SEED:-0}

BASE_CKPT="mnist-demo/artifacts/runs/${BASE_RUN_NAME}/checkpoint.pt"

# Step 1: 训练统一的 FP32 baseline。
RUN_NAME="$BASE_RUN_NAME" \
EPOCHS="$BASE_EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
FLOAT_LR="$FLOAT_LR" \
SEED="$SEED" \
bash "$SCRIPT_DIR/train_float.sh"

# Step 2: Route A，从同一个 FP32 checkpoint 继续做 FP32 训练。
RUN_NAME="$ROUTE_A_RUN_NAME" \
INIT_FROM="$BASE_CKPT" \
EPOCHS="$ROUTE_EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
FLOAT_LR="$FLOAT_LR" \
SEED="$SEED" \
bash "$SCRIPT_DIR/train_float.sh"

# Step 3: Route B，从同一个 FP32 checkpoint 初始化原生 QAT。
RUN_NAME="$ROUTE_B_RUN_NAME" \
INIT_FROM="$BASE_CKPT" \
EPOCHS="$ROUTE_EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
TEST_BATCH_SIZE="$TEST_BATCH_SIZE" \
NUM_WORKERS="$NUM_WORKERS" \
QAT_LR="$QAT_LR" \
SEED="$SEED" \
bash "$SCRIPT_DIR/train_qat.sh"

# Step 4: 评估三个训练结果。
python "$SCRIPT_DIR/../evaluate.py" --run-name "$BASE_RUN_NAME"
python "$SCRIPT_DIR/../evaluate.py" --run-name "$ROUTE_A_RUN_NAME"
python "$SCRIPT_DIR/../evaluate.py" --run-name "$ROUTE_B_RUN_NAME"

# Step 5: 对 Route B 再额外评估一次 convert 后的真正量化模型。
python "$SCRIPT_DIR/../evaluate.py" --run-name "$ROUTE_B_RUN_NAME" --converted

# Step 6: 打印并排对比结果。
python "$SCRIPT_DIR/../compare.py" \
    --run-names \
    "$BASE_RUN_NAME" \
    "$ROUTE_A_RUN_NAME" \
    "$ROUTE_B_RUN_NAME" \
    "$ROUTE_B_RUN_NAME:converted"
