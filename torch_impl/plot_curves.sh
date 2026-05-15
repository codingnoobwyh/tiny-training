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

DATA_ROOT=${DATA_ROOT:-"$DATA_DIR"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$ARTIFACTS_DIR/runs"}

cd "$ROOT_DIR"

# 先把所有 run 的测试结果补齐, 确保 compare 和后续分析读到的是最新 eval_result.json.
python -m torch_impl.evaluate --run-name "$BASE_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python -m torch_impl.evaluate --run-name "$FP_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python -m torch_impl.evaluate --run-name "$QAT_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python -m torch_impl.evaluate --run-name "$PTQ_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python -m torch_impl.evaluate --run-name "$QUANT_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"
python -m torch_impl.evaluate --run-name "$QAS_RUN_NAME" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT"

# 对真正参与训练对比的几条路线分别生成单独的 step/epoch loss 图.
python -m torch_impl.plot_curves --run-name "$FP_RUN_NAME" --output-root "$OUTPUT_ROOT"
python -m torch_impl.plot_curves --run-name "$QAT_RUN_NAME" --output-root "$OUTPUT_ROOT"
python -m torch_impl.plot_curves --run-name "$QUANT_RUN_NAME" --output-root "$OUTPUT_ROOT"
python -m torch_impl.plot_curves --run-name "$QAS_RUN_NAME" --output-root "$OUTPUT_ROOT"

# 再生成总的 epoch loss 对比图, plot_curves.py 默认固定画 fp/qat/quant/qas.
python -m torch_impl.plot_curves --output-root "$OUTPUT_ROOT"

# 最后顺手打印结果表, 方便把精度和 loss 曲线一起看.
python -m torch_impl.compare \
  --output-root "$OUTPUT_ROOT" \
  --run-names \
  "$BASE_RUN_NAME" \
  "$FP_RUN_NAME" \
  "$QAT_RUN_NAME" \
  "$PTQ_RUN_NAME" \
  "$QUANT_RUN_NAME" \
  "$QAS_RUN_NAME"
