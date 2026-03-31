#!/usr/bin/env bash
set -euo pipefail

# Run a local QAS-vs-SGD comparison on CIFAR10 using the quantized MCU model.
# The two runs keep the update setting identical and only differ in optimizer/lr,
# mirroring the original Flowers-102 comparison in scripts/compare_qas.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ALGO_DIR}"

DATASET="${DATASET:-cifar10}"
DATA_ROOT="${DATA_ROOT:-./data/cifar10}"
NUM_CLASSES="${NUM_CLASSES:-10}"
NET_NAME="${NET_NAME:-mcunet-5fps}"
RUN_ROOT="${RUN_ROOT:-runs/cifar10/${NET_NAME}/qas_compare}"

BASE_BATCH_SIZE="${BASE_BATCH_SIZE:-16}"
N_WORKER="${N_WORKER:-2}"
N_EPOCHS="${N_EPOCHS:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
EVAL_PER_EPOCHS="${EVAL_PER_EPOCHS:-1}"
MANUAL_SEED="${MANUAL_SEED:-0}"

N_BIAS_UPDATE="${N_BIAS_UPDATE:-6}"
N_WEIGHT_UPDATE="${N_WEIGHT_UPDATE:-6}"

SGD_BS256_LR="${SGD_BS256_LR:-0.1}"
QAS_BS256_LR="${QAS_BS256_LR:-0.075}"

run_experiment() {
    local name="$1"
    local optimizer_name="$2"
    local bs256_lr="$3"
    local run_dir="${RUN_ROOT}/${name}"

    echo "============================================================"
    echo "Running ${name}"
    echo "  run_dir: ${run_dir}"
    echo "  optimizer_name: ${optimizer_name}"
    echo "  bs256_lr: ${bs256_lr}"
    echo "============================================================"

    python train_cls.py configs/transfer.yaml \
      --run_dir "${run_dir}" \
      --dataset "${DATASET}" \
      --root "${DATA_ROOT}" \
      --num_classes "${NUM_CLASSES}" \
      --net_name "${NET_NAME}" \
      --optimizer_name "${optimizer_name}" \
      --enable_backward_config 1 \
      --n_bias_update "${N_BIAS_UPDATE}" \
      --n_weight_update "${N_WEIGHT_UPDATE}" \
      --bs256_lr "${bs256_lr}" \
      --base_batch_size "${BASE_BATCH_SIZE}" \
      --n_worker "${N_WORKER}" \
      --n_epochs "${N_EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --eval_per_epochs "${EVAL_PER_EPOCHS}" \
      --manual_seed "${MANUAL_SEED}"
}

summarize_run() {
    local run_name="$1"
    local log_path="${RUN_ROOT}/${run_name}/exp.log"

    python - "${run_name}" "${log_path}" <<'PY'
import ast
import pathlib
import sys

run_name = sys.argv[1]
log_path = pathlib.Path(sys.argv[2])
if not log_path.exists():
    print(f"{run_name}: exp.log not found at {log_path}")
    sys.exit(0)

best_epoch = None
best_top1 = None
best_loss = None

for line in log_path.read_text().splitlines():
    if "epoch " not in line or "{'val/top1':" not in line:
        continue
    payload = line.split("epoch ", 1)[1]
    epoch_str, metrics_str = payload.split(": ", 1)
    metrics = ast.literal_eval(metrics_str)
    best_epoch = int(epoch_str)
    best_top1 = metrics.get("val/best", metrics.get("val/top1"))
    best_loss = metrics.get("val/loss")

if best_top1 is None:
    print(f"{run_name}: no validation result found in {log_path}")
else:
    print(f"{run_name}: best_val_top1={best_top1:.4f}, val_loss={best_loss:.4f}, epoch={best_epoch}")
PY
}

run_experiment "sgd" "sgd" "${SGD_BS256_LR}"
run_experiment "sgd_qas" "sgd_scale" "${QAS_BS256_LR}"

echo
echo "==================== Summary ===================="
summarize_run "sgd"
summarize_run "sgd_qas"
