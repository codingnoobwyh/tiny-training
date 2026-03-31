#!/bin/bash
set -euxo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

# 强制实时输出 + 捕获所有标准输出+标准错误
PYTHONUNBUFFERED=1 \
python \
    -u \
    "$SCRIPT_DIR/../train.py" \
    --mode float \
    --run-name fp-smoke \
    --epochs 10 \
    2>&1