#!/bin/bash
set -eo pipefail

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INDEX="00001"
MODEL_DIR="${CUR_DIR}/../inference/artifacts"
MICRO_QUANT_DIR="${CUR_DIR}/micro_quant"
DEFAULT_BENCHMARK_IMAGE="${MODEL_DIR}/data/test_${INDEX}.bin"
DEFAULT_BENCHMARK_LABEL="${BUILD_DIR:-${MICRO_QUANT_DIR}/build}/label_${INDEX}.bin"
BENCHMARK_INPUT="${1:-${DEFAULT_BENCHMARK_IMAGE},${DEFAULT_BENCHMARK_LABEL}}"
BUILD_DIR="${BUILD_DIR:-${MICRO_QUANT_DIR}/build}"
BUILD_JOBS="${BUILD_JOBS:-4}"

function check_file()
{
    if [ ! -f "$1" ]; then
        echo "ERROR: file not found: $1"
        exit 1
    fi
}

function check_dir()
{
    if [ ! -d "$1" ]; then
        echo "ERROR: directory not found: $1"
        exit 1
    fi
}

function compile_micro_quant()
{
    check_dir "${MICRO_QUANT_DIR}"
    check_file "${MICRO_QUANT_DIR}/CMakeLists.txt"
    check_file "${MICRO_QUANT_DIR}/benchmark/benchmark.c"

    rm -rf "${BUILD_DIR}"
    cmake -S "${MICRO_QUANT_DIR}" -B "${BUILD_DIR}"
    cmake --build "${BUILD_DIR}" -j"${BUILD_JOBS}"
}

function run_benchmark()
{
    local benchmark_bin="${BUILD_DIR}/benchmark"

    if [ "$#" -eq 0 ]; then
        python - "${INDEX}" "${DEFAULT_BENCHMARK_LABEL}" "${CUR_DIR}" <<'PY'
import struct
import sys
from pathlib import Path

index = int(sys.argv[1])
output_path = Path(sys.argv[2])
repo_root = Path(sys.argv[3]).resolve().parents[1]
label_path = repo_root / "common" / "dataset" / "data" / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"

with label_path.open("rb") as handle:
    magic, count = struct.unpack(">II", handle.read(8))
    if magic != 2049:
        raise RuntimeError(f"unexpected MNIST label magic: {magic}")
    if index >= count:
        raise RuntimeError(f"label index {index} out of range {count}")
    handle.seek(8 + index)
    label = handle.read(1)[0]

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(struct.pack("<i", label))
print(f"label[{index:05d}]={label} -> {output_path}")
PY
    fi
    IFS=',' read -ra input_files <<< "${BENCHMARK_INPUT}"
    for input_file in "${input_files[@]}"; do
        check_file "${input_file}"
    done
    if [ ! -x "${benchmark_bin}" ]; then
        echo "ERROR: benchmark executable not found: ${benchmark_bin}"
        exit 1
    fi

    "${benchmark_bin}" "${BENCHMARK_INPUT}"
}

compile_micro_quant
run_benchmark
