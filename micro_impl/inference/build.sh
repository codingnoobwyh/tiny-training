#!/bin/bash
set -eo pipefail

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MSLITE_ROOT="/home/w00955485/hispark_ai/src/mindspore-lite"
SDK_PATH="/home/w00955485/fbb_ws63"
ADAPTOR_PATH="/home/w00955485/hispark_ai/src/adaptor"
MODEL_DIR="${CUR_DIR}/artifacts"
MODEL_FILE="${MODEL_DIR}/model.onnx"
EXPORT_ONNX_SCRIPT="${CUR_DIR}/export_onnx.py"
BENCHMARK_INPUT="${MODEL_DIR}/data/test_00000.bin"
BENCHMARK_GOLDEN="${MODEL_DIR}/golden.txt"
BENCHMARK_TEMPLATE="${CUR_DIR}/benchmark_template.c"
TARGET="${1:-riscv}"
MODE="${2:-fp32}"

SDK_SRC="${SDK_PATH}/src"
RISCV_TOOLCHAIN_PATH="${SDK_PATH}/src/tools/bin/compiler/riscv/cc_riscv32_musl_105/cc_riscv32_musl/bin"
CONFIG_FILE="${MODEL_DIR}/micro.cfg"
QUANT_CONFIG_FILE="${MODEL_DIR}/micro_quant.cfg"
MICRO_DIR="${MODEL_DIR}/micro"
QUANT_MICRO_DIR="${MODEL_DIR}/micro_quant"
MICRO_MS_FILE="${MODEL_DIR}/micro.ms"
QUANT_MICRO_MS_FILE="${MODEL_DIR}/micro_quant.ms"
MSLITE_PKG_PATH="${MSLITE_ROOT}/output/mindspore-lite-2.8.0-linux-x64"
CONVERTER_BIN="${MSLITE_PKG_PATH}/tools/converter/converter/converter_lite"
CONVERTER_LIB="${MSLITE_PKG_PATH}/tools/converter/lib"
OP_LIB="${MSLITE_PKG_PATH}/tools/codegen/lib/riscv/libnnacl.a"
WRAPPER_LIB="${MSLITE_PKG_PATH}/tools/codegen/lib/riscv/libwrapper.a"
CPU_OP_LIB="${MSLITE_PKG_PATH}/tools/codegen/lib/cpu/libnnacl.a"
CPU_WRAPPER_LIB="${MSLITE_PKG_PATH}/tools/codegen/lib/cpu/libwrapper.a"

case "${MODE}" in
    fp32)
        ACTIVE_CONFIG_FILE="${CONFIG_FILE}"
        ACTIVE_MICRO_DIR="${MICRO_DIR}"
        ACTIVE_MICRO_MS_FILE="${MICRO_MS_FILE}"
        ACTIVE_OUTPUT_PREFIX="${MODEL_DIR}/micro"
        ;;
    quant)
        ACTIVE_CONFIG_FILE="${QUANT_CONFIG_FILE}"
        ACTIVE_MICRO_DIR="${QUANT_MICRO_DIR}"
        ACTIVE_MICRO_MS_FILE="${QUANT_MICRO_MS_FILE}"
        ACTIVE_OUTPUT_PREFIX="${MODEL_DIR}/micro_quant"
        ;;
    *)
        echo "Usage: $0 {onnx|benchmark|riscv} [fp32|quant]"
        exit 1
        ;;
esac

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

function prepare_converter_env()
{
    check_file "${CONVERTER_BIN}"
    check_file "${MODEL_FILE}"
    check_dir "${CONVERTER_LIB}"

    export PATH="${MSLITE_PKG_PATH}/tools/converter/converter:${PATH}"
    export LD_LIBRARY_PATH="${CONVERTER_LIB}:${LD_LIBRARY_PATH:-}"
}

function export_onnx_model()
{
    check_file "${EXPORT_ONNX_SCRIPT}"
    python3 "${EXPORT_ONNX_SCRIPT}"
}

function set_micro_config()
{
    local enabled="$1"
    local config_file="${2:-${CONFIG_FILE}}"
    if grep -q '^enable_micro=' "${config_file}"; then
        sed -i "s|^enable_micro=.*|enable_micro=${enabled}|" "${config_file}"
    else
        echo "enable_micro=${enabled}" >> "${config_file}"
    fi
}

function generate_micro_ms()
{
    local config_file="${1:-${CONFIG_FILE}}"
    local output_prefix="${2:-${MODEL_DIR}/micro}"
    local ms_file="${3:-${MICRO_MS_FILE}}"

    export_onnx_model
    prepare_converter_env
    check_file "${config_file}"

    local temp_config="${config_file}.micro_ms.tmp"
    cp -f "${config_file}" "${temp_config}"
    set_micro_config false "${temp_config}"
    trap 'rm -f "${temp_config}"' RETURN
    rm -f "${ms_file}"
    converter_lite \
        --fmk=ONNX \
        --encryption=false \
        --inputDataFormat=NCHW \
        --outputDataFormat=NCHW \
        --outputFile="${output_prefix}" \
        --modelFile="${MODEL_FILE}" \
        --configFile="${temp_config}"
    trap - RETURN
    rm -f "${temp_config}"
}

function convert_model()
{
    local config_file="${1:-${CONFIG_FILE}}"
    local micro_dir="${2:-${MICRO_DIR}}"

    prepare_converter_env
    check_file "${config_file}"
    set_micro_config true "${config_file}"
    rm -rf "${micro_dir}"
    converter_lite \
        --fmk=ONNX \
        --encryption=false \
        --inputDataFormat=NCHW \
        --outputDataFormat=NCHW \
        --outputFile="${micro_dir}" \
        --modelFile="${MODEL_FILE}" \
        --configFile="${config_file}"
}

function compile_micro_riscv()
{
    local micro_dir="${1:-${MICRO_DIR}}"

    check_dir "${micro_dir}"
    check_file "${micro_dir}/CMakeLists.txt"
    check_file "${OP_LIB}"
    check_file "${WRAPPER_LIB}"
    check_file "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-gcc"
    check_file "${RISCV_TOOLCHAIN_PATH}/riscv32-linux-musl-g++"

    pushd "${micro_dir}"
    rm -rf build
    cmake -S . -B build \
        -D OP_LIB="${OP_LIB}" \
        -D WRAPPER_LIB="${WRAPPER_LIB}" \
        -D RISCV_TOOLCHAIN_PATH="${RISCV_TOOLCHAIN_PATH}" \
        -D PKG_PATH="${MSLITE_PKG_PATH}" \
        -D MS_ROOT_DIR="${MSLITE_PKG_PATH}"
    cmake --build build -j4
    popd
}

function compile_benchmark()
{
    local micro_dir="${1:-${MICRO_DIR}}"

    check_dir "${micro_dir}"
    check_file "${micro_dir}/CMakeLists.txt"
    check_file "${micro_dir}/benchmark/benchmark.c"
    check_file "${BENCHMARK_TEMPLATE}"
    check_file "${CPU_OP_LIB}"
    check_file "${CPU_WRAPPER_LIB}"

    cp -f "${BENCHMARK_TEMPLATE}" "${micro_dir}/benchmark/benchmark.c"

    pushd "${micro_dir}"
    sed -i "s|^set(CMAKE_C_COMPILER.*|set(OP_LIB \"${CPU_OP_LIB}\")|" CMakeLists.txt
    sed -i "s|^set(CMAKE_CXX_COMPILER.*|set(WRAPPER_LIB \"${CPU_WRAPPER_LIB}\")|" CMakeLists.txt
    sed -i "s|^set(CMAKE_C_FLAGS.*|set(MS_ROOT_DIR \"${MSLITE_PKG_PATH}\")|" CMakeLists.txt
    sed -i "s|^set(CMAKE_CXX_FLAGS.*|set(PKG_PATH \"${MSLITE_PKG_PATH}\")|" CMakeLists.txt
    if ! grep -q 'add_executable(benchmark' CMakeLists.txt; then
        cat >> CMakeLists.txt <<'EOF'

file(GLOB BENCHMARK_SRC ./benchmark/*.c)
add_executable(benchmark ${BENCHMARK_SRC})
target_link_libraries(benchmark PRIVATE micro_runtime)
EOF
    fi

    rm -rf build
    cmake -S . -B build
    cmake --build build -j4
    popd
}

function run_benchmark()
{
    local micro_dir="${1:-${MICRO_DIR}}"
    local benchmark_bin="${micro_dir}/build/benchmark"

    check_file "${BENCHMARK_INPUT}"
    check_file "${BENCHMARK_GOLDEN}"
    if [ ! -x "${benchmark_bin}" ]; then
        echo "ERROR: benchmark executable not found: ${benchmark_bin}"
        exit 1
    fi

    "${benchmark_bin}" "${BENCHMARK_INPUT}" "${BENCHMARK_GOLDEN}"
}

function enable_sdk_sample()
{
    local micro_dir="${1:-${MICRO_DIR}}"
    local samples_cmake="${SDK_SRC}/application/samples/CMakeLists.txt"
    local utils_cmake="${SDK_SRC}/middleware/utils/CMakeLists.txt"
    local target_config="${SDK_SRC}/build/config/target_config/ws63/config.py"

    check_file "${MODEL_DIR}/CMakeLists.txt"
    check_file "${MODEL_DIR}/ai_main.c"
    check_dir "${ADAPTOR_PATH}/adaptor"
    check_file "${ADAPTOR_PATH}/include/ai.h"
    check_file "${samples_cmake}"
    check_file "${utils_cmake}"
    check_file "${target_config}"

    mkdir -p "${SDK_SRC}/middleware/utils/ai_mcu/lib"
    cp -f "${micro_dir}/build/libmicro_runtime.a" "${SDK_SRC}/middleware/utils/ai_mcu/lib/"
    cp -f "${micro_dir}/build/src/libnet.a" "${SDK_SRC}/middleware/utils/ai_mcu/lib/"

    rm -rf "${SDK_SRC}/middleware/utils/ai_mcu/adaptor"
    mkdir -p "${SDK_SRC}/middleware/utils/ai_mcu"
    cp -rf "${ADAPTOR_PATH}/adaptor" "${SDK_SRC}/middleware/utils/ai_mcu/"

    mkdir -p "${SDK_SRC}/include/middleware/utils"
    cp -f "${ADAPTOR_PATH}/include/ai.h" "${SDK_SRC}/include/middleware/utils/ai.h"

    if ! grep -q '\$ENV{ENABLE_AI_CUSTOM_SAMPLE}' "${samples_cmake}"; then
        sed -i '/COMPONENT_NAME/a\\nset(CONFIG_ENABLE_AI_CUSTOM_SAMPLE "$ENV{ENABLE_AI_CUSTOM_SAMPLE}")' "${samples_cmake}"
    fi
    if ! grep -q "MaxPool2D" "${samples_cmake}"; then
        sed -i "/add_subdirectory_if_exist(custom)/i\\if(DEFINED CONFIG_ENABLE_AI_CUSTOM_SAMPLE)\\n  add_subdirectory(\\n    ${MODEL_DIR}\\n    \\${CMAKE_CURRENT_BINARY_DIR}/maxpool2d_build\\n  )\\nendif()\\n" "${samples_cmake}"
    fi
    if ! grep -q "ai_mcu/adaptor/cpu" "${utils_cmake}"; then
        echo "add_subdirectory_if_exist(ai_mcu/adaptor/cpu)" >> "${utils_cmake}"
    fi
    if ! grep -q "ai_adaptor_cpu" "${target_config}"; then
        python3 - "${target_config}" <<'PY'
import pathlib
import sys

config_path = pathlib.Path(sys.argv[1])
namespace = {}
exec(config_path.read_text(), namespace)
ram_components = namespace["target"]["ws63-liteos-app"]["ram_component"]
if "ai_adaptor_cpu" not in ram_components:
    ram_components.append("ai_adaptor_cpu")
config_path.write_text(
    "target = " + repr(namespace["target"]) + "\n"
    "target_copy = {}\n"
    "target_group = {}\n"
)
PY
    fi
}

function build_sdk()
{
    local micro_dir="${1:-${MICRO_DIR}}"

    check_file "${micro_dir}/build/libmicro_runtime.a"
    check_file "${micro_dir}/build/src/libnet.a"
    check_file "${SDK_SRC}/build.py"

    enable_sdk_sample "${micro_dir}"
    export LANG="C"
    export ENABLE_AI_CUSTOM_SAMPLE=y

    pushd "${SDK_SRC}"
    rm -rf \
        "${SDK_SRC}/output/ws63/acore/ws63-flashboot" \
        "${SDK_SRC}/output/ws63/acore/ws63-loaderboot" \
        "${SDK_SRC}/output/ws63/acore/ws63-liteos-app"
    if [ -f "${SDK_SRC}/application/wb02_3.mk" ]; then
        ./build.py -c ws63-flashboot
    fi
    ./build.py -c ws63-liteos-app
    popd

    cp -f \
        "${SDK_SRC}/output/ws63/fwpkg/ws63-liteos-app/ws63-liteos-app_all.fwpkg" \
        "${MODEL_DIR}/ws63-liteos-app_all.fwpkg"
}

case "${TARGET}" in
    onnx)
        export_onnx_model
        ;;
    benchmark)
        generate_micro_ms "${ACTIVE_CONFIG_FILE}" "${ACTIVE_OUTPUT_PREFIX}" "${ACTIVE_MICRO_MS_FILE}"
        convert_model "${ACTIVE_CONFIG_FILE}" "${ACTIVE_MICRO_DIR}"
        compile_benchmark "${ACTIVE_MICRO_DIR}"
        run_benchmark "${ACTIVE_MICRO_DIR}"
        ;;
    riscv)
        generate_micro_ms "${ACTIVE_CONFIG_FILE}" "${ACTIVE_OUTPUT_PREFIX}" "${ACTIVE_MICRO_MS_FILE}"
        convert_model "${ACTIVE_CONFIG_FILE}" "${ACTIVE_MICRO_DIR}"
        compile_micro_riscv "${ACTIVE_MICRO_DIR}"
        build_sdk "${ACTIVE_MICRO_DIR}"
        ;;
    *)
        echo "Usage: $0 {onnx|benchmark|riscv} [fp32|quant]"
        exit 1
        ;;
esac
