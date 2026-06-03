
cmake_minimum_required(VERSION 3.14)
project(micro_lib)

add_compile_definitions(BENCHMARK)

set(OP_LIB "/home/w00955485/hispark_ai/src/mindspore-lite/output/mindspore-lite-2.8.0-linux-x64/tools/codegen/lib/cpu/libnnacl.a")
set(WRAPPER_LIB "/home/w00955485/hispark_ai/src/mindspore-lite/output/mindspore-lite-2.8.0-linux-x64/tools/codegen/lib/cpu/libwrapper.a")
set(MS_ROOT_DIR "/home/w00955485/hispark_ai/src/mindspore-lite/output/mindspore-lite-2.8.0-linux-x64")
set(PKG_PATH "/home/w00955485/hispark_ai/src/mindspore-lite/output/mindspore-lite-2.8.0-linux-x64")
message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}, CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("OP_LIB: ${OP_LIB}")
message("WRAPPER_LIB: ${WRAPPER_LIB}")
message("MS_ROOT_DIR: ${MS_ROOT_DIR}")

set(HEADER_PATH ${PKG_PATH}/runtime)
set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)

add_subdirectory(src)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MS_ROOT_DIR}/include)
include_directories(${MS_ROOT_DIR})
include_directories(${OP_HEADER_PATH})
set(
    SOURCE_FILE
    src/model.c
    src/tensor.c
    src/allocator.c
    src/context.c
    src/model0/model0.c
    src/model0/net0.c
    src/model0/weight0.c
)
add_library(micro_runtime STATIC ${SOURCE_FILE})
target_link_libraries(micro_runtime net -lm -pthread)

file(GLOB BENCHMARK_SRC ./benchmark/*.c)
add_executable(benchmark ${BENCHMARK_SRC})
target_link_libraries(benchmark PRIVATE micro_runtime)
