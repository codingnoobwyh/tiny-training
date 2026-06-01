include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
set(OP_SRC
    common_func.c.o
    common_func_int8.c.o
    conv3x3_int8_low_memory.c.o
    fixed_point.c.o
    matmul_int8.c.o
    matmul_int8_wrapper.c.o
    pack_fp32.c.o
    pack_int8.c.o
    pooling_int8.c.o
    quant_dtype_cast_int8.c.o
    relux_int8.c.o
    transpose_fp32.c.o
    transpose_fp32_wrapper.c.o
    transpose_int8.c.o
)
file(GLOB_RECURSE NET_SRC
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/*.c
     )
add_library(net STATIC ${NET_SRC})
