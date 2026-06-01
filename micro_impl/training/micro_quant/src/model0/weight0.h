
/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "nnacl_c/common_func.h"
#include "nnacl_c/errorcode.h"
#include "nnacl_c/fp32/pack_fp32.h"
#include "nnacl_c/fp32/transpose_fp32.h"
#include "nnacl_c/int8/common_func_int8.h"
#include "nnacl_c/int8/conv3x3_int8_low_memory.h"
#include "nnacl_c/int8/fixed_point.h"
#include "nnacl_c/int8/matmul_int8.h"
#include "nnacl_c/int8/pack_int8.h"
#include "nnacl_c/int8/pooling_int8.h"
#include "nnacl_c/int8/quant_dtype_cast_int8.h"
#include "nnacl_c/int8/relux_int8.h"
#include "nnacl_c/int8/transpose_int8.h"
#include "nnacl_c/kernel/pooling.h"
#include "nnacl_c/transpose_parameter.h"
#include "wrapper/fp32/transpose_fp32_wrapper.h"
#include "wrapper/int8/matmul_int8_wrapper.h"
#include "nnacl_c/op_base.h"
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
extern unsigned char *m0_buffer;
extern uint8_t *m0_weight;
extern int *m0_shape;
extern int *m0_offset;
enum STATUS {
  RET_OK = 0,
  RET_ERROR = 1,
};

extern int m0_thread_num;
extern const int8_t m0_weight10[];  // 
extern const int32_t m0_weight11[];  // 
extern const int8_t m0_weight12[];  // 
extern const int32_t m0_weight13[];  // 
extern const int32_t m0_weight14[];  // 
extern const int32_t m0_weight15[];  // 
extern const int8_t m0_weight6[];  // fc1.weight
extern const int8_t m0_weight8[];  // fc2.weight
/// \brief Init model weight from buffer.

/// \param[in] weight_buffer The address of the weight binary file.
/// \param[in] weight_size The size of the weight file in bytes.
int Init0(void *weight_buffer, int weight_size);

