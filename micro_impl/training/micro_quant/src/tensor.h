
/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

#include "c_api/data_type_c.h"
#include "c_api/format_c.h"
#include "c_api/tensor_c.h"
#include <stdbool.h>
#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif

typedef struct {
  enum MSDataType type;
  enum MSFormat format;
  char *name;
  int ndim;
  int64_t *shape;
  void *data;
  int quant_nums;
  bool owned;
} MicroTensor; // if change MicroTensor parameter, need to update kMicroTensorSize

typedef struct {
  int num;
  MicroTensor **tensor;
} MicroTensorList;

typedef struct {
  int bit_num;
  double scale;
  int32_t zero_point;
  double min;
  double max;
} QuantParam;

enum TypeTransMode {
  TypeTransMode_FP32_TO_FP16 = 0,
  TypeTransMode_FP16_TO_FP32 = 1,
  TypeTransMode_UNSUPPORT = 2,
  TypeTransMode_MAX = TypeTransMode_UNSUPPORT
};

void *TransformInput(MSTensorHandle tensor, int expect_type, bool *type_changed);

#ifdef ENABLE_FP16
void Fp32CastToFp16(const float *input, float16_t *output, int number);
void Fp16CastToFp32(const float16_t *input, float *output, int number);
#endif

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_TENSOR_H_

