/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
#define MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_

#include "c_api/model_c.h"
#include "src/tensor.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct CalibTensor {
  char *tensor_name;
  int elemets_num_;
  float *data_;
} CalibTensor;
int ReadCalibData(const char *calib_data_path, CalibTensor **calib_tensots, int *calib_num);
int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num,
                   float cosine_distance_threshold);
void FreeCalibTensors(CalibTensor **calib_tensors, int calib_num);

#ifdef __cplusplus
}
#endif
#endif  // MINDSPORE_LITE_MICRO_CALIB_OUTPUT_H_
