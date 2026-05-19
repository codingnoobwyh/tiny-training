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

#include "load_input.h"
#include "calib_output.h"
#include "c_api/types_c.h"
#include "c_api/model_c.h"
#include "c_api/context_c.h"
#include "src/tensor.h"
#include <time.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define kMaxThreadNum 4
#define kBindDefault 1

void usage() {
  printf(
    "-- mindspore benchmark params usage:\n"
    "args[0]: executable file\n"
    "args[1]: inputs binary file\n"
    "args[2]: calibration file\n\n");
}

void PrintTensorHandle(MSTensorHandle tensor) {
  if (tensor == NULL) {
    printf("input tensor is null");
    return;
  }
  printf("name: %s, ", MSTensorGetName(tensor));
  MSDataType data_type = MSTensorGetDataType(tensor);
  printf("DataType: %d, ", data_type);
  size_t element_num = (size_t)(MSTensorGetElementNum(tensor));
  printf("Elements: %zu, ", element_num);
  printf("Shape: [");
  size_t shape_num = 0;
  const int64_t *dims = MSTensorGetShape(tensor, &shape_num);
  for (size_t i = 0; i < shape_num; i++) {
    printf("%d ", (int)dims[i]);
  }
  printf("], Data: \n");
  void *data = MSTensorGetMutableData(tensor);
  const size_t MAX_ELEMENT_NUM = 10;
  element_num = element_num > MAX_ELEMENT_NUM ? MAX_ELEMENT_NUM : element_num;
  switch (data_type) {
    case kMSDataTypeNumberTypeFloat32: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%.6f, ", ((float *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%" PRIi8, ((int8_t *)data)[i]);
      }
      printf("\n");
    } break;
    case kMSDataTypeNumberTypeUInt8: {
      for (size_t i = 0; i < element_num; i++) {
        printf("%u", ((uint8_t *)data)[i]);
      }
      printf("\n");
    } break;
    default:
      printf("Unsupported data type to print");
      break;
  }
}

int main(int argc, const char **argv) {
  printf("1. Create context\n");
  MSContextHandle ms_context_handle = MSContextCreate();

  printf("2. Create model\n");
  MSModelHandle model_handle = MSModelCreate();
  int ret = MSModelBuild(model_handle, NULL, 0, kMSModelTypeMindIR, ms_context_handle);
  MSContextDestroy(&ms_context_handle);
  if (ret != kMSStatusSuccess) {
    printf("MSModelBuild failed, ret: %d\n", ret);
    return ret;
  }

  // set model inputs tensor data
  printf("3. Get model inputs\n");
  MSTensorHandleArray inputs_handle = MSModelGetInputs(model_handle);
  if (inputs_handle.handle_list == NULL) {
    printf("MSModelGetInputs failed, ret: %d", ret);
    return ret;
  }
  size_t inputs_num = inputs_handle.handle_num;
  printf("input num: %zu\n", inputs_num);
  void *inputs_binbuf[inputs_num];
  int inputs_size[inputs_num];
  for (size_t i = 0; i < inputs_num; ++i) {
    MSTensorHandle tensor = inputs_handle.handle_list[i];
    inputs_size[i] = (int)MSTensorGetDataSize(tensor);
  }

  printf("4. Read inputs data from file\n");
  ret = ReadInputsFile((char *)(argv[1]), inputs_binbuf, inputs_size, (int)inputs_num);
  if (ret != 0) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("5. Set inputs data to tensor\n");
  for (size_t i = 0; i < inputs_num; ++i) {
    void *input_data = MSTensorGetMutableData(inputs_handle.handle_list[i]);
    memcpy(input_data, inputs_binbuf[i], inputs_size[i]);
    free(inputs_binbuf[i]);
    inputs_binbuf[i] = NULL;
  }

  printf("6. Get model outputs\n");
  MSTensorHandleArray outputs_handle = MSModelGetOutputs(model_handle);
  if (!outputs_handle.handle_list) {
    printf("MSModelGetOutputs failed, ret: %d", ret);
    return ret;
  }

  printf("7. Predict\n");
  ret = MSModelPredict(model_handle, inputs_handle, &outputs_handle, NULL, NULL);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(&model_handle);
    return ret;
  }

  printf("\n===== Run success =====\nPrint model outputs:\n");
  for (size_t i = 0; i < outputs_handle.handle_num; i++) {
    MSTensorHandle output = outputs_handle.handle_list[i];
    PrintTensorHandle(output);
  }

  if (argc >= 3) {
    CalibTensor *calib_tensors;
    int calib_num = 0;
    ret = ReadCalibData(argv[2], &calib_tensors, &calib_num);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    float cosine_distance_threshold = 0.9999;
    if (argc >= 9) {
      cosine_distance_threshold = atof(argv[8]);
    }
    ret = CompareOutputs(outputs_handle, &calib_tensors, calib_num, cosine_distance_threshold);
    if (ret != kMSStatusSuccess) {
      MSModelDestroy(&model_handle);
      return ret;
    }
    FreeCalibTensors(&calib_tensors, calib_num);
  }
  MSModelDestroy(&model_handle);
  return kMSStatusSuccess;
}
