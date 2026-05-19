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

#include "calib_output.h"
#include "c_api/status_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef ENABLE_FP16
#include <arm_neon.h>
#endif

#define kToleranceVal 0.0001f
#define kMaxOutput 5
#define kMaxTensorSize 400 * 400 * 4
#define kInt32Max 2147483647

int ReadCalibData(const char *calib_data_path, CalibTensor **calib_tensor_pointers, int *calib_num) {
  FILE *file = fopen(calib_data_path, "r");
  if (!file) {
    printf("Unable open %s", calib_data_path);
    return kMSStatusLiteError;
  }
  CalibTensor *calib_tensors = (CalibTensor *)malloc(kMaxOutput * sizeof(CalibTensor));
  if(calib_tensors == NULL) {
    printf("Malloc calib tensors failed.");
    return kMSStatusLiteError;
  }
  // read line by line
  char line[kMaxTensorSize];
  char *p;
  int i = 0;
  int elements = 1;
  *calib_num = 0;
  while (fgets(line, kMaxTensorSize, file) != NULL) {
    if (i == 0) {
      elements = 1;
      int j = 0;
      int dims = 0;
      p = strtok(line, " ");
      if (p == NULL) {
        printf("Parse tensor name failed.");
        return kMSStatusLiteError;
      }
      char* tensor_name = (char *)malloc(strlen(p)+1);
      if(tensor_name == NULL) {
        printf("Malloc tensor name failed.");
        return kMSStatusLiteError;
      }
      (void)strcpy(tensor_name, p);
      calib_tensors[*calib_num].tensor_name = tensor_name;
      while (p != NULL) {
        if (j == 1) {
          dims = atoi(p);
        }
        if (j >= 2 && j - 2 < dims) {
          if (elements > kInt32Max / atoi(p)) {
            printf("Elements calculation overflow.");
            return kMSStatusLiteError;
          }
          elements *= atoi(p);
          if (j - 2 == dims - 1) {
            calib_tensors[*calib_num].elemets_num_ = elements;
            break;
          }
        }
        p = strtok(NULL, " ");
        j++;
      }
      i++;
    } else {
      float *data = (float *)malloc(elements * sizeof(float));
      if(data == NULL) {
        printf("Malloc tensor data failed.");
        return kMSStatusLiteError;
      }
      p = strtok(line, " ");
      int k = 0;
      while (p != NULL) {
        data[k++] = atof(p);
        p = strtok(NULL, " ");
        if (k == elements) {
          calib_tensors[*calib_num].data_ = data;
          break;
        }
      }
      i--;
      (*calib_num)++;
    }
  }
  *calib_tensor_pointers = calib_tensors;
  fclose(file);
  return kMSStatusSuccess;
}

int CompareOutputs(MSTensorHandleArray outputs, CalibTensor **calib_tensors, int calib_num,
                   float cosine_distance_threshold) {
  if (outputs.handle_num != (size_t)calib_num) {
    printf("error, outputs and calibs size is mismatch\n");
    return kMSStatusLiteError;
  }
  size_t outputs_num = outputs.handle_num;
  bool is_success = true;
  for (size_t i = 0; i < outputs_num; ++i) {
    MicroTensor *output = (MicroTensor *)outputs.handle_list[i];
    if (!output || !output->data) {
      return kMSStatusLiteError;
    }
    CalibTensor *calib = calib_tensors[0];
    if (!calib || !calib[i].data_) {
      return kMSStatusLiteError;
    }
    if (strcmp(output->name, calib[i].tensor_name) != 0) {
      printf("warning, output tensor name is not equal to calib\n");
    }
    size_t elements = (size_t)MSTensorGetElementNum(output);
    if (elements != (size_t)calib[i].elemets_num_) {
      printf("error, output elements num is not equal to calib\n");
      return kMSStatusLiteError;
    }
    float cosin = 0.f, dot = 0.f, normx = 0.f, normy = 0.f;
    switch (output->type) {
      case kMSDataTypeNumberTypeFloat32: {
        float *float_output = (float *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          if (isnan(float_output[j]) || isinf(float_output[j]) || isnan(calib[i].data_[j]) ||
              isinf(calib[i].data_[j])) {
            printf("error, output data is nan or inf\n");
            return kMSStatusLiteError;
          }
          dot += float_output[j] * calib[i].data_[j];
          normx += float_output[j] * float_output[j];
          normy += calib[i].data_[j] * calib[i].data_[j];
        }
        break;
      }
#ifdef ENABLE_FP16
      case kMSDataTypeNumberTypeFloat16: {
        float16_t *float16_output = (float16_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          if (isnan(float16_output[j]) || isinf(float16_output[j]) || isnan(calib[i].data_[j]) ||
              isinf(calib[i].data_[j])) {
            printf("error, output data is nan or inf\n");
            return kMSStatusLiteError;
          }
          dot += float16_output[j] * calib[i].data_[j];
          normx += float16_output[j] * float16_output[j];
          normy += calib[i].data_[j] * calib[i].data_[j];
        }
        break;
      }
#endif
      case kMSDataTypeNumberTypeInt8: {
        int8_t *int_output = (int8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeBool:
      case kMSDataTypeNumberTypeUInt8: {
        uint8_t *int_output = (uint8_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      case kMSDataTypeNumberTypeInt32:
      case kMSDataTypeNumberTypeUInt32: {
        int32_t *int_output = (int32_t *)output->data;
        for (size_t j = 0; j < elements; ++j) {
          dot += (float) (int_output[j] * calib[i].data_[j]);
          normx += (float) (int_output[j] * int_output[j]);
          normy += (float)(calib[i].data_[j] * calib[i].data_[j]);
        }
        break;
      }
      default: {
        printf("unsupported tensor data type.\n");
      }
    }
    cosin = dot / (sqrt(normx) * sqrt(normy));
    if (cosin < cosine_distance_threshold) {
      printf("cos-similarity of %s is %f, less than %f.\n", output->name, cosin, cosine_distance_threshold);
      is_success = false;
    } else {
      printf("cos-similarity of %s is %f.\n", output->name, cosin);
    }
  }
  if (!is_success) {
    printf("compare outputs failed.\n");
    return kMSStatusLiteError;
  }
  printf("compare outputs success.\n");
  return kMSStatusSuccess;
}

void FreeCalibTensors(CalibTensor **calib_tensors_pointers, int calib_num) {
  CalibTensor *calib_tensors = *calib_tensors_pointers;
  if (calib_tensors) {
    for (int i = 0; i < calib_num; i++) {
      if (calib_tensors[i].data_) {
        free(calib_tensors[i].data_);
        calib_tensors[i].data_ = NULL;
      }
      if (calib_tensors[i].tensor_name) {
        free(calib_tensors[i].tensor_name);
        calib_tensors[i].tensor_name = NULL;
      }
    }
    free(calib_tensors);
    calib_tensors = NULL;
  }
}
