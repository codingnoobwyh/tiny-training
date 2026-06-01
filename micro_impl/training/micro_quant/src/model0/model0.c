
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
#include "src/tensor.h"
#include "src/context.h"
#include "c_api/model_c.h"
#include "src/model.h"
#include "src/model0/net0.h"
#include "src/allocator.h"
#include "src/model0/weight0.h"

MSStatus MSModelBuild0(MSModelHandle model, const void *model_data,
                       size_t data_size, const MSContextHandle model_context);
MSStatus MSModelResize0(MSModelHandle model, 
                       const MSTensorHandleArray inputs, MSShapeInfo *shape_infos, size_t shape_info_num);
MSStatus MSModelPredict0(MSModelHandle model, const MSTensorHandleArray inputs,
                         MSTensorHandleArray *output,
                         const MSKernelCallBackC before,
                         const MSKernelCallBackC after);
void Reset0();
void MSModelSetWorkspace0(MSModelHandle model, void *workspace, size_t workspace_size);
size_t MSModelCalcWorkspaceSize0(MSModelHandle model);
static MicroModel gModel0 = {.runtime_buffer = NULL,
                             .train_mode = false,
                             .inputs = {1, NULL},
                             .outputs = {1, NULL},
                             .build = MSModelBuild0,
                             .resize = MSModelResize0,
                             .predict = MSModelPredict0,
                             .set_work_space = NULL,
                             .calc_work_space = NULL,
                             .free_resource = Reset0};
MSModelHandle model0 = &gModel0;

void Reset0() {
  FreeResource0();
}
MSStatus MSModelCreate0(MicroModel *micro_model) {
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
void *runtime_buffer = GlobalMemory();
if (runtime_buffer == NULL) {
    return kMSStatusLiteNullptr;
  }
  micro_model->runtime_buffer = runtime_buffer;
  int ret = SetBuffer0(((MemBlock *)runtime_buffer)->addr);
  if (ret != kMSStatusSuccess) {
    return kMSStatusLiteMemoryFailed;
  }

  micro_model->train_mode = false;
  MSTensorHandleArray model_inputs;
  model_inputs.handle_num = 1;
  MicroTensor **input_tensors = malloc(1 * sizeof(MicroTensor *));
  model_inputs.handle_list = (MSTensorHandle *)(input_tensors);
  micro_model->inputs = model_inputs;
  input_tensors[0] = malloc(sizeof(MicroTensor));
  input_tensors[0]->type = kMSDataTypeNumberTypeFloat32;
  input_tensors[0]->format = kMSFormatNHWC;
  input_tensors[0]->ndim = 4;
  input_tensors[0]->shape = malloc(4 * sizeof(int64_t));
  input_tensors[0]->shape[0]= 1;
  input_tensors[0]->shape[1]= 1;
  input_tensors[0]->shape[2]= 28;
  input_tensors[0]->shape[3]= 28;
  input_tensors[0]->name = "input";
  input_tensors[0]->data = NULL;
  input_tensors[0]->owned = false;
  MSTensorHandleArray model_outputs;
  model_outputs.handle_num = 1;
  MicroTensor **output_tensors = malloc(1 * sizeof(MicroTensor *));
  model_outputs.handle_list = (MSTensorHandle *)(output_tensors);
  micro_model->outputs = model_outputs;
  output_tensors[0] = malloc(sizeof(MicroTensor));
  output_tensors[0]->type = kMSDataTypeNumberTypeFloat32;
  output_tensors[0]->format = kMSFormatNHWC;
  output_tensors[0]->ndim = 2;
  output_tensors[0]->shape = malloc(2 * sizeof(int64_t));
  output_tensors[0]->shape[0]= 1;
  output_tensors[0]->shape[1]= 10;
  output_tensors[0]->name = "logits";
  output_tensors[0]->data = NULL;
  output_tensors[0]->owned = false;
  return kMSStatusSuccess;
}

MSStatus MSModelBuild0(MSModelHandle model, const void *model_data, size_t data_size,
                      const MSContextHandle model_context) {
  if (model == NULL || model_context == NULL) {
    return kMSStatusLiteParamInvalid;
  }
  if (model_data == NULL && data_size != 0) {
    return kMSStatusLiteParamInvalid;
  }
  if (data_size != 0) {
    return kMSStatusLiteInputParamInvalid;
  }
  MicroModel *micro_model = (MicroModel *)model;
  int ret = MSModelCreate0(micro_model);
  if (ret != kMSStatusSuccess) {
    return ret;
  }
  ret = Init0(NULL, 0);
  return ret;
}
MSStatus MSModelResize0(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos, size_t shape_info_num) {
  if (model == NULL) {
    return kMSStatusLiteParamInvalid;
  }
  return kMSStatusLiteNotSupport;
}
int CopyOutputsData0(MSTensorHandleArray *outputs_ori, int *cur_out_types, bool *type_changed) {
  if (outputs_ori == NULL || cur_out_types == NULL || type_changed == NULL) {
    return kMSStatusLiteNullptr;
  }
  unsigned char *buffer[1] = {m0_buffer + 16832, };
  for (int i = 0; i < 1; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)outputs_ori->handle_list[i];
    int expect_type = micro_tensor->type;
    int cur_type = cur_out_types[i];
    if (expect_type == cur_type) {
      micro_tensor->data = buffer[i];
      micro_tensor->owned = false;
      continue;
    }
#ifdef ENABLE_FP16
    int type_trans_mode = TypeTransMode_MAX;
    if (expect_type == kMSDataTypeNumberTypeFloat16 && cur_type == kMSDataTypeNumberTypeFloat32) {
      type_trans_mode = TypeTransMode_FP32_TO_FP16;
    } else if (expect_type == kMSDataTypeNumberTypeFloat32 && cur_type == kMSDataTypeNumberTypeFloat16) {
      type_trans_mode = TypeTransMode_FP16_TO_FP32;
    }
    if (type_trans_mode == TypeTransMode_UNSUPPORT) {
      return kMSStatusLiteNotSupport;
    }
    int shape_size = micro_tensor->ndim;
    int num = 1;
    for (int i = 0; i < shape_size; ++i) {
      num *= micro_tensor->shape[i];
    }
    void *out_data = MSTensorGetMutableData(micro_tensor);
    if (type_trans_mode == TypeTransMode_FP32_TO_FP16) {
      Fp32CastToFp16((float *)(buffer[i]), (float16_t *)out_data, num);
      type_changed[i] = true;
    } else if (type_trans_mode == TypeTransMode_FP16_TO_FP32) {
      Fp16CastToFp32((float16_t *)(buffer[i]), (float *)out_data, num);
      type_changed[i] = true;
    }
#endif
  }
  return RET_OK;
}

MSStatus MSModelPredict0(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                         const MSKernelCallBackC before, const MSKernelCallBackC after) {

  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (outputs == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->runtime_buffer == NULL) {
    return kMSStatusLiteMemoryFailed;
  }
  if (inputs.handle_num != 1) {
    return kMSStatusLiteParamInvalid;
  }
  if (outputs->handle_num != 1) {
    return kMSStatusLiteParamInvalid;
  }
  const void *inputs_data_array[1];
  int expect_types[1] = {43, };
  bool type_changed[1] = {false, };
  for (int i = 0; i < 1; i++) {
    inputs_data_array[i] = TransformInput((MicroTensor *)inputs.handle_list[i], expect_types[i], &type_changed[i]);
  }
  SetInputs0(inputs_data_array, 1);
  Execute0(micro_model->train_mode);

  for (int i = 0; i < 1; i++) {
    if (type_changed[i]) {
      free((void *)inputs_data_array[i]);
    }
  }

  int cur_out_types[1] = {43, };
  bool out_type_changed[1] = {false, };
  MSStatus ret = CopyOutputsData0(outputs, cur_out_types, out_type_changed);
  return ret;
}
