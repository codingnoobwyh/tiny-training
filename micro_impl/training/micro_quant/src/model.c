
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
#include "src/model.h"
#include "c_api/model_c.h"
#include "include/model_handle.h"
#include "malloc.h"
#include "src/context.h"
#include "src/tensor.h"
#include "string.h"
#include "src/allocator.h"
size_t MSModelCalcWorkspaceSize(MSModelHandle model) {
  return 0;
}

void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size) {}

MSModelHandle MSModelCreate() { return model0; }

MSStatus MSModelBuild(MSModelHandle model, const void *model_data,
                      size_t data_size, MSModelType model_type,
                      const MSContextHandle model_context) {
  if (model_type != kMSModelTypeMindIR) {
    return kMSStatusLiteNotSupport;
  }
  if (model == NULL || model_context == NULL) {
    return kMSStatusLiteParamInvalid;
  }

  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->build == NULL) {
    return kMSStatusLiteNullptr;
  }

  MSStatus ret =
    micro_model->build(model, model_data, data_size, model_context);
  if (ret != kMSStatusSuccess) {
    MSModelDestroy(model);
  }
  return ret;
}

MSTensorHandleArray MSModelGetInputs(const MSModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    MSTensorHandleArray tmp = {0, NULL};
    return tmp;
  }
  return micro_model->inputs;
}

MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    MSTensorHandleArray tmp = {0, NULL};
    return tmp;
  }
  return micro_model->outputs;
}

MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->inputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->inputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->inputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name) == 0) {
      return micro_tensor;
    }
  }
  return NULL;
}

MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL || micro_model->outputs.handle_list == NULL) {
    return NULL;
  }
  for (size_t i = 0; i < micro_model->outputs.handle_num; i++) {
    MicroTensor *micro_tensor = (MicroTensor *)micro_model->outputs.handle_list[i];
    if (micro_tensor == NULL) {
      return NULL;
    }
    if (strcmp(micro_tensor->name, tensor_name) == 0) {
      return micro_tensor;
    }
  }
  return NULL;
}

MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos,
                       size_t shape_info_num) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->resize == NULL) {
    return kMSStatusLiteNullptr;
  }
  return micro_model->resize(model, inputs, shape_infos, shape_info_num);
}


MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs,
                        const MSKernelCallBackC before, const MSKernelCallBackC after) {
  MicroModel *micro_model = (MicroModel *)model;
  if (micro_model == NULL) {
    return kMSStatusLiteNullptr;
  }
  if (micro_model->predict == NULL) {
    return kMSStatusLiteNullptr;
  }
  return micro_model->predict(model, inputs, outputs, before, after);
}


void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs) {
 if (inputs.handle_list == NULL) {
   return;
 }
 for (size_t i = 0; i < inputs.handle_num; i++) {
   MicroTensor *micro_tensor = inputs.handle_list[i];
   if (micro_tensor == NULL) {
     continue;
   }
   if (micro_tensor->data != NULL && micro_tensor->owned) {
     free(micro_tensor->data);
     micro_tensor->data = NULL;
     micro_tensor->owned = false;
   }
   if (micro_tensor->shape) {
     free(micro_tensor->shape);
     micro_tensor->shape = NULL;
   }
   free(micro_tensor);
   micro_tensor = NULL;
 }
 free(inputs.handle_list);
 inputs.handle_list = NULL;
}

void MSModelDestroy(MSModelHandle *model) {
  if (model == NULL || *model == NULL) {
    return;
  }
  if (*model) {
    MicroModel *micro_model = (MicroModel *)*model;
    if (micro_model->runtime_buffer) {
      micro_model->runtime_buffer = NULL;
    }
    MSTensorHandleArrayDestroy(micro_model->inputs);
    MSTensorHandleArrayDestroy(micro_model->outputs);
    micro_model->inputs.handle_list = NULL;
    micro_model->outputs.handle_list = NULL;
    micro_model->free_resource();
    DecRefCount();
  }
}
