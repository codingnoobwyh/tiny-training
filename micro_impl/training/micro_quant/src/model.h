
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
#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_

#include "c_api/model_c.h"

typedef MSStatus (*ModelBuild)(MSModelHandle model, const void *model_data,
                               size_t data_size,
                               const MSContextHandle model_context);

typedef MSStatus (*ModelPredict)(MSModelHandle model,
                                 const MSTensorHandleArray inputs,
                                 MSTensorHandleArray *outputs,
                                 const MSKernelCallBackC before,
                                 const MSKernelCallBackC after);

typedef void (*FreeResource)();

typedef void (*ModelSetWorkspace)(MSModelHandle model, void *workspace, size_t workspace_size);

typedef size_t (*ModelCalcWorkspaceSize)(MSModelHandle model);

typedef MSStatus (*ModelResize)(MSModelHandle model,
                                const MSTensorHandleArray inputs,
                                MSShapeInfo *shape_infos,
                                size_t shape_info_num);

typedef struct {
  void *runtime_buffer;
  bool train_mode;  // true: train mode, false: eval mode
  MSTensorHandleArray inputs;
  MSTensorHandleArray outputs;
  ModelBuild build;
  ModelResize resize;
  ModelSetWorkspace set_work_space;
  ModelCalcWorkspaceSize calc_work_space;
  FreeResource free_resource;
  ModelPredict predict;
} MicroModel;
void MSTensorHandleArrayDestroy(MSTensorHandleArray inputs);
#endif // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_MODEL_H_

