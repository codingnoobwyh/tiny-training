
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

#include "context.h"
#include <stdlib.h>
#include <string.h>

MSContextHandle MSContextCreate() {
  MicroContext *micro_context = (MicroContext *)malloc(sizeof(MicroContext));
  if (micro_context == NULL) {
    return NULL;
  }
  micro_context->enable_parallel_ = false;
  micro_context->thread_num_ = 1;
  micro_context->affinity_core_list_ = NULL;
  micro_context->core_num = 0;
  micro_context->affinity_mode = 0;
  return micro_context;
}

void MSContextDestroy(MSContextHandle *context) {
  if (context == NULL) {
    return;
  }
  MicroContext *micro_context = (MicroContext *)(*context);
  if (micro_context) {
    free(micro_context);
    micro_context = NULL;
  }
  *context = NULL;
}

void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) {
}

int32_t MSContextGetThreadNum(const MSContextHandle context) {
  return 1;
}

void MSContextSetThreadAffinityMode(MSContextHandle context, int mode) {
}

int MSContextGetThreadAffinityMode(const MSContextHandle context) {
  return 0;
}
