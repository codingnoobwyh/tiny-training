
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

#ifndef MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_
#define MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_

#include <stdbool.h>
#include "c_api/context_c.h"

typedef struct MicroContext {
  char* vendor_name_;
  int thread_num_; /**< thread number config for thread pool */
  bool enable_parallel_;
  int* affinity_core_list_; /**< explicitly specify the core to be bound. priority use affinity core list */
  int core_num;
  int affinity_mode;
} MicroContext;

#endif  // MINDSPORE_LITE_MICRO_LIBRARY_SOURCE_CONTEXT_H_
