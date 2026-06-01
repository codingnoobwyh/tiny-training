
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
#include "src/allocator.h"
#include "stdatomic.h"
#include "stdlib.h"
#include <stdbool.h>

MemBlock *mem_block = NULL;

  void DecRefCount() {
    while (mem_block != NULL) {
      MemBlock *next = mem_block->next;
      free(mem_block);
      mem_block = next;
    }
  }

  void *GlobalMemory() {
size_t init_size = 16900;

    bool expected = false;
    mem_block = malloc(sizeof(MemBlock) + init_size);
    mem_block->occupied = false;
    mem_block->size = init_size;
    mem_block->addr = (char *)mem_block + sizeof(MemBlock);
    mem_block->next = NULL;
    return mem_block;
  }
  
  void *Malloc(size_t size) {
    bool expected = false;
    MemBlock *pre = mem_block;
    MemBlock *cur = mem_block;
    MemBlock *find = NULL;
    while (cur != NULL) {
      if (cur->size < size) {
        break;
      }
      if (!cur->occupied) {
        find = cur;
      }
      pre = cur;
      cur = cur->next;
    }
    MemBlock *block = malloc(sizeof(MemBlock) + size);
    block->occupied = true;
    block->size = size;
    block->addr = (char *)block + sizeof(MemBlock);
    block->next = NULL;
    block->next = pre->next;
    pre->next = block;
    return block;
}
  
  bool LockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = true;
    return expected;
  }

  bool UnLockBuffer(void *block) {
    MemBlock *m_block = block;
    bool expected = true;
    return expected;
  }
  