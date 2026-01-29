/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace clutra::detail::utils {

template<size_t Capacity>
struct SharedQueue {
  int tail;
  int head;
  uint32_t vertices[Capacity];
  uint32_t degrees[Capacity];

  __host__ static size_t getSizeInBytes() {
    return sizeof(SharedQueue<Capacity>);
  }

  __device__ void init() { tail = 0; head = 0;}

  __forceinline__ __device__ int push(uint32_t vertex, uint32_t degree) {
  const int loc = atomicAdd(&tail, 1);
    vertices[loc] = vertex;
    degrees[loc] = degree;
    return loc;
  }

  __forceinline__ __device__ bool pop(uint32_t& vertex, uint32_t& degree) {
    if (head < 0 || head >= tail) {
      return false; // empty
    }
    
    vertex = vertices[head];
    degree = degrees[head];
    
    if (threadIdx.x == 0) {
      atomicAdd(&head, 1);
    }
    return true;
  }

  __device__ int size() const { return tail; }
};

}