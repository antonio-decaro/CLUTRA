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
  uint32_t vertices[Capacity];
  uint32_t degrees[Capacity];

  __device__ void init() { tail = 0; }

  __forceinline__ __device__ int push(uint32_t vertex, uint32_t degree) {
  const int loc = atomicAdd(&tail, 1);
    vertices[loc] = vertex;
    degrees[loc] = degree;
    return loc;
  }

  __device__ int size() const { return tail; }
};

}