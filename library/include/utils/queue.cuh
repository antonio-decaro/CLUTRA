/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace clutra::detail::utils {

template <size_t Capacity>
struct SharedQueue {
  int tail;
  int head;
  uint32_t vertices[Capacity];
  uint32_t degrees[Capacity];

  __host__ static size_t getSizeInBytes() { return sizeof(SharedQueue<Capacity>); }

  __device__ void init() {
    tail = 0;
    head = 0;
  }

  __forceinline__ __device__ int push(uint32_t vertex, uint32_t degree) {
    const int loc = atomicAdd(&tail, 1);
    vertices[loc] = vertex;
    degrees[loc] = degree;
    return loc;
  }

  __forceinline__ __device__ bool pop(uint32_t& vertex, uint32_t& degree) {
    cuda::atomic_ref<int, cuda::thread_scope_device> tail_ref(tail);
    const int tail_snapshot = tail_ref.load(cuda::memory_order_relaxed);

    if (head < 0 || head >= tail_snapshot) {
      return false;  // empty
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

template <typename T>
class WorkQueue {
public:
  __host__ WorkQueue(size_t capacity) {
    cudaMalloc(&data, capacity * sizeof(T));
    cudaMalloc(&head, sizeof(uint32_t));
    cudaMalloc(&tail, sizeof(uint32_t));
    cudaMalloc(&lock, sizeof(int16_t));

    uint32_t zero = 0;
    int zero_lock = 0;
    cudaMemcpy(&head, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&tail, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&lock, &zero_lock, sizeof(int), cudaMemcpyHostToDevice);
  }

  __host__ ~WorkQueue() {
    cudaFree(&data);
    cudaFree(&head);
    cudaFree(&tail);
    cudaFree(&lock);
  }

  __device__ void push(uint32_t size) {}

  __device__ bool pop() { return false; }

  __device__ bool steal() { return false; }

private:
  T* data;
  volatile uint32_t* head;
  volatile uint32_t* tail;
  volatile int* lock;

  __device__ void acquire() {
    while (*lock == 1 || atomicCAS((int*)lock, 0, 1) != 0)  // reduce the number of atomic operations
      ;
  }

  __device__ void release() { atomicExch((int*)lock, 0); }
};

}  // namespace clutra::detail::utils
