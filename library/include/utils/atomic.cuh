/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace clutra::detail::atomic {

template <typename T>
concept Lock = requires(T lock) {
  { lock.acquire() } -> std::same_as<void>;
  { lock.release() } -> std::same_as<void>;
  { lock.try_acquire() } -> std::same_as<bool>;
};

struct SpinLock {
  unsigned int word;

  __device__ __forceinline__ void acquire() {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> L(word);

    int k = 0;
    while (true) {
      unsigned int expected = 0;

      // Try to grab the lock: 0 -> 1
      if (L.compare_exchange_weak(expected, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed)) {
        return;  // acquired
      }

      // Backoff (important on H100 to avoid hammering L2/atomics)
      int spin = 1 << (k < 8 ? k : 8);  // up to 256 iterations
#pragma unroll 1
      for (int i = 0; i < spin; ++i) {
        __nanosleep(50);
      }
      ++k;
    }
  }

  __device__ __forceinline__ void release() {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> L(word);
    L.store(0, cuda::memory_order_release);
  }

  __device__ __forceinline__ bool try_acquire() {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> L(word);
    unsigned int expected = 0;
    return L.compare_exchange_strong(expected, 1, cuda::memory_order_acquire, cuda::memory_order_relaxed);
  }
};

struct TicketLock {
  unsigned int next;
  unsigned int serving;

  __device__ void acquire() {
    auto r_next = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(next);
    auto r_serving = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(serving);

    unsigned int my = r_next.fetch_add(1, cuda::memory_order_relaxed);

    while (r_serving.load(cuda::memory_order_acquire) != my) {
      __nanosleep(50);  // important on H100
    }
  }

  __device__ void release() {
    auto r_serving = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(serving);
    r_serving.fetch_add(1, cuda::memory_order_release);
  }

  __device__ bool try_acquire() {
    auto r_next = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(next);
    auto r_serving = cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(serving);

    unsigned int my = r_next.load(cuda::memory_order_relaxed);
    if (r_next.compare_exchange_strong(my, my + 1, cuda::memory_order_acquire, cuda::memory_order_relaxed)) {
      while (r_serving.load(cuda::memory_order_acquire) != my) {
        __nanosleep(50);  // important on H100
      }
      return true;
    }
    return false;
  }
};

}  // namespace clutra::detail::atomic
