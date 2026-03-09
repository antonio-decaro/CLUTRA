/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <stealer/stealer_config.cuh>

namespace clutra::stealer {

struct StealQueueDescriptor {
  int* head;
  int* tail;
  uintptr_t payload0;
  uintptr_t payload1;
  uintptr_t payload2;
};

struct StealerDevice {
  StealerConfig config{};

  StealerDevice() = default;

  explicit StealerDevice(const StealerConfig& cfg) : config(cfg) {}

  __forceinline__ __device__ bool isIntraClusterStealingEnabled() const {
    return config.intra_cluster_stealing_enabled;
  }

  __forceinline__ __device__ int getPreferredClusterSize() const { return config.preferred_cluster_size; }

  __forceinline__ __device__ int getLocalStealingChunkSize() const { return config.local_stealing_chunk_size; }

  __forceinline__ __device__ bool isInterClusterStealingEnabled() const {
    return config.inter_cluster_stealing_enabled;
  }

  __forceinline__ __device__ int getGlobalStealingChunkSize() const { return config.global_stealing_chunk_size; }

  template <size_t BlockSize>
  struct SharedState {};

  template <size_t BlockSize, typename BuildDescriptorFn>
  __device__ void init(SharedState<BlockSize>& state, BuildDescriptorFn&& build_descriptor) const;

  template <size_t BlockSize, typename ProcessStealFn>
  __device__ void runLocalStealLoop(SharedState<BlockSize>& state, int chunk_size, ProcessStealFn&& process) const;

  template <size_t BlockSize>
  __device__ void finalize() const;

  template <size_t BlockSize>
  __device__ void setReady(SharedState<BlockSize>&, bool) const;
};

struct NullStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;

  template <size_t BlockSize>
  struct SharedState : StealerDevice::SharedState<BlockSize> {};

  template <size_t BlockSize, typename BuildDescriptorFn>
  __device__ void init(SharedState<BlockSize>& state, BuildDescriptorFn&& build_descriptor) const {
    (void)state;
    (void)build_descriptor;
  }

  template <size_t BlockSize, typename ProcessStealFn>
  __device__ void runLocalStealLoop(SharedState<BlockSize>& state, int chunk_size, ProcessStealFn&& process) const {
    (void)state;
    (void)chunk_size;
    (void)process;
  }

  template <size_t BlockSize>
  __device__ void finalize() const {}

  template <size_t BlockSize>
  __device__ void setReady(SharedState<BlockSize>& state, bool ready) const {
    (void)state;
    (void)ready;
  }
};

struct BasicStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;

  template <size_t BlockSize>
  struct SharedState : StealerDevice::SharedState<BlockSize> {
    int16_t victim_rank;
    int16_t steal_count;
    int16_t steal_tail;
    bool is_finished;
    int is_ready;
    bool* is_finished_ptr[8];
    int* is_ready_ptr[8];
    StealQueueDescriptor queues[8];
  };

  template <size_t BlockSize, typename BuildDescriptorFn>
  __device__ void init(SharedState<BlockSize>& state, BuildDescriptorFn&& build_descriptor) const;

  template <size_t BlockSize, typename ProcessStealFn>
  __device__ void runLocalStealLoop(SharedState<BlockSize>& state, int chunk_size, ProcessStealFn&& process) const;

  template <size_t BlockSize>
  __device__ void finalize() const;

  template <size_t BlockSize>
  __device__ void setReady(SharedState<BlockSize>& state, bool ready) const;
};

template <size_t BlockSize, typename BuildDescriptorFn>
__device__ void StealerDevice::init(SharedState<BlockSize>& state, BuildDescriptorFn&& build_descriptor) const {
  (void)state;
  (void)build_descriptor;
}

template <size_t BlockSize, typename ProcessStealFn>
__device__ void
StealerDevice::runLocalStealLoop(SharedState<BlockSize>& state, int chunk_size, ProcessStealFn&& process) const {
  (void)state;
  (void)chunk_size;
  (void)process;
}

template <size_t BlockSize>
__device__ void StealerDevice::finalize() const {}

template <size_t BlockSize>
__device__ void StealerDevice::setReady(SharedState<BlockSize>&, bool) const {}

template <size_t BlockSize, typename BuildDescriptorFn>
__device__ void BasicStealerDevice::init(SharedState<BlockSize>& state, BuildDescriptorFn&& build_descriptor) const {
#if __CUDA_ARCH__ >= 900
  auto cluster = cooperative_groups::this_cluster();
  cluster.sync();

  state.is_finished = false;
  state.is_ready = 0;
  state.victim_rank = -1;
  state.steal_count = 0;

  for (int i = 0; i < cluster.dim_blocks().x; ++i) {
    state.is_finished_ptr[i] = cluster.map_shared_rank(&state.is_finished, i);
    state.is_ready_ptr[i] = cluster.map_shared_rank(&state.is_ready, i);
    state.queues[i] = build_descriptor(cluster, i);
  }
#else
  (void)state;
  (void)build_descriptor;
#endif
}

template <size_t BlockSize, typename ProcessStealFn>
__device__ void
BasicStealerDevice::runLocalStealLoop(SharedState<BlockSize>& state, int chunk_size, ProcessStealFn&& process) const {
#if __CUDA_ARCH__ >= 900
  if (chunk_size < 1) {
    chunk_size = 1;
  }

  auto block = cooperative_groups::this_thread_block();
  while (true) {
    state.is_finished = true;

    cooperative_groups::invoke_one(block, [&]() {
      auto cluster = cooperative_groups::this_cluster();
      while (true) {
        state.steal_count = 0;
        state.victim_rank = -1;
        bool all_finished = true;

        for (int16_t victim_offset = 1; victim_offset < cluster.dim_blocks().x; ++victim_offset) {
          const int16_t victim_rank = (cluster.block_rank() + victim_offset) % cluster.dim_blocks().x;
          auto* victim_ready_ptr = state.is_ready_ptr[victim_rank];
          auto* victim_finished_ptr = state.is_finished_ptr[victim_rank];

          if (!(*victim_finished_ptr)) {
            all_finished = false;
          } else {
            continue;
          }

          if (atomicAdd(victim_ready_ptr, 0) == 0) {
            all_finished = false;
            continue;
          }

          auto& victim_desc = state.queues[victim_rank];
          cuda::atomic_ref<int, cuda::thread_scope_device> head_ref(*victim_desc.head);
          cuda::atomic_ref<int, cuda::thread_scope_device> tail_ref(*victim_desc.tail);
          const int head_snapshot = head_ref.load(cuda::memory_order_relaxed);
          const int tail_snapshot = tail_ref.load(cuda::memory_order_relaxed);
          const int available = tail_snapshot - head_snapshot;

          if (available > chunk_size) {
            if (atomicCAS(victim_desc.tail, tail_snapshot, tail_snapshot - chunk_size) != tail_snapshot) {
              continue;
            }
            state.steal_tail = tail_snapshot;
            state.steal_count = chunk_size;
            state.victim_rank = victim_rank;
            break;
          }
        }

        if (state.steal_count > 0 || all_finished) {
          break;
        }
      }
    });

    block.sync();
    const int steal_count = state.steal_count;
    if (steal_count == 0) {
      break;
    }

    const auto& victim_desc = state.queues[state.victim_rank];
    const int steal_begin_index = state.steal_tail - state.steal_count;
    for (int i = 0; i < steal_count; ++i) {
      process(victim_desc, steal_begin_index + i, i, steal_count);
    }
    __syncthreads();
  }
#else
  (void)state;
  (void)chunk_size;
  (void)process;
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::finalize() const {
#if __CUDA_ARCH__ >= 900
  cooperative_groups::this_cluster().sync();
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::setReady(SharedState<BlockSize>& state, bool ready) const {
#if __CUDA_ARCH__ >= 900
  atomicExch(&state.is_ready, ready ? 1 : 0);
#else
  (void)state;
  (void)ready;
#endif
}

}  // namespace clutra::stealer
