/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cooperative_groups.h>
#include <stealer/stealer_device.cuh>
#include <utils/queue.cuh>

namespace cg = cooperative_groups;

namespace clutra::stealer {

template <size_t BlockSize>
__device__ void StealerDevice::init(clutra::detail::utils::SharedQueue<BlockSize>*,
                                    SharedState<BlockSize>*) const {}

template <size_t BlockSize>
__device__ int StealerDevice::attemptStealing(SharedState<BlockSize>*,
                                              int) const {
  return 0;
}

template <size_t BlockSize>
__device__ void StealerDevice::steal(SharedState<BlockSize>*,
                                     int,
                                     uint32_t*,
                                     uint32_t*) const {}

template <size_t BlockSize>
__device__ void StealerDevice::finalize() const {}

template <size_t BlockSize>
__device__ void BasicStealerDevice::init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                                         SharedState<BlockSize>* state) const {
#if __CUDA_ARCH__ >= 900
  if (!config.intra_cluster_stealing_enabled) {
    return;
  }
  auto cluster = cg::this_cluster();
  cluster.sync();
  state->cluster_queues[cluster.block_rank()] = local_queue;
  for (int i = 0; i < cluster.dim_blocks().x; ++i) {
    if (i != cluster.block_rank()) {
      state->cluster_queues[i] = cluster.map_shared_rank(local_queue, i);
    }
  }
  state->victim_rank = -1;
  state->steal_count = 0;
#endif
}

template <size_t BlockSize>
__device__ int BasicStealerDevice::attemptStealing(SharedState<BlockSize>* state,
                                                   int chunk_size) const {
#if __CUDA_ARCH__ >= 900
  if (!config.intra_cluster_stealing_enabled) {
    return 0;
  }

  auto cluster = cg::this_cluster();
  if (threadIdx.x == 0) {
    state->steal_count = 0;
    state->victim_rank = -1;
    for (int victim_offset = 1; victim_offset < cluster.dim_blocks().x; ++victim_offset) {
      int potential_victim_rank = (cluster.block_rank() + victim_offset) % cluster.dim_blocks().x;
      auto* victim_queue = state->cluster_queues[potential_victim_rank];
      if (victim_queue->head < victim_queue->tail - (chunk_size + 16)) {
        state->steal_tail = atomicSub(&(victim_queue->tail), chunk_size);
        state->steal_count = chunk_size;
        state->victim_rank = potential_victim_rank;
        break;
      }
    }
  }
  __syncthreads();
  return state->steal_count;
#else
  (void)state;
  (void)chunk_size;
  return 0;
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::steal(SharedState<BlockSize>* state,
                                          int i,
                                          uint32_t* vertex,
                                          uint32_t* degree) const {
#if __CUDA_ARCH__ >= 900
  if (state->victim_rank == -1 || i < 0 || i >= state->steal_count) {
    *vertex = 0;
    *degree = 0;
    return;
  }
  auto* victim_queue = state->cluster_queues[state->victim_rank];
  const int index = state->steal_tail - state->steal_count + i;
  *vertex = victim_queue->vertices[index];
  *degree = victim_queue->degrees[index];
#else
  (void)state;
  (void)i;
  *vertex = 0;
  *degree = 0;
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::finalize() const {
#if __CUDA_ARCH__ >= 900
  if (!config.intra_cluster_stealing_enabled) {
    return;
  }
  auto cluster = cg::this_cluster();
  cluster.sync();
#endif
}

template __device__ void StealerDevice::init<256>(clutra::detail::utils::SharedQueue<256>*,
                                                  StealerDevice::SharedState<256>*) const;
template __device__ void BasicStealerDevice::init<256>(clutra::detail::utils::SharedQueue<256>*,
                                                       BasicStealerDevice::SharedState<256>*) const;
template __device__ int StealerDevice::attemptStealing<256>(StealerDevice::SharedState<256>*,
                                                            int) const;
template __device__ int BasicStealerDevice::attemptStealing<256>(BasicStealerDevice::SharedState<256>*,
                                                                 int) const;
template __device__ void StealerDevice::steal<256>(StealerDevice::SharedState<256>*,
                                                   int,
                                                   uint32_t*,
                                                   uint32_t*) const;
template __device__ void BasicStealerDevice::steal<256>(BasicStealerDevice::SharedState<256>*,
                                                        int,
                                                        uint32_t*,
                                                        uint32_t*) const;
template __device__ void StealerDevice::finalize<256>() const;
template __device__ void BasicStealerDevice::finalize<256>() const;

} // namespace clutra::stealer
