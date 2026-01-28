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
                                    SharedState<BlockSize>&) const {}

template <size_t BlockSize>
__device__ int StealerDevice::attemptStealing(SharedState<BlockSize>&,
                                              int) const {
  return 0;
}

template <size_t BlockSize>
__device__ void StealerDevice::steal(SharedState<BlockSize>&,
                                     int,
                                     uint32_t&,
                                     uint32_t&) const {}

template <size_t BlockSize>
__device__ void StealerDevice::finalize() const {}

template <size_t BlockSize>
__device__ void BasicStealerDevice::init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                                         SharedState<BlockSize>& state) const {
#if __CUDA_ARCH__ >= 900
  auto cluster = cg::this_cluster();
  cluster.sync();
  state.cluster_queues[cluster.block_rank()] = local_queue;
  for (int i = 0; i < cluster.dim_blocks().x; ++i) {
    if (i != cluster.block_rank()) {
      state.cluster_queues[i] = cluster.map_shared_rank(local_queue, i);
    }
  }
  state.victim_rank = -1;
  state.steal_count = 0;
#endif
}

template <size_t BlockSize>
__device__ int BasicStealerDevice::attemptStealing(SharedState<BlockSize>& state,
                                                   int chunk_size) const {
#if __CUDA_ARCH__ >= 900
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  cg::invoke_one(block,[&]() {
    state.steal_count = 0;
    state.victim_rank = -1;
    for (int victim_offset = 1; victim_offset < cluster.dim_blocks().x; ++victim_offset) {
      int potential_victim_rank = (cluster.block_rank() + victim_offset) % cluster.dim_blocks().x;
      auto* victim_queue = state.cluster_queues[potential_victim_rank];
      const int tail_snapshot = victim_queue->tail;
      if (victim_queue->head < tail_snapshot - (chunk_size)) {
        if (atomicCAS(&(victim_queue->tail), tail_snapshot, tail_snapshot - chunk_size) != tail_snapshot) {
          // Another block beat us to stealing from this victim; try next.
          continue;
        }
        state.steal_tail = tail_snapshot;
        state.steal_count = chunk_size;
        state.victim_rank = potential_victim_rank;
        // printf("Stealing from block %d by block %d: steal_count=%d\n",
        //        potential_victim_rank, cluster.block_rank(), state.steal_count);
        break;
      }
    }
  });
  block.sync();
  return state.steal_count;
#else
  (void)state;
  (void)chunk_size;
  return 0;
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::steal(SharedState<BlockSize>& state,
                                          int i,
                                          uint32_t& vertex,
                                          uint32_t& degree) const {
#if __CUDA_ARCH__ >= 900
  if (state.victim_rank == -1 || i < 0 || i >= state.steal_count) {
    vertex = 0;
    degree = 0;
    return;
  }
  auto* victim_queue = state.cluster_queues[state.victim_rank];
  const int index = state.steal_tail - state.steal_count + i;
  vertex = victim_queue->vertices[index];
  degree = victim_queue->degrees[index];
#else
  (void)state;
  (void)i;
  vertex = 0;
  degree = 0;
#endif
}

template <size_t BlockSize>
__device__ void BasicStealerDevice::finalize() const {
#if __CUDA_ARCH__ >= 900
  auto cluster = cg::this_cluster();
  cluster.sync();
#endif
}

#define INSTANTIATE_STEALER_DEVICE_TEMPLATES(BlockSize) \
  template struct StealerDevice::SharedState<BlockSize>; \
  template struct BasicStealerDevice::SharedState<BlockSize>; \
  template __device__ void StealerDevice::init<BlockSize>(clutra::detail::utils::SharedQueue<BlockSize>*, \
                                                  StealerDevice::SharedState<BlockSize>&) const; \
  template __device__ void BasicStealerDevice::init<BlockSize>(clutra::detail::utils::SharedQueue<BlockSize>*, \
                                                       BasicStealerDevice::SharedState<BlockSize>&) const; \
  template __device__ int StealerDevice::attemptStealing<BlockSize>(StealerDevice::SharedState<BlockSize>&, \
                                                            int) const; \
  template __device__ int BasicStealerDevice::attemptStealing<BlockSize>(BasicStealerDevice::SharedState<BlockSize>&, \
                                                                 int) const; \
  template __device__ void StealerDevice::steal<BlockSize>(StealerDevice::SharedState<BlockSize>&, \
                                                   int, \
                                                   uint32_t&, \
                                                   uint32_t&) const; \
  template __device__ void BasicStealerDevice::steal<BlockSize>(BasicStealerDevice::SharedState<BlockSize>&, \
                                                        int, \
                                                        uint32_t&, \
                                                        uint32_t&) const; \
  template __device__ void StealerDevice::finalize<BlockSize>() const; \
  template __device__ void BasicStealerDevice::finalize<BlockSize>() const;


INSTANTIATE_STEALER_DEVICE_TEMPLATES(256)
INSTANTIATE_STEALER_DEVICE_TEMPLATES(512)
INSTANTIATE_STEALER_DEVICE_TEMPLATES(1024)

#undef INSTANTIATE_STEALER_DEVICE_TEMPLATES

} // namespace clutra::stealer
