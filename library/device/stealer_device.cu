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
__device__ void StealerDevice::setReady(SharedState<BlockSize>&,
                                        bool) const {}

template <size_t BlockSize>
__device__ void BasicStealerDevice::init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                                         SharedState<BlockSize>& state) const {
#if __CUDA_ARCH__ >= 900
  auto cluster = cg::this_cluster();
  cluster.sync();
  state.cluster_queues[cluster.block_rank()] = local_queue;
  state.is_finished = false;
  state.is_ready = 0;
  for (int i = 0; i < cluster.dim_blocks().x; ++i) {
    state.is_finished_ptr[i] = cluster.map_shared_rank(&state.is_finished, i);
    state.is_ready_ptr[i] = cluster.map_shared_rank(&state.is_ready, i);
    state.cluster_queues[i] = cluster.map_shared_rank(local_queue, i);
    // if (i != cluster.block_rank()) {
    // }
  }
  state.victim_rank = -1;
  state.steal_count = 0;
#endif
}

template <size_t BlockSize>
__device__ int BasicStealerDevice::attemptStealing(SharedState<BlockSize>& state,
                                                   int chunk_size) const {
#if __CUDA_ARCH__ >= 900
  state.is_finished = true;
  auto cluster = cg::this_cluster();
  auto block = cg::this_thread_block();
  cg::invoke_one(block,[&]() {
    // constexpr int GUARD = 16;
    while (true) {
      state.steal_count = 0;
      state.victim_rank = -1;
      bool all_finished = true;
      for (int16_t victim_offset = 1; victim_offset < cluster.dim_blocks().x; ++victim_offset) {
        int16_t potential_victim_rank = (cluster.block_rank() + victim_offset) % cluster.dim_blocks().x;
        auto* victim_queue = state.cluster_queues[potential_victim_rank];
        auto* victim_ready_ptr = state.is_ready_ptr[potential_victim_rank];
        auto* victim_finished_ptr = state.is_finished_ptr[potential_victim_rank];
        if (!(*victim_finished_ptr)) {
          all_finished = false;
        } else {
          continue;
        }
        if (atomicAdd(victim_ready_ptr, 0) == 0) {
          all_finished = false;
          continue;
        }
        cuda::atomic_ref<int, cuda::thread_scope_device> head_ref(victim_queue->head);
        cuda::atomic_ref<int, cuda::thread_scope_device> tail_ref(victim_queue->tail);
        const int tail_snapshot = tail_ref.load(cuda::memory_order_relaxed);
        if (tail_snapshot - head_ref.load(cuda::memory_order_relaxed) > chunk_size) {
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
      if (state.steal_count > 0 || all_finished) {
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

template <size_t BlockSize>
__device__ void BasicStealerDevice::setReady(SharedState<BlockSize>& state,
  bool ready) const {
#if __CUDA_ARCH__ >= 900
  atomicExch(&state.is_ready, ready ? 1 : 0);
#else
  (void)state;
  (void)ready;
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
  template __device__ void BasicStealerDevice::finalize<BlockSize>() const; \
  template __device__ void StealerDevice::setReady<BlockSize>(StealerDevice::SharedState<BlockSize>&, \
                                                      bool) const; \
  template __device__ void BasicStealerDevice::setReady<BlockSize>(BasicStealerDevice::SharedState<BlockSize>&, \
                                                           bool) const;


INSTANTIATE_STEALER_DEVICE_TEMPLATES(256)
INSTANTIATE_STEALER_DEVICE_TEMPLATES(512)
INSTANTIATE_STEALER_DEVICE_TEMPLATES(1024)

#undef INSTANTIATE_STEALER_DEVICE_TEMPLATES

} // namespace clutra::stealer
