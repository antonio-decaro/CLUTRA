/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/concept.hpp>
#include <graph/graph.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/logging.cuh>
#include <utils/profile.cuh>
#include <utils/queue.cuh>

namespace clutra::operators::advance::detail {

constexpr uint32_t ADVANCE_WARP_SIZE = 32;

__device__ __forceinline__ uint32_t nextStealRngState(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

__device__ __forceinline__ uint32_t mapVictimExcludingSelf(uint32_t sample,
                                                           uint32_t my_cluster,
                                                           uint32_t num_clusters) {
  const uint32_t victim_space = num_clusters - 1U;
  const uint32_t mapped = sample % victim_space;
  return (mapped >= my_cluster) ? (mapped + 1U) : mapped;
}

template <typename LockType, typename StealerDeviceT>
__device__ __forceinline__ bool
tryGlobalClusterSteal(clutra::detail::utils::WorkQueueView<uint32_t, LockType>& local_cluster_queue,
                      clutra::detail::utils::WorkQueueView<uint32_t, LockType>* cluster_work_queues,
                      StealerDeviceT stealer,
                      uint32_t& steal_state) {
#if __CUDA_ARCH__ >= 900
  constexpr uint32_t STEAL_SUCCESS_MASK = 0x80000000U;
  constexpr uint32_t CURSOR_MASK = 0x7FFFFFFFU;
  // constexpr uint32_t WARP_RANDOM_ROUNDS = 1U;
  constexpr uint32_t RNG_SALT = 0xA57D3C29U;
  auto cluster = cooperative_groups::this_cluster();
  uint32_t* steal_state_ptr = cluster.map_shared_rank(&steal_state, 0);

  if (cluster.block_rank() == 0 && threadIdx.x < ADVANCE_WARP_SIZE) {
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & (ADVANCE_WARP_SIZE - 1U);
    const unsigned int warp_mask = __activemask();
    const uint32_t cluster_size = static_cast<uint32_t>(cluster.dim_blocks().x);
    const uint32_t my_cluster = static_cast<uint32_t>(blockIdx.x) / cluster_size;
    const uint32_t num_clusters = static_cast<uint32_t>(gridDim.x) / cluster_size;
    const uint32_t warp_random_rounds = ceil(logf(static_cast<float>(num_clusters)));
    uint32_t next_cursor = 0;
    bool stole_any = false;
    if (num_clusters > 1) {
      int requested = stealer.getGlobalStealingChunkSize();
      if (requested < 1) {
        requested = 1;
      }

      const uint32_t cursor = (*steal_state_ptr) & CURSOR_MASK;
      uint32_t start = cursor;
      if (start >= num_clusters || start == my_cluster) {
        start = (my_cluster + 1U) % num_clusters;
      }

      // Warp-cooperative random victim selection with deterministic seeding.
      uint32_t lane_rng_state = cursor ^ (my_cluster << 16) ^ num_clusters ^ RNG_SALT ^ ((lane + 1U) * 0x9E3779B9U);
      if (lane_rng_state == 0U) {
        lane_rng_state = RNG_SALT ^ (lane + 1U);
      }

      for (uint32_t round = 0; round < warp_random_rounds; ++round) {
        const bool lane_active = lane < (num_clusters - 1U);
        const uint32_t proposed_victim =
            mapVictimExcludingSelf(nextStealRngState(lane_rng_state), my_cluster, num_clusters);
        bool candidate_available = false;
        if (lane_active) {
          // Lock-free hint: avoid taking victim queue locks during broad probing.
          candidate_available = cluster_work_queues[proposed_victim].hasWorkRelaxed();
        }

        const unsigned int candidate_mask = __ballot_sync(warp_mask, lane_active && candidate_available);
        if (candidate_mask == 0U) {
          continue;
        }

        const int winner_lane = __ffs(static_cast<int>(candidate_mask)) - 1;
        const uint32_t victim = __shfl_sync(warp_mask, proposed_victim, winner_lane);
        int stolen = 0;
        if (lane == 0U) {
          stolen = cluster_work_queues[victim].popChunkFromTail(local_cluster_queue.data, requested);
          local_cluster_queue.setTail(stolen);
          local_cluster_queue.setHead(0);
          // printf("Cluster %u stealing from cluster %u, requested %d, stolen %d\n", my_cluster, victim, requested,
          //        stolen);
          if (stolen > 0) {
            stole_any = true;
            next_cursor = (victim + 1U) % num_clusters;
          }
        }
        stolen = __shfl_sync(warp_mask, stolen, 0);
        if (stolen > 0) {
          break;
        }
      }

      // Fallback to a full probe to preserve progress guarantees.
      // if (!stole_any && lane == 0U) {
      //   for (uint32_t i = 0; i < num_clusters - 1U; ++i) {
      //     const uint32_t victim = (start + i) % num_clusters;
      //     if (victim == my_cluster || !cluster_work_queues[victim].hasWorkRelaxed()) {
      //       continue;
      //     }
      //     const int stolen = cluster_work_queues[victim].popChunkFromTail(local_cluster_queue.data, requested);
      //     local_cluster_queue.setTail(stolen);
      //     local_cluster_queue.setHead(0);
      //
      //     if (stolen > 0) {
      //       stole_any = true;
      //       next_cursor = (victim + 1U) % num_clusters;
      //       break;
      //     }
      //   }
      // }

      if (!stole_any && lane == 0U) {
        next_cursor = (start + 1U) % num_clusters;
      }
    } else {
      next_cursor = 0;
    }
    if (lane == 0U) {
      *steal_state_ptr = (stole_any ? STEAL_SUCCESS_MASK : 0U) | (next_cursor & CURSOR_MASK);
    }
  }

  cluster.sync();
  const bool has_new_work = ((*steal_state_ptr & STEAL_SUCCESS_MASK) != 0U);
  return has_new_work;
#else
  (void)local_cluster_queue;
  (void)cluster_work_queues;
  (void)stealer;
  (void)steal_state;
  return false;
#endif
}
}  // namespace clutra::operators::advance::detail
