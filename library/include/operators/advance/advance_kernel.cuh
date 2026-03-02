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

constexpr int ADVANCE_WARP_SIZE = 32;

template <advance_direction Direction,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename FrontierDevT,
          typename LambdaT>
__device__ __forceinline__ void processVertexRange(GraphDevT graph_dev,
                                                   FrontierDevT out_dev_frontier,
                                                   LambdaT functor,
                                                   uint32_t vertex,
                                                   uint32_t degree,
                                                   uint32_t lane,
                                                   uint32_t stride) {
  auto start = graph_dev.begin(vertex);
  for (uint32_t edge_offset = lane; edge_offset < degree; edge_offset += stride) {
    auto n = start + edge_offset;
    const auto edge = n.getIndex();
    const auto weight = graph_dev.getEdgeWeight(edge);
    const auto neighbor = *n;
    if constexpr (Direction == advance_direction::push) {
      if (functor(vertex, neighbor, edge, weight)) {
        out_dev_frontier.insert(neighbor);
      }
    } else {
      if (functor(neighbor, vertex, edge, weight)) {
        out_dev_frontier.insert(vertex);
      }
    }
  }
}

template <view View, typename FronterDevT>
__device__ uint32_t
getAssignedVertex(const FronterDevT& in_dev_frontier, uint32_t coarsening_factor, uint32_t gid, uint32_t tid) {
  if constexpr (View == view::graph) {
    return (gid * blockDim.x) + tid;
  } else if constexpr (View == view::frontier) {
    const int offsets_size = in_dev_frontier.getOffsetsSize()[0];
    const uint16_t bitmap_range = in_dev_frontier.getBitmapRange();
    const int* bitmap_offsets = in_dev_frontier.getOffsets();

    // fetch assigned vertex
    const uint32_t actual_id_offset = (gid * coarsening_factor) + (tid / bitmap_range);
    uint32_t assigned_vertex;
    if (actual_id_offset < offsets_size) {
      assigned_vertex = (bitmap_offsets[actual_id_offset] * bitmap_range) + (tid % bitmap_range);
    } else {
      assigned_vertex = UINT32_MAX;
    }
    return assigned_vertex;
  } else {
    return UINT32_MAX;
  }
}

template <view View, advance_direction Direction, graph::detail::DeviceGraphConcept GraphDevT, typename FrontierDevT>
__device__ __forceinline__ bool
checkVertexActive(const GraphDevT& graph_dev, const FrontierDevT& in_dev_frontier, uint32_t vertex) {
  if constexpr (View == view::frontier) {
    if constexpr (Direction == advance_direction::push) {
      return (vertex < graph_dev.getVertexCount()) && in_dev_frontier.check(vertex);
    } else {
      return (vertex < graph_dev.getVertexCount()) && !in_dev_frontier.check(vertex);
    }
  } else {  // View == view::graph
    return vertex < graph_dev.getVertexCount();
  }
}

template <view View,
          advance_direction Direction,
          size_t BlockSize,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename InFrontierDevT,
          typename OutFrontierDevT,
          typename StealerDeviceT,
          typename LambdaT>
__device__ __forceinline__ void processTile(GraphDevT graph_dev,
                                            InFrontierDevT in_dev_frontier,
                                            OutFrontierDevT out_dev_frontier,
                                            int coarsening_factor,
                                            uint32_t tile_gid,
                                            clutra::detail::utils::SharedQueue<BlockSize>& cta_queue,
                                            clutra::detail::utils::SharedQueue<ADVANCE_WARP_SIZE>& warp_queue,
                                            StealerDeviceT stealer,
                                            typename StealerDeviceT::template SharedState<BlockSize>& stealer_state,
                                            LambdaT functor,
                                            int tid,
                                            int lane,
                                            int block_dim) {
  __syncthreads();
  if (tid == 0) {
    stealer.setReady(stealer_state, false);
    cta_queue.init();
  }
  if (lane == 0) {
    warp_queue.init();
  }

  __syncthreads();

  const uint32_t assigned_vertex = getAssignedVertex<View>(in_dev_frontier, coarsening_factor, tile_gid, tid);
  const bool vertex_active = checkVertexActive<View, Direction>(graph_dev, in_dev_frontier, assigned_vertex);
  if (vertex_active) {
    const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
    const uint32_t cta_threshold = block_dim;

    if (n_edges >= cta_threshold) {
      cta_queue.push(assigned_vertex, n_edges);
    } else {
      warp_queue.push(assigned_vertex, n_edges);
    }
  }

  __syncthreads();
  if (tid == 0) {
    stealer.setReady(stealer_state, true);
  }

  uint32_t vertex = 0;
  uint32_t degree = 0;
  while (cta_queue.pop(vertex, degree)) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, vertex, degree, tid, block_dim);
    __syncthreads();
  }

  for (int i = 0; i < warp_queue.size(); ++i) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, warp_queue.vertices[i], warp_queue.degrees[i],
                                  lane, ADVANCE_WARP_SIZE);
  }
}

template <advance_direction Direction,
          size_t BlockSize,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename OutFrontierDevT,
          typename StealerDeviceT,
          typename LambdaT>
__device__ __forceinline__ void
runLocalStealLoop(GraphDevT graph_dev,
                  OutFrontierDevT out_dev_frontier,
                  StealerDeviceT stealer,
                  typename StealerDeviceT::template SharedState<BlockSize>& stealer_state,
                  LambdaT functor,
                  int tid,
                  int block_dim) {
  while (true) {
    const int steal_count =
        stealer.template attemptStealing<BlockSize>(stealer_state, stealer.getLocalStealingChunkSize());
    if (steal_count == 0) {
      break;
    }
    uint32_t steal_vertex = 0;
    uint32_t steal_degree = 0;
    for (int i = 0; i < steal_count; ++i) {
      stealer.template steal<BlockSize>(stealer_state, i, steal_vertex, steal_degree);
      processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, steal_vertex, steal_degree, tid, block_dim);
    }
    __syncthreads();
  }
}

template <size_t BlockSize>
size_t getAdvanceSharedMemorySize(size_t stealer_shared_size) {
  size_t shared_size = 0;
  // CTA queue
  shared_size += clutra::detail::utils::SharedQueue<BlockSize>::getSizeInBytes();
  // Warp queues
  shared_size +=
      (BlockSize / ADVANCE_WARP_SIZE) * clutra::detail::utils::SharedQueue<ADVANCE_WARP_SIZE>::getSizeInBytes();
  // Stealer shared state
  shared_size += stealer_shared_size;
  return shared_size;
}

template <typename WorkQueue>
__device__ __forceinline__ void populateClusterQueue(WorkQueue& cluster_queue, size_t work_tiles) {
  if (threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 900
    auto cluster = cooperative_groups::this_cluster();
    if (cluster.block_rank() == 0) {
      const uint32_t cluster_size = static_cast<uint32_t>(cluster.dim_blocks().x);
      const uint32_t cluster_idx = static_cast<uint32_t>(blockIdx.x / cluster_size);
      const uint32_t num_clusters = static_cast<uint32_t>(gridDim.x / cluster_size);
      for (uint32_t i = cluster_idx; i < work_tiles; i += num_clusters) {
        cluster_queue.push(i);
      }
    }
#else
    for (uint32_t i = static_cast<uint32_t>(blockIdx.x); i < work_tiles; i += static_cast<uint32_t>(gridDim.x)) {
      cluster_queue.push(i);
    }
#endif
  }
#if __CUDA_ARCH__ >= 900
  cooperative_groups::this_cluster().sync();
#endif
}

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
  constexpr uint32_t WARP_RANDOM_ROUNDS = 1U;
  constexpr uint32_t RNG_SALT = 0xA57D3C29U;
  auto cluster = cooperative_groups::this_cluster();
  uint32_t* steal_state_ptr = cluster.map_shared_rank(&steal_state, 0);

  if (cluster.block_rank() == 0 && threadIdx.x < ADVANCE_WARP_SIZE) {
    const uint32_t lane = static_cast<uint32_t>(threadIdx.x) & (ADVANCE_WARP_SIZE - 1U);
    const unsigned int warp_mask = __activemask();
    const uint32_t cluster_size = static_cast<uint32_t>(cluster.dim_blocks().x);
    const uint32_t my_cluster = static_cast<uint32_t>(blockIdx.x) / cluster_size;
    const uint32_t num_clusters = static_cast<uint32_t>(gridDim.x) / cluster_size;
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

      for (uint32_t round = 0; round < WARP_RANDOM_ROUNDS; ++round) {
        const bool lane_active = lane < (num_clusters - 1U);
        const uint32_t proposed_victim =
            mapVictimExcludingSelf(nextStealRngState(lane_rng_state), my_cluster, num_clusters);
        bool candidate_available = false;
        if (lane_active) {
          candidate_available = cluster_work_queues[proposed_victim].size() > 0U;
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
      //   for (uint32_t i = 0; i < num_clusters - 1; ++i) {
      //     const uint32_t victim = (start + i) % num_clusters;
      //     if (victim == my_cluster) {
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
  // cluster.sync();
  return has_new_work;
#else
  (void)local_cluster_queue;
  (void)cluster_work_queues;
  (void)stealer;
  (void)steal_state;
  return false;
#endif
}

template <view View,
          advance_direction Direction,
          size_t BlockSize,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename InFrontierDevT,
          typename OutFrontierDevT,
          typename LockType,
          typename StealerDeviceT,
          typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              InFrontierDevT in_dev_frontier,
                              OutFrontierDevT out_dev_frontier,
                              int coarsening_factor,
                              size_t work_tiles,
                              clutra::detail::utils::WorkQueueView<uint32_t, LockType>* cluster_work_queues,
                              StealerDeviceT stealer,
                              LambdaT functor) {
  static_assert(BlockSize % ADVANCE_WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ clutra::detail::utils::SharedQueue<BlockSize> cta_queue;
  __shared__ clutra::detail::utils::SharedQueue<ADVANCE_WARP_SIZE> warp_queues[BlockSize / ADVANCE_WARP_SIZE];
  __shared__ typename StealerDeviceT::template SharedState<BlockSize> stealer_state;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (ADVANCE_WARP_SIZE - 1);
  const int block_dim = blockDim.x;
  auto& warp_queue = warp_queues[warp_id];
  auto& cluster_queue = clutra::detail::utils::getCurrentClusterQueueView(cluster_work_queues);
  __shared__ uint32_t global_steal_state;

  if (tid == 0) {
    global_steal_state = 0;
  }
  __syncthreads();

  if (stealer.isIntraClusterStealingEnabled()) {
    stealer.template init<BlockSize>(&cta_queue, stealer_state);
  }

  populateClusterQueue(cluster_queue, work_tiles);

  // uint32_t tile = 0;
  // bool has_tile = false;
  __shared__ uint32_t tile;
  while (true) {
    if (tid == 0 && !cluster_queue.pop(tile)) {
      tile = UINT32_MAX;
    }
    __syncthreads();
    if (tile == UINT32_MAX) {
      break;
    }
    processTile<View, Direction, BlockSize>(graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile,
                                            cta_queue, warp_queue, stealer, stealer_state, functor, tid, lane,
                                            block_dim);
  }
  if (stealer.isIntraClusterStealingEnabled()) {
    runLocalStealLoop<Direction, BlockSize>(graph_dev, out_dev_frontier, stealer, stealer_state, functor, tid,
                                            block_dim);
  }

  while (stealer.isInterClusterStealingEnabled()) {
    const bool has_new_work = tryGlobalClusterSteal(cluster_queue, cluster_work_queues, stealer, global_steal_state);
    if (!has_new_work) {
      break;
    }
    while (true) {
      if (tid == 0 && !cluster_queue.pop(tile)) {  // Short circuit if no tile to steal
        tile = UINT32_MAX;
      }
      __syncthreads();
      if (tile == UINT32_MAX) {
        break;
      }
      processTile<View, Direction, BlockSize>(graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile,
                                              cta_queue, warp_queue, stealer, stealer_state, functor, tid, lane,
                                              block_dim);
    }
    if (stealer.isIntraClusterStealingEnabled()) {
      runLocalStealLoop<Direction, BlockSize>(graph_dev, out_dev_frontier, stealer, stealer_state, functor, tid,
                                              block_dim);
    }
  }

  stealer.template finalize<BlockSize>();
}

__device__ uint32_t getAssignedEdge(uint32_t tile_gid, uint32_t tid) {
  return static_cast<uint32_t>(tile_gid) * blockDim.x + tid;
}

template <size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename StealerDeviceT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev, size_t total_iters, StealerDeviceT stealer, LambdaT functor) {

  static_assert(BlockSize % ADVANCE_WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ clutra::detail::utils::SharedQueue<BlockSize> cta_queue;
  __shared__ clutra::detail::utils::SharedQueue<ADVANCE_WARP_SIZE> warp_queues[BlockSize / ADVANCE_WARP_SIZE];
  __shared__ typename StealerDeviceT::template SharedState<BlockSize> stealer_state;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (ADVANCE_WARP_SIZE - 1);
  const int block_dim = blockDim.x;
  auto& warp_queue = warp_queues[warp_id];

  if (stealer.isIntraClusterStealingEnabled()) {
    stealer.template init<BlockSize>(&cta_queue, stealer_state);
  }

  for (size_t iter = 0; iter < total_iters; ++iter) {  // TODO change with stealing with ptx
    const uint32_t tile_gid = static_cast<uint32_t>(blockIdx.x + (iter * gridDim.x));

    __syncthreads();
    if (tid == 0) {
      stealer.setReady(stealer_state, false);
      cta_queue.init();
    }
    if (lane == 0) {
      warp_queue.init();
    }

    __syncthreads();

    const uint32_t edge_index = getAssignedEdge(tile_gid, tid);
    if (edge_index < graph_dev.getEdgeCount()) {
      const auto src = graph_dev.getSourceVertex(edge_index);
      const auto dst = graph_dev.getDestinationVertex(edge_index);
      size_t tot_degree = graph_dev.getDegree(src) + graph_dev.getDegree(dst);
      size_t cta_threshold = block_dim;
      if (tot_degree >= cta_threshold) {
        cta_queue.push(edge_index, tot_degree);
      } else {
        warp_queue.push(edge_index, tot_degree);
      }
    }

    __syncthreads();

    if (tid == 0) {
      stealer.setReady(stealer_state, true);
    }

    uint32_t edge, degree;
    while (cta_queue.pop(edge, degree)) {
      const auto weight = graph_dev.getEdgeWeight(edge);
      const auto src = graph_dev.getSourceVertex(edge);
      const auto dst = graph_dev.getDestinationVertex(edge);
      functor(src, dst, edge, weight);
    }

    for (int i = tid; i < warp_queue.size(); ++i) {
      const uint32_t edge = warp_queue.vertices[i];
      const auto weight = graph_dev.getEdgeWeight(edge);
      const auto src = graph_dev.getSourceVertex(edge);
      const auto dst = graph_dev.getDestinationVertex(edge);
      functor(src, dst, edge, weight);
    }
  }

  if (!stealer.isIntraClusterStealingEnabled()) {
    return;
  }

  while (true) {
    const int steal_count =
        stealer.template attemptStealing<BlockSize>(stealer_state, stealer.getLocalStealingChunkSize());
    if (steal_count == 0) {
      break;
    }
    uint32_t steal_edge = 0;
    uint32_t steal_degree = 0;
    for (int i = 0; i < steal_count; ++i) {
      stealer.template steal<BlockSize>(stealer_state, i, steal_edge, steal_degree);
      const auto weight = graph_dev.getEdgeWeight(steal_edge);
      const auto src = graph_dev.getSourceVertex(steal_edge);
      const auto dst = graph_dev.getDestinationVertex(steal_edge);
      functor(src, dst, steal_edge, weight);
    }
    __syncthreads();
  }
}

}  // namespace clutra::operators::advance::detail
