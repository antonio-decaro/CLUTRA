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
#include <operators/advance/global_stealing.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/logging.cuh>
#include <utils/profile.cuh>
#include <utils/queue.cuh>

namespace clutra::operators::advance::detail {

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

  const int warp_queue_size = warp_queue.size();
  for (int i = 0; i < warp_queue_size; ++i) {
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
  stealer.template runLocalStealLoop<BlockSize>(
      stealer_state, stealer.getLocalStealingChunkSize(),
      [&](const clutra::stealer::StealQueueDescriptor& desc, int queue_index, int steal_index, int steal_count) {
        (void)steal_index;
        (void)steal_count;
        const auto* vertices = reinterpret_cast<const uint32_t*>(desc.payload0);
        const auto* degrees = reinterpret_cast<const uint32_t*>(desc.payload1);
        const uint32_t steal_vertex = vertices[queue_index];
        const uint32_t steal_degree = degrees[queue_index];
        processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, steal_vertex, steal_degree, tid, block_dim);
      });
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
    stealer.template init<BlockSize>(stealer_state, [&](auto& cluster, int rank) {
      auto* q = cluster.map_shared_rank(&cta_queue, rank);
      return clutra::stealer::StealQueueDescriptor{&q->head, &q->tail, reinterpret_cast<uintptr_t>(q->vertices),
                                                   reinterpret_cast<uintptr_t>(q->degrees), 0U};
    });
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

}  // namespace clutra::operators::advance::detail
