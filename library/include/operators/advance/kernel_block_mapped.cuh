/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/block/block_scan.cuh>

#include <operators/advance/kernel_bucketing.cuh>

namespace clutra::operators::advance::detail {

template <typename T>
__device__ __forceinline__ uint32_t blockMappedUpperBound(const T* values, uint32_t n, T value) {
  uint32_t left = 0;
  uint32_t right = n;
  while (left < right) {
    const uint32_t mid = left + ((right - left) >> 1);
    if (values[mid] <= value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template <advance_direction Direction,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename OutFrontierDevT,
          typename LambdaT>
__device__ __forceinline__ void processBlockMappedEdgeRange(GraphDevT graph_dev,
                                                            OutFrontierDevT out_dev_frontier,
                                                            LambdaT functor,
                                                            uint32_t source,
                                                            uint32_t begin_edge,
                                                            uint32_t end_edge,
                                                            int tid,
                                                            int block_dim) {
  static_assert(Direction == advance_direction::push, "Block-mapped advance currently supports push direction only.");

  for (uint32_t edge = begin_edge + static_cast<uint32_t>(tid); edge < end_edge;
       edge += static_cast<uint32_t>(block_dim)) {
    if (edge >= graph_dev.getEdgeCount()) {
      continue;
    }
    const auto neighbor = graph_dev.getDestinationVertex(edge);
    const auto weight = graph_dev.getEdgeWeight(edge);
    if (functor(source, neighbor, edge, weight)) {
      out_dev_frontier.insert(neighbor);
    }
  }
}

template <view View,
          advance_direction Direction,
          size_t BlockSize,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename InFrontierDevT,
          typename OutFrontierDevT,
          typename LambdaT>
__device__ __forceinline__ void
processBlockMappedTile(GraphDevT graph_dev,
                       InFrontierDevT in_dev_frontier,
                       OutFrontierDevT out_dev_frontier,
                       int coarsening_factor,
                       uint32_t tile_gid,
                       LambdaT functor,
                       uint32_t* vertices,
                       uint32_t* start_edges,
                       uint32_t* scan_begins,
                       uint32_t* scan_ends,
                       uint32_t* total_edges,
                       typename cub::BlockScan<uint32_t, BlockSize>::TempStorage& scan_storage,
                       int tid,
                       int block_dim) {
  static_assert(Direction == advance_direction::push, "Block-mapped advance currently supports push direction only.");

  const uint32_t assigned_vertex = getAssignedVertex<View>(in_dev_frontier, coarsening_factor, tile_gid, tid);
  const bool vertex_active = checkVertexActive<View, Direction>(graph_dev, in_dev_frontier, assigned_vertex);

  const uint32_t degree = vertex_active ? static_cast<uint32_t>(graph_dev.getDegree(assigned_vertex)) : 0U;
  vertices[tid] = vertex_active ? assigned_vertex : UINT32_MAX;
  start_edges[tid] = vertex_active ? static_cast<uint32_t>(graph_dev.getFirstNeighbor(assigned_vertex)) : 0U;

  uint32_t exclusive_begin = 0;
  uint32_t aggregate_degree = 0;
  using block_scan_t = cub::BlockScan<uint32_t, BlockSize>;
  block_scan_t(scan_storage).ExclusiveSum(degree, exclusive_begin, aggregate_degree);

  scan_begins[tid] = exclusive_begin;
  scan_ends[tid] = exclusive_begin + degree;

  if (tid == 0) {
    total_edges[0] = aggregate_degree;
  }
  __syncthreads();

  const uint32_t block_total_edges = total_edges[0];
  for (uint32_t edge_rank = static_cast<uint32_t>(tid); edge_rank < block_total_edges;
       edge_rank += static_cast<uint32_t>(block_dim)) {
    const uint32_t slot = blockMappedUpperBound(scan_ends, static_cast<uint32_t>(BlockSize), edge_rank);
    if (slot >= BlockSize) {
      continue;
    }

    const uint32_t source = vertices[slot];
    if (source == UINT32_MAX) {
      continue;
    }

    const uint32_t local_edge_offset = edge_rank - scan_begins[slot];
    const uint32_t edge = start_edges[slot] + local_edge_offset;
    if (edge >= graph_dev.getEdgeCount()) {
      continue;
    }

    const auto neighbor = graph_dev.getDestinationVertex(edge);
    const auto weight = graph_dev.getEdgeWeight(edge);
    if (functor(source, neighbor, edge, weight)) {
      out_dev_frontier.insert(neighbor);
    }
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
__device__ __forceinline__ void
processBlockMappedTileWithLocalStealing(GraphDevT graph_dev,
                                        InFrontierDevT in_dev_frontier,
                                        OutFrontierDevT out_dev_frontier,
                                        int coarsening_factor,
                                        uint32_t tile_gid,
                                        clutra::detail::utils::SharedQueueBlockMapped<BlockSize>& cta_queue,
                                        StealerDeviceT stealer,
                                        typename StealerDeviceT::template SharedState<BlockSize>& stealer_state,
                                        LambdaT functor,
                                        uint32_t* vertices,
                                        uint32_t* start_edges,
                                        uint32_t* scan_begins,
                                        uint32_t* scan_ends,
                                        uint32_t* total_edges,
                                        typename cub::BlockScan<uint32_t, BlockSize>::TempStorage& scan_storage,
                                        int tid,
                                        int block_dim) {
  static_assert(Direction == advance_direction::push, "Block-mapped advance currently supports push direction only.");
  constexpr uint32_t STEAL_VERTEX_THRESHOLD_MULTIPLIER = 4U;

  __syncthreads();
  if (tid == 0) {
    stealer.setReadyBlockMapped(stealer_state, false);
    cta_queue.init();
  }
  __syncthreads();

  const uint32_t assigned_vertex = getAssignedVertex<View>(in_dev_frontier, coarsening_factor, tile_gid, tid);
  const bool vertex_active = checkVertexActive<View, Direction>(graph_dev, in_dev_frontier, assigned_vertex);
  const uint32_t degree = vertex_active ? static_cast<uint32_t>(graph_dev.getDegree(assigned_vertex)) : 0U;
  const uint32_t begin_edge = vertex_active ? static_cast<uint32_t>(graph_dev.getFirstNeighbor(assigned_vertex)) : 0U;
  const uint32_t steal_threshold = static_cast<uint32_t>(block_dim) * STEAL_VERTEX_THRESHOLD_MULTIPLIER;
  const bool queue_candidate = vertex_active && degree >= steal_threshold;

  vertices[tid] = vertex_active ? assigned_vertex : UINT32_MAX;
  start_edges[tid] = begin_edge;

  if (queue_candidate && degree > 0U) {
    cta_queue.push(assigned_vertex, begin_edge, begin_edge + degree);
  }

  const uint32_t scan_degree = queue_candidate ? 0U : degree;
  uint32_t exclusive_begin = 0;
  uint32_t aggregate_degree = 0;
  using block_scan_t = cub::BlockScan<uint32_t, BlockSize>;
  block_scan_t(scan_storage).ExclusiveSum(scan_degree, exclusive_begin, aggregate_degree);
  scan_begins[tid] = exclusive_begin;
  scan_ends[tid] = exclusive_begin + scan_degree;

  __syncthreads();
  if (tid == 0) {
    total_edges[0] = aggregate_degree;
  }
  __syncthreads();

  const uint32_t block_total_edges = total_edges[0];
  for (uint32_t edge_rank = static_cast<uint32_t>(tid); edge_rank < block_total_edges;
       edge_rank += static_cast<uint32_t>(block_dim)) {
    const uint32_t slot = blockMappedUpperBound(scan_ends, static_cast<uint32_t>(BlockSize), edge_rank);
    if (slot >= BlockSize) {
      continue;
    }
    const uint32_t source = vertices[slot];
    if (source == UINT32_MAX) {
      continue;
    }
    const uint32_t local_edge_offset = edge_rank - scan_begins[slot];
    const uint32_t edge = start_edges[slot] + local_edge_offset;
    if (edge >= graph_dev.getEdgeCount()) {
      continue;
    }
    const auto neighbor = graph_dev.getDestinationVertex(edge);
    const auto weight = graph_dev.getEdgeWeight(edge);
    if (functor(source, neighbor, edge, weight)) {
      out_dev_frontier.insert(neighbor);
    }
  }

  __syncthreads();
  if (tid == 0) {
    stealer.setReadyBlockMapped(stealer_state, true);
  }

  uint32_t source = 0;
  uint32_t steal_begin_edge = 0;
  uint32_t steal_end_edge = 0;
  while (cta_queue.pop(source, steal_begin_edge, steal_end_edge)) {
    processBlockMappedEdgeRange<Direction>(graph_dev, out_dev_frontier, functor, source, steal_begin_edge,
                                           steal_end_edge, tid, block_dim);
    __syncthreads();
  }
}

template <advance_direction Direction,
          size_t BlockSize,
          graph::detail::DeviceGraphConcept GraphDevT,
          typename OutFrontierDevT,
          typename StealerDeviceT,
          typename LambdaT>
__device__ __forceinline__ void
runLocalBlockMappedStealLoop(GraphDevT graph_dev,
                             OutFrontierDevT out_dev_frontier,
                             StealerDeviceT stealer,
                             typename StealerDeviceT::template SharedState<BlockSize>& stealer_state,
                             LambdaT functor,
                             int tid,
                             int block_dim) {
  while (true) {
    const int steal_count =
        stealer.template attemptStealingBlockMapped<BlockSize>(stealer_state, stealer.getLocalStealingChunkSize());
    if (steal_count == 0) {
      break;
    }

    for (int i = 0; i < steal_count; ++i) {
      uint32_t source = 0;
      uint32_t begin_edge = 0;
      uint32_t end_edge = 0;
      stealer.template stealBlockMapped<BlockSize>(stealer_state, i, source, begin_edge, end_edge);
      processBlockMappedEdgeRange<Direction>(graph_dev, out_dev_frontier, functor, source, begin_edge, end_edge, tid,
                                             block_dim);
    }
    __syncthreads();
  }
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
__global__ void advanceKernelBlockMapped(GraphDevT graph_dev,
                                         InFrontierDevT in_dev_frontier,
                                         OutFrontierDevT out_dev_frontier,
                                         int coarsening_factor,
                                         size_t work_tiles,
                                         clutra::detail::utils::WorkQueueView<uint32_t, LockType>* cluster_work_queues,
                                         StealerDeviceT stealer,
                                         LambdaT functor) {
  static_assert(BlockSize % ADVANCE_WARP_SIZE == 0, "BlockSize must be multiple of warp size");
  static_assert(Direction == advance_direction::push, "Block-mapped advance currently supports push direction only.");

  __shared__ uint32_t vertices[BlockSize];
  __shared__ uint32_t start_edges[BlockSize];
  __shared__ uint32_t scan_begins[BlockSize];
  __shared__ uint32_t scan_ends[BlockSize];
  __shared__ uint32_t total_edges[1];
  __shared__ typename cub::BlockScan<uint32_t, BlockSize>::TempStorage scan_storage;

  __shared__ clutra::detail::utils::SharedQueueBlockMapped<BlockSize> cta_queue;
  __shared__ typename StealerDeviceT::template SharedState<BlockSize> stealer_state;

  const int tid = threadIdx.x;
  const int block_dim = blockDim.x;
  auto& cluster_queue = clutra::detail::utils::getCurrentClusterQueueView(cluster_work_queues);
  __shared__ uint32_t global_steal_state;

  if (tid == 0) {
    global_steal_state = 0;
  }
  __syncthreads();

  if (stealer.isIntraClusterStealingEnabled()) {
    stealer.template initBlockMapped<BlockSize>(&cta_queue, stealer_state);
  }

  populateClusterQueue(cluster_queue, work_tiles);

  __shared__ uint32_t tile;
  while (true) {
    if (tid == 0 && !cluster_queue.pop(tile)) {
      tile = UINT32_MAX;
    }
    __syncthreads();
    if (tile == UINT32_MAX) {
      break;
    }

    if (stealer.isIntraClusterStealingEnabled()) {
      processBlockMappedTileWithLocalStealing<View, Direction, BlockSize>(
          graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile, cta_queue, stealer, stealer_state,
          functor, vertices, start_edges, scan_begins, scan_ends, total_edges, scan_storage, tid, block_dim);
    } else {
      processBlockMappedTile<View, Direction, BlockSize>(
          graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile, functor, vertices, start_edges,
          scan_begins, scan_ends, total_edges, scan_storage, tid, block_dim);
    }
  }

  if (stealer.isIntraClusterStealingEnabled()) {
    runLocalBlockMappedStealLoop<Direction, BlockSize>(graph_dev, out_dev_frontier, stealer, stealer_state, functor,
                                                       tid, block_dim);
  }

  while (stealer.isInterClusterStealingEnabled()) {
    const bool has_new_work = tryGlobalClusterSteal(cluster_queue, cluster_work_queues, stealer, global_steal_state);
    if (!has_new_work) {
      break;
    }

    while (true) {
      if (tid == 0 && !cluster_queue.pop(tile)) {
        tile = UINT32_MAX;
      }
      __syncthreads();
      if (tile == UINT32_MAX) {
        break;
      }

      if (stealer.isIntraClusterStealingEnabled()) {
        processBlockMappedTileWithLocalStealing<View, Direction, BlockSize>(
            graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile, cta_queue, stealer, stealer_state,
            functor, vertices, start_edges, scan_begins, scan_ends, total_edges, scan_storage, tid, block_dim);
      } else {
        processBlockMappedTile<View, Direction, BlockSize>(
            graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, tile, functor, vertices, start_edges,
            scan_begins, scan_ends, total_edges, scan_storage, tid, block_dim);
      }
    }

    if (stealer.isIntraClusterStealingEnabled()) {
      runLocalBlockMappedStealLoop<Direction, BlockSize>(graph_dev, out_dev_frontier, stealer, stealer_state, functor,
                                                         tid, block_dim);
    }
  }

  stealer.template finalize<BlockSize>();
}

}  // namespace clutra::operators::advance::detail
