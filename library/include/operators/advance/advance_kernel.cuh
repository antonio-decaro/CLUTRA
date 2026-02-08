/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <frontier/frontier.cuh>
#include <operators/advance/options.hpp>
#include <utils/profile.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/queue.cuh>
#include <utils/logging.cuh>
#include <stealer/stealer.cuh>

namespace cg = cooperative_groups;

namespace clutra::operators::advance::detail {

template<advance_direction Direction, graph::detail::DeviceGraphConcept GraphDevT, typename FrontierDevT, typename LambdaT>
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

template<view View, typename FronterDevT>
__device__ uint32_t getAssignedVertex(const FronterDevT& in_dev_frontier, uint32_t coarsening_factor, uint32_t gid, uint32_t tid) {
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

template<view View, advance_direction Direction, graph::detail::DeviceGraphConcept GraphDevT, typename FrontierDevT>
__device__ __forceinline__ bool checkVertexActive(const GraphDevT& graph_dev, const FrontierDevT& in_dev_frontier, uint32_t vertex) {
  if constexpr (View == view::frontier) {
    if constexpr (Direction == advance_direction::push) {
      return (vertex < graph_dev.getVertexCount()) && in_dev_frontier.check(vertex);
    } else {
      return (vertex < graph_dev.getVertexCount()) && !in_dev_frontier.check(vertex);
    }
  } else { // View == view::graph
    return vertex < graph_dev.getVertexCount();
  }
}

template<size_t BlockSize>
size_t getAdvanceSharedMemorySize(size_t stealer_shared_size) {
  constexpr size_t WARP_SIZE = 32;
  size_t shared_size = 0;
  // CTA queue
  shared_size += clutra::detail::utils::SharedQueue<BlockSize>::getSizeInBytes();
  // Warp queues
  shared_size += (BlockSize / WARP_SIZE) * clutra::detail::utils::SharedQueue<WARP_SIZE>::getSizeInBytes();
  // Stealer shared state
  shared_size += stealer_shared_size;
  return shared_size;
}

template<view View, advance_direction Direction, size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename InFrontierDevT, typename OutFrontierDevT, typename StealerDeviceT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              InFrontierDevT in_dev_frontier,
                              OutFrontierDevT out_dev_frontier,
                              int coarsening_factor,
                              size_t total_iters,
                              StealerDeviceT stealer,
                              LambdaT functor) {
  constexpr int WARP_SIZE = 32;
  static_assert(BlockSize % WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ clutra::detail::utils::SharedQueue<BlockSize> cta_queue;
  __shared__ clutra::detail::utils::SharedQueue<WARP_SIZE> warp_queues[BlockSize / WARP_SIZE];
  __shared__ typename StealerDeviceT::template SharedState<BlockSize> stealer_state;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (WARP_SIZE - 1);
  const int block_dim = blockDim.x;
  auto& warp_queue = warp_queues[warp_id];

  if (stealer.isStealingEnabled()) { stealer.template init<BlockSize>(&cta_queue, stealer_state); }

  for (size_t iter = 0; iter < total_iters; ++iter) { // TODO change with stealing with ptx 
    const uint32_t tile_gid = static_cast<uint32_t>(blockIdx.x + (iter * gridDim.x));

    __syncthreads();
    if (tid == 0)  {
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

    uint32_t vertex, degree;
    while (cta_queue.pop(vertex, degree)) {
      processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, vertex, degree, tid, block_dim);
      __syncthreads();
    }
    
    for (int i = 0; i < warp_queue.size(); ++i) {
      processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, warp_queue.vertices[i], warp_queue.degrees[i], lane, WARP_SIZE);
    }

  }
  if (!stealer.isStealingEnabled()) { return ; }

  while (true) {
    const int steal_count = stealer.template attemptStealing<BlockSize>(stealer_state, stealer.getStealingChunkSize());
    if (steal_count == 0) {
      break;
    }
    uint32_t steal_vertex = 0;
    uint32_t steal_degree = 0;
    for (int i = 0; i < steal_count; ++i) {
      stealer.template steal<BlockSize>(stealer_state, i, steal_vertex, steal_degree);
      processVertexRange<Direction>(graph_dev,
        out_dev_frontier,
        functor,
        steal_vertex,
        steal_degree,
        tid,
        block_dim);
      }
    __syncthreads();
  }

  stealer.template finalize<BlockSize>();
}

__device__ uint32_t getAssignedEdge(uint32_t tile_gid, uint32_t tid) {
  return static_cast<uint32_t>(tile_gid) * blockDim.x + tid;
}

template<size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename StealerDeviceT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              size_t total_iters,
                              StealerDeviceT stealer,
                              LambdaT functor) {

  constexpr int WARP_SIZE = 32;
  static_assert(BlockSize % WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ clutra::detail::utils::SharedQueue<BlockSize> cta_queue;
  __shared__ clutra::detail::utils::SharedQueue<WARP_SIZE> warp_queues[BlockSize / WARP_SIZE];
  __shared__ typename StealerDeviceT::template SharedState<BlockSize> stealer_state;

  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & (WARP_SIZE - 1);
  const int block_dim = blockDim.x;
  auto& warp_queue = warp_queues[warp_id];

  if (stealer.isStealingEnabled()) { stealer.template init<BlockSize>(&cta_queue, stealer_state); }

  for (size_t iter = 0; iter < total_iters; ++iter) { // TODO change with stealing with ptx 
    const uint32_t tile_gid = static_cast<uint32_t>(blockIdx.x + (iter * gridDim.x));

    __syncthreads();
    if (tid == 0)  {
      stealer.setReady(stealer_state, false);
      cta_queue.init();
    }
    if (lane == 0) {
      warp_queue.init();
    }

    __syncthreads();

    const uint32_t edge_index = getAssignedEdge(tile_gid, tid);
    if (edge_index < graph_dev.getEdgeCount()) {
      const auto weight = graph_dev.getEdgeWeight(edge_index);
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

  if (!stealer.isStealingEnabled()) { return ; }

  while (true) {
    const int steal_count = stealer.template attemptStealing<BlockSize>(stealer_state, stealer.getStealingChunkSize());
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

} // namespace clutra::operators::advance::detail
