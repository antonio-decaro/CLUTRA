/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <frontier/frontier.cuh>
#include <operators/advance/options.hpp>
#include <memory>
#include <utils/profile.cuh>
#include <utils/device.cuh>
#include <concepts>

namespace clutra::operators::advance::detail {

template<advance_direction Direction, graph::detail::DeviceGraphConcept GraphDevT, typename FrontierDevT, typename LambdaT>
__device__ inline void processVertexRange(GraphDevT graph_dev,
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

template<typename FronterDevT>
__device__ uint32_t getAssignedVertex(const FronterDevT& in_dev_frontier, uint32_t coarsening_factor, uint32_t gid, uint32_t tid) {
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
}

template<advance_direction Direction, typename FrontierDevT>
__device__ __forceinline__ bool checkVertexActive(const FrontierDevT& in_dev_frontier, uint32_t vertex) {
  if constexpr (Direction == advance_direction::push) {
    return in_dev_frontier.check(vertex);
  } else {
    return !in_dev_frontier.check(vertex);
  }
}

template<advance_direction Direction, size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename InFrontierDevT, typename OutFrontierDevT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              InFrontierDevT in_dev_frontier,
                              OutFrontierDevT out_dev_frontier,
                              int coarsening_factor,
                              LambdaT functor) {
  constexpr int WARP_SIZE = 32;
  static_assert(BlockSize % WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ uint32_t n_edges_cta[BlockSize];
  __shared__ uint32_t n_edges_warp[BlockSize];
  __shared__ uint32_t warp_reduce[BlockSize];
  __shared__ uint32_t warp_reduce_tail[BlockSize / WARP_SIZE];
  __shared__ uint32_t cta_reduce[BlockSize];
  __shared__ uint32_t cta_reduce_tail;
  __shared__ uint32_t thread_reduce_vertices[BlockSize];
  __shared__ uint32_t thread_reduce_degrees[BlockSize];
  __shared__ uint32_t thread_reduce_tail[BlockSize / WARP_SIZE];
  
  // fetch frontier info
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  uint32_t assigned_vertex = getAssignedVertex(in_dev_frontier, coarsening_factor, blockIdx.x, threadIdx.x);

  // init computation
  if (lane == 0) {
    warp_reduce_tail[warp_id] = 0;
    thread_reduce_tail[warp_id] = 0;
  }
  if (threadIdx.x == 0) cta_reduce_tail = 0;

  __syncthreads();

  // classify vertices by degree
  const uint32_t warp_offset = warp_id * WARP_SIZE;
  const bool vertex_active = assigned_vertex < graph_dev.getVertexCount() && in_dev_frontier.check(assigned_vertex);
  if (vertex_active) {
    const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
    const uint32_t cta_threshold = blockDim.x * blockDim.x;
    if (n_edges >= cta_threshold) {
      const uint32_t loc = atomicAdd(&cta_reduce_tail, 1);
      n_edges_cta[loc] = n_edges;
      cta_reduce[loc] = assigned_vertex;
    } else if (n_edges >= WARP_SIZE) {
      const uint32_t loc = atomicAdd(&warp_reduce_tail[warp_id], 1);
      n_edges_warp[warp_offset + loc] = n_edges;
      warp_reduce[warp_offset + loc] = assigned_vertex;
    } else {
      const int loc = atomicAdd(&thread_reduce_tail[warp_id], 1);
      const int write_idx = warp_offset + loc;
      thread_reduce_vertices[write_idx] = assigned_vertex;
      thread_reduce_degrees[write_idx] = n_edges;
    }
  }

  __syncthreads();

  // process CTA large degree vertices
  for (int i = 0; i < cta_reduce_tail; ++i) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, cta_reduce[i], n_edges_cta[i], threadIdx.x, blockDim.x);
  }
  
  // process warp large degree vertices
  for (int i = 0; i < warp_reduce_tail[warp_id]; ++i) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, warp_reduce[warp_offset + i], n_edges_warp[warp_offset + i], lane, WARP_SIZE);
  }

  // process small degree vertices
  const uint32_t tiny_tail = thread_reduce_tail[warp_id];
  for (int i = 0; i < tiny_tail; ++i) {
    const int tiny_idx = warp_offset + i;
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, thread_reduce_vertices[tiny_idx], thread_reduce_degrees[tiny_idx], lane, WARP_SIZE);
  }
}

template<advance_direction Direction, clutra::graph::detail::GraphConcept GraphT, typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 256;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto graph_dev = (Direction == advance_direction::pull) ? graph.getTransposedDeviceGraph() : graph.getDeviceGraph();

  const bool invert = (Direction == advance_direction::pull); // In pull mode, we consider inactive vertices as active.
  input_frontier.computeActiveFrontier(invert);

  // compute launch informations
  const size_t coarsening_factor = CU_SIZE  / 32 /* Warp Size */;
  const size_t bitmap_range = in_dev_frontier.getBitmapRange();
  if (bitmap_range != 32) {
    throw std::runtime_error("Advance operator currently supports only frontiers with bitmap range equal to 32.");
  }
  const size_t active_size = input_frontier.getActiveFrontierSize();

  const size_t block_size = CU_SIZE;
  const size_t grid_size = ((active_size * bitmap_range) + block_size - 1) / block_size;

  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel");

  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();

    detail::advanceKernel<Direction, CU_SIZE><<<grid_size, block_size>>>(graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    detail::advanceKernel<Direction, CU_SIZE><<<grid_size, block_size>>>(graph_dev, in_dev_frontier, frontier::detail::NullFrontierDevice{}, coarsening_factor, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}
}