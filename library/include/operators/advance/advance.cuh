/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/graph.cuh>
#include <frontier/frontier.cuh>
#include <utils/profile.cuh>
#include <utils/device.cuh>



namespace clutra::operators::advance {

namespace detail {

template<size_t BlockSize, typename GraphDevT, typename FrontierDevT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev, FrontierDevT in_dev_frontier, FrontierDevT out_dev_frontier,int coarsening_factor, LambdaT functor) {
  __shared__ uint32_t n_edges_cta[BlockSize];
  __shared__ uint32_t n_edges_warp[BlockSize];
  __shared__ uint32_t warp_reduce[BlockSize];
  __shared__ uint32_t warp_reduce_tail[BlockSize / 32];
  __shared__ uint32_t cta_reduce[BlockSize];
  __shared__ uint32_t cta_reduce_tail;
  __shared__ uint32_t thread_reduce_vertices[BlockSize];
  __shared__ uint32_t thread_reduce_degrees[BlockSize];
  __shared__ uint32_t thread_reduce_tail[BlockSize / 32];
  
  // fetch frontier info
  const int warp_id = threadIdx.x / 32;
  const int offsets_size = in_dev_frontier.getOffsetsSize()[0];
  const uint16_t bitmap_range = in_dev_frontier.getBitmapRange();
  const int* bitmap_offsets = in_dev_frontier.getOffsets();

  // fetch assigned vertex
  const uint32_t actual_id_offset = (blockIdx.x * coarsening_factor) + (threadIdx.x / bitmap_range);
  if (actual_id_offset >= offsets_size) return;
  const auto assigned_vertex = (bitmap_offsets[actual_id_offset] * bitmap_range) + (threadIdx.x % bitmap_range);

  // init computation
  if ((threadIdx.x % 32) == 0) {
    warp_reduce_tail[warp_id] = 0;
    thread_reduce_tail[warp_id] = 0;
  }
  if (threadIdx.x == 0) cta_reduce_tail = 0;

  __syncthreads();

  // classify vertices by degree
  const uint32_t offset = warp_id * 32;
  if (assigned_vertex < graph_dev.getVertexCount() && in_dev_frontier.check(assigned_vertex)) {
    const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
    if (n_edges >= blockDim.x * blockDim.x) {
      const uint32_t loc = atomicAdd(&cta_reduce_tail, 1);
      n_edges_cta[loc] = n_edges;
      cta_reduce[loc] = assigned_vertex;
    } else if (n_edges >= 32) {
      const uint32_t loc = atomicAdd(&warp_reduce_tail[warp_id], 1);
      n_edges_warp[offset + loc] = n_edges;
      warp_reduce[offset + loc] = assigned_vertex;
    } else {
      const int loc = atomicAdd(&thread_reduce_tail[warp_id], 1);
      const int write_idx = (warp_id * 32) + loc;
      thread_reduce_vertices[write_idx] = assigned_vertex;
      thread_reduce_degrees[write_idx] = n_edges;
    }
  }

  __syncthreads();

  // process CTA large degree vertices
  for (int i = 0; i < cta_reduce_tail; ++i) {
    const auto vertex = cta_reduce[i];
    const uint32_t n_edges = n_edges_cta[i];
    auto start = graph_dev.begin(vertex);

    for (int j = threadIdx.x; j < n_edges; j += blockDim.x) {
      auto n = start + j;
      const auto edge = n.getIndex();
      const auto weight = graph_dev.getEdgeWeight(edge);
      const auto neighbor = *n;
      if (functor(vertex, neighbor, edge, weight)) {
        out_dev_frontier.insert(neighbor);
      }
    }
  }
  
  // process warp large degree vertices
  for (int i = 0; i < warp_reduce_tail[warp_id]; ++i) {
    const auto vertex = warp_reduce[offset + i];
    const uint32_t n_edges = n_edges_warp[offset + i];
    auto start = graph_dev.begin(vertex);

    for (int j = threadIdx.x % 32; j < n_edges; j += 32) {
      auto n = start + j;
      const auto edge = n.getIndex();
      const auto weight = graph_dev.getEdgeWeight(edge);
      const auto neighbor = *n;
      if (functor(vertex, neighbor, edge, weight)) {
        out_dev_frontier.insert(neighbor);
      }
    }
  }

  // process small degree vertices
  const uint32_t tiny_offset = warp_id * 32;
  const uint32_t tiny_tail = thread_reduce_tail[warp_id];
  for (int i = 0; i < tiny_tail; ++i) {
    const auto vertex = thread_reduce_vertices[tiny_offset + i];
    const auto n_edges = thread_reduce_degrees[tiny_offset + i];
    auto start = graph_dev.begin(vertex);
    for (int j = threadIdx.x % 32; j < n_edges; j += 32) {
      auto n = start + j;
      const auto edge = n.getIndex();
      const auto weight = graph_dev.getEdgeWeight(edge);
      auto neighbor = *n;
      if (functor(vertex, neighbor, edge, weight)) {
        out_dev_frontier.insert(neighbor);
      }
    }
}

} // namespace clutra::operators::advance::detail

template<typename GraphT, typename FrontierT, typename LambdaT>
void frontier(const GraphT& graph, const FrontierT& input_frontier, FrontierT& output_frontier, LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto out_dev_frontier = output_frontier.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  input_frontier.computeActiveFrontier();

  // compute launch informations
  const size_t coarsening_factor = CU_SIZE  / 32 /* Warp Size */;
  const size_t bitmap_range = in_dev_frontier.getBitmapRange();
  const size_t active_size = input_frontier.getActiveFrontierSize();

  const size_t block_size = coarsening_factor * bitmap_range;
  const size_t grid_size = ((active_size * bitmap_range) + block_size - 1) / block_size;

  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel");
  detail::advanceKernel<CU_SIZE><<<grid_size, block_size>>>(graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor, std::forward<LambdaT>(functor));
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}


}// namespace clutra::operators::advance
