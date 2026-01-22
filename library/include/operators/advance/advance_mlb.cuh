/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <frontier/frontier.cuh>
#include <operators/advance/options.hpp>
#include <memory>
#include <utils/profile.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <stealer/stealer.cuh>
#include <utils/queue.cuh>
#include <spdlog/spdlog.h>

namespace cg = cooperative_groups;

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

template<advance_direction Direction, size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename InFrontierDevT, typename OutFrontierDevT, typename DerivedStealerT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              InFrontierDevT in_dev_frontier,
                              OutFrontierDevT out_dev_frontier,
                              int coarsening_factor,
                              clutra::stealer::Stealer<DerivedStealerT> stealer,
                              LambdaT functor) {
  constexpr int WARP_SIZE = 32;
  static_assert(BlockSize % WARP_SIZE == 0, "BlockSize must be multiple of warp size");

  __shared__ clutra::detail::utils::SharedQueue<BlockSize> stealing_queue;
  __shared__ clutra::detail::utils::SharedQueue<WARP_SIZE> warp_queues[BlockSize / WARP_SIZE];

  // fetch frontier info
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  auto& warp_queue = warp_queues[warp_id];

  // fetch assigned vertex
  uint32_t assigned_vertex = getAssignedVertex(in_dev_frontier, coarsening_factor, blockIdx.x, threadIdx.x);

  // init computation
  if (threadIdx.x == 0)  {
    stealing_queue.init();
  }
  if (lane == 0) {
    warp_queue.init();
  }

  __syncthreads();

  // classify vertices by degree
  const bool vertex_active = assigned_vertex < graph_dev.getVertexCount() && in_dev_frontier.check(assigned_vertex);
  if (vertex_active) {
    const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
    const uint32_t cta_threshold = blockDim.x;// * blockDim.x;

    if (n_edges >= cta_threshold) {
      stealing_queue.push(assigned_vertex, n_edges);
    } else {
      warp_queue.push(assigned_vertex, n_edges);
    }
  }

  __syncthreads();

  // process CTA large degree vertices
  for (int i = 0; i < stealing_queue.size(); ++i) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, stealing_queue.vertices[i], stealing_queue.degrees[i], threadIdx.x, blockDim.x);
  }
  
  // process warp large degree vertices
  for (int i = 0; i < warp_queue.size(); ++i) {
    processVertexRange<Direction>(graph_dev, out_dev_frontier, functor, warp_queue.vertices[i], warp_queue.degrees[i], lane, WARP_SIZE);
  }
}

template<advance_direction Direction, clutra::graph::detail::GraphConcept GraphT, typename DerivedStealerT, typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::Stealer<DerivedStealerT>& stealer,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 256;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto graph_dev = (Direction == advance_direction::pull) ? graph.getTransposedDeviceGraph() : graph.getDeviceGraph();

  const bool invert = (Direction == advance_direction::pull); // In pull mode, we consider inactive vertices as active.
  input_frontier.computeActiveFrontier(invert);

  // compute launch informations
  const size_t coarsening_factor = CU_SIZE / 32 /* Warp Size */;
  const size_t bitmap_range = in_dev_frontier.getBitmapRange();
  if (bitmap_range != 32) {
    throw std::runtime_error("Advance operator currently supports only frontiers with bitmap range equal to 32.");
  }
  const size_t active_size = input_frontier.getActiveFrontierSize();

  const size_t block_size = CU_SIZE;
  const size_t grid_size = ((active_size * bitmap_range) + block_size - 1) / block_size;
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, active_size, stealer);

  spdlog::debug("Advance Operator Launch - Direction: {}, Grid Size: {} (was {}), Block Size: {}, Cluster Size: {}", 
                (Direction == advance_direction::push) ? "Push" : "Pull",
                launch_config.grid_size,
                grid_size,
                launch_config.block_size,
                launch_config.cluster_size);
                
  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel", "core");

  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function = detail::advanceKernel<Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), decltype(out_dev_frontier), DerivedStealerT, LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function,
                                                 graph_dev,
                                                 in_dev_frontier, 
                                                 out_dev_frontier, 
                                                 coarsening_factor, 
                                                 stealer, 
                                                 std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function = detail::advanceKernel<Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), frontier::detail::NullFrontierDevice, DerivedStealerT, LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function, 
                                                 graph_dev, 
                                                 in_dev_frontier, 
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 coarsening_factor, 
                                                 stealer, 
                                                 std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}
} // namespace clutra::operators::advance::detail
