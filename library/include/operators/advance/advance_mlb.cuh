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
#include <utils/logging.cuh>

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

template<advance_direction Direction, size_t BlockSize, graph::detail::DeviceGraphConcept GraphDevT, typename InFrontierDevT, typename OutFrontierDevT, typename StealerDeviceT, typename LambdaT>
__global__ void advanceKernel(GraphDevT graph_dev,
                              InFrontierDevT in_dev_frontier,
                              OutFrontierDevT out_dev_frontier,
                              int coarsening_factor,
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

  const uint32_t active_size = in_dev_frontier.getOffsetsSize()[0];
  const uint32_t bitmap_range = in_dev_frontier.getBitmapRange();
  const size_t work_tiles = ((static_cast<size_t>(active_size) * bitmap_range) + block_dim - 1) / block_dim;
  const size_t total_iters = (work_tiles + gridDim.x - 1) / gridDim.x;

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

    const uint32_t assigned_vertex = getAssignedVertex(in_dev_frontier, coarsening_factor, tile_gid, tid);
    const bool vertex_active = assigned_vertex < graph_dev.getVertexCount() && in_dev_frontier.check(assigned_vertex);
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

template<advance_direction Direction, clutra::graph::detail::GraphConcept GraphT, typename DerivedStealerT, typename DeviceStealerT, typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
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
  const size_t work_tiles = ((active_size * bitmap_range) + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = clutra::detail::device::getMaxOccupancyGridSize(device_id,
                                                  block_size,
                                                  smem,
                                                  advanceKernel<Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), frontier::detail::NullFrontierDevice, decltype(stealer.getDeviceStealer()), LambdaT>);
  // const size_t grid_size = 1024; //clutra::detail::device::getMaxNumBlocks(block_size, device_id);
  // const size_t grid_size = (active_size * bitmap_range + (block_size - 1)) / block_size;
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles, stealer);

  clutra::detail::log("Advance Operator Launch - Active Size: {}, Direction: {}, Grid Size: {} (was {}), Block Size: {}, Cluster Size: {}, SMEM: {}, Stealing Enabled: {}",
                   active_size,
                   (Direction == advance_direction::push) ? "Push" : "Pull",
                   launch_config.grid_size,
                   grid_size,
                   launch_config.block_size,
                   launch_config.cluster_size,
                   smem,
                   stealer.isIntraClusterStealingEnabled() ? "Yes" : "No");
                
  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function = detail::advanceKernel<Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), decltype(out_dev_frontier), decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function,
                                                 graph_dev,
                                                 in_dev_frontier, 
                                                 out_dev_frontier, 
                                                 coarsening_factor,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function = detail::advanceKernel<Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), frontier::detail::NullFrontierDevice, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function, 
                                                 graph_dev, 
                                                 in_dev_frontier, 
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 coarsening_factor,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}
} // namespace clutra::operators::advance::detail
