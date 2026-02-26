/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "utils/atomic.cuh"
#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/concept.hpp>
#include <graph/graph.cuh>
#include <memory>
#include <operators/advance/advance_kernel.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/profile.cuh>
#include <utils/queue.cuh>

namespace clutra::operators::advance::detail {

/**
 * Launch kernel for advance operator with input frontier (used in both push and pull modes).
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename InputFrontierT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,  // TODO fix according to the ClusterQueue
                  InputFrontierT& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto graph_dev = (Direction == advance_direction::pull) ? graph.getTransposedDeviceGraph() : graph.getDeviceGraph();
  using LockType = clutra::detail::atomic::SpinLock;

  const bool invert = (Direction == advance_direction::pull);  // In pull mode, we consider inactive vertices as active.
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
  const size_t grid_size = clutra::detail::device::getMaxOccupancyGridSize(
      device_id, block_size, smem,
      advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier),
                    frontier::detail::NullFrontierDevice, LockType, decltype(stealer.getDeviceStealer()), LambdaT>);
  // const size_t grid_size = 1024; //clutra::detail::device::getMaxNumBlocks(block_size, device_id);
  // const size_t grid_size = (active_size * bitmap_range + (block_size - 1)) / block_size;
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  const uint32_t num_clusters = static_cast<uint32_t>(launch_config.grid_size / launch_config.cluster_size);
  const uint32_t tiles_per_cluster =
      static_cast<uint32_t>(work_tiles == 0 ? 1 : (work_tiles + num_clusters - 1) / num_clusters);
  clutra::detail::utils::ClusterWorkQueues<uint32_t> work_queues(num_clusters, tiles_per_cluster);

  clutra::detail::log("Advance Operator Launch - Active Size: {}, Direction: {}, Grid Size: {} (was {}), Block Size: "
                      "{}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, Local Chunk: {}, "
                      "Global Chunk: {}",
                      active_size, (Direction == advance_direction::push) ? "Push" : "Pull", launch_config.grid_size,
                      grid_size, launch_config.block_size, launch_config.cluster_size, smem,
                      stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No", stealer.getLocalStealingChunkSize(),
                      stealer.getGlobalStealingChunkSize());

  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function =
        detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier),
                              decltype(out_dev_frontier), LockType, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, in_dev_frontier, out_dev_frontier, coarsening_factor,
        work_tiles, work_queues.deviceViews(), stealer_dev, std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function =
        detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier),
                              frontier::detail::NullFrontierDevice, LockType, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, in_dev_frontier, frontier::detail::NullFrontierDevice{},
        coarsening_factor, work_tiles, work_queues.deviceViews(), stealer_dev, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

/**
 * Launch advance kernel without input frontier (i.e., all vertices are active).
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  if constexpr (Direction == advance_direction::pull) {
    throw std::runtime_error("Advance operator in pull mode requires an input frontier.");
  }
  using LockType = clutra::detail::atomic::TicketLock;
  auto graph_dev = graph.getDeviceGraph();

  const size_t active_size = graph.getVertexCount();

  const size_t block_size = CU_SIZE;
  const size_t work_tiles = (active_size + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = clutra::detail::device::getMaxOccupancyGridSize(
      device_id, block_size, smem,
      advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev), frontier::detail::NullFrontierDevice,
                    frontier::detail::NullFrontierDevice, LockType, decltype(stealer.getDeviceStealer()), LambdaT>);
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  const size_t num_clusters = launch_config.grid_size / launch_config.cluster_size;
  const size_t tiles_per_cluster = work_tiles == 0 ? 1 : (work_tiles + num_clusters - 1) / num_clusters;
  clutra::detail::utils::ClusterWorkQueues<uint32_t, LockType> work_queues(num_clusters, tiles_per_cluster);

  clutra::detail::log("Advance Operator Launch - Active Size: {}, Direction: Push, Grid Size: {} (was {}), Block Size: "
                      "{}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, Local Chunk: {}, "
                      "Global Chunk: {}",
                      active_size, launch_config.grid_size, grid_size, launch_config.block_size,
                      launch_config.cluster_size, smem, stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No", stealer.getLocalStealingChunkSize(),
                      stealer.getGlobalStealingChunkSize());

  // launch advance kernel
  clutra::profile::KernelProfiler profiler("advanceKernel", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function =
        detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                              frontier::detail::NullFrontierDevice, decltype(out_dev_frontier), LockType,
                              decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, frontier::detail::NullFrontierDevice{}, out_dev_frontier, 1,
        work_tiles, work_queues.deviceViews(), stealer_dev, std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function =
        detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                              frontier::detail::NullFrontierDevice, frontier::detail::NullFrontierDevice, LockType,
                              decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, frontier::detail::NullFrontierDevice{},
        frontier::detail::NullFrontierDevice{}, 1, work_tiles, work_queues.deviceViews(), stealer_dev,
        std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

/**
 * Launch edge-based advance kernel (i.e., all edges are processed).
 */
template <clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchEdgeKernel(const GraphT& graph,
                      const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                      LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  auto graph_dev = graph.getDeviceGraph();

  const size_t edge_count = graph.getEdgeCount();

  const size_t block_size = CU_SIZE;
  const size_t work_tiles = (edge_count + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = clutra::detail::device::getMaxOccupancyGridSize(
      device_id, block_size, smem,
      advanceKernel<CU_SIZE, decltype(graph_dev), decltype(stealer.getDeviceStealer()), LambdaT>);
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  const size_t total_iters = (work_tiles + launch_config.grid_size - 1) / launch_config.grid_size;

  clutra::detail::log("Advance Edge Operator Launch - Edge Count: {}, Grid Size: {} (was {}), Block Size: {}, Cluster "
                      "Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, Local Chunk: {}, Global Chunk: {}",
                      edge_count, launch_config.grid_size, grid_size, launch_config.block_size,
                      launch_config.cluster_size, smem, stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No", stealer.getLocalStealingChunkSize(),
                      stealer.getGlobalStealingChunkSize());

  clutra::profile::KernelProfiler profiler("advanceKernelEdge", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  auto& kernel_launch_function = detail::advanceKernel<CU_SIZE, decltype(graph_dev), decltype(stealer_dev), LambdaT>;
  clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev, total_iters,
                                               stealer_dev, std::forward<LambdaT>(functor));

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}
}  // namespace clutra::operators::advance::detail
