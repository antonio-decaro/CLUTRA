/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <graph/graph.cuh>
#include <graph/concept.hpp>
#include <frontier/frontier.cuh>
#include <operators/advance/options.hpp>
#include <operators/advance/advance_kernel.cuh>
#include <memory>
#include <utils/profile.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <stealer/stealer.cuh>

namespace clutra::operators::advance::detail {

/**
 * Launch kernel for advance operator with input frontier (used in both push and pull modes).
 */
template<advance_direction Direction, clutra::graph::detail::GraphConcept GraphT, typename InputFrontierT, typename DerivedStealerT, typename DeviceStealerT, typename LambdaT>
void launchKernel(const GraphT& graph,
                  InputFrontierT& input_frontier,
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
                                                  advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), frontier::detail::NullFrontierDevice, decltype(stealer.getDeviceStealer()), LambdaT>);
  // const size_t grid_size = 1024; //clutra::detail::device::getMaxNumBlocks(block_size, device_id);
  // const size_t grid_size = (active_size * bitmap_range + (block_size - 1)) / block_size;
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles, stealer);

  const size_t total_iters = (work_tiles + launch_config.grid_size - 1) / launch_config.grid_size;

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
    auto& kernel_launch_function = detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), decltype(out_dev_frontier), decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function,
                                                 graph_dev,
                                                 in_dev_frontier, 
                                                 out_dev_frontier, 
                                                 coarsening_factor,
                                                 total_iters,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function = detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier), frontier::detail::NullFrontierDevice, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function, 
                                                 graph_dev, 
                                                 in_dev_frontier, 
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 coarsening_factor,
                                                 total_iters,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}


/**
 * Launch advance kernel without input frontier (i.e., all vertices are active).
 */
template<advance_direction Direction, clutra::graph::detail::GraphConcept GraphT, typename DerivedStealerT, typename DeviceStealerT, typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  if constexpr (Direction == advance_direction::pull) {
    throw std::runtime_error("Advance operator in pull mode requires an input frontier.");
  }
  auto graph_dev = graph.getDeviceGraph();

  const size_t active_size = graph.getVertexCount();

  const size_t block_size = CU_SIZE;
  const size_t work_tiles = (active_size + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = clutra::detail::device::getMaxOccupancyGridSize(device_id,
                                                  block_size,
                                                  smem,
                                                  advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev), frontier::detail::NullFrontierDevice, frontier::detail::NullFrontierDevice, decltype(stealer.getDeviceStealer()), LambdaT>);
  const size_t cluster_size = stealer.getPreferredClusterSize();
  auto launch_config = clutra::detail::kernels::adjustLaunchConfig(grid_size, block_size, cluster_size, work_tiles, stealer);

  const size_t total_iters = (work_tiles + launch_config.grid_size - 1) / launch_config.grid_size;

  clutra::detail::log("Advance Operator Launch - Active Size: {}, Direction: Push, Grid Size: {} (was {}), Block Size: {}, Cluster Size: {}, SMEM: {}, Stealing Enabled: {}",
                   active_size,
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
    auto& kernel_launch_function = detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev), frontier::detail::NullFrontierDevice, decltype(out_dev_frontier), decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function,
                                                 graph_dev,
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 out_dev_frontier, 
                                                 1,
                                                 total_iters,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  } else {
    // Use a null frontier when the caller does not need to store output.
    auto& kernel_launch_function = detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev), frontier::detail::NullFrontierDevice, frontier::detail::NullFrontierDevice, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, 
                                                 kernel_launch_function, 
                                                 graph_dev, 
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 frontier::detail::NullFrontierDevice{}, 
                                                 1,
                                                 total_iters,
                                                 stealer_dev, 
                                                 std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}
} // namespace clutra::operators::advance::detail
