/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/concept.hpp>
#include <graph/graph.cuh>
#include <operators/advance/kernel_block_mapped.cuh>
#include <operators/advance/kernel_bucketing.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/profile.cuh>

namespace clutra::operators::advance::detail {

template <typename StealerT>
inline void validateStealingSupport(const StealerT& stealer, int device_id) {
  if (stealer.isIntraClusterStealingEnabled() && !clutra::detail::kernels::isClusterLaunchSupported(device_id)) {
    throw std::runtime_error("Local stealing requires cluster launch support on the current device.");
  }

  // if (stealer.isInterClusterStealingEnabled()) {
  //   const bool supported =
  //       hasPtxClusterLaunchControlApi() && clutra::detail::kernels::isClusterLaunchControlSupported(device_id);
  //   if (!supported) {
  //     throw std::runtime_error("Global stealing requires SM100+ cluster launch control support and a CUDA toolkit with "
  //                              "cluster-launch-control PTX APIs.");
  //   }
  // }
}

template <typename StealerT>
inline size_t resolveClusterSize(const StealerT& stealer, int device_id, size_t preferred_cluster_size) {
  if (!stealer.isIntraClusterStealingEnabled() && !stealer.isInterClusterStealingEnabled()) {
    return 1;
  }
  if (!clutra::detail::kernels::isClusterLaunchSupported(device_id)) {
    return 1;
  }
  return preferred_cluster_size;
}

/**
 * Launch bucketing kernel for advance operator with input frontier
 * (used in both push and pull modes).
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename InputFrontierT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernelBucketing(const GraphT& graph,
                           InputFrontierT& input_frontier,
                           clutra::frontier::FrontierMLB<>* output_frontier,
                           const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                           LambdaT&& functor) {
  constexpr size_t CU_SIZE = 512;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto graph_dev = (Direction == advance_direction::pull) ? graph.getTransposedDeviceGraph() : graph.getDeviceGraph();

  const bool invert = (Direction == advance_direction::pull);
  input_frontier.computeActiveFrontier(invert);

  const size_t coarsening_factor = CU_SIZE / 32;
  const size_t bitmap_range = in_dev_frontier.getBitmapRange();
  if (bitmap_range != 32) {
    throw std::runtime_error("Advance operator currently supports only frontiers with bitmap range equal to 32.");
  }
  const size_t active_size = input_frontier.getActiveFrontierSize();

  const size_t block_size = CU_SIZE;
  const size_t work_tiles = ((active_size * bitmap_range) + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  validateStealingSupport(stealer, device_id);

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = work_tiles;
  const size_t cluster_size = resolveClusterSize(stealer, device_id, stealer.getPreferredClusterSize());
  auto launch_config = clutra::detail::kernels::fetchLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  clutra::detail::log(
      "Advance Operator Launch - LB: bucketing, Active Size: {}, Direction: {}, Grid Size: {} (was {}), "
      "Block Size: {}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, Global Mode: {}, "
      "Local Chunk: {}, Global Chunk: {}",
      active_size, (Direction == advance_direction::push) ? "Push" : "Pull", launch_config.grid_size, grid_size,
      launch_config.block_size, launch_config.cluster_size, smem,
      stealer.isIntraClusterStealingEnabled() ? "Yes" : "No", stealer.isInterClusterStealingEnabled() ? "Yes" : "No",
      stealer.isInterClusterStealingEnabled() ? "SM100 cancel" : "disabled", stealer.getLocalStealingChunkSize(),
      stealer.getGlobalStealingChunkSize());

  clutra::profile::KernelProfiler profiler("advanceKernelBucketing", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function =
        detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier),
                              decltype(out_dev_frontier), decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev, in_dev_frontier,
                                                 out_dev_frontier, coarsening_factor, work_tiles, stealer_dev,
                                                 std::forward<LambdaT>(functor));
  } else {
    auto& kernel_launch_function =
        detail::advanceKernel<view::frontier, Direction, CU_SIZE, decltype(graph_dev), decltype(in_dev_frontier),
                              frontier::detail::NullFrontierDevice, decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev, in_dev_frontier,
                                                 frontier::detail::NullFrontierDevice{}, coarsening_factor, work_tiles,
                                                 stealer_dev, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

/**
 * Launch block-mapped kernel for advance operator with input frontier.
 * Currently supported for push direction only.
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename InputFrontierT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernelBlockMapped(const GraphT& graph,
                             InputFrontierT& input_frontier,
                             clutra::frontier::FrontierMLB<>* output_frontier,
                             const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                             LambdaT&& functor) {
  static_assert(Direction == advance_direction::push, "Block-mapped advance currently supports push direction only.");

  constexpr size_t CU_SIZE = 512;
  auto in_dev_frontier = input_frontier.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  input_frontier.computeActiveFrontier(false);

  const size_t coarsening_factor = CU_SIZE / 32;
  const size_t bitmap_range = in_dev_frontier.getBitmapRange();
  if (bitmap_range != 32) {
    throw std::runtime_error("Advance operator currently supports only frontiers with bitmap range equal to 32.");
  }
  const size_t active_size = input_frontier.getActiveFrontierSize();

  const size_t block_size = CU_SIZE;
  const size_t work_tiles = ((active_size * bitmap_range) + block_size - 1) / block_size;

  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  validateStealingSupport(stealer, device_id);

  constexpr size_t smem = 0;
  const size_t grid_size = work_tiles;
  const size_t cluster_size = resolveClusterSize(stealer, device_id, stealer.getPreferredClusterSize());
  auto launch_config = clutra::detail::kernels::fetchLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  clutra::detail::log("Advance Operator Launch - LB: block_mapped, Active Size: {}, Direction: Push, Grid Size: {} "
                      "(was {}), Block Size: {}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, "
                      "Global Mode: {}, Local Chunk: {}, Global Chunk: {}",
                      active_size, launch_config.grid_size, grid_size, launch_config.block_size,
                      launch_config.cluster_size, smem, stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "SM100 cancel" : "disabled",
                      stealer.getLocalStealingChunkSize(), stealer.getGlobalStealingChunkSize());

  clutra::profile::KernelProfiler profiler("advanceKernelBlockMapped", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function =
        detail::advanceKernelBlockMapped<view::frontier, Direction, CU_SIZE, decltype(graph_dev),
                                         decltype(in_dev_frontier), decltype(out_dev_frontier), decltype(stealer_dev),
                                         LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev, in_dev_frontier,
                                                 out_dev_frontier, coarsening_factor, work_tiles, stealer_dev,
                                                 std::forward<LambdaT>(functor));
  } else {
    auto& kernel_launch_function =
        detail::advanceKernelBlockMapped<view::frontier, Direction, CU_SIZE, decltype(graph_dev),
                                         decltype(in_dev_frontier), frontier::detail::NullFrontierDevice,
                                         decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev, in_dev_frontier,
                                                 frontier::detail::NullFrontierDevice{}, coarsening_factor, work_tiles,
                                                 stealer_dev, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

/**
 * Runtime dispatch for frontier-based advance with explicit load-balancing
 * policy.
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename InputFrontierT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,
                  InputFrontierT& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  load_balance load_balance,
                  LambdaT&& functor) {
  switch (load_balance) {
    case load_balance::bucketing:
      launchKernelBucketing<Direction>(graph, input_frontier, output_frontier, stealer, std::forward<LambdaT>(functor));
      break;
    case load_balance::block_mapped:
      if constexpr (Direction == advance_direction::push) {
        launchKernelBlockMapped<Direction>(graph, input_frontier, output_frontier, stealer,
                                           std::forward<LambdaT>(functor));
      } else {
        throw std::runtime_error("Block-mapped advance currently supports push direction only.");
      }
      break;
    default:
      throw std::runtime_error("Unsupported advance load-balancing strategy.");
  }
}

/**
 * Default frontier-based advance launcher (backward compatible):
 * defaults to bucketing.
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename InputFrontierT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,
                  InputFrontierT& input_frontier,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  launchKernel<Direction>(graph, input_frontier, output_frontier, stealer, load_balance::bucketing,
                          std::forward<LambdaT>(functor));
}

/**
 * Launch advance kernel without input frontier (i.e., all vertices are active).
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernelGraphBucketing(const GraphT& graph,
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
  validateStealingSupport(stealer, device_id);

  const size_t smem = getAdvanceSharedMemorySize<CU_SIZE>(stealer.template getSharedStateSizeInBytes<CU_SIZE>());
  const size_t grid_size = work_tiles;
  const size_t cluster_size = resolveClusterSize(stealer, device_id, stealer.getPreferredClusterSize());
  auto launch_config = clutra::detail::kernels::fetchLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  clutra::detail::log("Advance Operator Launch - LB: bucketing, Active Size: {}, Direction: Push, Grid Size: {} (was "
                      "{}), Block Size: {}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, "
                      "Global Mode: {}, Local Chunk: {}, Global Chunk: {}",
                      active_size, launch_config.grid_size, grid_size, launch_config.block_size,
                      launch_config.cluster_size, smem, stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "SM100 cancel" : "disabled",
                      stealer.getLocalStealingChunkSize(), stealer.getGlobalStealingChunkSize());

  clutra::profile::KernelProfiler profiler("advanceKernelBucketing", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function = detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                                                         frontier::detail::NullFrontierDevice,
                                                         decltype(out_dev_frontier), decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev,
                                                 frontier::detail::NullFrontierDevice{}, out_dev_frontier, 1,
                                                 work_tiles, stealer_dev, std::forward<LambdaT>(functor));
  } else {
    auto& kernel_launch_function =
        detail::advanceKernel<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                              frontier::detail::NullFrontierDevice, frontier::detail::NullFrontierDevice,
                              decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, frontier::detail::NullFrontierDevice{},
        frontier::detail::NullFrontierDevice{}, 1, work_tiles, stealer_dev, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

/**
 * Launch block-mapped kernel without input frontier (i.e., all vertices are
 * active). This maps contiguous vertex ranges to blocks.
 */
template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernelGraphBlockMapped(const GraphT& graph,
                                  clutra::frontier::FrontierMLB<>* output_frontier,
                                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                                  LambdaT&& functor) {
  if constexpr (Direction == advance_direction::pull) {
    throw std::runtime_error("Advance operator in pull mode requires an input frontier.");
  }

  constexpr size_t CU_SIZE = 512;
  auto graph_dev = graph.getDeviceGraph();

  const size_t active_size = graph.getVertexCount();
  const size_t block_size = CU_SIZE;
  const size_t coarsening_factor = 1;
  const size_t work_tiles = (active_size + block_size - 1) / block_size;
  int device_id = 0;
  CUDA_CHECK(cudaGetDevice(&device_id));
  validateStealingSupport(stealer, device_id);

  constexpr size_t smem = 0;
  const size_t grid_size = work_tiles;
  const size_t cluster_size = resolveClusterSize(stealer, device_id, stealer.getPreferredClusterSize());
  auto launch_config = clutra::detail::kernels::fetchLaunchConfig(grid_size, block_size, cluster_size, work_tiles);

  clutra::detail::log("Advance Operator Launch - LB: block_mapped, Active Size: {}, Direction: Push, Grid Size: {} "
                      "(was {}), Block Size: {}, Cluster Size: {}, SMEM: {}, Local Stealing: {}, Global Stealing: {}, "
                      "Local Chunk: {}, Global Chunk: {}",
                      active_size, launch_config.grid_size, grid_size, launch_config.block_size,
                      launch_config.cluster_size, smem, stealer.isIntraClusterStealingEnabled() ? "Yes" : "No",
                      stealer.isInterClusterStealingEnabled() ? "Yes" : "No", stealer.getLocalStealingChunkSize(),
                      stealer.getGlobalStealingChunkSize());

  clutra::profile::KernelProfiler profiler("advanceKernelBlockMapped", "core");

  auto stealer_dev = stealer.getDeviceStealer();
  if (output_frontier != nullptr) {
    auto out_dev_frontier = output_frontier->getDeviceFrontier();
    auto& kernel_launch_function =
        detail::advanceKernelBlockMapped<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                                         frontier::detail::NullFrontierDevice, decltype(out_dev_frontier),
                                         decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(
        launch_config, kernel_launch_function, graph_dev, frontier::detail::NullFrontierDevice{}, out_dev_frontier,
        coarsening_factor, work_tiles, stealer_dev, std::forward<LambdaT>(functor));
  } else {
    auto& kernel_launch_function =
        detail::advanceKernelBlockMapped<view::graph, Direction, CU_SIZE, decltype(graph_dev),
                                         frontier::detail::NullFrontierDevice, frontier::detail::NullFrontierDevice,
                                         decltype(stealer_dev), LambdaT>;
    clutra::detail::kernels::launchClusterKernel(launch_config, kernel_launch_function, graph_dev,
                                                 frontier::detail::NullFrontierDevice{},
                                                 frontier::detail::NullFrontierDevice{}, coarsening_factor, work_tiles,
                                                 stealer_dev, std::forward<LambdaT>(functor));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  load_balance load_balance,
                  LambdaT&& functor) {
  switch (load_balance) {
    case load_balance::bucketing:
      launchKernelGraphBucketing<Direction>(graph, output_frontier, stealer, std::forward<LambdaT>(functor));
      break;
    case load_balance::block_mapped:
      launchKernelGraphBlockMapped<Direction>(graph, output_frontier, stealer, std::forward<LambdaT>(functor));
      break;
    default:
      throw std::runtime_error("Unsupported advance load-balancing strategy.");
  }
}

template <advance_direction Direction,
          clutra::graph::detail::GraphConcept GraphT,
          typename DerivedStealerT,
          typename DeviceStealerT,
          typename LambdaT>
void launchKernel(const GraphT& graph,
                  clutra::frontier::FrontierMLB<>* output_frontier,
                  const clutra::stealer::StealerBase<DerivedStealerT, DeviceStealerT>& stealer,
                  LambdaT&& functor) {
  launchKernel<Direction>(graph, output_frontier, stealer, load_balance::bucketing, std::forward<LambdaT>(functor));
}

}  // namespace clutra::operators::advance::detail
