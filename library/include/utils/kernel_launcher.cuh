/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "misc.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <stealer/stealer.cuh>
#include <utils/logging.cuh>

namespace clutra::detail::kernels {

struct LaunchConfig {
  size_t grid_size;
  size_t block_size;
  size_t cluster_size;
};

// Generic CUDA kernel for single-threaded execution of any lambda

inline bool isClusterLaunchSupported(int device = 0) {
  int cluster_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, device));
  return cluster_supported != 0;
}

inline bool isClusterLaunchControlSupported(int device = 0) {
  int cc_major = 0;
  int cluster_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch, device));
  return (cc_major >= 10) && (cluster_supported != 0);
}

template <typename KernelT, typename... Args>
inline void launchClusterKernelImpl(size_t grid_size,
                                    size_t block_size,
                                    size_t cluster_size,
                                    size_t dynamic_smem_bytes,
                                    cudaStream_t stream,
                                    KernelT kernel,
                                    Args... args) {
  if (grid_size == 0) {
    throw std::runtime_error("Kernel launch requested with grid_size == 0.");
  }
  if (block_size == 0) {
    throw std::runtime_error("Kernel launch requested with block_size == 0.");
  }
  if (cluster_size == 0) {
    cluster_size = 1;
  }

  if (cluster_size > 1) {
    const size_t remainder = grid_size % cluster_size;
    if (remainder != 0) {
      grid_size += (cluster_size - remainder);
    }
  }

  cudaLaunchConfig_t config = {};
  config.gridDim.x = grid_size;
  config.gridDim.y = 1;
  config.gridDim.z = 1;
  config.blockDim.x = block_size;
  config.blockDim.y = 1;
  config.blockDim.z = 1;
  config.dynamicSmemBytes = dynamic_smem_bytes;
  config.stream = stream;
  cudaLaunchAttribute attr[1];

  // Cluster launch attributes are optional: avoid requesting them when
  // cluster_size==1 to keep execution compatible with non-cluster devices.
  if (cluster_size > 1) {
    int device_id = 0;
    CUDA_CHECK(cudaGetDevice(&device_id));
    if (!isClusterLaunchSupported(device_id)) {
      throw std::runtime_error("Cluster launch requested but not supported by the current device.");
    }

    int max_cluster_size = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialClusterSize(&max_cluster_size, kernel, &config));
    if (max_cluster_size <= 0) {
      cluster_size = 1;
    } else if (cluster_size > static_cast<size_t>(max_cluster_size)) {
      cluster_size = static_cast<size_t>(max_cluster_size);
    }

    if (cluster_size > 1) {
      attr[0].id = cudaLaunchAttributeClusterDimension;
      attr[0].val.clusterDim.x = cluster_size;
      attr[0].val.clusterDim.y = 1;
      attr[0].val.clusterDim.z = 1;
      config.attrs = attr;
      config.numAttrs = 1;
    }
  } else {
    config.attrs = nullptr;
    config.numAttrs = 0;
  }

  clutra::detail::log("Launching kernel with grid_size={}, block_size={}, cluster_size={}, dynamic_smem_bytes={}",
                      config.gridDim.x, config.blockDim.x, cluster_size, config.dynamicSmemBytes);

  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <typename KernelT, typename... Args>
inline void launchClusterKernel(const LaunchConfig& config,
                                size_t dynamic_smem_bytes,
                                cudaStream_t stream,
                                KernelT kernel,
                                Args... args) {
  launchClusterKernelImpl(config.grid_size, config.block_size, config.cluster_size, dynamic_smem_bytes, stream, kernel,
                          args...);
}

template <typename KernelT, typename... Args>
inline void launchClusterKernel(const LaunchConfig& config, KernelT kernel, Args... args) {
  launchClusterKernelImpl(config.grid_size, config.block_size, config.cluster_size, 0, 0, kernel, args...);
}

template <typename KernelT, typename... Args>
inline void
launchClusterKernel(const LaunchConfig& config, const size_t& dynamic_smem_bytes, KernelT kernel, Args... args) {
  launchClusterKernelImpl(config.grid_size, config.block_size, config.cluster_size, dynamic_smem_bytes, 0, kernel,
                          args...);
}

/**
 * Fetch the cluster size for intra-cluster work stealing.
 * If intra-cluster work stealing is disabled, return 1.
 * Adjust the grid size to be a multiple of the cluster size.
 * @param preferred_grid_size Preferred grid size (will be modified if needed).
 * @param preferred_block_size Preferred block size.
 * @param preferred_cluster_size Preferred cluster size (will be modified if needed).
 * @param workload_size Total workload size.
 * @return LaunchConfig: Adjusted LaunchConfig with grid size, block size, and cluster size.
 */
inline LaunchConfig fetchLaunchConfig(const size_t& preferred_grid_size,
                                      const size_t& preferred_block_size,
                                      const size_t& preferred_cluster_size,
                                      const size_t& workload_size) {
  size_t grid_size = preferred_grid_size;
  size_t block_size = preferred_block_size;
  size_t cluster_size = preferred_cluster_size;

  if (block_size == 0) {
    throw std::runtime_error("Block size must be greater than zero.");
  }

  // Treat a zero preferred cluster size as "no clustering" instead of
  // allowing undefined modulo-by-zero behavior.
  if (cluster_size == 0) {
    cluster_size = 1;
  }

  if (preferred_grid_size == 0 && preferred_block_size > 0 && workload_size > 0) {
    grid_size = (workload_size + preferred_block_size - 1) / preferred_block_size;
  }

  if (grid_size < cluster_size) {
    cluster_size = 1;
  }

  // Round up the grid size to a multiple of the cluster size when needed.
  const size_t remainder = grid_size % cluster_size;
  if (remainder != 0) {
    grid_size += (cluster_size - remainder);
  }

  return {.grid_size = grid_size, .block_size = block_size, .cluster_size = cluster_size};
}

}  // namespace clutra::detail::kernels
