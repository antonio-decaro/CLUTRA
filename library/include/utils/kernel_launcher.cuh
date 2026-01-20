/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>
#include "misc.cuh"

namespace clutra::detail::kernels {

// Generic CUDA kernel for single-threaded execution of any lambda

inline bool isClusterLaunchSupported(int device = 0) {
  int cluster_supported = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&cluster_supported,
                                    cudaDevAttrClusterLaunch,
                                    device));
  return cluster_supported != 0;
}

template<typename KernelT, typename... Args>
inline void launchClusterKernelImpl(const size_t& grid_size,
                                    const size_t& block_size,
                                    const size_t& cluster_size,
                                    size_t dynamic_smem_bytes,
                                    cudaStream_t stream,
                                    KernelT kernel,
                                    Args... args) {
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
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster_size;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  config.attrs = attr;
  config.numAttrs = 1;

  CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template<typename KernelT, typename... Args>
inline void launchClusterKernel(const size_t& grid_size,
                                const size_t& block_size,
                                const size_t& cluster_size,
                                size_t dynamic_smem_bytes,
                                cudaStream_t stream,
                                KernelT kernel,
                                Args... args) {
  launchClusterKernelImpl(grid_size,
                          block_size,
                          cluster_size,
                          dynamic_smem_bytes,
                          stream,
                          kernel,
                          args...);
}

template<typename KernelT, typename... Args>
inline void launchClusterKernel(const size_t& grid_size,
                                const size_t& block_size,
                                const size_t& cluster_size,
                                KernelT kernel,
                                Args... args) {
  launchClusterKernelImpl(grid_size,
                          block_size,
                          cluster_size,
                          0,
                          0,
                          kernel,
                          args...);
}

template<typename KernelT, typename... Args>
inline void launchClusterKernel(const size_t& grid_size,
                                const size_t& block_size,
                                const size_t& cluster_size,
                                const size_t& dynamic_smem_bytes,
                                KernelT kernel,
                                Args... args) {
  launchClusterKernelImpl(grid_size,
                          block_size,
                          cluster_size,
                          dynamic_smem_bytes,
                          0,
                          kernel,
                          args...);
}

}
