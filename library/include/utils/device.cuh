/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <cuda_runtime.h>
#include <algorithm>
#include <string>

namespace clutra::detail::device {

// get the number of SMs available on the current device
__host__ inline int getNumSMs(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  return prop.multiProcessorCount;
};

__host__ inline std::string getDeviceName(int device_id) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);
  return std::string(prop.name);
}

__host__ inline int getNumBlocks(size_t num_threads, size_t block_size, int device_id) {
  int num_sms = getNumSMs(device_id);
  int max_blocks_per_sm;
  cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device_id);
  int max_blocks = max_blocks_per_sm * num_sms;
  int req_blocks = (num_threads + block_size - 1) / block_size;
  return std::min(max_blocks, req_blocks);
}

__host__ inline size_t getMaxNumBlocks(size_t block_size, int device_id) {
  int num_sms = getNumSMs(device_id);
  int max_blocks_per_sm;
  cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device_id);
  return max_blocks_per_sm * num_sms;
}

template <typename T>
__host__ inline size_t getMaxOccupancyGridSize(int device_id, size_t block_size, size_t smem_bytes, T&& kernel) {
  int num_sms = getNumSMs(device_id);
  int maxBlocksPerSM;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocksPerSM,
      kernel,
      block_size,
      smem_bytes);

  int grid = num_sms * maxBlocksPerSM; // full occupancy persistent grid
  return grid;
}

} // namespace clutra::detail::device
