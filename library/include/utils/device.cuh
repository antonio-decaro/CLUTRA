/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "misc.cuh"
#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace clutra::detail::device {

// get the number of SMs available on the current device
__host__ inline int getNumSMs(int device_id) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return prop.multiProcessorCount;
};

__host__ inline std::string getDeviceName(int device_id) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
  return std::string(prop.name);
}

__host__ inline int getNumBlocks(size_t num_threads, size_t block_size, int device_id) {
  int num_sms = getNumSMs(device_id);
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device_id));
  int max_blocks = max_blocks_per_sm * num_sms;
  int req_blocks = (num_threads + block_size - 1) / block_size;
  return std::min(max_blocks, req_blocks);
}

__host__ inline size_t getMaxNumBlocks(size_t block_size, int device_id) {
  (void)block_size;
  int num_sms = getNumSMs(device_id);
  int max_blocks_per_sm = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device_id));
  return max_blocks_per_sm * num_sms;
}

template <typename T>
__host__ inline size_t getMaxOccupancyGridSize(int device_id, size_t block_size, size_t smem_bytes, T&& kernel) {
  if (block_size == 0) {
    throw std::runtime_error("getMaxOccupancyGridSize: block_size must be greater than zero.");
  }
  int num_sms = getNumSMs(device_id);
  int maxBlocksPerSM = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, kernel, block_size, smem_bytes));

  int grid = num_sms * maxBlocksPerSM;  // full occupancy persistent grid
  if (grid <= 0) {
    throw std::runtime_error("getMaxOccupancyGridSize: computed zero/negative grid size.");
  }
  return grid;
}

} // namespace clutra::detail::device
