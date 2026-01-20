/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>
#include <type_traits>

namespace clutra::stealer {

/**
 * @brief Configuration for the Stealer component.
 * @details This structure holds configuration options for enabling or disabling
 * various stealing strategies such as thread stealing, block stealing, and grid stealing.
 * @note This object should be device-compatible.
 */
class StealerConfig final {
public:
  bool thread_stealing_enabled = false;
  bool block_stealing_enabled = false;
  bool grid_stealing_enabled = false;
};

static_assert(std::is_trivially_copyable_v<StealerConfig>,
              "StealerConfig must be trivially copyable for device use");
static_assert(std::is_standard_layout_v<StealerConfig>,
              "StealerConfig should be standard layout");

/**
 * @brief Stealer component for managing work stealing strategies.
 * @details This class implements various work stealing strategies based on the provided configuration.
 * @note This object should be device-compatible.
 */
class Stealer {
public:
  __host__ Stealer(const StealerConfig& config) : _config(config) {}
  
  __forceinline__ __host__ __device__ bool isThreadStealingEnabled() const { return _config.thread_stealing_enabled; }
  __forceinline__ __host__ __device__ bool isBlockStealingEnabled() const { return _config.block_stealing_enabled; }
  __forceinline__ __host__ __device__ bool isGridStealingEnabled() const { return _config.grid_stealing_enabled; }
  
  __device__ void steal();

private:
  StealerConfig _config;
};

static_assert(std::is_trivially_copyable_v<Stealer>,
              "Stealer must be trivially copyable for device use");
static_assert(std::is_standard_layout_v<Stealer>,
              "Stealer should be standard layout");
              
} // namespace clutra::stealer
