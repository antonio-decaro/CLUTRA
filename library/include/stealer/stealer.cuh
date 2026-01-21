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
struct StealerConfig final {
  bool intra_cluster_stealing_enabled = false;
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
template <typename DerivedT>
class Stealer {
public:
  __host__ Stealer() : _config() {}
  __host__ Stealer(const StealerConfig& config) : _config(config) {}
  
  __forceinline__ __host__ __device__ bool isIntraClusterStealingEnabled() const { return _config.intra_cluster_stealing_enabled; }
      
  __forceinline__ __device__ void steal() { static_cast<DerivedT*>(this)->steal_impl(); }

protected:
  StealerConfig _config;
};

/**
 * @brief A no-operation stealer implementation.
 * @details This stealer does not perform any stealing operation.
 * @note This object should be device-compatible.
 */
class NullStealer : public Stealer<NullStealer> {
public:
  __forceinline__ __device__ void steal_impl() {}
};
              
} // namespace clutra::stealer
