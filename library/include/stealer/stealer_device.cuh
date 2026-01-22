/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <stealer/stealer_config.cuh>

namespace clutra::stealer {

struct NullStealerDevice {
  __forceinline__ __device__ void init() const {}
  __forceinline__ __device__ void steal() const {} 
};

struct BasicStealerDevice {
  StealerConfig config{};

  __forceinline__ __device__ void init() const {
#if __CUDA_ARCH__ >= 900
    if (config.intra_cluster_stealing_enabled) {
      auto cluster = cooperative_groups::this_cluster();
      cluster.sync();
    }
#endif
  }

  __forceinline__ __device__ void steal() const {
    // Device-side stealing logic can be implemented here.
  }
};

} // namespace clutra::stealer
