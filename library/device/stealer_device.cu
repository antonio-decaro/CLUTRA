/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cooperative_groups.h>
#include <stealer/stealer_device.cuh>

namespace cg = cooperative_groups;

namespace clutra::stealer {

__device__ void BasicStealerDevice::init() const {
#if __CUDA_ARCH__ >= 900
  if (config.intra_cluster_stealing_enabled) {
    auto cluster = cg::this_cluster();
    cluster.sync();
  }
#endif
}

__device__ void BasicStealerDevice::steal() const {
  // Device-side stealing logic can be implemented here.
}

} // namespace clutra::stealer
