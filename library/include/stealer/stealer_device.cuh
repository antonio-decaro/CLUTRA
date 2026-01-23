/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <stealer/stealer_config.cuh>

namespace cg = cooperative_groups;

namespace clutra::stealer {

struct StealerDevice {
  StealerConfig config{};

  StealerDevice() = default;
  StealerDevice(const StealerConfig& cfg) : config(cfg) {}
  __device__ bool isStealingEnabled() const;
};

struct NullStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;
  
  __device__ void init() const;
  __device__ void steal() const;
};

struct BasicStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;
  __device__ void init() const;
  __device__ void steal() const;
};

} // namespace clutra::stealer
