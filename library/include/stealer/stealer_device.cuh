/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <stealer/stealer_config.cuh>

namespace clutra::stealer {

struct NullStealerDevice {
  __device__ void init() const;
  __device__ void steal() const;
};

struct BasicStealerDevice {
  StealerConfig config{};

  __device__ void init() const;
  __device__ void steal() const;
};

} // namespace clutra::stealer
