/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cuda_runtime.h>
#include <stealer/stealer_config.cuh>
#include <stealer/stealer_device.cuh>

namespace clutra::stealer {

/**
 * @brief Host-side stealer configuration and state.
 * @details Owns host configuration and produces a device-side view for kernels.
 */
template <typename DerivedT, typename DeviceStealerT>
class StealerBase {
public:
  __host__ StealerBase() : _config() {}
  __host__ StealerBase(const StealerConfig& config) : _config(config) {}

  __host__ void enableIntraClusterStealing() { _config.intra_cluster_stealing_enabled = true; }
  __host__ void disableIntraClusterStealing() { _config.intra_cluster_stealing_enabled = false; }
  
  __host__ bool isIntraClusterStealingEnabled() const { return _config.intra_cluster_stealing_enabled; }
  __host__ int getPreferredClusterSize() const { return _config.preferred_cluster_size; }

  __host__ DeviceStealerT device_view() const { return DeviceStealerT{_config}; }

protected:
  StealerConfig _config;
};

/**
 * @brief A no-operation stealer implementation.
 * @details This stealer does not perform any stealing operation.
 */
class NullStealer : public StealerBase<NullStealer, NullStealerDevice> {
public:
  using device_type = NullStealerDevice;
  __host__ NullStealer() : StealerBase<NullStealer, NullStealerDevice>() {}
};

class BasicStealer : public StealerBase<BasicStealer, BasicStealerDevice> {
public:
  using device_type = BasicStealerDevice;
  __host__ BasicStealer(const StealerConfig& config = {}) : StealerBase<BasicStealer, BasicStealerDevice>(config) {}
};

} // namespace clutra::stealer
