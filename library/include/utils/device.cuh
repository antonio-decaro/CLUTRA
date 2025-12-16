/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <cuda_runtime.h>

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

}