/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string>
#include <cuda_runtime.h>

namespace clutra::detail::device {

// get the number of SMs available on the current device
__host__ inline int getNumSMs() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  return prop.multiProcessorCount;
};

__host__ inline std::string getDeviceName() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  return std::string(prop.name);
}

}