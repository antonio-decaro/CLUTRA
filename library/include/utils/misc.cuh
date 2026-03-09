/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <string>

#define CUDA_CHECK(call)                                                                                               \
  do {                                                                                                                 \
    cudaError_t err = (call);                                                                                          \
    if (err != cudaSuccess) {                                                                                          \
      std::cerr << "CUDA error in file '" << __FILE__ << "' in line " << __LINE__ << ": " << cudaGetErrorString(err)   \
                << "." << std::endl;                                                                                   \
      std::exit(EXIT_FAILURE);                                                                                         \
    }                                                                                                                  \
  } while (0)

namespace clutra::detail::kernels {

// Generic CUDA kernel for single-threaded execution of any lambda
template <typename Func>
__global__ void executeKernel(Func func) {
  func();
}
}  // namespace clutra::detail::kernels
