/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <frontier.cuh>
#include <utils.cuh>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>

namespace clutra::frontier {

namespace detail {
__global__ void mergeLevelKernel(clutra::detail::types::bitmap_type_t* dst,
                                   const clutra::detail::types::bitmap_type_t* src,
                                   size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (; idx < n; idx += stride) {
    dst[idx] |= src[idx];
  }
}

__global__ void intersectMLBFrontierKernel(clutra::detail::types::bitmap_type_t* dst,
                                              const clutra::detail::types::bitmap_type_t* src,
                                              size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  for (; idx < n; idx += stride) {
    dst[idx] &= src[idx];
  }
}

template<typename DeviceFrontier>
__global__ void computeActiveFrontierKernel(DeviceFrontier bitmap,
                                            size_t level_size,
                                            uint32_t range,
                                            bool invert) {
  using bitmap_type = typename DeviceFrontier::bitmap_type;
  extern __shared__ unsigned char smem[];
  __shared__ uint32_t local_size;
  __shared__ uint32_t global_offset;

  int* local_offsets = reinterpret_cast<int*>(smem);

  if (threadIdx.x == 0) {
    local_size = 0;
    global_offset = 0;
  }
  __syncthreads();

  size_t gid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (gid < level_size) {
    bitmap_type data = bitmap.getData(1)[gid];
    const bitmap_type max_value = static_cast<bitmap_type>(~static_cast<bitmap_type>(0));
    for (uint16_t bit = 0; bit < range; ++bit) {
      bool is_active = (data & (static_cast<bitmap_type>(1) << bit)) != 0;
      if ((!invert && !is_active)
          || (invert && (is_active && bitmap.getData(0)[bit + (gid * range)] == max_value))) {
        continue;
      }
      uint32_t slot = atomicAdd(&local_size, 1);
      local_offsets[slot] = static_cast<int>(bit + (gid * range));
    }
  }

  __syncthreads();

  uint32_t block_count = local_size;
  if (threadIdx.x == 0 && block_count > 0) {
    global_offset = atomicAdd(bitmap.getOffsetsSize(), block_count);
  }
  __syncthreads();

  if (block_count == 0) {
    return;
  }

  for (uint32_t i = threadIdx.x; i < block_count; i += blockDim.x) {
    bitmap.getOffsets()[global_offset + i] = local_offsets[i];
  }
}
} // namespace detail

template<typename T, size_t Levels>
FrontierMLB<T, Levels>::FrontierMLB(size_t num_elems) : _bitmap(num_elems) {
  bitmap_type* ptr[Levels];
#pragma unroll
  for (size_t i = 0; i < Levels; i++) {
    CUDA_CHECK(cudaMalloc(&ptr[i], _bitmap.getBitmapSize(i) * sizeof(bitmap_type)));
    CUDA_CHECK(cudaMemset(ptr[i], 0, _bitmap.getBitmapSize(i) * sizeof(bitmap_type)));
  }
  int* offsets;
  CUDA_CHECK(cudaMalloc(&offsets, _bitmap.getBitmapSize() * sizeof(int)));
  CUDA_CHECK(cudaMemset(offsets, 0, _bitmap.getBitmapSize() * sizeof(int)));
  uint32_t* offsets_size;
  CUDA_CHECK(cudaMalloc(&offsets_size, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(offsets_size, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaHostAlloc(&_host_offsets_size, sizeof(uint32_t), cudaHostAllocPortable));
  *_host_offsets_size = 0;

  _bitmap.setData(ptr);
  _bitmap.setOffsets(offsets);
  _bitmap.setOffsetsSize(offsets_size);
}

template<typename T, size_t Levels>
FrontierMLB<T, Levels>::~FrontierMLB() {
#pragma unroll
  for (size_t i = 0; i < Levels; i++) {
    CUDA_CHECK(cudaFree(_bitmap.getData(i)));
  }
  CUDA_CHECK(cudaFree(_bitmap.getOffsets()));
  _bitmap.setOffsets(nullptr);
  CUDA_CHECK(cudaFree(_bitmap.getOffsetsSize()));
  _bitmap.setOffsetsSize(nullptr);
  if (_host_offsets_size != nullptr) {
    CUDA_CHECK(cudaFreeHost(_host_offsets_size));
    _host_offsets_size = nullptr;
  }
}

template<typename T, size_t Levels>
bool FrontierMLB<T, Levels>::empty() const {
  auto bitmap = this->getDeviceFrontier();
  size_t bitmap_size = bitmap.getBitmapSize(Levels - 1);

  thrust::device_ptr<bitmap_type> dev_ptr(bitmap.getData(Levels - 1));
  clutra::profile::KernelProfiler profiler("emptyKernel", "core");
  bitmap_type result = thrust::reduce(
      thrust::device,
      dev_ptr,
      dev_ptr + bitmap_size);
  profiler.stop();
  return !result;
}

template<typename T, size_t Levels>
bool FrontierMLB<T, Levels>::check(size_t idx) const {
  auto bitmap = this->getDeviceFrontier();
  bool* d_result;
  bool h_result;
  clutra::profile::KernelProfiler profiler("checkKernel", "operational");
  CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));

  clutra::detail::kernels::executeKernel<<<1, 1>>>([=, d_result = d_result] __device__() { *d_result = bitmap.check(idx); });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_result));
  profiler.stop();
  return h_result;
}

template<typename T, size_t Levels>
bool FrontierMLB<T, Levels>::insert(size_t idx) {
  auto bitmap = this->getDeviceFrontier();
  clutra::profile::KernelProfiler profiler("insertKernel", "operational");
  clutra::detail::kernels::executeKernel<<<1, 1>>>([=] __device__() { bitmap.insert(idx); });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
  return true;
}

template<typename T, size_t Levels>
bool FrontierMLB<T, Levels>::remove(size_t idx) {
  auto bitmap = this->getDeviceFrontier();
  clutra::profile::KernelProfiler profiler("removeKernel", "operational");
  clutra::detail::kernels::executeKernel<<<1, 1>>>([=] __device__() { bitmap.remove(idx); });
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
  return true;
}

template<typename T, size_t Levels>
size_t FrontierMLB<T, Levels>::size() const {
  auto bitmap = this->getDeviceFrontier();
  size_t frontier_size = this->getBitmapSize();

  auto count_bits_functor = [] __device__(bitmap_type val) -> size_t {
    size_t count = 0;
    while (val) {
      count += val & 1;
      val >>= 1;
    }
    return count;
  };

  clutra::profile::KernelProfiler profiler("sizeKernel", "operational");
  thrust::device_ptr<bitmap_type> dev_ptr(bitmap.getData());
  size_t result = thrust::transform_reduce(
      thrust::device,
      dev_ptr,
      dev_ptr + frontier_size,
      count_bits_functor,
      static_cast<size_t>(0),
      thrust::plus<size_t>());

  profiler.stop();
  return result;
}

template<typename T, size_t Levels>
FrontierMLB<T, Levels>& FrontierMLB<T, Levels>::operator=(const FrontierMLB& other) {
  if (this == &other) {
    return *this;
  }
  for (size_t i = 0; i < Levels; i++) {
    CUDA_CHECK(cudaMemcpy(_bitmap.getData(i), other._bitmap.getData(i), _bitmap.getBitmapSize(i) * sizeof(bitmap_type), cudaMemcpyDeviceToDevice));
  }
  if (_host_offsets_size != nullptr && other._host_offsets_size != nullptr) {
    *_host_offsets_size = *other._host_offsets_size;
  }
  return *this;
}

template<typename T, size_t Levels>
void FrontierMLB<T, Levels>::merge(FrontierMLB<T>& other) {
  clutra::profile::KernelProfiler profiler("mergeLevelKernel", "operational");
  for (size_t level = 0; level < Levels; ++level) {
    size_t n = _bitmap.getBitmapSize(level);
    if (n == 0) {
      continue;
    }
    bitmap_type* dst = _bitmap.getData(level);
    const bitmap_type* src = other._bitmap.getData(level);

    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks <= 0) {
      continue;
    }

    detail::mergeLevelKernel<<<blocks, threads>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

template<typename T, size_t Levels>
void FrontierMLB<T, Levels>::intersect(FrontierMLB<T>& other) {
  clutra::profile::KernelProfiler profiler("computeActiveFrontierKernel", "operational");
  for (size_t level = 0; level < Levels; ++level) {
    size_t n = _bitmap.getBitmapSize(level);
    if (n == 0) {
      continue;
    }
    bitmap_type* dst = _bitmap.getData(level);
    const bitmap_type* src = other._bitmap.getData(level);

    const int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    if (blocks <= 0) {
      continue;
    }

    detail::intersectMLBFrontierKernel<<<blocks, threads>>>(dst, src, n);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

template<typename T, size_t Levels>
void FrontierMLB<T, Levels>::clear() {
  clutra::profile::KernelProfiler profiler("clearKernel", "core");
#pragma unroll
  for (size_t i = 0; i < Levels; i++) {
    CUDA_CHECK(cudaMemset(_bitmap.getData(i), 0, _bitmap.getBitmapSize(i) * sizeof(bitmap_type)));
  }
  CUDA_CHECK(cudaMemset(_bitmap.getOffsets(), 0, _bitmap.getBitmapSize() * sizeof(int)));
  CUDA_CHECK(cudaMemset(_bitmap.getOffsetsSize(), 0, sizeof(uint32_t)));
  if (_host_offsets_size != nullptr) { *_host_offsets_size = 0; }
  profiler.stop();
}

template<typename T, size_t Levels>
void FrontierMLB<T, Levels>::computeActiveFrontier(bool invert) const {
  auto bitmap = this->getDeviceFrontier();
  size_t level_size = bitmap.getBitmapSize(1);
  uint32_t range = bitmap.getBitmapRange();

  CUDA_CHECK(cudaMemset(bitmap.getOffsetsSize(), 0, sizeof(uint32_t)));
  if (level_size == 0 || range == 0) {
    if (_host_offsets_size != nullptr) { *_host_offsets_size = 0; }
    return;
  }

  const int threads = 256;
  int blocks = static_cast<int>((level_size + threads - 1) / threads);
  int shared_mem_size = threads * range * sizeof(int); // each thread can store up to 'range' offsets
  clutra::profile::KernelProfiler profiler("computeActiveFrontierKernel", "core");
  detail::computeActiveFrontierKernel<<<blocks, threads, shared_mem_size>>>(bitmap, level_size, range, invert);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  profiler.stop();
}

template<typename T, size_t Levels>
size_t FrontierMLB<T, Levels>::getActiveFrontierSize() const {
  auto bitmap = this->getDeviceFrontier();
  clutra::profile::KernelProfiler profiler("getActiveFrontierSize", "core");
  if (_host_offsets_size != nullptr) {
    CUDA_CHECK(cudaMemcpy(_host_offsets_size, bitmap.getOffsetsSize(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
  }
  uint32_t value = (_host_offsets_size != nullptr ? *_host_offsets_size : 0);
  profiler.stop();
  return static_cast<size_t>(value);
}

// Explicit instantiation(s) for commonly used template arguments
template class FrontierMLB<uint32_t, 2>;
template class FrontierMLB<uint64_t, 2>;

} // namespace clutra::frontier
