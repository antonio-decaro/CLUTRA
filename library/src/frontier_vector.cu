/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#include <frontier/frontier.cuh>
#include <graph/graph.cuh>
#include <utils/profile.cuh>
#include <stdexcept>
#include <type_traits>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <algorithm>

namespace clutra::frontier {

VectorFrontier::VectorFrontier(size_t capacity) : _vector_frontier(capacity) {
  CUDA_CHECK(cudaMallocHost(&_host_frontier_size, sizeof(uint32_t), cudaHostAllocPortable));
  *_host_frontier_size = 0;

  uint32_t* device_size_ptr = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&device_size_ptr, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_size_ptr, _host_frontier_size, sizeof(uint32_t), cudaMemcpyHostToDevice));

  _vector_frontier.setSize(device_size_ptr);

  uint32_t* device_data_ptr = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&device_data_ptr, capacity * sizeof(uint32_t)));
  _vector_frontier.setData(device_data_ptr);
}

VectorFrontier::VectorFrontier(const VectorFrontier& other) : VectorFrontier(other.getCapacity()) {
  CUDA_CHECK(cudaMemcpy(_host_frontier_size, other._host_frontier_size, sizeof(uint32_t), cudaMemcpyHostToHost));
  CUDA_CHECK(cudaMemcpy(_vector_frontier.getData(), other._vector_frontier.getData(),
                        other.getCapacity() * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(_vector_frontier._size, other._vector_frontier._size,
                        sizeof(uint32_t), cudaMemcpyDeviceToDevice));
}

VectorFrontier::VectorFrontier(VectorFrontier&& other) noexcept
    : _vector_frontier(other._vector_frontier), _host_frontier_size(other._host_frontier_size) {
  other._vector_frontier = detail::VectorFrontierDevice();
  other._host_frontier_size = nullptr;
}

VectorFrontier::~VectorFrontier() {
  if (_vector_frontier.getData() != nullptr) {
    CUDA_CHECK(cudaFree(_vector_frontier.getData()));
    _vector_frontier.setData(nullptr);
  }
  if (_vector_frontier._size != nullptr) {
    CUDA_CHECK(cudaFree(_vector_frontier._size));
    _vector_frontier.setSize(nullptr);
  }
  if (_host_frontier_size != nullptr) {
    CUDA_CHECK(cudaFreeHost(_host_frontier_size));
    _host_frontier_size = nullptr;
  }
}

size_t VectorFrontier::size() const {
  CUDA_CHECK(cudaMemcpy(_host_frontier_size, _vector_frontier._size, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  return static_cast<size_t>(*_host_frontier_size);
};

bool VectorFrontier::empty() const {
  return size() == 0;
}

bool VectorFrontier::insert(size_t idx) {
  size_t current_size = size();
  if (current_size >= getCapacity()) {
    return false;
  }
  CUDA_CHECK(cudaMemcpy(_vector_frontier.getData() + current_size, &idx, sizeof(uint32_t), cudaMemcpyHostToDevice));
  uint32_t new_size = static_cast<uint32_t>(current_size + 1);
  CUDA_CHECK(cudaMemcpy(_vector_frontier._size, &new_size, sizeof(uint32_t), cudaMemcpyHostToDevice));
  return true;
}

void VectorFrontier::clear() {
  uint32_t zero = 0;
  CUDA_CHECK(cudaMemcpy(_vector_frontier._size, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));
}

size_t VectorFrontier::getCapacity() const {
  return static_cast<size_t>(_vector_frontier.getCapacity());
}

void VectorFrontier::invalidateDuplicates() {
  // Using thrust to remove duplicates
  thrust::device_ptr<uint32_t> dev_ptr(_vector_frontier.getData());
  size_t current_size = size();

  // Sort the frontier
  thrust::sort(thrust::device, dev_ptr, dev_ptr + current_size);

  // Remove duplicates
  auto new_end = thrust::unique(thrust::device, dev_ptr, dev_ptr + current_size);
  size_t new_size = static_cast<size_t>(new_end - dev_ptr);

  // Update the size
  CUDA_CHECK(cudaMemcpy(_vector_frontier._size, &new_size, sizeof(uint32_t), cudaMemcpyHostToDevice));
  *_host_frontier_size = static_cast<uint32_t>(new_size);
}

VectorFrontier& VectorFrontier::operator=(const VectorFrontier& other) {
  if (this != &other) {
    VectorFrontier temp(other);
    VectorFrontier::swap(*this, temp);
  }
  return *this;
}

VectorFrontier& VectorFrontier::operator=(VectorFrontier&& other) noexcept {
  if (this != &other) {
    VectorFrontier::swap(*this, other);
  }
  return *this;
}

} // namespace clutra::frontier