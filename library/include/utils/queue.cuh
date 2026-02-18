/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <stdexcept>
#include <utils/atomic.cuh>

namespace clutra::detail::utils {

template <size_t Capacity>
struct SharedQueue {
  int tail;
  int head;
  uint32_t vertices[Capacity];
  uint32_t degrees[Capacity];

  __host__ static size_t getSizeInBytes() { return sizeof(SharedQueue<Capacity>); }

  __device__ void init() {
    tail = 0;
    head = 0;
  }

  __forceinline__ __device__ int push(uint32_t vertex, uint32_t degree) {
    const int loc = atomicAdd(&tail, 1);
    vertices[loc] = vertex;
    degrees[loc] = degree;
    return loc;
  }

  __forceinline__ __device__ bool pop(uint32_t& vertex, uint32_t& degree) {
    cuda::atomic_ref<int, cuda::thread_scope_device> tail_ref(tail);
    const int tail_snapshot = tail_ref.load(cuda::memory_order_relaxed);

    if (head < 0 || head >= tail_snapshot) {
      return false;  // empty
    }

    vertex = vertices[head];
    degree = degrees[head];

    if (threadIdx.x == 0) {
      atomicAdd(&head, 1);
    }
    return true;
  }

  __device__ int size() const { return tail; }
};

template <typename T, typename LockType>
struct WorkQueueView {
  T* data;
  uint32_t* head;
  uint32_t* tail;
  LockType* lock;
  uint32_t capacity;

  __device__ __forceinline__ void push(const T& item) {
    auto pos = atomicAdd(tail, 1U);
    data[pos % capacity] = item;
  }

  __device__ __forceinline__ bool pop(T& out) {
    lock->acquire();
    if (*head >= *tail) {
      lock->release();
      return false;  // empty
    }
    auto pos = *head;
    (*head)++;
    lock->release();
    out = data[pos % capacity];

    return true;
  }

  __device__ __forceinline__ bool steal(T& out) { return false; }
};

template <typename T, clutra::detail::atomic::Lock LockType = clutra::detail::atomic::SpinLock>
class WorkQueue {
public:
  __host__ explicit WorkQueue(uint32_t capacity) : capacity_(capacity) {
    cudaMalloc(&data_, capacity_ * sizeof(T));
    cudaMalloc(&head_, sizeof(uint32_t));
    cudaMalloc(&tail_, sizeof(uint32_t));
    cudaMalloc(&lock_, sizeof(LockType));

    cudaMemset(head_, 0, sizeof(uint32_t));
    cudaMemset(tail_, 0, sizeof(uint32_t));
    cudaMemset(lock_, 0, sizeof(LockType));
  }

  __host__ ~WorkQueue() {
    cudaFree(data_);
    cudaFree(head_);
    cudaFree(tail_);
    cudaFree(lock_);
  }

  __host__ WorkQueueView<T, LockType> deviceView() const { return {data_, head_, tail_, lock_, capacity_}; }

private:
  uint32_t capacity_;
  T* data_;
  uint32_t* head_;
  uint32_t* tail_;
  LockType* lock_;
};

template <typename T, clutra::detail::atomic::Lock LockType = clutra::detail::atomic::SpinLock>
class ClusterWorkQueues {
public:
  __host__ ClusterWorkQueues(uint32_t cluster_count, uint32_t queue_capacity)
      : cluster_count_(cluster_count), queue_capacity_(queue_capacity) {
    if (cluster_count_ == 0 || queue_capacity_ == 0) {
      throw std::invalid_argument("ClusterWorkQueues requires cluster_count > 0 and queue_capacity > 0.");
    }

    const size_t total_capacity = static_cast<size_t>(cluster_count_) * queue_capacity_;

    cudaMalloc(&data_, total_capacity * sizeof(T));
    cudaMalloc(&head_, cluster_count_ * sizeof(uint32_t));
    cudaMalloc(&tail_, cluster_count_ * sizeof(uint32_t));
    cudaMalloc(&lock_, cluster_count_ * sizeof(LockType));
    cudaMalloc(&views_, cluster_count_ * sizeof(WorkQueueView<T, LockType>));

    cudaMemset(head_, 0, cluster_count_ * sizeof(uint32_t));
    cudaMemset(tail_, 0, cluster_count_ * sizeof(uint32_t));
    cudaMemset(lock_, 0, cluster_count_ * sizeof(LockType));

    WorkQueueView<T, LockType>* host_views = new WorkQueueView<T, LockType>[cluster_count_];
    for (uint32_t cluster = 0; cluster < cluster_count_; ++cluster) {
      host_views[cluster] = {
          data_ + (static_cast<size_t>(cluster) * queue_capacity_),
          head_ + cluster,
          tail_ + cluster,
          lock_ + cluster,
          queue_capacity_,
      };
    }
    cudaMemcpy(views_, host_views, cluster_count_ * sizeof(WorkQueueView<T, LockType>), cudaMemcpyHostToDevice);
    delete[] host_views;
  }

  ClusterWorkQueues(const ClusterWorkQueues&) = delete;
  ClusterWorkQueues& operator=(const ClusterWorkQueues&) = delete;

  __host__ ClusterWorkQueues(ClusterWorkQueues&& other) noexcept
      : cluster_count_(other.cluster_count_), queue_capacity_(other.queue_capacity_), data_(other.data_),
        head_(other.head_), tail_(other.tail_), lock_(other.lock_), views_(other.views_) {
    other.cluster_count_ = 0;
    other.queue_capacity_ = 0;
    other.data_ = nullptr;
    other.head_ = nullptr;
    other.tail_ = nullptr;
    other.lock_ = nullptr;
    other.views_ = nullptr;
  }

  __host__ ClusterWorkQueues& operator=(ClusterWorkQueues&& other) noexcept {
    if (this == &other) {
      return *this;
    }

    cudaFree(data_);
    cudaFree(head_);
    cudaFree(tail_);
    cudaFree(lock_);
    cudaFree(views_);

    cluster_count_ = other.cluster_count_;
    queue_capacity_ = other.queue_capacity_;
    data_ = other.data_;
    head_ = other.head_;
    tail_ = other.tail_;
    lock_ = other.lock_;
    views_ = other.views_;

    other.cluster_count_ = 0;
    other.queue_capacity_ = 0;
    other.data_ = nullptr;
    other.head_ = nullptr;
    other.tail_ = nullptr;
    other.lock_ = nullptr;
    other.views_ = nullptr;
    return *this;
  }

  __host__ ~ClusterWorkQueues() {
    cudaFree(data_);
    cudaFree(head_);
    cudaFree(tail_);
    cudaFree(lock_);
    cudaFree(views_);
  }

  __host__ WorkQueueView<T, LockType>* deviceViews() const { return views_; }

private:
  uint32_t cluster_count_{0};
  uint32_t queue_capacity_{0};
  T* data_{nullptr};
  uint32_t* head_{nullptr};
  uint32_t* tail_{nullptr};
  LockType* lock_{nullptr};
  WorkQueueView<T, LockType>* views_{nullptr};
};

template <typename T, typename LockType>
__device__ __forceinline__ WorkQueueView<T, LockType>&
getCurrentClusterQueueView(WorkQueueView<T, LockType>* cluster_views) {
#if __CUDA_ARCH__ >= 900
  auto cluster = cooperative_groups::this_cluster();
  const uint32_t cluster_idx = static_cast<uint32_t>(blockIdx.x / cluster.dim_blocks().x);
#else
  const uint32_t cluster_idx = static_cast<uint32_t>(blockIdx.x);
#endif
  return cluster_views[cluster_idx];
}

}  // namespace clutra::detail::utils
