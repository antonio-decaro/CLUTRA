/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <stealer/stealer_config.cuh>
#include <utils/queue.cuh>

namespace cg = cooperative_groups;

namespace clutra::stealer {

struct StealerDevice {
  StealerConfig config{};

  StealerDevice() = default;
  StealerDevice(const StealerConfig& cfg) : config(cfg) {}
  __device__ bool isStealingEnabled() const;

  template <size_t BlockSize>
  struct SharedState {};

  template <size_t BlockSize>
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>*,
                       SharedState<BlockSize>*) const;

  template <size_t BlockSize>
  __device__ int attemptStealing(SharedState<BlockSize>*,
                                 int) const;

  template <size_t BlockSize>
  __device__ void steal(SharedState<BlockSize>*,
                        int,
                        uint32_t*,
                        uint32_t*) const;

  template <size_t BlockSize>
  __device__ void finalize() const;
};

struct NullStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;

  template <size_t BlockSize>
  struct SharedState : StealerDevice::SharedState<BlockSize> {};

  template <size_t BlockSize>
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                       SharedState<BlockSize>* state) const {}

  template <size_t BlockSize>
  __device__ int attemptStealing(SharedState<BlockSize>* state,
                                 int chunk_size) const {return 0;}

  template <size_t BlockSize>
  __device__ void steal(SharedState<BlockSize>* state,
                        int i,
                        uint32_t*,
                        uint32_t*) const {}

  template <size_t BlockSize>
  __device__ void finalize() const {}
};

struct BasicStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;
  template <size_t BlockSize>
  struct SharedState : StealerDevice::SharedState<BlockSize> {
    int victim_rank;
    int steal_count;
    int steal_tail;
    clutra::detail::utils::SharedQueue<BlockSize>* cluster_queues[4];
  };

  template <size_t BlockSize>
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                       SharedState<BlockSize>* state) const;

  template <size_t BlockSize>
  __device__ int attemptStealing(SharedState<BlockSize>* state,
                                 int chunk_size) const;

  template <size_t BlockSize>
  __device__ void steal(SharedState<BlockSize>* state,
                        int i,
                        uint32_t*,
                        uint32_t*) const;

  template <size_t BlockSize>
  __device__ void finalize() const;
};

} // namespace clutra::stealer
