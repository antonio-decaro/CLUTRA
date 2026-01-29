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
  /**
   * @brief Returns true if intra-cluster stealing is enabled and available.
   */
  __forceinline__ __device__ bool isStealingEnabled() const {return config.intra_cluster_stealing_enabled;}

  /**
   * @brief Returns the preferred cluster size for stealing.
   */
  __forceinline__ __device__ int getPreferredClusterSize() const {return config.preferred_cluster_size;}

  /**
   * @brief Returns the stealing chunk size.
   */
  __forceinline__ __device__ int getStealingChunkSize() const {return config.stealing_chunk_size;}

  template <size_t BlockSize>
  /**
   * @brief Per-block shared state for the stealer implementation.
   */
  struct SharedState {};

  template <size_t BlockSize>
  /**
   * @brief Initialize stealer shared state and cluster queue mapping.
   * @param local_queue Pointer to this block's CTA queue.
   * @param state Pointer to the stealer shared state.
   * @note Expected to be called once per block before the main loop.
   */
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>*,
                       SharedState<BlockSize>&) const;

  template <size_t BlockSize>
  /**
   * @brief Attempt to steal a chunk from a victim block.
   * @param state Pointer to the stealer shared state.
   * @param chunk_size Requested number of items to steal (implementation may clamp).
   * @return 0 if no work was stolen; otherwise the number of items stolen.
   * @note This method is responsible for selecting the victim and performing atomics.
   */
  __device__ int attemptStealing(SharedState<BlockSize>&,
                                 int) const;

  template <size_t BlockSize>
  /**
   * @brief Read the i-th stolen item (vertex, degree) from the current victim.
   * @param state Pointer to the stealer shared state.
   * @param i Index within the stolen chunk [0, steal_count).
   * @param vertex Output vertex.
   * @param degree Output degree.
   * @note Calling this before a successful attemptStealing results in a no-op.
   */
  __device__ void steal(SharedState<BlockSize>&,
                        int,
                        uint32_t&,
                        uint32_t&) const;

  template <size_t BlockSize>
  /**
   * @brief Finalize stealing for the block (e.g., cluster sync).
   */
  __device__ void finalize() const;
};

struct NullStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;

  template <size_t BlockSize>
  /**
   * @brief No-op shared state for NullStealerDevice.
   */
  struct SharedState : StealerDevice::SharedState<BlockSize> {};

  template <size_t BlockSize>
  /**
   * @brief No-op init for NullStealerDevice.
   */
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                       SharedState<BlockSize>& state) const {}

  template <size_t BlockSize>
  /**
   * @brief No-op stealing attempt for NullStealerDevice.
   */
  __device__ int attemptStealing(SharedState<BlockSize>& state,
                                 int chunk_size) const {return 0;}

  template <size_t BlockSize>
  /**
   * @brief No-op steal for NullStealerDevice.
   */
  __device__ void steal(SharedState<BlockSize>& state,
                        int i,
                        uint32_t&,
                        uint32_t&) const {}

  template <size_t BlockSize>
  /**
   * @brief No-op finalize for NullStealerDevice.
   */
  __device__ void finalize() const {}
};

struct BasicStealerDevice : StealerDevice {
  using StealerDevice::StealerDevice;
  template <size_t BlockSize>
  /**
   * @brief Shared state for BasicStealerDevice (victim info + cluster queues).
   */
  struct SharedState : StealerDevice::SharedState<BlockSize> {
    uint32_t victim_rank;
    uint32_t steal_count;
    uint32_t steal_tail;
    clutra::detail::utils::SharedQueue<BlockSize>* cluster_queues[8];
  };

  template <size_t BlockSize>
  /**
   * @brief Initialize cluster queue mapping for BasicStealerDevice.
   */
  __device__ void init(clutra::detail::utils::SharedQueue<BlockSize>* local_queue,
                       SharedState<BlockSize>& state) const;

  template <size_t BlockSize>
  /**
   * @brief Attempt to steal a chunk from another block.
   */
  __device__ int attemptStealing(SharedState<BlockSize>& state,
                                 int chunk_size) const;

  template <size_t BlockSize>
  /**
   * @brief Fetch the i-th stolen item from the current victim.
   * @note Calling this before a successful attemptStealing results in a no-op.
   */
  __device__ void steal(SharedState<BlockSize>& state,
                        int i,
                        uint32_t&,
                        uint32_t&) const;

  template <size_t BlockSize>
  /**
   * @brief Finalize stealing for the block (e.g., cluster sync).
   */
  __device__ void finalize() const;
};

} // namespace clutra::stealer
