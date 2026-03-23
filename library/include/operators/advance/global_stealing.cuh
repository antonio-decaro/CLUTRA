/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <frontier/frontier.cuh>
#include <graph/concept.hpp>
#include <graph/graph.cuh>
#include <operators/advance/options.hpp>
#include <stealer/stealer.cuh>
#include <utils/device.cuh>
#include <utils/kernel_launcher.cuh>
#include <utils/logging.cuh>
#include <utils/profile.cuh>

#if __CUDA_ARCH__ >= 1000
#define CLUTRA_HAS_PTX_CLUSTER_LAUNCH_CONTROL_API 1
#endif

#if __CUDA_ARCH__ >= 1000
#include <cuda/ptx>
#endif

namespace clutra::operators::advance::detail {

constexpr uint32_t ADVANCE_WARP_SIZE = 32;

#if __CUDA_ARCH__ >= 1000
namespace ptx = cuda::ptx;
#endif

__host__ __device__ constexpr bool hasPtxClusterLaunchControlApi() {
#if __CUDA_ARCH__ >= 1000
  return true;
#else
  return false;
#endif
}

__device__ __forceinline__ void initClusterLaunchControl(uint64_t& mbarrier) {
#if __CUDA_ARCH__ >= 1000
  auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() == 0) {
    ptx::mbarrier_init(&mbarrier, 1);
    ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
  }
  __syncthreads();
#else
  (void)mbarrier;
#endif
}

__device__ __forceinline__ bool
tryAcquireCanceledCta(uint4& result, uint64_t& mbarrier, int& phase, uint32_t& canceled_cta_x) {
#if __CUDA_ARCH__ >= 1000
  auto block = cooperative_groups::this_thread_block();
  auto cluster = cooperative_groups::this_cluster();

  // Keep cancellation responses serialized across the cluster and avoid races
  // between generic/shared reads and async-proxy writes to `result`.
  cluster.sync();

  if (cluster.thread_rank() == 0) {
    ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_acquire, ptx::space_cluster, ptx::scope_cluster);
    cooperative_groups::invoke_one(cooperative_groups::coalesced_threads(),
                                   [&]() { ptx::clusterlaunchcontrol_try_cancel_multicast(&result, &mbarrier); });
  }

  if (block.thread_rank() == 0) {
    ptx::mbarrier_arrive_expect_tx(ptx::sem_relaxed, ptx::scope_cluster, ptx::space_shared, &mbarrier,
                                   static_cast<uint32_t>(sizeof(uint4)));
  }

  while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cluster, &mbarrier, phase)) {}
  phase ^= 1;

  const bool success = ptx::clusterlaunchcontrol_query_cancel_is_canceled(result);
  if (!success) {
    return false;
  }

  canceled_cta_x = static_cast<uint32_t>(ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(result));
  canceled_cta_x += static_cast<uint32_t>(cluster.block_rank());

  ptx::fence_proxy_async_generic_sync_restrict(ptx::sem_release, ptx::space_shared, ptx::scope_cluster);
  return true;
#else
  (void)result;
  (void)mbarrier;
  (void)phase;
  (void)canceled_cta_x;
  return false;
#endif
}

}  // namespace clutra::operators::advance::detail
