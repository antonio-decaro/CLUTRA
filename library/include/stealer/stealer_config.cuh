/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>

namespace clutra::stealer {

/**
 * @brief Configuration for the Stealer component.
 * @details This structure holds configuration options for enabling or disabling
 * various stealing strategies such as thread stealing, block stealing, and grid stealing.
 * @note This object should be device-compatible.
 */
struct StealerConfig final {
  bool intra_cluster_stealing_enabled = false;
  int preferred_cluster_size = 4;
  int stealing_chunk_size = 16;
};

static_assert(std::is_trivially_copyable_v<StealerConfig>,
              "StealerConfig must be trivially copyable for device use");
static_assert(std::is_standard_layout_v<StealerConfig>,
              "StealerConfig should be standard layout");

} // namespace clutra::stealer
