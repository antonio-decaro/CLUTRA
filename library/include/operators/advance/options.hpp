/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace clutra::operators::advance::detail {

/**
 * @enum advance_direction
 * @brief Enumeration for advance operation direction.
 */
enum class advance_direction {
  push,
  pull
};

/**
 * @enum view
 * @brief Enumeration for view types in advance operation.
 */
enum class view {
  frontier,
  graph,
};

}
