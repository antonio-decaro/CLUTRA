/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdint>

namespace clutra::detail::types {

using bitmap_type_t = uint32_t;
using index_t = uint32_t;
using offset_t = uint32_t;
constexpr std::size_t byte_size = 8;
constexpr std::size_t CU_SIZE = 256;

}