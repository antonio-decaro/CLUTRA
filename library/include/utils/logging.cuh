#pragma once

#include <cstdlib>
#include <fmt/core.h>
#include <iostream>
#include <string_view>
#include <utility>

namespace clutra::detail {

template <typename... Args>
inline void log(std::string_view message, Args&&... args) {
  if (std::getenv("CLUTRA_DEBUG") == nullptr) {
    return;
  }

  std::cerr << "\x1b[35m" << fmt::format(fmt::runtime(message), std::forward<Args>(args)...) << "\x1b[0m" << std::endl;
}
}  // namespace clutra::detail
