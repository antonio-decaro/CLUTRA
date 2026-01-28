#pragma once

#include <fmt/core.h>
#include <iostream>
#include <string_view>
#include <utility>

namespace clutra::detail {

template<typename... Args>
inline void log(std::string_view message, Args&&... args) {
#if defined(CLUTRA_DEBUG_LOG) && CLUTRA_DEBUG_LOG
  std::cerr << "\x1b[35m"
            << fmt::format(fmt::runtime(message), std::forward<Args>(args)...)
            << "\x1b[0m" << std::endl;
#else
  (void)message;
  (void)sizeof...(args);
#endif
}
} // namespace clutra::detail
