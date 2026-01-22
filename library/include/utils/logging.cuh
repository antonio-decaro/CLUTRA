#pragma once

#include <spdlog/spdlog.h>

// Debug logging macro, compiled out when SPDLOG_ACTIVE_LEVEL disables debug.
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
#define CLUTRA_LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#else
#define CLUTRA_LOG_DEBUG(...) ((void)0)
#endif

// Initialize runtime logging level in Debug builds (host code only).
inline void clutraInitLogging() {
#if SPDLOG_ACTIVE_LEVEL <= SPDLOG_LEVEL_DEBUG
  spdlog::set_level(spdlog::level::debug);
#endif
}
