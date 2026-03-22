/*
 * Copyright (c) 2026 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <map>
#include <mutex>
#include <string>
#include <utils/misc.cuh>
#include <vector>

namespace clutra::profile {

class KernelProfilerManager {
public:
  static KernelProfilerManager& instance() {
    static KernelProfilerManager manager;
    return manager;
  }

  void record(const char* label, const char* stage, float ms) {
    std::lock_guard<std::mutex> lock(_mutex);
    const std::string stage_name = stage ? stage : "default";
    const std::string kernel_name = label ? label : "kernel";
    StageStat& stage_stats = _stats[stage_name];
    KernelStat& stat = stage_stats.kernels[kernel_name];
    stat.count += 1;
    stat.total_ms += ms;
    stat.events_ms.push_back(ms);
    stage_stats.total_ms += ms;
  }

  void reset() {
    std::lock_guard<std::mutex> lock(_mutex);
    _stats.clear();
  }

  void printSummary(bool detail = false) const {
    std::lock_guard<std::mutex> lock(_mutex);
    std::printf("==== CUDA Kernel Profiling Summary ====\n");
    double grand_total = 0.0;
    for (const auto& stage_pair : _stats) {
      const std::string& stage = stage_pair.first;
      const StageStat& stage_stat = stage_pair.second;
      std::printf("-- Stage: %s (total: %.3f ms) --\n", stage.c_str(), stage_stat.total_ms);
      for (const auto& kernel_pair : stage_stat.kernels) {
        const std::string& kernel = kernel_pair.first;
        const KernelStat& stat = kernel_pair.second;
        if (detail) {
          std::printf("  %s -> runs: %llu, events: ", kernel.c_str(), static_cast<unsigned long long>(stat.count));
          for (size_t i = 0; i < stat.events_ms.size(); ++i) {
            std::printf("%.3f ms%s", stat.events_ms[i], (i + 1 < stat.events_ms.size()) ? ", " : "");
          }
          std::printf("\n");
        } else {
          double avg = stat.count ? (stat.total_ms / static_cast<double>(stat.count)) : 0.0;
          std::printf("  %s -> runs: %llu, total: %.3f ms, avg: %.3f ms\n", kernel.c_str(),
                      static_cast<unsigned long long>(stat.count), stat.total_ms, avg);
        }
      }
      grand_total += stage_stat.total_ms;
    }
    std::printf("Grand total time: %.3f ms\n", grand_total);
    std::printf("=======================================\n");
  }

private:
  struct KernelStat {
    uint64_t count = 0;
    double total_ms = 0.0;
    std::vector<double> events_ms;
  };

  struct StageStat {
    std::map<std::string, KernelStat> kernels;
    double total_ms = 0.0;
  };

  mutable std::mutex _mutex;
  std::map<std::string, StageStat> _stats;
};

class KernelProfiler {
public:
  KernelProfiler(const char* label, const char* stage = "default") : _label(label), _stage(stage), _stopped(false) {
    CUDA_CHECK(cudaEventCreate(&_start));
    CUDA_CHECK(cudaEventCreate(&_stop));
    CUDA_CHECK(cudaEventRecord(_start));
  }

  ~KernelProfiler() {
    stop();
    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_stop));
  }

  void stop() {
    if (_stopped) {
      return;
    }
    CUDA_CHECK(cudaEventRecord(_stop));
    CUDA_CHECK(cudaEventSynchronize(_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, _start, _stop));
    KernelProfilerManager::instance().record(_label ? _label : "kernel", _stage ? _stage : "default", ms);
    _stopped = true;
  }

private:
  const char* _label;
  const char* _stage;
  bool _stopped;
  cudaEvent_t _start{};
  cudaEvent_t _stop{};
};

}  // namespace clutra::profile
